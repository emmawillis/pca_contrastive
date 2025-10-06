#!/usr/bin/env python3
# train_min_coral.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_picai_slices import PicaiSliceDataset
from ISUPMedSAM import MedSAMSliceSpatialAttn   # make sure this import matches your module path
from segment_anything import sam_model_registry

# --- optional: AUC ---
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

LABEL_NAMES = ["ISUP01", "ISUP23", "ISUP45"]  # c0,c1,c2

# ----------------- helpers -----------------
def collate_resize_1024(batch):
    imgs, labels = [], []
    extras_keys = [k for k in batch[0].keys() if k not in ("image", "label")]
    extras = {k: [] for k in extras_keys}
    for s in batch:
        x = s["image"].unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False).squeeze(0)
        imgs.append(x)
        labels.append(torch.as_tensor(s["label"], dtype=torch.long))
        for k in extras_keys:
            extras[k].append(s[k])
    return {"image": torch.stack(imgs, 0),
            "label": torch.stack(labels, 0),
            **extras}

def map_isup3(y6: int) -> int:
    if y6 in (0,1): return 0
    if y6 in (2,3): return 1
    if y6 in (4,5): return 2
    raise ValueError(f"bad label6={y6}")

def class_weights_from_train(df: pd.DataFrame, target: str):
    """Return torch.FloatTensor of class weights (mean-normalized)."""
    y = df["label6"].map(map_isup3) if target == "isup3" else df["label6"]
    classes = sorted(int(c) for c in y.unique())
    cnt = Counter(int(v) for v in y.tolist())
    K, N = len(classes), len(y)
    ws = [N / (K * max(1, cnt.get(c, 0))) for c in classes]
    m = sum(ws)/len(ws)
    ws = [w/m for w in ws]
    return torch.tensor(ws, dtype=torch.float32), classes

# Oversample slices that intersect lesions (informative slices).
def make_pos_sampler(df: pd.DataFrame, pos_ratio: float = 0.33, seed: int | None = None):
    is_pos = df["has_lesion"].astype(int).values
    n_pos = int(is_pos.sum()); n_neg = len(is_pos) - n_pos
    assert n_pos > 0, "No positive slices in train folds."
    w_neg = 1.0
    w_pos = (pos_ratio/(1-pos_ratio)) * (n_neg/max(1,n_pos))
    w = np.where(is_pos==1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)

def macro_f1_from_probs(probs: torch.Tensor, y: torch.Tensor) -> float:
    y_pred = probs.argmax(dim=1).cpu().numpy()
    y_true = y.cpu().numpy()
    K = probs.shape[1]
    f1s = []
    for c in range(K):
        tp = ((y_pred==c)&(y_true==c)).sum()
        fp = ((y_pred==c)&(y_true!=c)).sum()
        fn = ((y_pred!=c)&(y_true==c)).sum()
        prec = tp/(tp+fp) if tp+fp>0 else 0.0
        rec  = tp/(tp+fn) if tp+fn>0 else 0.0
        f1s.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
    return float(np.mean(f1s))

def per_class_metrics_from_probs(probs: torch.Tensor, y: torch.Tensor):
    """
    Args:
      probs: (N,K) probabilities (sum to 1 across classes)
      y:     (N,) int labels
    Returns:
      accs: dict {class_idx: acc}
      aucs: dict {class_idx: auc or nan}
      macro_auc: float or None
    """
    K = probs.shape[1]
    y_np = y.cpu().numpy()
    y_pred = probs.argmax(dim=1).cpu().numpy()

    # per-class accuracy
    accs = {}
    for c in range(K):
        mask = (y_np == c)
        accs[c] = float((y_pred[mask] == c).mean()) if mask.sum() > 0 else float("nan")

    aucs = {}
    macro_auc = None
    if _HAS_SK:
        probs_np = probs.detach().cpu().numpy()
        auc_vals = []
        for c in range(K):
            y_bin = (y_np == c).astype(np.int32)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                try:
                    auc = roc_auc_score(y_bin, probs_np[:, c])
                    aucs[c] = float(auc)
                    auc_vals.append(auc)
                except Exception:
                    aucs[c] = float("nan")
            else:
                aucs[c] = float("nan")
        if len(auc_vals) > 0:
            macro_auc = float(np.nanmean(auc_vals))
    else:
        aucs = {c: None for c in range(K)}
        macro_auc = None

    return accs, aucs, macro_auc

# ---- Confusion matrix (no sklearn needed) ----
def confusion_matrix_from_probs(probs: torch.Tensor, y: torch.Tensor, K: int = 3) -> torch.Tensor:
    """Return KxK matrix with counts; rows=true, cols=pred."""
    preds = probs.argmax(dim=1).cpu()
    y_cpu = y.cpu()
    cm = torch.zeros((K, K), dtype=torch.long)
    for t, p in zip(y_cpu, preds):
        cm[t.long(), p.long()] += 1
    return cm

def print_confusion_matrix(cm: torch.Tensor, labels=LABEL_NAMES):
    K = cm.shape[0]
    header = "true\\pred " + " ".join(f"{labels[j]:>7}" for j in range(K))
    print("Confusion matrix (val): rows=true, cols=pred")
    print(header)
    for i in range(K):
        row = " ".join(f"{int(cm[i, j]):7d}" for j in range(K))
        print(f"{labels[i]:>9} {row}")

# ----------------- CORAL (ordinal) -----------------
class OrdinalHead(nn.Module):
    """Tiny ordinal head that turns an embedding (B,d) into K-1 logits for CORAL."""
    def __init__(self, d_in: int, K: int = 3, p_drop: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(d_in, 1)                       # scalar score z
        # Initialize ordered cutpoints for K=3 => 2 thresholds
        self.cutpoints = nn.Parameter(torch.tensor([-0.5, 0.5], dtype=torch.float))

    def forward(self, h: torch.Tensor):                    # h: (B, d_in)
        h = self.dropout(h)
        z = self.fc(h)                                     # (B,1)
        return z - self.cutpoints[None, :]                 # (B, K-1)

def coral_loss(ord_logits: torch.Tensor, y: torch.Tensor, K: int = 3):
    # ord_logits: (B, K-1), y in {0..K-1}
    y = y.view(-1, 1)
    targets = (y > torch.arange(K-1, device=y.device)).float()  # (B, K-1)
    return F.binary_cross_entropy_with_logits(ord_logits, targets)

def coral_to_probs(ord_logits: torch.Tensor) -> torch.Tensor:
    # ord_logits: (B,K-1) -> probs: (B,K) for K=3
    p_gt = torch.sigmoid(ord_logits)    # P(y>k)
    p_le = 1.0 - p_gt                   # P(y<=k)
    p0 = p_le[:, 0]
    p1 = p_gt[:, 0] * p_le[:, 1]
    p2 = p_gt[:, 1]
    probs = torch.stack([p0, p1, p2], dim=1)  # (B,3)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

# ----------------- train / val -----------------
@torch.no_grad()
def infer_embedding_dim(model: nn.Module, device: str) -> int:
    model.eval()
    dummy = torch.zeros(1, 3, 1024, 1024, device=device)
    logits0, emb0 = model(dummy)   # model returns (logits, emb)
    return int(emb0.shape[1])

def run_epoch(loader, model, ord_head, optimizer=None, device="cuda", return_outputs=False):
    train_mode = optimizer is not None
    model.train(train_mode)
    ord_head.train(train_mode)

    total_loss, total_correct, total_n = 0.0, 0, 0
    all_probs, all_y = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        # forward through MedSAM slice model to get embedding
        _, emb = model(x)                 # ignore the model's CE logits
        ord_logits = ord_head(emb)        # (B, K-1)
        loss = coral_loss(ord_logits, y, K=3)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(ord_head.parameters()), 1.0)
            optimizer.step()

        probs = coral_to_probs(ord_logits)  # (B, K)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += (probs.argmax(1) == y).sum().item()
        total_n += x.size(0)
        all_probs.append(probs.detach())
        all_y.append(y.detach())

    avg_loss = total_loss / max(1,total_n)
    acc = total_correct / max(1,total_n)
    probs_cat = torch.cat(all_probs) if all_probs else torch.empty(0)
    y_cat = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)
    f1 = macro_f1_from_probs(probs_cat, y_cat) if probs_cat.numel() else 0.0

    if return_outputs:
        return avg_loss, acc, f1, probs_cat, y_cat
    return avg_loss, acc, f1

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--outdir", default="./runs/simple_coral")
    p.add_argument("--target", choices=["isup3","isup6"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3") #holding back 4 as test set
    p.add_argument("--folds_val", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()!=""]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()!=""]

    # -------- dataset --------
    train_ds = PicaiSliceDataset(
        manifest_csv=args.manifest,
        folds=folds_train,
        use_skip=True,
        target=args.target,
        channels=("path_T2","path_ADC","path_HBV"),
        missing_channel_mode="zeros",
        pct_lower=0.5, pct_upper=99.5,
        cache_size=64,
    )
    val_ds = PicaiSliceDataset(
        manifest_csv=args.manifest,
        folds=folds_val,
        use_skip=True,
        target=args.target,
        channels=("path_T2","path_ADC","path_HBV"),
        missing_channel_mode="zeros",
        pct_lower=0.5, pct_upper=99.5,
        cache_size=32,
    )

    # class weights computed but not used by CORAL (kept for reference)
    w_ce, classes_present = class_weights_from_train(train_ds.df, target=args.target)
    n_classes = len(classes_present)
    assert n_classes == 3, "This CORAL setup expects 3 classes (ISUP01/23/45)."

    # sampler to bump lesion slice rate
    sampler = make_pos_sampler(train_ds.df, pos_ratio=args.pos_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_1024)

    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_1024)

    # -------- model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,                 # unused for CORAL head, but fine to keep
        proj_dim=128, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,
        allow_var_size=False,
        pixel_mean_std=None,
    ).to(device)

    # freeze encoder for a stable baseline
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Build tiny CORAL head on top of the embedding
    d_in = infer_embedding_dim(model, device)
    ord_head = OrdinalHead(d_in=d_in, K=n_classes, p_drop=0.3).to(device)

    # Optimizer: train model head(s) that are still requires_grad + ordinal head
    params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(ord_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    best_f1 = -1.0
    best_path = outdir / "ckpt_best.pt"

    # -------- loop --------
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, model, ord_head, optimizer=optimizer, device=device)
        va_loss, va_acc, va_f1, va_probs, va_y = run_epoch(val_loader, model, ord_head, optimizer=None, device=device, return_outputs=True)

        # Per-class metrics on validation (from probabilities)
        per_acc, per_auc, macro_auc = per_class_metrics_from_probs(va_probs, va_y)

        # Confusion matrix (rows=true, cols=pred)
        cm = confusion_matrix_from_probs(va_probs, va_y, K=n_classes)

        # pretty print
        pcs = "  ".join([f"acc[c{c}]={per_acc[c]:.3f}" if not np.isnan(per_acc[c]) else f"acc[c{c}]=NA"
                         for c in range(va_probs.shape[1])])
        if _HAS_SK and macro_auc is not None:
            aucs = "  ".join([f"auc[c{c}]={per_auc[c]:.3f}" if per_auc[c] is not None else f"auc[c{c}]=NA"
                               for c in range(va_probs.shape[1])])
            extra = f" | {pcs} | {aucs} | macroAUC={macro_auc:.3f}"
        else:
            extra = f" | {pcs} | (sklearn not available: AUC skipped)"

        print(f"[{epoch:03d}] "
              f"train: loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"val: loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}{extra}")

        print_confusion_matrix(cm, labels=LABEL_NAMES)

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ord_head": ord_head.state_dict(),
            }, best_path)
            print(f"  ↳ saved best to {best_path} (macro-F1={best_f1:.4f})")

if __name__ == "__main__":
    main()
