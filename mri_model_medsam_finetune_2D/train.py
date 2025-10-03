#!/usr/bin/env python3
# train_min.py
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
from ISUPMedSAM import MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry

# --- optional: AUC ---
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ----------------- helpers -----------------
import torch.nn.functional as F

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

def make_pos_sampler(df: pd.DataFrame, pos_ratio: float = 0.33, seed: int = 1337):
    is_pos = df["has_lesion"].astype(int).values
    n_pos = int(is_pos.sum()); n_neg = len(is_pos) - n_pos
    assert n_pos > 0, "No positive slices in train folds."
    w_neg = 1.0
    w_pos = (pos_ratio/(1-pos_ratio)) * (n_neg/max(1,n_pos))
    w = np.where(is_pos==1, w_pos, w_neg).astype(np.float64)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True, generator=g)

def macro_f1(logits: torch.Tensor, y: torch.Tensor, K: int) -> float:
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true = y.cpu().numpy()
    f1s = []
    for c in range(K):
        tp = ((y_pred==c)&(y_true==c)).sum()
        fp = ((y_pred==c)&(y_true!=c)).sum()
        fn = ((y_pred!=c)&(y_true==c)).sum()
        prec = tp/(tp+fp) if tp+fp>0 else 0.0
        rec  = tp/(tp+fn) if tp+fn>0 else 0.0
        f1s.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
    return float(np.mean(f1s))

# ----------------- train / val -----------------
def run_epoch(loader, model, loss_fn, optimizer=None, device="cuda", return_outputs=False):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_logits, all_y = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits, _ = model(x)
        loss = loss_fn(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
        all_logits.append(logits.detach())
        all_y.append(y.detach())

    avg_loss = total_loss / max(1,total_n)
    acc = total_correct / max(1,total_n)
    logits_cat = torch.cat(all_logits) if all_logits else torch.empty(0)
    y_cat = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)
    K = logits_cat.shape[1] if logits_cat.ndim==2 else 0
    f1 = macro_f1(logits_cat, y_cat, K) if K else 0.0

    if return_outputs:
        return avg_loss, acc, f1, logits_cat, y_cat
    return avg_loss, acc, f1

def per_class_metrics(logits: torch.Tensor, y: torch.Tensor):
    """
    Returns:
      - per-class accuracy dict {class_idx: acc}
      - per-class AUC dict {class_idx: auc}  (if sklearn available)
      - macro_auc (float or None if unavailable)
    Assumes labels are contiguous 0..K-1.
    """
    K = logits.shape[1]
    y_np = y.cpu().numpy()
    y_pred = logits.argmax(dim=1).cpu().numpy()

    # per-class accuracy
    accs = {}
    for c in range(K):
        mask = (y_np == c)
        if mask.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = float((y_pred[mask] == c).mean())

    # per-class AUC (OvR)
    aucs = {}
    macro_auc = None
    if _HAS_SK:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        auc_vals = []
        for c in range(K):
            y_bin = (y_np == c).astype(np.int32)
            # Need both pos and neg to compute AUC
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                try:
                    auc = roc_auc_score(y_bin, probs[:, c])
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

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--outdir", default="./runs/simple")
    p.add_argument("--target", choices=["isup3","isup6"], default="isup3")
    p.add_argument("--folds_train", default="0,2,3,4")
    p.add_argument("--folds_val", default="1")
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
        pct_lower=0.5, pct_upper=99.5,   # per-slice 0.5–99.5% clip → [0,1]
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

    # class weights from TRAIN distribution
    w_ce, classes_present = class_weights_from_train(train_ds.df, target=args.target)
    n_classes = len(classes_present)
    w_ce = w_ce.to(device)

    # sampler to bump lesion slice rate
    sampler = make_pos_sampler(train_ds.df, pos_ratio=args.pos_ratio)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_1024)

    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_1024)

    # -------- model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,
        proj_dim=128, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,              # pre-neck + spatial attention
        allow_var_size=False,           # we resize/pad to 1024 inside the model
        pixel_mean_std=None,            # inputs already in [0,1]
    ).to(device)

    # freeze encoder for a stable baseline
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss(weight=w_ce)

    best_f1 = -1.0
    best_path = outdir / "ckpt_best.pt"

    # -------- loop --------
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, model, criterion, optimizer=optimizer, device=device)
        va_loss, va_acc, va_f1, va_logits, va_y = run_epoch(val_loader, model, criterion, optimizer=None, device=device, return_outputs=True)

        # Per-class metrics on validation
        per_acc, per_auc, macro_auc = per_class_metrics(va_logits, va_y)
        # pretty print
        pcs = "  ".join([f"acc[c{c}]={per_acc[c]:.3f}" if not np.isnan(per_acc[c]) else f"acc[c{c}]=NA"
                         for c in range(va_logits.shape[1])])
        if _HAS_SK and macro_auc is not None:
            aucs = "  ".join([f"auc[c{c}]={per_auc[c]:.3f}" if per_auc[c] is not None else f"auc[c{c}]=NA"
                               for c in range(va_logits.shape[1])])
            extra = f" | {pcs} | {aucs} | macroAUC={macro_auc:.3f}"
        else:
            extra = f" | {pcs} | (sklearn not available: AUC skipped)"

        print(f"[{epoch:03d}] "
              f"train: loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"val: loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}{extra}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
            print(f"  ↳ saved best to {best_path} (macro-F1={best_f1:.4f})")

if __name__ == "__main__":
    main()
