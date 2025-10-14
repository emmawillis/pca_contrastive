#!/usr/bin/env python3
# train_triplet_logreg.py
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
from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry

from triplet_loss_utils import (
    get_histo_by_isup,
    triplet_loss_batch,
)

# --- sklearn bits (required for LR); AUC optional ---
try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
    )
    from sklearn.linear_model import LogisticRegression
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ----------------- helpers -----------------
def collate_resize_to_imgsize(batch):
    imgs, labels = [], []
    extras_keys = [k for k in batch[0].keys() if k not in ("image", "label")]
    extras = {k: [] for k in extras_keys}
    for s in batch:
        x = s["image"].unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
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

# oversample the slices that actually intersect the lesions, since other slices are set to isup 0
def make_pos_sampler(df: pd.DataFrame, pos_ratio: float = 0.33, seed: int = 1337):
    is_pos = df["has_lesion"].astype(int).values
    n_pos = int(is_pos.sum()); n_neg = len(is_pos) - n_pos
    assert n_pos > 0, "No positive slices in train folds."
    w_neg = 1.0
    w_pos = (pos_ratio/(1-pos_ratio)) * (n_neg/max(1,n_pos))
    w = np.where(is_pos==1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)

# ---- Triplet train/val (encoder-only training) ----
def run_epoch_triplet(loader, model, triplet_fn, optimizer=None, device="cuda"):
    train_mode = optimizer is not None
    if train_mode: model.train(True)
    else:          model.train(False)

    total_loss, total_n = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            _, emb = model(x)                    # (logits unused), emb: [B,D]
            loss = triplet_fn(emb, y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs

    avg_loss = total_loss / max(1, total_n)
    return avg_loss

# ---- Embedding extraction ----
@torch.no_grad()
def extract_embeddings(loader, model, device="cuda"):
    model.eval()
    embs, ys = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        _, emb = model(x)
        embs.append(emb.cpu())
        ys.append(y.cpu())
    X = torch.cat(embs, 0).numpy()
    y = torch.cat(ys, 0).numpy()
    return X, y

# ---- LR eval on embeddings ----
def eval_with_logreg(X_train, y_train, X_val, y_val, n_classes, max_iter=5):
    if not _HAS_SK:
        raise RuntimeError("scikit-learn is required for LogisticRegression evaluation but is not available.")

    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class="auto",        # 'multinomial' if supported by solver
        solver="lbfgs",
        n_jobs=None,
        class_weight=None
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    bacc = float(balanced_accuracy_score(y_val, y_pred))
    f1_macro = float(f1_score(y_val, y_pred, average="macro"))

    # Per-class accuracy
    per_acc = {}
    for c in range(n_classes):
        mask = (y_val == c)
        per_acc[c] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")

    # AUC (OvR) if probabilities exist and both classes present
    per_auc = {c: float("nan") for c in range(n_classes)}
    macro_auc = float("nan")
    try:
        probs = clf.predict_proba(X_val)  # [N, K]
        auc_vals = []
        for c in range(n_classes):
            y_bin = (y_val == c).astype(np.int32)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                auc = roc_auc_score(y_bin, probs[:, c])
                per_auc[c] = float(auc)
                auc_vals.append(auc)
        if len(auc_vals) > 0:
            macro_auc = float(np.nanmean(auc_vals))
    except Exception:
        pass

    # Confusion matrix (rows=true, cols=pred)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(n_classes)))

    return {
        "acc": acc,
        "bacc": bacc,
        "f1_macro": f1_macro,
        "per_acc": per_acc,
        "per_auc": per_auc,
        "macro_auc": macro_auc,
        "cm": cm,
        "clf": clf,
    }

def format_confusion_matrix(cm: np.ndarray, n_classes: int):
    labels = ["ISUP01","ISUP23","ISUP45"] if n_classes == 3 else ["ISUP0","ISUP1","ISUP2","ISUP3","ISUP4","ISUP5"]
    header = "true\\pred " + " ".join(f"{lbl:>7}" for lbl in labels)
    lines = ["Confusion matrix (LR val): rows=true, cols=pred", header]
    for i in range(n_classes):
        row = " ".join(f"{int(cm[i, j]):7d}" for j in range(n_classes))
        lines.append(f"{labels[i]:>9} {row}")
    return "\n".join(lines)

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--outdir", default="./runs/simple_triplet_lr")
    p.add_argument("--target", choices=["isup3","isup6"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3") # holding back 4 as test set
    p.add_argument("--folds_val", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)      # number of triplet epochs (each followed by LR eval)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--histo_dir", required=True)        # folder of .npy histo encodings
    p.add_argument("--histo_marksheet_dir", required=True)
    p.add_argument("--lr_max_iter", type=int, default=5, help="LogReg max_iter per evaluation")
    args = p.parse_args()

    if not _HAS_SK:
        raise RuntimeError("scikit-learn not available; install it to use LogisticRegression evaluation.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- MRI dataset --------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()!=""]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()!=""]

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

    # class info
    w_ce, classes_present = class_weights_from_train(train_ds.df, target=args.target)
    n_classes = len(classes_present)

    # sampler to bump lesion slice rate
    sampler = make_pos_sampler(train_ds.df, pos_ratio=args.pos_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_to_imgsize)

    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_to_imgsize)
    
    # -------- histo dataset for triplet anchors/pos/negs --------
    train_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "train"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "train.csv"),
        num_classes=n_classes,
    )
    val_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "val"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "val.csv"),
        num_classes=n_classes,
    )

    # -------- model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,
        proj_dim=128, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,
        pixel_mean_std=None,
    ).to(device)

    # Train **encoder only** (head not used for triplet)
    for p in model.parameters():            # freeze all
        p.requires_grad = False
    for p in model.encoder.parameters():    # unfreeze encoder
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr, weight_decay=args.wd)

    # triplet criteria (train/val)
    def train_triplet(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return triplet_loss_batch(embeddings, labels, train_histo_buckets, margin=0.4)

    def val_triplet(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return triplet_loss_batch(embeddings, labels, val_histo_buckets, margin=0.4)

    best_val_bacc = -1.0
    best_path = outdir / "ckpt_best.pt"

    # -------- loop --------
    for epoch in range(1, args.epochs + 1):
        # 1) Encoder training epoch (triplet)
        tr_loss = run_epoch_triplet(train_loader, model, train_triplet, optimizer=optimizer, device=device)
        va_loss = run_epoch_triplet(val_loader,   model, val_triplet,   optimizer=None,     device=device)

        # 2) Embed train/val with current encoder
        X_tr, y_tr = extract_embeddings(train_loader, model, device=device)
        X_va, y_va = extract_embeddings(val_loader,   model, device=device)

        # 3) Train a simple classifier on train embeddings, eval on val (max_iter ~ "5 epochs")
        lr_metrics = eval_with_logreg(
            X_tr, y_tr, X_va, y_va,
            n_classes=n_classes,
            max_iter=args.lr_max_iter
        )

        # 4) Log nicely (including confusion matrix)
        per_acc_str = "  ".join([
            f"acc[c{c}]={lr_metrics['per_acc'][c]:.3f}" if not np.isnan(lr_metrics['per_acc'][c]) else f"acc[c{c}]=NA"
            for c in range(n_classes)
        ])
        if not np.isnan(lr_metrics["macro_auc"]):
            per_auc_str = "  ".join([
                f"auc[c{c}]={lr_metrics['per_auc'][c]:.3f}" if not np.isnan(lr_metrics['per_auc'][c]) else f"auc[c{c}]=NA"
                for c in range(n_classes)
            ])
            auc_part = f" | {per_auc_str} | macroAUC={lr_metrics['macro_auc']:.3f}"
        else:
            auc_part = " | (AUC unavailable)"

        print(f"[{epoch:03d}] triplet: train loss {tr_loss:.4f} | val loss {va_loss:.4f} || "
              f"LR(val): acc {lr_metrics['acc']:.4f} BAL-acc {lr_metrics['bacc']:.4f} f1 {lr_metrics['f1_macro']:.4f} | "
              f"{per_acc_str}{auc_part}")

        cm_str = format_confusion_matrix(lr_metrics["cm"], n_classes=n_classes)
        print(cm_str)

        # 5) Save 'best' based on **Balanced Accuracy** on validation
        if lr_metrics["bacc"] > best_val_bacc:
            best_val_bacc = lr_metrics["bacc"]
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
            print(f"  ↳ new best (val BAL-acc={best_val_bacc:.4f}) saved to {best_path}")

if __name__ == "__main__":
    main()
