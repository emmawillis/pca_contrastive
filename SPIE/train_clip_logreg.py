#!/usr/bin/env python3
# train_triplet_logreg.py
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry

from triplet_loss_utils import (
    get_histo_by_isup,
    get_random_sample,   # for sampling positives per class
)
from info_loss_utils import SymmetricInfoNCELoss
import train_utils
from train_utils import (
    build_datasets_and_loaders,
    EarlyStopper,
)

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression


# ---- InfoNCE batch wrapper (replaces triplet_loss_batch) ----
def infonce_loss_batch(embeddings: torch.Tensor,
                       labels: torch.Tensor,
                       histo_buckets,
                       loss_obj: SymmetricInfoNCELoss) -> torch.Tensor:
    """
    Build a batch of positives from histo_buckets (one positive per label)
    and compute symmetric InfoNCE with the new API:

        loss = loss_obj(mri_feats=embeddings, positive_hist=pos_hist)
    """
    device = embeddings.device
    dtype = embeddings.dtype

    pos_list = []
    for i in range(labels.size(0)):
        lbl = int(labels[i].item())
        pos = get_random_sample(lbl, histo_buckets)  # 1D tensor [D]
        pos_list.append(pos)

    pos_hist = torch.stack(pos_list, dim=0).to(device=device, dtype=dtype, non_blocking=True)

    # Dim check (should match projection dim)
    if embeddings.size(1) != pos_hist.size(1):
        raise ValueError(f"Embedding dim mismatch: MRI feats {embeddings.size(1)} vs hist {pos_hist.size(1)}")

    return loss_obj(embeddings, pos_hist)


# ---- InfoNCE train/val (encoder-only training) ----
def run_epoch_triplet(loader, model, loss_fn_builder, optimizer=None, device="cuda"):
    """
    Keeps the original structure/signature, but 'loss_fn_builder' is a callable
    that returns a loss tensor from (embeddings, labels).
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_n = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            _, emb = model(x)                    # (logits unused), emb: [B,D]
            loss = loss_fn_builder(emb, y)

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
    X = torch.cat(embs, 0).numpy() if embs else np.empty((0, 0), dtype=np.float32)
    y = torch.cat(ys, 0).numpy() if ys else np.empty((0,), dtype=np.int64)
    return X, y


# ---- LR eval on embeddings ----
def eval_with_logreg(X_train, y_train, X_val, y_val, n_classes, max_iter=5):
    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class="auto",
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

    cm = confusion_matrix(y_val, y_pred, labels=list(range(n_classes)))
    return {
        "acc": acc, "bacc": bacc, "f1_macro": f1_macro,
        "per_acc": per_acc, "per_auc": per_auc, "macro_auc": macro_auc,
        "cm": cm, "clf": clf,
    }


# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--outdir", default="./runs/simple_triplet_lr")
    p.add_argument("--target", choices=["isup3","isup6", "binary_low_high", "binary_all"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3") # holding back 4 as test set
    p.add_argument("--folds_val", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)      # number of encoder epochs (each followed by LR eval)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--histo_dir", required=True)        # folder of .npy histo encodings
    p.add_argument("--histo_marksheet_dir", required=True)
    p.add_argument("--lr_max_iter", type=int, default=5, help="LogReg max_iter per evaluation")
    p.add_argument("--provider", default="karolinska")
    p.add_argument("--proj_dim", type=int, required=True)
    # InfoNCE temperature
    p.add_argument("--infonce_tau", type=float, default=0.07)

    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True,
                help="If true, drop rows with skip==1. Use --no-use-skip to include them.")
    p.add_argument("--label6_column", default="label6")

    args = p.parse_args()
    print("SCRIPT: train_clip_logreg.py (info loss)")
    print("ARGS: ", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- MRI dataset / loaders (shared helper) --------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    (train_ds, val_ds,train_loader, val_loader, _w_ce_unused, classes_present, n_classes) = build_datasets_and_loaders(
        manifest=args.manifest,
        folds_train=folds_train,
        folds_val=folds_val,
        target=args.target,
        use_skip=args.use_skip,
        label6_column=args.label6_column,
        batch_size=args.batch_size,
        pos_ratio=args.pos_ratio,
    )

    # -------- histo dataset for positives (class buckets) --------
    train_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "train"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "train.csv"),
        num_classes=n_classes,
        provider=args.provider
    )
    val_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "val"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "val.csv"),
        num_classes=n_classes,
        provider=args.provider
    )

    # -------- model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,
        proj_dim=args.proj_dim, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,
        pixel_mean_std=None,
    ).to(device)

    # Train **encoder only** (head not used for InfoNCE training)
    for p in model.parameters():            # freeze all
        p.requires_grad = False
    for p in model.encoder.parameters():    # unfreeze encoder
        p.requires_grad = True
    for p in model.proj.parameters():       # unfreeze the projection
        p.requires_grad = True

    proj_lr = args.lr
    enc_lr = proj_lr * 0.1

    optimizer = torch.optim.AdamW(
        [
            {"params": model.proj.parameters(),    "lr": proj_lr, "weight_decay": args.wd},
            {"params": model.encoder.parameters(), "lr": enc_lr,  "weight_decay": args.wd},
        ]
    )
    print(f"[InfoNCE] lr_proj={proj_lr:g} | lr_enc={enc_lr:g} | tau={args.infonce_tau:g}")

    # Instantiate the Symmetric InfoNCE loss
    infonce = SymmetricInfoNCELoss(temperature=args.infonce_tau)

    # loss builders (train/val) that match the old signature used by run_epoch_triplet
    def train_infonce(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return infonce_loss_batch(embeddings, labels, train_histo_buckets, loss_obj=infonce)

    def val_infonce(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return infonce_loss_batch(embeddings, labels, val_histo_buckets, loss_obj=infonce)

    early = EarlyStopper(patience=100)
    best_path = outdir / "ckpt_best.pt"

    # -------- loop --------
    for epoch in range(1, args.epochs + 1):
        # 1) Encoder training epoch (InfoNCE)
        tr_loss = run_epoch_triplet(train_loader, model, train_infonce, optimizer=optimizer, device=device)
        va_loss = run_epoch_triplet(val_loader,   model, val_infonce,   optimizer=None,     device=device)

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

        print(f"[{epoch:03d}] infonce: train loss {tr_loss:.4f} | val loss {va_loss:.4f} || "
              f"LR(val): acc {lr_metrics['acc']:.4f} BAL-acc {lr_metrics['bacc']:.4f} f1 {lr_metrics['f1_macro']:.4f} | "
              f"{per_acc_str}{auc_part}")

        cm_str = train_utils.format_confusion_matrix(lr_metrics["cm"], n_classes=n_classes)
        print(cm_str)

        # 5) Save/early-stop on **Balanced Accuracy** (validation)
        if early.update(lr_metrics["bacc"], model, best_path):
            print(f"  ↳ new best (val BAL-acc={early.best:.4f}) saved to {best_path}")
        else:
            print(f"  ↳ no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= early.patience:
                print(f"Early stopping at epoch {epoch}.")


if __name__ == "__main__":
    main()
