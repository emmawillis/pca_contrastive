#!/usr/bin/env python3
# train_triplet_logreg.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry

from triplet_loss_utils import (
    get_histo_by_isup,
    triplet_loss_batch,
)
import train_utils
from train_utils import (
    build_datasets_and_loaders,
    EarlyStopper,
    wandb_init, wandb_log, wandb_finish,
)

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

# ---- Triplet train/val (encoder-only training) ----
def run_epoch_triplet(loader, model, triplet_fn, optimizer=None, device="cuda"):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_n = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            _, emb = model(x)                    # emb: [B,D]
            loss = triplet_fn(emb, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs
    return total_loss / max(1, total_n)

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

# ---- LR eval on embeddings (fit) ----
def eval_with_logreg(X_train, y_train, X_val, y_val, n_classes, max_iter=5):
    clf = LogisticRegression(
        max_iter=max_iter, multi_class="auto", solver="lbfgs",
        n_jobs=None, class_weight=None
    )
    clf.fit(X_train, y_train)
    return eval_with_existing_clf(X_val, y_val, n_classes, clf, return_clf=True)

# ---- LR eval on embeddings (pre-fit clf) ----
def eval_with_existing_clf(X, y, n_classes, clf, return_clf=False):
    y_pred = clf.predict(X)
    acc = float(accuracy_score(y, y_pred))
    bacc = float(balanced_accuracy_score(y, y_pred))
    f1_macro = float(f1_score(y, y_pred, average="macro"))

    per_acc = {}
    for c in range(n_classes):
        mask = (y == c)
        per_acc[c] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")

    per_auc = {c: float("nan") for c in range(n_classes)}
    macro_auc = float("nan")
    try:
        probs = clf.predict_proba(X)  # [N, K]
        auc_vals = []
        for c in range(n_classes):
            y_bin = (y == c).astype(np.int32)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                auc = roc_auc_score(y_bin, probs[:, c])
                per_auc[c] = float(auc)
                auc_vals.append(auc)
        if len(auc_vals) > 0:
            macro_auc = float(np.nanmean(auc_vals))
    except Exception:
        pass

    cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))
    out = {
        "acc": acc, "bacc": bacc, "f1_macro": f1_macro,
        "per_acc": per_acc, "per_auc": per_auc, "macro_auc": macro_auc,
        "cm": cm,
    }
    if return_clf:
        out["clf"] = clf
    return out

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--outdir", default="./runs/simple_triplet_lr")
    p.add_argument("--target", choices=["isup3","isup6", "binary_low_high", "binary_all"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3")       # keep 4 as test by default
    p.add_argument("--folds_val", default="0")
    p.add_argument("--folds_test", default="4")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)       # triplet epochs
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--histo_dir", required=True)           # folder of .npy histo encodings
    p.add_argument("--histo_marksheet_dir", required=True)
    p.add_argument("--lr_max_iter", type=int, default=5, help="LogReg max_iter per evaluation")
    p.add_argument("--provider", default="karolinska")
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, drop rows with skip==1. Use --no-use-skip to include them.")
    p.add_argument("--label6_column", default="label6")
    # W&B
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="Log epoch metrics to Weights & Biases.")
    p.add_argument("--wandb_project", default="mri-training")
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    print("SCRIPT: train_triplet_logreg.py")
    print("ARGS: ", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- MRI dataset / loaders (shared helper; includes TEST) --------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.folds_test.split(",") if s.strip()]

    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader,
     _w_ce_unused, classes_present, n_classes) = build_datasets_and_loaders(
        manifest=args.manifest,
        folds_train=folds_train,
        folds_val=folds_val,
        folds_test=folds_test,
        target=args.target,
        use_skip=args.use_skip,
        label6_column=args.label6_column,
        batch_size=args.batch_size,
        pos_ratio=args.pos_ratio,
    )

    # -------- histo dataset for triplet anchors/pos/negs --------
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

    # Train **encoder + proj** (head unused for triplet)
    for p in model.parameters():            # freeze all
        p.requires_grad = False
    for p in model.encoder.parameters():    # unfreeze encoder
        p.requires_grad = True
    for p in model.proj.parameters():       # unfreeze projection
        p.requires_grad = True

    proj_lr = args.lr
    enc_lr = proj_lr * 0.1
    optimizer = torch.optim.AdamW(
        [
            {"params": model.proj.parameters(),    "lr": proj_lr, "weight_decay": args.wd},
            {"params": model.encoder.parameters(), "lr": enc_lr,  "weight_decay": args.wd},
        ]
    )
    print(f"[triplet] lr_proj={proj_lr:g} | lr_enc={enc_lr:g} | triplet_margin={args.triplet_margin:g}")

    # Triplet criteria (train/val)
    def train_triplet(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return triplet_loss_batch(embeddings, labels, train_histo_buckets, num_classes=n_classes, margin=args.triplet_margin)

    def val_triplet(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return triplet_loss_batch(embeddings, labels, val_histo_buckets, num_classes=n_classes, margin=args.triplet_margin)

    # W&B
    wb = wandb_init(bool(args.wandb), args.wandb_project, args.wandb_run_name, config=vars(args))

    early = EarlyStopper(patience=args.patience)
    best_path = outdir / "ckpt_best.pt"

    best_lr_clf = None
    last_lr_clf = None

    # -------- loop --------
    for epoch in range(1, args.epochs + 1):
        # 1) Encoder training epoch (triplet)
        tr_loss = run_epoch_triplet(train_loader, model, train_triplet, optimizer=optimizer, device=device)
        va_loss = run_epoch_triplet(val_loader,   model, val_triplet,   optimizer=None,     device=device)

        # 2) Embed train/val with current encoder
        X_tr, y_tr = extract_embeddings(train_loader, model, device=device)
        X_va, y_va = extract_embeddings(val_loader,   model, device=device)

        # 3) Train LR on train embeddings, eval on val
        lr_metrics = eval_with_logreg(
            X_tr, y_tr, X_va, y_va,
            n_classes=n_classes,
            max_iter=args.lr_max_iter
        )
        last_lr_clf = lr_metrics["clf"]

        # 4) Pretty print
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
        print(train_utils.format_confusion_matrix(lr_metrics["cm"], n_classes=n_classes))

        # 5) W&B log
        if wb is not None:
            lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
            payload = {
                "epoch": epoch,
                "lr_proj": lrs[0] if lrs else None,
                "lr_enc":  lrs[1] if len(lrs) > 1 else None,
                "triplet/train_loss": tr_loss,
                "triplet/val_loss": va_loss,
                "lr_val/acc": lr_metrics["acc"],
                "lr_val/bacc": lr_metrics["bacc"],
                "lr_val/f1_macro": lr_metrics["f1_macro"],
                "lr_val/macro_auc": lr_metrics["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"lr_val/acc_c{c}"] = lr_metrics["per_acc"][c]
                payload[f"lr_val/auc_c{c}"] = lr_metrics["per_auc"][c]
            wandb_log(wb, payload)

        # 6) Early stop on BAL-ACC; keep best encoder in memory and best LR clf
        if early.update(lr_metrics["bacc"], model, best_path):
            best_lr_clf = lr_metrics["clf"]
            print(f"  ↳ new best (val BAL-acc={early.best:.4f}) saved snapshot (encoder) + LR clf (in-memory)")
        else:
            print(f"  ↳ no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # -------- Load in-memory best encoder (no disk reload) --------
    if not early.load_best_into(model, strict=False):
        print("[warn] No validation improvement recorded; using last model state for final eval.")
        if best_lr_clf is None:
            best_lr_clf = last_lr_clf  # fallback

    model.to(device).eval()

    # -------- Final TEST eval using best encoder + best LR clf --------
    if test_loader is not None and best_lr_clf is not None:
        X_te, y_te = extract_embeddings(test_loader, model, device=device)
        test_metrics = eval_with_existing_clf(X_te, y_te, n_classes, best_lr_clf, return_clf=False)

        per_acc_str_t = "  ".join([
            f"acc[c{c}]={test_metrics['per_acc'][c]:.3f}" if not np.isnan(test_metrics['per_acc'][c]) else f"acc[c{c}]=NA"
            for c in range(n_classes)
        ])
        if not np.isnan(test_metrics["macro_auc"]):
            per_auc_str_t = "  ".join([
                f"auc[c{c}]={test_metrics['per_auc'][c]:.3f}" if not np.isnan(test_metrics['per_auc'][c]) else f"auc[c{c}]=NA"
                for c in range(n_classes)
            ])
            auc_part_t = f" | {per_auc_str_t} | macroAUC={test_metrics['macro_auc']:.3f}"
        else:
            auc_part_t = " | (AUC unavailable)"

        print(f"[TEST] LR(test): acc {test_metrics['acc']:.4f} BAL-acc {test_metrics['bacc']:.4f} "
              f"f1 {test_metrics['f1_macro']:.4f} | {per_acc_str_t}{auc_part_t}")
        print(train_utils.format_confusion_matrix(test_metrics["cm"], n_classes=n_classes))

        if wb is not None:
            payload = {
                "final_test/acc": test_metrics["acc"],
                "final_test/bacc": test_metrics["bacc"],
                "final_test/f1_macro": test_metrics["f1_macro"],
                "final_test/macro_auc": test_metrics["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"final_test/acc_c{c}"] = test_metrics["per_acc"][c]
                payload[f"final_test/auc_c{c}"] = test_metrics["per_auc"][c]
            wandb_log(wb, payload)
    else:
        print("[TEST] Skipping test evaluation (no test loader or no best LR clf).")

    wandb_finish(wb)

if __name__ == "__main__":
    main()
