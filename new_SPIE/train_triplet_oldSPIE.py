#!/usr/bin/env python3
# train_triplet_oldSPIE.py
#
# SPIE-style two-phase training:
#   Phase 1: Triplet loss to align MedSAM embeddings with histopath encodings
#            - MedSAM frozen
#            - CNN + projection trained
#            - LR proxy each epoch, early stopping
#   Phase 2: Classification head training (Cross-Entropy)
#            - Everything frozen except classifier (and optionally proj)
#            - Early stopping on balanced accuracy
#
# At the end:
#   - Evaluate on test set
#   - Save val/test embeddings
#   - Generate UMAP for MRI-test embeddings
#   - Generate UMAP for histopath-test encodings

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import umap.umap_ as umap
from cnn_frozen import MRIClassifierFrozenCNN
from dataset_frozen_medsam import PicaiSliceFrozenEncodingDataset

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression

import wandb

from segment_anything import sam_model_registry

# old SPIE CNN model
from cnn import MRIClassifierCNN

import train_utils
from train_utils import (
    build_datasets_and_loaders,
    evaluate_loader,
    format_perclass_acc_auc,
    format_sens_spec,
    format_confusion_matrix,
    per_class_operating_points,
    EarlyStopper,
    set_seed,
    wandb_init, wandb_log, wandb_finish,
)

from triplet_loss_utils import (
    get_histo_by_isup,
    triplet_loss_batch,
)

# ----------------------------------------------------------
# helper: run triplet epoch
# ----------------------------------------------------------
def run_epoch_triplet(loader, model, histo_dict, margin, optimizer=None, device="cuda"):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_n = 0.0, 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        # logit unused
        _, emb = model(x)

        loss = triplet_loss_batch(
            embeddings=emb,
            labels=y,
            histo_dict=histo_dict,
            margin=margin,
            num_classes=len(histo_dict),
        )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)


# ----------------------------------------------------------
# extract (X,y) embedding arrays
# ----------------------------------------------------------
@torch.no_grad()
def extract_embeddings(loader, model, device="cuda"):
    model.eval()
    embs, ys = [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        _, emb = model(x)
        embs.append(emb.cpu())
        ys.append(y.cpu())
    if not embs:
        return np.empty((0, 0)), np.empty((0,), dtype=np.int64)
    return torch.cat(embs, 0).numpy(), torch.cat(ys, 0).numpy()


# ----------------------------------------------------------
# Evaluate LR classifier on val split
# ----------------------------------------------------------
def eval_with_logreg(X_train, y_train, X_val, y_val, n_classes, max_iter=5):
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="auto",
        n_jobs=None
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    bacc = float(balanced_accuracy_score(y_val, y_pred))
    f1m  = float(f1_score(y_val, y_pred, average="macro"))

    per_acc = {}
    per_auc = {}
    auc_vals = []

    try:
        probs = clf.predict_proba(X_val)
    except Exception:
        probs = None

    for c in range(n_classes):
        mask = (y_val == c)
        per_acc[c] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")

        if probs is not None:
            y_bin = (y_val == c).astype(np.int32)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                auc = roc_auc_score(y_bin, probs[:, c])
                per_auc[c] = float(auc)
                auc_vals.append(auc)
            else:
                per_auc[c] = float("nan")
        else:
            per_auc[c] = float("nan")

    macro_auc = float(np.nanmean(auc_vals)) if auc_vals else float("nan")

    return {
        "clf": clf,
        "acc": acc,
        "bacc": bacc,
        "f1_macro": f1m,
        "per_acc": per_acc,
        "per_auc": per_auc,
        "macro_auc": macro_auc
    }


# ----------------------------------------------------------
# Run CE epoch (classifier-only)
# ----------------------------------------------------------
def run_epoch_ce(loader, model, w_ce, optimizer=None, device="cuda"):
    train_mode = optimizer is not None
    model.train(train_mode)

    ce = nn.CrossEntropyLoss(weight=w_ce)

    total_loss, total_n, total_correct = 0.0, 0, 0
    preds, trues = [], []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits, _ = model(x)
        loss = ce(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(1)
        total_correct += (pred == y).sum().item()

        preds.append(pred.cpu())
        trues.append(y.cpu())

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)

    if preds:
        yp = torch.cat(preds).numpy()
        yt = torch.cat(trues).numpy()
        f1m = float(f1_score(yt, yp, average="macro"))
        bacc = float(balanced_accuracy_score(yt, yp))
    else:
        f1m, bacc = 0.0, 0.0

    return avg_loss, acc, f1m, bacc


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-frozen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--medsam_encodings", default="/home/ewillis/projects/aip-medilab/shared/picai/picai_medsam_zero_shot_encodings")

    p.add_argument("--seed", type=int, default=42)

    # Dataset
    p.add_argument("--manifest", required=True)
    p.add_argument("--target",
                   choices=["isup3", "isup6", "binary_low_high", "binary_all", "isup0145"],
                   default="isup3")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val",   default="0")
    p.add_argument("--folds_test",  default="4")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label6_column", default="label6")

    # MedSAM + model
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--strip_neck", action=argparse.BooleanOptionalAction, default=False)

    # Triplet phase
    p.add_argument("--histo_dir", required=True)
    p.add_argument("--histo_marksheet_dir", required=True)
    p.add_argument("--provider", default="all")
    p.add_argument("--triplet_epochs", type=int, default=50)
    p.add_argument("--triplet_patience", type=int, default=10)
    p.add_argument("--triplet_lr", type=float, default=3e-4)
    p.add_argument("--triplet_wd", type=float, default=1e-4)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument("--lr_max_iter", type=int, default=5)

    # Head phase
    p.add_argument("--head_epochs", type=int, default=50)
    p.add_argument("--head_patience", type=int, default=10)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--head_wd", type=float, default=1e-4)
    p.add_argument("--train_proj", action="store_true")

    # Output & wandb
    p.add_argument("--outdir", default="./runs/triplet_oldSPIE")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", default="mri-training")
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    print("SCRIPT: train_triplet_oldSPIE.py")
    print("ARGS:", args)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    # Parse folds
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.folds_test.split(",") if s.strip()]

    # wandb
    wb = wandb_init(bool(args.wandb),
                    args.wandb_project,
                    args.wandb_run_name,
                    config=vars(args))
    if wb is not None:
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        wandb.define_metric("test/*",  step_metric="epoch")
        wandb.define_metric("aux/*",   step_metric="epoch")

    # ===========================
    # BUILD DATA
    # ===========================
    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader,
     w_ce, classes_present, n_classes) = build_datasets_and_loaders(
        manifest=args.manifest,
        folds_train=folds_train,
        folds_val=folds_val,
        folds_test=folds_test,
        target=args.target,
        use_skip=args.use_skip,
        label6_column=args.label6_column,
        batch_size=args.batch_size,
        pos_ratio=args.pos_ratio,
        use_frozen=args.use_frozen,
        medsam_encodings=args.medsam_encodings
    )
    w_ce = w_ce.to(device)

    # ===========================
    # HISTO BUCKETS (train / val / test)
    # ===========================
    # Train buckets
    train_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "train"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "train.csv"),
        num_classes=n_classes,
        provider=args.provider
    )

    # Val buckets
    val_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "val"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "val.csv"),
        num_classes=n_classes,
        provider=args.provider
    )

    # Test histo buckets (for UMAP)
    test_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "test"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "test.csv"),
        num_classes=n_classes,
        provider=args.provider
    )

    # ===========================
    # Build model
    # ===========================
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)

    if args.use_frozen:
        model = MRIClassifierFrozenCNN(
            num_classes=n_classes,     # or whatever you're predicting
            proj_dim=args.proj_dim       # or any projection size you want
        ).to(device)

    else:
        model = MRIClassifierCNN(
            sam_model=sam,
            num_classes=n_classes,
            proj_dim=args.proj_dim,
            use_pre_neck=args.strip_neck
        ).to(device)
        # Freeze MedSAM ALWAYS (SPIE style)
        for p in model.encoder.parameters():
            p.requires_grad = False

    # ======================================
    # PHASE 1 — TRIPLET ALIGNMENT
    # ======================================
    print("\n==============================")
    print("PHASE 1: Triplet Alignment")
    print("==============================\n")

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze projection + CNN encoder (but not MedSAM)
    for p in model.cnn.parameters():
        p.requires_grad = True
    for p in model.projection.parameters():
        p.requires_grad = True

    optimizer_triplet = torch.optim.Adam(
        [
            {"params": model.cnn.parameters(), "lr": args.triplet_lr, "weight_decay": args.triplet_wd},
            {"params": model.projection.parameters(), "lr": args.triplet_lr, "weight_decay": args.triplet_wd},
        ]
    )

    early_triplet = EarlyStopper(patience=args.triplet_patience)
    best_lr_clf = None
    last_lr_clf = None

    for epoch in range(1, args.triplet_epochs + 1):
        tr_loss = run_epoch_triplet(
            train_loader, model,
            histo_dict=train_histo_buckets,
            margin=args.triplet_margin,
            optimizer=optimizer_triplet,
            device=device
        )
        va_loss = run_epoch_triplet(
            val_loader, model,
            histo_dict=val_histo_buckets,
            margin=args.triplet_margin,
            optimizer=None,
            device=device
        )

        # LR proxy
        X_tr, y_tr = extract_embeddings(train_loader, model, device=device)
        X_va, y_va = extract_embeddings(val_loader,   model, device=device)

        lr_metrics = eval_with_logreg(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va,   y_val=y_va,
            n_classes=n_classes,
            max_iter=args.lr_max_iter
        )
        last_lr_clf = lr_metrics["clf"]

        print(f"[TRIP {epoch:03d}] "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"LR_val_bacc={lr_metrics['bacc']:.4f}")

        # wandb
        if wb is not None:
            payload = {
                "epoch": epoch,
                "aux/triplet/train_loss": tr_loss,
                "aux/triplet/val_loss": va_loss,
                "aux/lr_val/bacc": lr_metrics["bacc"],
                "aux/lr_val/acc": lr_metrics["acc"],
                "aux/lr_val/f1_macro": lr_metrics["f1_macro"],
                "aux/lr_val/macro_auc": lr_metrics["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"aux/lr_val/acc_c{c}"] = lr_metrics["per_acc"][c]
                payload[f"aux/lr_val/auc_c{c}"] = lr_metrics["per_auc"][c]
            wandb_log(wb, payload)

        # Early stopping
        if early_triplet.update(lr_metrics["bacc"], model, save_path=outdir / "ckpt_triplet_best.pt"):
            best_lr_clf = lr_metrics["clf"]
            print("  ↳ new best triplet model")
        else:
            print(f"  ↳ no improvement ({early_triplet.num_bad}/{early_triplet.patience})")
            if early_triplet.num_bad >= early_triplet.patience:
                print("Triplet early stopping.")
                break

    # reload best
    if not early_triplet.load_best_into(model, strict=False):
        print("[triplet][WARN] no improvement recorded")
        if best_lr_clf is None:
            best_lr_clf = last_lr_clf

    # ======================================
    # PHASE 2 — HEAD TRAINING
    # ======================================
    print("\n==============================")
    print("PHASE 2: Head (classification) training")
    print("==============================\n")

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classifier head (+ projection optionally)
    trainables = list(model.classifier.parameters())

    for p in trainables:
        p.requires_grad = True

    optimizer_head = torch.optim.Adam(
        trainables,
        lr=args.head_lr,
        weight_decay=args.head_wd
    )
    early_head = EarlyStopper(patience=args.head_patience)

    for epoch in range(1, args.head_epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_bacc = run_epoch_ce(
            train_loader, model, w_ce=w_ce,
            optimizer=optimizer_head, device=device
        )
        val_res = evaluate_loader(val_loader, model, collect_outputs=True, w_ce=w_ce, device=device, n_classes=n_classes)

        pcs, auc_str = format_perclass_acc_auc(
            val_res["per_acc"], val_res["per_auc"], val_res["macro_auc"], n_classes
        )
        extra2 = format_sens_spec(
            val_res["per_tpr"], val_res["per_tnr"],
            val_res["macro_tpr"], val_res["macro_tnr"],
            n_classes
        )

        print(f"[HEAD {epoch:03d}] "
              f"train_loss={tr_loss:.4f} train_bacc={tr_bacc:.4f} || "
              f"val_loss={val_res['loss']:.4f} val_bacc={val_res['bacc']:.4f} "
              f"{pcs}{auc_str}{extra2}")
        print(format_confusion_matrix(val_res["cm"], n_classes))

        if wb is not None:
            payload = {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/bacc": tr_bacc,
                "aux/train/acc": tr_acc,
                "aux/train/f1_macro": tr_f1,
                "val/loss": val_res["loss"],
                "val/bacc": val_res["bacc"],
                "val/macro_auc": val_res["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"val/acc_c{c}"] = val_res["per_acc"][c]
                payload[f"val/auc_c{c}"] = val_res["per_auc"][c]
            payload["aux/val/macro_tpr"] = val_res["macro_tpr"]
            payload["aux/val/macro_tnr"] = val_res["macro_tnr"]
            wandb_log(wb, payload)

        # Early stop
        if early_head.update(val_res["bacc"], model, save_path=outdir / "ckpt_head_best.pt"):
            print("  ↳ new best head model")
        else:
            print(f"  ↳ no improvement ({early_head.num_bad}/{early_head.patience})")
            if early_head.num_bad >= early_head.patience:
                print("Head early stopping.")
                break

    # reload best
    if not early_head.load_best_into(model, strict=False):
        print("[head][WARN] no improvement recorded")

    # ======================================
    # FINAL VAL
    # ======================================
    val_final = evaluate_loader(val_loader, model, collect_outputs=True, w_ce=w_ce, device=device, n_classes=n_classes)
    torch.save({
        "embeddings": val_final["embeddings"].cpu(),
        "labels": torch.tensor(val_ds.df[args.label6_column].values)
    }, outdir / "val_embeddings.pt")

    # ======================================
    # FINAL TEST
    # ======================================
    if test_loader is None:
        print("[TEST] No test folds provided. Done.")
        wandb_finish(wb)
        return

    test_final = evaluate_loader(test_loader, model, collect_outputs=True, w_ce=w_ce, device=device, n_classes=n_classes)
    emb_test = test_final["embeddings"].cpu()
    y_test = test_final["labels"]

    torch.save({"embeddings": emb_test, "labels": y_test}, outdir / "test_embeddings.pt")

    print("\n===== FINAL TEST METRICS =====")
    pcs_t, auc_t = format_perclass_acc_auc(
        test_final["per_acc"], test_final["per_auc"], test_final["macro_auc"], n_classes
    )
    extra_t = format_sens_spec(
        test_final["per_tpr"], test_final["per_tnr"],
        test_final["macro_tpr"], test_final["macro_tnr"],
        n_classes
    )
    # --- CHANGED LINE: now includes f1 and matches other script's style ---
    print(
        "[TEST] "
        f"loss {test_final['loss']:.4f} "
        f"acc {test_final['acc']:.4f} "
        f"BAL-acc {test_final['bacc']:.4f} "
        f"f1 {test_final['f1_macro']:.4f} | "
        f"{pcs_t}{auc_t}{extra_t}"
    )
    print(format_confusion_matrix(test_final["cm"], n_classes))

    # ============================
    # UMAP: MRI TEST EMBEDDINGS
    # ============================
    print("\n[UMAP] Computing MRI-test UMAP…")
    reducer_mri = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=args.seed
    )
    Z_mri = reducer_mri.fit_transform(emb_test.numpy())

    plt.figure(figsize=(6, 5))
    plt.scatter(Z_mri[:, 0], Z_mri[:, 1],
                c=y_test.numpy(), cmap="tab10", s=6)
    plt.title("MRI Test Embeddings (UMAP)")
    plt.tight_layout()
    plt.savefig(figdir / "umap_mri_test.png", dpi=300)
    plt.close()

    print(f"Saved MRI test UMAP → {figdir/'umap_mri_test.png'}")

    # ============================
    # UMAP: HISTO TEST ENCODINGS
    # ============================
    print("[UMAP] Computing Histopathology-test UMAP…")

    # Flatten all histo vectors into numpy array
    histo_embs = []
    histo_labels = []
    for cls, lst in enumerate(test_histo_buckets):
        for v in lst:
            histo_embs.append(v.numpy())
            histo_labels.append(cls)

    if len(histo_embs) > 0:
        histo_embs = np.stack(histo_embs, axis=0)
        histo_labels = np.array(histo_labels)

        reducer_histo = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=args.seed
        )
        Z_histo = reducer_histo.fit_transform(histo_embs)

        plt.figure(figsize=(6, 5))
        plt.scatter(Z_histo[:, 0], Z_histo[:, 1],
                    c=histo_labels, cmap="tab10", s=6)
        plt.title("Histopathology Test Encodings (UMAP)")
        plt.tight_layout()
        plt.savefig(figdir / "umap_histo_test.png", dpi=300)
        plt.close()

        print(f"Saved HISTO test UMAP → {figdir/'umap_histo_test.png'}")
    else:
        print("[UMAP] WARNING: No histopathology vectors found in test set.")

    wandb_finish(wb)


if __name__ == "__main__":
    main()
