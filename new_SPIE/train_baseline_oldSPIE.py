#!/usr/bin/env python3
# train_baseline_oldSPIE.py
#
# Baseline SPIE-style training:
# - Frozen MedSAM encoder (optionally strip neck)
# - MRIClassifierCNN (old SPIE architecture)
# - Weighted Cross-Entropy classification
# - Early stopping on VAL balanced accuracy
# - Final evaluation on TEST (AUC, per-class AUC, sensitivity@40/60/90% spec)
# - Save validation & test embeddings
# - Save UMAP of MRI-test embeddings
# - wandb logging in the same style as new scripts

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score
import umap.umap_ as umap
import matplotlib.pyplot as plt

import wandb

from segment_anything import sam_model_registry
from train_utils import (
    build_datasets_and_loaders,
    evaluate_loader,
    format_perclass_acc_auc,
    format_sens_spec,
    sensitivity_at_specificity,
    per_class_operating_points,
    format_confusion_matrix,
    EarlyStopper,
    set_seed,
    wandb_init, wandb_log, wandb_finish,
)

# old SPIE CNN
from cnn import MRIClassifierCNN
from cnn_frozen import MRIClassifierFrozenCNN
from dataset_frozen_medsam import PicaiSliceFrozenEncodingDataset


# ----------------------------------------------------
# One epoch CE
# ----------------------------------------------------
def run_epoch_ce(loader, model, loss_fn, optimizer=None, device="cuda"):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_n, total_correct = 0.0, 0, 0
    all_pred, all_true = [], []

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

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(1)
        total_correct += (pred == y).sum().item()
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)

    if all_pred:
        y_pred_np = torch.cat(all_pred).numpy()
        y_true_np = torch.cat(all_true).numpy()
        f1m = float(f1_score(y_true_np, y_pred_np, average="macro"))
        bacc = float(balanced_accuracy_score(y_true_np, y_pred_np))
    else:
        f1m, bacc = 0.0, 0.0

    return avg_loss, acc, f1m, bacc


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-frozen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--medsam_encodings", default="/home/ewillis/projects/aip-medilab/shared/picai/picai_medsam_zero_shot_encodings")

    p.add_argument("--seed", type=int, default=42)

    # Data
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

    # Model
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--proj_dim", type=int, default=128,
                   help="Projection dimension for MRIClassifierCNN (SPIE used 128).")
    p.add_argument("--strip_neck", action=argparse.BooleanOptionalAction, default=False)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)

    # Output
    p.add_argument("--outdir", default="./runs/baseline_oldSPIE")

    # wandb
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", default="mri-training")
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    print("SCRIPT: train_baseline_oldSPIE.py")
    print("ARGS:", args)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    # ----------------------------
    # Parse folds
    # ----------------------------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.folds_test.split(",") if s.strip()]

    # ----------------------------
    # wandb init
    # ----------------------------
    wb = wandb_init(bool(args.wandb),
                    args.wandb_project,
                    args.wandb_run_name,
                    config=vars(args))
    if wb is not None:
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        wandb.define_metric("test/*",  step_metric="epoch")
        wandb.define_metric("aux/*",   step_metric="epoch")

    # ----------------------------
    # Dataset loaders
    # ----------------------------
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

    # ----------------------------
    # Build model
    # ----------------------------
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

        # freeze MedSAM
        for p_ in model.encoder.parameters():
            p_.requires_grad = False

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.wd
    )
    ce_loss = nn.CrossEntropyLoss(weight=w_ce)

    early = EarlyStopper(patience=args.patience)
    best_path = outdir / "ckpt_best.pt"

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_bacc = run_epoch_ce(
            train_loader, model, ce_loss, optimizer=optimizer, device=device
        )

        val = evaluate_loader(val_loader, model, w_ce=w_ce, collect_outputs=True, device=device, n_classes=n_classes)
        pcs, auc_part = format_perclass_acc_auc(
            val["per_acc"], val["per_auc"], val["macro_auc"], n_classes
        )
        extra2 = format_sens_spec(
            val["per_tpr"], val["per_tnr"],
            val["macro_tpr"], val["macro_tnr"],
            n_classes
        )

        print(f"[{epoch:03d}] "
              f"train: loss {tr_loss:.4f} bacc {tr_bacc:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} || "
              f"val: loss {val['loss']:.4f} acc {val['acc']:.4f} BAL-acc {val['bacc']:.4f} "
              f"f1 {val['f1_macro']:.4f} | {pcs}{auc_part}{extra2}")
        print(format_confusion_matrix(val["cm"], n_classes))

        # wandb logging
        if wb is not None:
            payload = {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/bacc": tr_bacc,
                "aux/train/acc": tr_acc,
                "aux/train/f1_macro": tr_f1,
                "val/loss": val["loss"],
                "val/bacc": val["bacc"],
                "val/macro_auc": val["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"val/acc_c{c}"] = val["per_acc"][c]
                payload[f"val/auc_c{c}"] = val["per_auc"][c]
            payload["aux/val/macro_tpr"] = val["macro_tpr"]
            payload["aux/val/macro_tnr"] = val["macro_tnr"]
            wandb_log(wb, payload)

        if early.update(val["bacc"], model, best_path):
            print(f"  ↳ saved best (BAL-acc={val['bacc']:.4f})")
        else:
            print(f"  ↳ no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= early.patience:
                print("Early stopping.")
                break

    # ----------------------------
    # reload best
    # ----------------------------
    loaded = early.load_best_into(model, strict=False)
    if not loaded:
        print("[warn] No improvement logged; using last model state.")

    model.eval()

    # ----------------------------------------------------
    # Final Validation (embedding save)
    # ----------------------------------------------------
    val_final = evaluate_loader(val_loader, model, collect_outputs=True, w_ce=w_ce, device=device, n_classes=n_classes)
    torch.save({
        "embeddings": val_final["embeddings"].cpu(),
        "labels": torch.tensor(val_ds.df[args.label6_column].values)
    }, outdir / "val_embeddings.pt")

    # ----------------------------------------------------
    # Final Test (embedding + UMAP)
    # ----------------------------------------------------
    if test_loader is not None:
        test_final = evaluate_loader(
            test_loader, model, collect_outputs=True, w_ce=w_ce,
            device=device, n_classes=n_classes
        )

        # ---- NEW: print full test metrics (mirroring val print) ----
        pcs_test, auc_part_test = format_perclass_acc_auc(
            test_final["per_acc"], test_final["per_auc"], test_final["macro_auc"], n_classes
        )
        extra2_test = format_sens_spec(
            test_final["per_tpr"], test_final["per_tnr"],
            test_final["macro_tpr"], test_final["macro_tnr"],
            n_classes
        )

        print(
            "[TEST] "
            f"loss {test_final['loss']:.4f} "
            f"acc {test_final['acc']:.4f} "
            f"BAL-acc {test_final['bacc']:.4f} "
            f"f1 {test_final['f1_macro']:.4f} | "
            f"{pcs_test}{auc_part_test}{extra2_test}"
        )
        print(format_confusion_matrix(test_final["cm"], n_classes))

        # Save test embeddings
        emb_test = test_final["embeddings"].cpu()
        y_test = test_final["labels"]
        torch.save({"embeddings": emb_test, "labels": y_test}, outdir / "test_embeddings.pt")

        # ================================
        # ### UMAP SECTION (MRI test)
        # ================================
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=args.seed
        )
        Z = reducer.fit_transform(emb_test.numpy())

        plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], c=y_test.numpy(), cmap="tab10", s=6)
        plt.title("MRI Test Embeddings (UMAP)")
        plt.savefig(figdir / "umap_mri_test.png", dpi=300)
        plt.close()

        print(f"Saved MRI-test UMAP → {figdir/'umap_mri_test.png'}")

    wandb_finish(wb)


if __name__ == "__main__":
    main()
