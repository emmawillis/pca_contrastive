#!/usr/bin/env python3
# train_head_from_encoder.py

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry
import train_utils

# shared helpers
from train_utils import (
    build_datasets_and_loaders,   # now returns train/val/test loaders
    evaluate_loader,              # can collect logits/labels/embeddings
    format_perclass_acc_auc,
    format_sens_spec,
    print_operating_points_table,
    EarlyStopper,
    load_encoder_and_projector,   # load encoder.* and proj.* from checkpoint
    save_embeddings,
    wandb_init, wandb_log, wandb_finish,
)

def run_eval_print(val_loader, model, w_ce, device, n_classes):
    """Eval (no embeddings) + formatted strings for stdout."""
    val = evaluate_loader(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=False)
    pcs, auc_part = format_perclass_acc_auc(val["per_acc"], val["per_auc"], val["macro_auc"], n_classes)
    extra2 = format_sens_spec(val["per_tpr"], val["per_tnr"], val["macro_tpr"], val["macro_tnr"], n_classes)
    return val, pcs, auc_part, extra2

def run_epoch_ce(loader, model, w_ce, optimizer=None, device="cuda"):
    """Train/eval one epoch with CE; returns (loss, acc, f1_macro)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    ce = nn.CrossEntropyLoss(weight=w_ce)  # w_ce should already be on device
    total_loss, total_n, total_correct = 0.0, 0, 0

    # for f1
    all_pred, all_true = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

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

        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)

    # compute macro F1 on the fly
    if all_pred:
        from sklearn.metrics import f1_score
        y_pred_np = torch.cat(all_pred).numpy()
        y_true_np = torch.cat(all_true).numpy()
        f1m = float(f1_score(y_true_np, y_pred_np, average="macro"))
    else:
        f1m = 0.0

    return avg_loss, acc, f1m

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True, help="Base SAM/MedSAM checkpoint (full model).")
    p.add_argument("--encoder_ckpt", required=True, help="Checkpoint with encoder/proj weights to load.")
    p.add_argument("--outdir", default="./runs/head_finetune")
    p.add_argument("--target", choices=["isup3","isup6", "binary_all", "binary_low_high"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val", default="0")
    p.add_argument("--folds_test", default="4", help="Folds evaluated as held-out test at the end.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--train_proj", action="store_true",
                   help="Also train the projection MLP along with the classifier head.")
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, drop rows with skip==1. Use --no-use-skip to include them.")
    p.add_argument("--label6_column", default="label6")
    # W&B
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="Log epoch metrics to Weights & Biases.")
    p.add_argument("--wandb_project", default="mri-training")
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    print("SCRIPT: train_head_from_encoder.py")
    print("ARGS:", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- datasets/loaders via shared helper (with TEST) --------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.folds_test.split(",") if s.strip()]

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
    )
    w_ce = w_ce.to(device)

    # -------- model --------
    sam = sam_model_registry["vit_b"]()
    # Load base SAM/MedSAM weights (full)
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)

    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,
        proj_dim=args.proj_dim, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,
        pixel_mean_std=None,
    ).to(device)

    # Load encoder-only (and proj) weights from encoder_ckpt
    load_encoder_and_projector(model, Path(args.encoder_ckpt))

    # Freeze everything, then unfreeze head (+ optionally projection MLP)
    for p_ in model.parameters():
        p_.requires_grad = False
    head_params = list(model.head.parameters())
    trainable_params = head_params
    if args.train_proj:
        trainable_params += list(model.proj.parameters())
    for p_ in trainable_params:
        p_.requires_grad = True

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    # W&B
    wb = wandb_init(bool(args.wandb), args.wandb_project, args.wandb_run_name, config=vars(args))

    early = EarlyStopper(patience=args.patience)
    best_path = outdir / "ckpt_head_best.pt"

    # -------- training loop (head/proj only) --------
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_f1 = run_epoch_ce(train_loader, model, w_ce=w_ce, optimizer=optimizer, device=device)
        val, pcs, auc_part, extra2 = run_eval_print(val_loader, model, w_ce, device, n_classes)

        print(f"[{epoch:03d}] head-train: loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} || "
              f"val: loss {val['loss']:.4f} acc {val['acc']:.4f} BAL-acc {val['bacc']:.4f} "
              f"f1 {val['f1_macro']:.4f} | {pcs}{auc_part}{extra2}")
        print(train_utils.format_confusion_matrix(val["cm"], n_classes=n_classes))

        # wandb logging
        if wb is not None:
            lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
            payload = {
                "epoch": epoch,
                "lr": lrs[0] if lrs else None,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "train/f1_macro": tr_f1,
                "val/loss": val["loss"],
                "val/acc": val["acc"],
                "val/bacc": val["bacc"],
                "val/f1_macro": val["f1_macro"],
                "val/macro_auc": val["macro_auc"],
                "val/macro_tpr": val["macro_tpr"],
                "val/macro_tnr": val["macro_tnr"],
            }
            for c in range(n_classes):
                if c in val["per_auc"]:
                    payload[f"val/auc_c{c}"] = val["per_auc"][c]
                if c in val["per_acc"]:
                    payload[f"val/acc_c{c}"] = val["per_acc"][c]
            wandb_log(wb, payload)

        if early.update(val["bacc"], model, best_path):
            print(f"  ↳ new best (val BAL-acc={val['bacc']:.4f}) saved to {best_path}")
        else:
            print(f"  ↳ no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= early.patience:
                print(f"Early stopping triggered at epoch {epoch}: no BAL-acc improvement for {early.patience} epochs.")
                break

    # Keep last state if no improvement; otherwise load in-memory best
    if not early.load_best_into(model, strict=False):
        print("[warn] No validation improvement recorded; using last model state for final eval.")
    model.to(device)
    model.eval()

    # -------- Final VAL (collect embeddings) --------
    val_final = evaluate_loader(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=True)
    logits_val = val_final["logits"]; y_val_t = val_final["labels"]; val_embs = val_final["embeddings"]
    y_val = y_val_t.numpy() if y_val_t.numel() else np.empty((0,), dtype=np.int64)
    probs_val = torch.softmax(logits_val, dim=1).numpy() if logits_val.numel() else np.empty((0, n_classes), dtype=np.float32)
    spec_targets = (0.8, 0.9, 0.95, 0.975, 0.99)
    per_cls_val, macro_val = train_utils.per_class_operating_points(y_val, probs_val, spec_targets)
    print_operating_points_table(per_cls_val, macro_val, spec_targets, dataset_label="validation")

    # -------- TEST (collect embeddings) --------
    if test_loader is not None:
        test_final = evaluate_loader(test_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=True)
        pcs_t, auc_part_t = format_perclass_acc_auc(test_final["per_acc"], test_final["per_auc"], test_final["macro_auc"], n_classes)
        extra2_t = format_sens_spec(test_final["per_tpr"], test_final["per_tnr"], test_final["macro_tpr"], test_final["macro_tnr"], n_classes)
        print(f"[TEST] loss {test_final['loss']:.4f} acc {test_final['acc']:.4f} f1 {test_final['f1_macro']:.4f} | "
              f"{pcs_t}{auc_part_t}{extra2_t}")
        print(train_utils.format_confusion_matrix(test_final["cm"], n_classes=n_classes))

        logits_test = test_final["logits"]; y_test_t = test_final["labels"]; test_embs = test_final["embeddings"]
        y_test = y_test_t.numpy() if y_test_t.numel() else np.empty((0,), dtype=np.int64)
        probs_test = torch.softmax(logits_test, dim=1).numpy() if logits_test.numel() else np.empty((0, n_classes), dtype=np.float32)
        per_cls_test, macro_test = train_utils.per_class_operating_points(y_test, probs_test, spec_targets)
        print_operating_points_table(per_cls_test, macro_test, spec_targets, dataset_label="test")

        # W&B final test summary
        if wb is not None:
            summary = {
                "final_test/loss": test_final["loss"],
                "final_test/acc": test_final["acc"],
                "final_test/bacc": test_final["bacc"],
                "final_test/f1_macro": test_final["f1_macro"],
                "final_test/macro_auc": test_final["macro_auc"],
                "final_test/macro_tpr": test_final["macro_tpr"],
                "final_test/macro_tnr": test_final["macro_tnr"],
            }
            for c in range(n_classes):
                if c in test_final["per_auc"]:
                    summary[f"final_test/auc_c{c}"] = test_final["per_auc"][c]
                if c in test_final["per_acc"]:
                    summary[f"final_test/acc_c{c}"] = test_final["per_acc"][c]
            wandb_log(wb, summary)

        # Save TEST embeddings from best ckpt
        save_embeddings(outdir / "test_embeddings", "test.pt", test_embs, y_test_t)
    else:
        print("[TEST] No test folds provided; skipping test evaluation.")

    # Save VAL embeddings from best ckpt
    save_embeddings(outdir / "val_embeddings", "val.pt", val_embs, y_val_t)

    wandb_finish(wb)

if __name__ == "__main__":
    main()
