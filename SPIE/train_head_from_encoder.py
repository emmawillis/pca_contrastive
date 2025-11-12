#!/usr/bin/env python3
# train_head_from_encoder.py

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry
import train_utils

# shared helpers
from train_utils import (
    build_datasets_and_loaders,   # now used below
    evaluate_loader,
    format_perclass_acc_auc,
    format_sens_spec,
    print_operating_points_table,
    EarlyStopper,
    load_encoder_and_projector,            # shared loader for encoder-only weights
)

@torch.no_grad()
def run_eval_print(val_loader, model, w_ce, device, n_classes):
    """Run evaluation with shared helper and format strings for logging."""
    val = evaluate_loader(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes)
    pcs, auc_part = format_perclass_acc_auc(val["per_acc"], val["per_auc"], val["macro_auc"], n_classes)
    extra2 = format_sens_spec(val["per_tpr"], val["per_tnr"], val["macro_tpr"], val["macro_tnr"], n_classes)
    return val, pcs, auc_part, extra2

def run_epoch_ce(loader, model, w_ce, optimizer=None, device="cuda"):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_n = 0.0, 0
    ce = nn.CrossEntropyLoss(weight=w_ce)  # (keep as-is)

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

        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)

    return total_loss / max(1, total_n)

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True, help="Base SAM/MedSAM checkpoint (full model).")
    p.add_argument("--encoder_ckpt", required=True, help="Checkpoint from triplet+LR script (will load encoder.* only).")
    p.add_argument("--outdir", default="./runs/head_finetune")
    p.add_argument("--target", choices=["isup3","isup6", "binary_all", "binary_low_high"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val", default="0")
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

    args = p.parse_args()
    print("SCRIPT: train_head_from_encoder.py")
    print("ARGS:", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- datasets/loaders via shared helper --------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    (train_ds, val_ds,
     train_loader, val_loader,
     w_ce, classes_present, n_classes) = build_datasets_and_loaders(
        manifest=args.manifest,
        folds_train=folds_train,
        folds_val=folds_val,
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

    # Load encoder-only weights from encoder_ckpt via shared helper
    load_encoder_and_projector(model, Path(args.encoder_ckpt))

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classifier head (+ optionally projection MLP)
    head_params = list(model.head.parameters())
    trainable_params = head_params
    if args.train_proj:
        trainable_params += list(model.proj.parameters())

    for p in trainable_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr_head, weight_decay=args.wd)

    early = EarlyStopper(patience=args.patience)
    best_path = outdir / "ckpt_head_best.pt"

    # -------- training loop (head only) --------
    for epoch in range(1, args.epochs+1):
        tr_loss = run_epoch_ce(train_loader, model, w_ce=w_ce, optimizer=optimizer, device=device)
        val, pcs, auc_part, extra2 = run_eval_print(val_loader, model, w_ce, device, n_classes)

        print(f"[{epoch:03d}] head-train: loss {tr_loss:.4f} || "
              f"val: loss {val['loss']:.4f} acc {val['acc']:.4f} "
              f"BAL-acc {val['bacc']:.4f} f1 {val['f1_macro']:.4f} | "
              f"{pcs}{auc_part}{extra2}")
        # In your original head script you printed the return value:
        print(train_utils.format_confusion_matrix(val["cm"], n_classes=n_classes))

        if early.update(val["bacc"], model, best_path):
            print(f"  ↳ new best (val BAL-acc={val['bacc']:.4f}) saved to {best_path}")
        else:
            print(f"  ↳ no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= early.patience:
                print(f"Early stopping triggered at epoch {epoch}: no BAL-acc improvement for {early.patience} epochs.")
                break

    # -------- Final model: Sensitivity at fixed specificity (on val) --------
    # If no best was saved (e.g., early stop without improvement), save last model to allow metric computation
    if not best_path.exists():
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, best_path)
        print(f"[warn] No best checkpoint existed; saved last model to {best_path} for final metrics.")

    # reload best head
    ckpt = torch.load(best_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits, _ = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    logits_val = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, n_classes))
    y_val = torch.cat(all_y, dim=0).numpy() if all_y else np.empty((0,), dtype=np.int64)
    probs_val = torch.softmax(logits_val, dim=1).numpy()

    spec_targets = (0.8, 0.9, 0.95, 0.975, 0.99)
    per_cls, macro = train_utils.per_class_operating_points(y_val, probs_val, spec_targets)
    print_operating_points_table(per_cls, macro, spec_targets)

if __name__ == "__main__":
    main()
