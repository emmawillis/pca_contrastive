#!/usr/bin/env python3
# train.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_picai_slices import PicaiSliceDataset, map_binary_all, map_binary_low_high, map_isup3
from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry
import train_utils

# ----------------- train / val -----------------
def run_epoch(loader, model, loss_fn, optimizer=None, device="cuda",
              return_outputs=False, epoch_idx=None, probe_first_k=0, label_names=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_logits, all_y = [], []
    for bi, batch in enumerate(loader):
        # ---- tiny probe: print the first K batch distributions in epoch 1 ----
        if train_mode and probe_first_k and (epoch_idx == 1) and (bi < probe_first_k):
            y = batch["label"].cpu().numpy()
            hl = np.asarray(batch.get("has_lesion", []))
            if hl.size == 0:
                hl = np.zeros_like(y, dtype=np.int64)
            pos = int(hl.sum()); tot = int(hl.shape[0])
            # per-label counts
            uniq, cnts = np.unique(y, return_counts=True)
            if label_names is None:
                label_names = [f"c{i}" for i in range(int(y.max())+1)]
            lab_str = " | ".join(
                f"{(label_names[u] if 0 <= u < len(label_names) else f'c{u}')}={c} ({c/tot:.0%})"
                for u, c in zip(uniq, cnts)
            )
            print(f"[probe][epoch1 batch {bi:02d}] has_lesion: {pos}/{tot} ({pos/tot:.0%}) || labels: {lab_str}")

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

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)

    logits_cat = torch.cat(all_logits) if all_logits else torch.empty(0)
    y_cat = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)

    if logits_cat.ndim == 2 and logits_cat.numel() > 0:
        K = logits_cat.shape[1]
        y_pred = logits_cat.argmax(dim=1).cpu().numpy()
        y_true = y_cat.cpu().numpy()
        f1 = float(f1_score(y_true, y_pred,
                            average="macro",
                            labels=list(range(K)),
                            zero_division=0))
    else:
        f1 = 0.0

    if return_outputs:
        return avg_loss, acc, f1, logits_cat, y_cat
    return avg_loss, acc, f1

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--outdir", default="./runs/simple")
    p.add_argument("--target", choices=["isup3","isup6","binary_low_high", "binary_all"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3") # holding back 4 as test set
    p.add_argument("--folds_val", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4) # 1e-6 TODO try 1e-5 to have bigger rate on clssifier , 1e-7
    p.add_argument("--wd", type=float, default=1e-4) # 0 TODO maybe try 1e-2
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True,
                help="If true, drop rows with skip==1. Use --no-use-skip to include them.")

    # TODO double check how cropping is working - centered???
    # TODO if new lr doesnt work, try LIGHT augmentation 
    # TODO log distribution of batches to check 33% have lesion
        # [probe][epoch1 batch 00] has_lesion: 1/16 (6%) || labels: ISUP01=15 (94%) | ISUP23=1 (6%)
        # [probe][epoch1 batch 01] has_lesion: 5/16 (31%) || labels: ISUP01=11 (69%) | ISUP23=5 (31%)

    # TODO use 50% pos_ratio for binary 

    # --- NEW: number of epochs to keep the MedSAM encoder frozen before unfreezing ---
    p.add_argument("--freeze_epochs", type=int, default=2,
                   help="Freeze MedSAM encoder for this many epochs, then unfreeze.")
    args = p.parse_args()

    print("ARGS: ", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()!=""]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()!=""]

    # -------- dataset --------
    train_ds = PicaiSliceDataset(
        manifest_csv=args.manifest,
        folds=folds_train,
        use_skip=args.use_skip,
        target=args.target,
        channels=("path_T2","path_ADC","path_HBV"),
        missing_channel_mode="zeros",
        pct_lower=0.5, pct_upper=99.5,   # per-slice 0.5–99.5% clip → [0,1]
        cache_size=64,
    )
    val_ds = PicaiSliceDataset(
        manifest_csv=args.manifest,
        folds=folds_val,
        use_skip=args.use_skip,
        target=args.target,
        channels=("path_T2","path_ADC","path_HBV"),
        missing_channel_mode="zeros",
        pct_lower=0.5, pct_upper=99.5,
        cache_size=32,
    )

    # class weights from TRAIN distribution
    w_ce, classes_present = train_utils.class_weights_from_train(train_ds.df, target=args.target)
    n_classes = len(classes_present)
    w_ce = w_ce.to(device)

    # sampler to bump lesion slice rate
    sampler = train_utils.make_pos_sampler(train_ds.df, pos_ratio=args.pos_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True,
                              collate_fn=train_utils.collate_resize_to_imgsize)

    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=train_utils.collate_resize_to_imgsize)

    # -------- model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,
        proj_dim=args.proj_dim, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,              # pre-neck + spatial attention
        pixel_mean_std=None,            # inputs already in [0,1]
    ).to(device)

    # --- Freeze encoder for warmup ---
    for p_ in model.encoder.parameters():
        p_.requires_grad = False
    # Keep a handle to encoder params for later unfreezing
    encoder_params = list(model.encoder.parameters())

    # Optimizer for currently-trainable params (encoder excluded)
    optimizer = torch.optim.AdamW(
        (p_ for p_ in model.parameters() if p_.requires_grad),
        lr=args.lr, weight_decay=args.wd
    )
    criterion = nn.CrossEntropyLoss(weight=w_ce)

    best_bacc = -1.0
    best_path = outdir / "ckpt_best.pt"

    # ---- Early stopping state ----
    patience = 10
    no_improve = 0

    # -------- loop --------
    for epoch in range(1, args.epochs+1):
        # --- Unfreeze after warmup ---
        if epoch == args.freeze_epochs + 1:
            for p_ in encoder_params:
                p_.requires_grad = True
            # Add encoder params as a new param group (often with lower LR)
            base_lr = args.lr
            enc_lr = base_lr * 0.1
            optimizer.add_param_group({
                "params": encoder_params,
                "lr": enc_lr,
                "weight_decay": args.wd,
            })
            print(f"→ Unfroze encoder at epoch {epoch}; added to optimizer with lr={enc_lr:g}")

        tr_loss, tr_acc, tr_f1 = run_epoch(
            train_loader, model, criterion, optimizer=optimizer, device=device,
            epoch_idx=epoch, probe_first_k=8, label_names=train_utils.get_label_names(args.target)  # <= log 2 batches in epoch 1
        )
        va_loss, va_acc, va_f1, va_logits, va_y = run_epoch(
            val_loader, model, criterion, optimizer=None, device=device, return_outputs=True
        )
        # Per-class metrics on validation
        per_acc, per_auc, balanced_acc, macro_auc = train_utils.per_class_metrics(va_logits, va_y)
        # pretty print
        pcs = "  ".join([f"acc[c{c}]={per_acc[c]:.3f}" if not np.isnan(per_acc[c]) else f"acc[c{c}]=NA"
                         for c in range(va_logits.shape[1])])
        aucs = "  ".join([f"auc[c{c}]={per_auc[c]:.3f}" if per_auc[c] is not None else f"auc[c{c}]=NA"
                            for c in range(va_logits.shape[1])])
        extra = f" | {pcs} | {aucs} | macroAUC={macro_auc:.3f}"

        cm = confusion_matrix(va_y, va_logits, labels=list(range(n_classes)))

        # ---- NEW: Sensitivity & Specificity per epoch (from confusion matrix) ----
        per_tpr, per_tnr, macro_tpr, macro_tnr = train_utils.tpr_tnr_from_confusion(cm)
        sens_str = "  ".join([f"sens[c{c}]={per_tpr[c]:.3f}" if not np.isnan(per_tpr[c]) else f"sens[c{c}]=NA"
                              for c in range(n_classes)])
        spec_str = "  ".join([f"spec[c{c}]={per_tnr[c]:.3f}" if not np.isnan(per_tnr[c]) else f"spec[c{c}]=NA"
                              for c in range(n_classes)])
        extra2 = f" | macroSens={macro_tpr:.3f} macroSpec={macro_tnr:.3f} | {sens_str} | {spec_str}"

        print(f"[{epoch:03d}] "
              f"train: loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"val: loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}{extra}{extra2}")
        train_utils.format_confusion_matrix(cm, n_classes=n_classes)

        # ---- Model selection & early stopping tracking ----
        if balanced_acc > best_bacc:
            best_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists right now
            best_bacc = balanced_acc
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
            print(f"  ↳ saved best to {best_path} (BAL-acc={balanced_acc:.4f})")
            no_improve = 0  # reset patience
        else:
            no_improve += 1
            print(f"  ↳ no improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}: no BAL-acc improvement for {patience} epochs.")
                break

    # -------- Final model: Sensitivity at fixed specificity (on val) --------
    # reload best
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
    logits_val = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, n_classes)
    y_val = torch.cat(all_y, dim=0).numpy() if all_y else np.empty((0,), dtype=np.int64)
    probs_val = torch.softmax(logits_val, dim=1).numpy()

    spec_targets = (0.8, 0.9, 0.95, 0.975, 0.99)
    per_cls, macro = train_utils.per_class_operating_points(y_val, probs_val, spec_targets)

    print("\n=== Final model: Sensitivity at fixed specificity (validation) ===")
    header = ["class", "AUC"] + [f"Sens@Spec{int(100*s)}" for s in spec_targets]
    print(" | ".join(f"{h:>12}" for h in header))
    for c, stats in enumerate(per_cls):
        row = [f"c{c}", f"{stats['auc']:.3f}"] + [f"{stats[f'sens_at_spec_{int(100*s)}']:.3f}" for s in spec_targets]
        print(" | ".join(f"{r:>12}" for r in row))
    row = ["macro", f"{macro['auc']:.3f}"] + [f"{macro[f'sens_at_spec_{int(100*s)}']:.3f}" for s in spec_targets]
    print(" | ".join(f"{r:>12}" for r in row))

if __name__ == "__main__":
    main()
