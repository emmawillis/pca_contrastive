#!/usr/bin/env python3
# train_head_from_encoder.py

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

# sklearn metrics for evaluation
try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
    )
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

def make_pos_sampler(df: pd.DataFrame, pos_ratio: float = 0.33, seed: int = 1337):
    """Oversample lesion-intersecting slices."""
    is_pos = df["has_lesion"].astype(int).values
    n_pos = int(is_pos.sum()); n_neg = len(is_pos) - n_pos
    assert n_pos > 0, "No positive slices in train folds."
    w_neg = 1.0
    w_pos = (pos_ratio/(1-pos_ratio)) * (n_neg/max(1,n_pos))
    w = np.where(is_pos==1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)

def per_class_acc(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    accs = {}
    for c in range(n_classes):
        mask = (y_true == c)
        accs[c] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
    return accs

def confusion_matrix_str(cm: np.ndarray, n_classes: int):
    labels = ["ISUP01","ISUP23","ISUP45"] if n_classes == 3 else ["ISUP0","ISUP1","ISUP2","ISUP3","ISUP4","ISUP5"]
    header = "true\\pred " + " ".join(f"{lbl:>7}" for lbl in labels)
    lines = ["Confusion matrix (val): rows=true, cols=pred", header]
    for i in range(n_classes):
        row = " ".join(f"{int(cm[i, j]):7d}" for j in range(n_classes))
        lines.append(f"{labels[i]:>9} {row}")
    return "\n".join(lines)

# ---- Sensitivity & Specificity from confusion matrix (NumPy) ----
def tpr_tnr_from_confusion_np(cm_np: np.ndarray):
    """
    cm_np: KxK counts; rows=true, cols=pred. NumPy array.
    Returns:
      per_class_tpr (list), per_class_tnr (list), macro_tpr (float), macro_tnr (float)
    """
    K = cm_np.shape[0]
    per_tpr, per_tnr = [], []
    total = cm_np.sum()
    for c in range(K):
        TP = cm_np[c, c]
        FN = cm_np[c, :].sum() - TP
        FP = cm_np[:, c].sum() - TP
        TN = total - TP - FN - FP
        tpr = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # sensitivity
        tnr = TN / (TN + FP) if (TN + FP) > 0 else np.nan  # specificity
        per_tpr.append(tpr); per_tnr.append(tnr)
    macro_tpr = float(np.nanmean(per_tpr)) if len(per_tpr) else float("nan")
    macro_tnr = float(np.nanmean(per_tnr)) if len(per_tnr) else float("nan")
    return per_tpr, per_tnr, macro_tpr, macro_tnr

# ---- Sensitivity at fixed specificity (final model; needs sklearn) ----
def sensitivity_at_specificity(y_true, y_score, spec_targets=(0.8, 0.9, 0.95, 0.975, 0.99)):
    from sklearn.metrics import roc_curve, auc  # import here to respect _HAS_SK
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    out = {"auc": auc(fpr, tpr)}
    for s in spec_targets:
        mask = spec >= s
        sens = tpr[mask].max() if mask.any() else 0.0
        out[f"sens_at_spec_{int(100*s)}"] = float(sens)
    return out

def per_class_operating_points(y_np, probs_np, spec_targets=(0.8, 0.9, 0.95, 0.975, 0.99)):
    K = probs_np.shape[1]
    per_cls = []
    keys = ["auc"] + [f"sens_at_spec_{int(100*s)}" for s in spec_targets]
    for c in range(K):
        y_bin = (y_np == c).astype(np.int32)
        # Guard: need both positives and negatives present
        if y_bin.sum() == 0 or (len(y_bin) - y_bin.sum()) == 0:
            per_cls.append({"auc": float("nan"), **{f"sens_at_spec_{int(100*s)}": float("nan") for s in spec_targets}})
            continue
        res = sensitivity_at_specificity(y_bin, probs_np[:, c], spec_targets)
        per_cls.append(res)
    macro = {k: float(np.nanmean([d[k] for d in per_cls])) for k in keys}
    return per_cls, macro

@torch.no_grad()
def evaluate(loader, model, w_ce, device="cuda", n_classes=3):
    model.eval()
    ys, yps, prob_list = [], [], []
    ce_loss = nn.CrossEntropyLoss(reduction="sum", weight=w_ce)  # sum to average later
    total_loss, total_n = 0.0, 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits, _ = model(x)
        loss = ce_loss(logits, y)
        probs = torch.softmax(logits, dim=1)

        total_loss += float(loss.item())
        total_n += x.size(0)
        ys.append(y.cpu())
        yps.append(logits.argmax(1).cpu())
        prob_list.append(probs.cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(yps).numpy()
    probs  = torch.cat(prob_list).numpy()  # [N, K]
    avg_loss = total_loss / max(1, total_n)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))

    # per-class acc
    pacc = per_class_acc(y_true, y_pred, n_classes)

    # AUCs (OvR)
    per_auc = {c: float("nan") for c in range(n_classes)}
    macro_auc = float("nan")
    if _HAS_SK:
        auc_vals = []
        for c in range(n_classes):
            y_bin = (y_true == c).astype(np.int32)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                try:
                    auc = roc_auc_score(y_bin, probs[:, c])
                    per_auc[c] = float(auc)
                    auc_vals.append(auc)
                except Exception:
                    pass
        if len(auc_vals) > 0:
            macro_auc = float(np.nanmean(auc_vals))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # ---- NEW: Sensitivity & Specificity per epoch (from confusion matrix) ----
    per_tpr, per_tnr, macro_tpr, macro_tnr = tpr_tnr_from_confusion_np(cm)

    return {
        "loss": avg_loss,
        "acc": acc,
        "bacc": bacc,
        "f1_macro": f1m,
        "per_acc": pacc,
        "per_auc": per_auc,
        "macro_auc": macro_auc,
        "cm": cm,
        "per_tpr": per_tpr,
        "per_tnr": per_tnr,
        "macro_tpr": macro_tpr,
        "macro_tnr": macro_tnr,
    }

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


# ----------------- loading encoder-only weights -----------------
def load_encoder_only(model: MedSAMSliceSpatialAttn, ckpt_path: Path):
    """
    Load only the encoder.* weights from a state_dict saved by the previous script.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)
    enc_pref = "encoder."
    enc_sd = {}
    for k, v in sd.items():
        if k.startswith(enc_pref):
            enc_sd[k[len(enc_pref):]] = v  # strip "encoder." prefix for submodule load
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"[load_encoder_only] Missing keys in encoder: {missing}")
    if unexpected:
        print(f"[load_encoder_only] Unexpected keys in encoder: {unexpected}")
    print(f"[load_encoder_only] Loaded encoder weights from {ckpt_path}")


# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True, help="Base SAM/MedSAM checkpoint (full model).")
    p.add_argument("--encoder_ckpt", required=True, help="Checkpoint from triplet+LR script (will load encoder.* only).")
    p.add_argument("--outdir", default="./runs/head_finetune")
    p.add_argument("--target", choices=["isup3","isup6"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr_head", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--train_proj", action="store_true",
                   help="Also train the projection MLP along with the classifier head.")
    args = p.parse_args()
    print("ARGS: ", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- datasets --------
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
    w_ce = w_ce.to(device)

    # sampler to bump lesion slice rate (optional; mirrors encoder training)
    sampler = make_pos_sampler(train_ds.df, pos_ratio=args.pos_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_to_imgsize)

    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_resize_to_imgsize)

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

    # Load encoder-only weights from encoder_ckpt
    load_encoder_only(model, Path(args.encoder_ckpt))

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

    best_bacc = -1.0
    best_path = outdir / "ckpt_head_best.pt"

    # ---- Early stopping state (same as train.py) ----
    patience = 10
    no_improve = 0

    # -------- training loop (head only) --------
    for epoch in range(1, args.epochs+1):
        tr_loss = run_epoch_ce(train_loader, model, w_ce=w_ce, optimizer=optimizer, device=device)
        val_metrics = evaluate(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes)

        # pretty print
        per_acc_str = "  ".join([
            f"acc[c{c}]={val_metrics['per_acc'][c]:.3f}" if not np.isnan(val_metrics['per_acc'][c]) else f"acc[c{c}]=NA"
            for c in range(n_classes)
        ])
        if not np.isnan(val_metrics["macro_auc"]):
            per_auc_str = "  ".join([
                f"auc[c{c}]={val_metrics['per_auc'][c]:.3f}" if not np.isnan(val_metrics['per_auc'][c]) else f"auc[c{c}]=NA"
                for c in range(n_classes)
            ])
            auc_part = f" | {per_auc_str} | macroAUC={val_metrics['macro_auc']:.3f}"
        else:
            auc_part = " | (AUC unavailable)"

        # ---- per-epoch Sensitivity/Specificity (from confusion matrix) ----
        per_tpr = val_metrics["per_tpr"]
        per_tnr = val_metrics["per_tnr"]
        macro_tpr = val_metrics["macro_tpr"]
        macro_tnr = val_metrics["macro_tnr"]
        sens_str = "  ".join([f"sens[c{c}]={per_tpr[c]:.3f}" if not np.isnan(per_tpr[c]) else f"sens[c{c}]=NA"
                              for c in range(n_classes)])
        spec_str = "  ".join([f"spec[c{c}]={per_tnr[c]:.3f}" if not np.isnan(per_tnr[c]) else f"spec[c{c}]=NA"
                              for c in range(n_classes)])
        extra2 = f" | macroSens={macro_tpr:.3f} macroSpec={macro_tnr:.3f} | {sens_str} | {spec_str}"

        print(f"[{epoch:03d}] head-train: loss {tr_loss:.4f} || "
              f"val: loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} "
              f"BAL-acc {val_metrics['bacc']:.4f} f1 {val_metrics['f1_macro']:.4f} | "
              f"{per_acc_str}{auc_part}{extra2}")
        print(confusion_matrix_str(val_metrics["cm"], n_classes=n_classes))

        # ---- Model selection & early stopping tracking (balanced accuracy) ----
        if val_metrics["bacc"] > best_bacc:
            best_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists right now
            best_bacc = val_metrics["bacc"]
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
            print(f"  ↳ new best (val BAL-acc={best_bacc:.4f}) saved to {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  ↳ no improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}: no BAL-acc improvement for {patience} epochs.")
                break

    # -------- Final model: Sensitivity at fixed specificity (on val) --------
    if _HAS_SK:
        # If no best was saved (e.g., early stop without improvement), save last model to allow metric computation
        if not best_path.exists():
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
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
        per_cls, macro = per_class_operating_points(y_val, probs_val, spec_targets)

        print("\n=== Final model: Sensitivity at fixed specificity (validation) ===")
        header = ["class", "AUC"] + [f"Sens@Spec{int(100*s)}" for s in spec_targets]
        print(" | ".join(f"{h:>12}" for h in header))
        for c, stats in enumerate(per_cls):
            row = [f"c{c}", f"{stats['auc']:.3f}" if not np.isnan(stats['auc']) else "NA"] + [
                f"{stats[f'sens_at_spec_{int(100*s)}']:.3f}" if not np.isnan(stats[f'sens_at_spec_{int(100*s)}']) else "NA"
                for s in spec_targets
            ]
            print(" | ".join(f"{r:>12}" for r in row))
        row = ["macro", f"{macro['auc']:.3f}" if not np.isnan(macro['auc']) else "NA"] + [
            f"{macro[f'sens_at_spec_{int(100*s)}']:.3f}" if not np.isnan(macro[f'sens_at_spec_{int(100*s)}']) else "NA"
            for s in spec_targets
        ]
        print(" | ".join(f"{r:>12}" for r in row))
    else:
        print("\n[warn] sklearn not available: skipped final Sens@Spec computation.")

if __name__ == "__main__":
    main()
