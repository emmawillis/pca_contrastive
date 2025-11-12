import numpy as np
import pandas as pd
from collections import Counter

import torch

from dataset_picai_slices import map_binary_all, map_binary_low_high, map_isup3, PicaiSliceDataset

from torch.utils.data import DataLoader, WeightedRandomSampler

from ISUPMedSAM import IMG_SIZE
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, auc, roc_curve
)
import torch.nn.functional as F

def get_label_names(target = None, n_classes = None):
    if target == "isup3" or n_classes == 3:
        return ["ISUP01", "ISUP23", "ISUP45"]  # c0,c1,c2
    elif target == "binary_all" or n_classes == 2:
        return ["no csPCa", "yes csPCa"]
    elif target == "binary_low_high":
        return ["LOW(ISUP01)", "HIGH(ISUP45)"]
    else:
        return ["ISUP0", "ISUP1", "ISUP2", "ISUP3", "ISUP4", "ISUP5"]


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

def make_pos_sampler(df: pd.DataFrame, pos_ratio: float = 0.33, seed: int = 1337):
    """Oversample lesion-intersecting slices."""
    is_pos = df["has_lesion"].astype(int).values
    n_pos = int(is_pos.sum()); n_neg = len(is_pos) - n_pos
    assert n_pos > 0, "No positive slices in train folds."
    w_neg = 1.0
    w_pos = (pos_ratio/(1-pos_ratio)) * (n_neg/max(1,n_pos))
    w = np.where(is_pos==1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)


def class_weights_from_train(df: pd.DataFrame, target: str):
    """Return torch.FloatTensor of class weights (mean-normalized)."""
    if target == "isup3":
        y = df["label6"].map(map_isup3)
    elif target == "binary_low_high":
        y = df["label6"].map(map_binary_low_high)
    elif target == "binary_all":
        y = df["label6"].map(map_binary_all)
    else:
        y = df["label6"]
    classes = sorted(int(c) for c in y.unique())
    cnt = Counter(int(v) for v in y.tolist())
    K, N = len(classes), len(y)
    ws = [N / (K * max(1, cnt.get(c, 0))) for c in classes]
    m = sum(ws)/len(ws)
    ws = [w/m for w in ws]
    return torch.tensor(ws, dtype=torch.float32), classes

# ---- Sensitivity & Specificity from confusion matrix ----
def tpr_tnr_from_confusion(cm: np.ndarray):
    """
    Compute per-class Sensitivity (TPR) and Specificity (TNR) from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (K, K), rows=true classes, cols=predicted classes.
        Must be integer-like counts. (Produced by sklearn.metrics.confusion_matrix.)

    Returns
    -------
    per_tpr : list[float]
        Per-class true positive rate (sensitivity/recall).
    per_tnr : list[float]
        Per-class true negative rate (specificity).
    macro_tpr : float
        Mean of per-class TPR, ignoring NaNs when a class has no positives.
    macro_tnr : float
        Mean of per-class TNR, ignoring NaNs when a class has no negatives.
    """
    cm_np = np.asarray(cm, dtype=np.int64)
    K = cm_np.shape[0]

    per_tpr, per_tnr = [], []
    total = cm_np.sum()

    for c in range(K):
        TP = int(cm_np[c, c])
        FN = int(cm_np[c, :].sum() - TP)
        FP = int(cm_np[:, c].sum() - TP)
        TN = int(total - TP - FN - FP)

        # Sensitivity (recall)
        tpr = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        # Specificity
        tnr = TN / (TN + FP) if (TN + FP) > 0 else np.nan

        per_tpr.append(float(tpr))
        per_tnr.append(float(tnr))

    macro_tpr = float(np.nanmean(per_tpr)) if per_tpr else float("nan")
    macro_tnr = float(np.nanmean(per_tnr)) if per_tnr else float("nan")
    return per_tpr, per_tnr, macro_tpr, macro_tnr

# ---- Sensitivity at fixed specificity (final model; needs sklearn) ----
def sensitivity_at_specificity(y_true, y_score, spec_targets=(0.8, 0.9, 0.95, 0.975, 0.99)):
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

def format_confusion_matrix(cm: np.ndarray, n_classes: int):
    labels = get_label_names(n_classes=n_classes)
    header = "true\\pred " + " ".join(f"{lbl:>7}" for lbl in labels)
    lines = ["Confusion matrix (LR val): rows=true, cols=pred", header]
    for i in range(n_classes):
        row = " ".join(f"{int(cm[i, j]):7d}" for j in range(n_classes))
        lines.append(f"{labels[i]:>9} {row}")
    return "\n".join(lines)

def per_class_metrics(logits: torch.Tensor, y: torch.Tensor):
    """
    Returns:
      - per-class accuracy dict {class_idx: acc}
      - per-class AUC dict {class_idx: auc}  (if sklearn available)
      - balanced_acc (macro over per-class acc)
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

    balanced_acc = float(np.nanmean(list(accs.values()))) if accs else float("nan")

    return accs, aucs, balanced_acc, macro_auc


def build_datasets_and_loaders(
    manifest: str,
    folds_train, folds_val,
    target: str,
    use_skip: bool = True,
    label6_column: str = 'label6',
    channels=("path_T2","path_ADC","path_HBV"),
    missing_channel_mode="zeros",
    pct_lower: float = 0.5, pct_upper: float = 99.5,
    cache_train: int = 64, cache_val: int = 32,
    batch_size: int = 16,
    pos_ratio: float = 0.33,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_ds = PicaiSliceDataset(
        manifest_csv=manifest,
        folds=folds_train,
        use_skip=use_skip,
        label6_column=label6_column,
        target=target,
        channels=channels,
        missing_channel_mode=missing_channel_mode,
        pct_lower=pct_lower, pct_upper=pct_upper,
        cache_size=cache_train,
    )
    val_ds = PicaiSliceDataset(
        manifest_csv=manifest,
        folds=folds_val,
        use_skip=use_skip,
        label6_column=label6_column,
        target=target,
        channels=channels,
        missing_channel_mode=missing_channel_mode,
        pct_lower=pct_lower, pct_upper=pct_upper,
        cache_size=cache_val,
    )
    # weights / classes
    w_ce, classes_present = class_weights_from_train(train_ds.df, target=target)
    n_classes = len(classes_present)

    # sampler for positive ratio
    sampler = make_pos_sampler(train_ds.df, pos_ratio=pos_ratio)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_resize_to_imgsize
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_resize_to_imgsize
    )
    return train_ds, val_ds, train_loader, val_loader, w_ce, classes_present, n_classes


# 2) Standard evaluation on a loader (used by both scripts) --------------------
@torch.no_grad()
def evaluate_loader(loader, model, w_ce: torch.Tensor, device="cuda", n_classes: int = 3):
    model.eval()
    ce_loss = nn.CrossEntropyLoss(reduction="sum", weight=w_ce.to(device))
    ys, yps, logits_list = [], [], []
    total_loss, total_n = 0.0, 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits, _ = model(x)
        loss = ce_loss(logits, y)

        total_loss += float(loss.item())
        total_n += x.size(0)
        ys.append(y.cpu())
        yps.append(logits.argmax(1).cpu())
        logits_list.append(logits.cpu())

    if total_n == 0:
        # empty loader fallback
        empty = {
            "loss": 0.0, "acc": 0.0, "bacc": 0.0, "f1_macro": 0.0,
            "per_acc": {c: np.nan for c in range(n_classes)},
            "per_auc": {c: np.nan for c in range(n_classes)},
            "macro_auc": np.nan,
            "cm": np.zeros((n_classes, n_classes), dtype=int),
            "per_tpr": {c: np.nan for c in range(n_classes)},
            "per_tnr": {c: np.nan for c in range(n_classes)},
            "macro_tpr": np.nan, "macro_tnr": np.nan,
        }
        return empty

    y_all = torch.cat(ys)
    y_pred_all = torch.cat(yps).numpy()
    logits_all = torch.cat(logits_list)

    avg_loss = total_loss / total_n
    y_np = y_all.numpy()

    acc = float(accuracy_score(y_np, y_pred_all))
    bacc = float(balanced_accuracy_score(y_np, y_pred_all))
    f1m = float(f1_score(y_np, y_pred_all, average="macro"))

    per_acc, per_auc, _bacc_from_fn, macro_auc = per_class_metrics(logits_all, y_all)

    cm = confusion_matrix(y_np, y_pred_all, labels=list(range(n_classes)))
    per_tpr, per_tnr, macro_tpr, macro_tnr = tpr_tnr_from_confusion(cm)

    return {
        "loss": avg_loss, "acc": acc, "bacc": bacc, "f1_macro": f1m,
        "per_acc": per_acc, "per_auc": per_auc, "macro_auc": macro_auc,
        "cm": cm, "per_tpr": per_tpr, "per_tnr": per_tnr,
        "macro_tpr": macro_tpr, "macro_tnr": macro_tnr,
    }


# 3) Pretty-print helpers (shared) ---------------------------------------------
def format_perclass_acc_auc(per_acc: dict, per_auc: dict, macro_auc: float, n_classes: int):
    pcs = "  ".join([
        f"acc[c{c}]={per_acc[c]:.3f}" if not np.isnan(per_acc[c]) else f"acc[c{c}]=NA"
        for c in range(n_classes)
    ])
    if np.isnan(macro_auc):
        auc_part = " | (AUC unavailable)"
    else:
        aucs = "  ".join([
            f"auc[c{c}]={per_auc[c]:.3f}" if not np.isnan(per_auc[c]) else f"auc[c{c}]=NA"
            for c in range(n_classes)
        ])
        auc_part = f" | {aucs} | macroAUC={macro_auc:.3f}"
    return pcs, auc_part

def format_sens_spec(per_tpr: dict, per_tnr: dict, macro_tpr: float, macro_tnr: float, n_classes: int):
    sens_str = "  ".join([f"sens[c{c}]={per_tpr[c]:.3f}" if not np.isnan(per_tpr[c]) else f"sens[c{c}]=NA"
                          for c in range(n_classes)])
    spec_str = "  ".join([f"spec[c{c}]={per_tnr[c]:.3f}" if not np.isnan(per_tnr[c]) else f"spec[c{c}]=NA"
                          for c in range(n_classes)])
    extra2 = f" | macroSens={macro_tpr:.3f} macroSpec={macro_tnr:.3f} | {sens_str} | {spec_str}"
    return extra2

def print_operating_points_table(per_cls: list, macro: dict, spec_targets):
    print("\n=== Final model: Sensitivity at fixed specificity (validation) ===")
    header = ["class", "AUC"] + [f"Sens@Spec{int(100*s)}" for s in spec_targets]
    print(" | ".join(f"{h:>12}" for h in header))
    for c, stats in enumerate(per_cls):
        row = [f"c{c}", f"{stats['auc']:.3f}" if not np.isnan(stats['auc']) else "NA"] + [
            f"{stats[f'sens_at_spec_{int(100*s)}']:.3f}"
            if not np.isnan(stats[f'sens_at_spec_{int(100*s)}']) else "NA"
            for s in spec_targets
        ]
        print(" | ".join(f"{r:>12}" for r in row))
    row = ["macro", f"{macro['auc']:.3f}" if not np.isnan(macro['auc']) else "NA"] + [
        f"{macro[f'sens_at_spec_{int(100*s)}']:.3f}"
        if not np.isnan(macro[f'sens_at_spec_{int(100*s)}']) else "NA"
        for s in spec_targets
    ]
    print(" | ".join(f"{r:>12}" for r in row))


# 4) Model/optimizer utilities -------------------------------------------------
# train_utils.py
def load_encoder_and_projector(model, ckpt_path: Path):
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)

    enc_sd  = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    proj_sd = {k[len("proj."):]:    v for k, v in sd.items() if k.startswith("proj.")}

    missing_e, unexpected_e = model.encoder.load_state_dict(enc_sd, strict=False)
    if missing_e:    print(f"[load_encoder_and_projector] Missing encoder keys: {missing_e}")
    if unexpected_e: print(f"[load_encoder_and_projector] Unexpected encoder keys: {unexpected_e}")

    missing_p, unexpected_p = model.proj.load_state_dict(proj_sd, strict=False)
    if missing_p:    print(f"[load_encoder_and_projector] Missing proj keys: {missing_p}")
    if unexpected_p: print(f"[load_encoder_and_projector] Unexpected proj keys: {unexpected_p}")

    print(f"[load_encoder_and_projector] Loaded encoder+proj from {ckpt_path}")

def unfreeze_and_add_param_group(encoder_params, optimizer, base_lr, wd, lr_mult=0.1):
    for p in encoder_params:
        p.requires_grad = True
    optimizer.add_param_group({
        "params": encoder_params,
        "lr": base_lr * lr_mult,
        "weight_decay": wd,
    })


# 5) Early stopping helper -----------------------------------------------------
@dataclass
class EarlyStopper:
    patience: int
    best: float = float("-inf")
    num_bad: int = 0

    def update(self, metric: float, model: torch.nn.Module, save_path: Path, tag: str = "model") -> bool:
        if metric > self.best:
            self.best = metric
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, save_path)
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            return False
