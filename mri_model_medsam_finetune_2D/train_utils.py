import numpy as np
import pandas as pd
from collections import Counter

from sklearn.metrics import auc, roc_auc_score, roc_curve
import torch

from dataset_picai_slices import map_binary_all, map_binary_low_high, map_isup3
from torch.utils.data import DataLoader, WeightedRandomSampler

from ISUPMedSAM import IMG_SIZE

def get_label_names(target):
    if target == "isup3":
        return ["ISUP01", "ISUP23", "ISUP45"]  # c0,c1,c2
    elif target == "binary_low_high":
        return ["LOW(ISUP01)", "HIGH(ISUP45)"]
    elif target == "binary_all":
        return ["no csPCa", "yes csPCa"]
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
def tpr_tnr_from_confusion(cm: torch.Tensor):
    """
    cm: KxK counts; rows=true, cols=pred.
    Returns:
      per_class_tpr (list), per_class_tnr (list), macro_tpr (float), macro_tnr (float)
    """
    K = cm.shape[0]
    cm_np = cm.cpu().numpy().astype(np.int64)
    per_tpr, per_tnr = [], []
    for c in range(K):
        TP = cm_np[c, c]
        FN = cm_np[c, :].sum() - TP
        FP = cm_np[:, c].sum() - TP
        TN = cm_np.sum() - TP - FN - FP
        tpr = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # sensitivity
        tnr = TN / (TN + FP) if (TN + FP) > 0 else np.nan  # specificity
        per_tpr.append(tpr); per_tnr.append(tnr)
    macro_tpr = float(np.nanmean(per_tpr)) if len(per_tpr) else float("nan")
    macro_tnr = float(np.nanmean(per_tnr)) if len(per_tnr) else float("nan")
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
    labels = get_label_names(n_classes)
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
