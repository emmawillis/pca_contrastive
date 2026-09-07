import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.builder import create_model


device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class SlideBagOfEncodingsDataset(torch.utils.data.Dataset):
    """Minimal dataset wrapper around pre-computed slide encodings."""

    def __init__(self, encodings_root: str, split_csv: str):
        self.encodings_root = encodings_root
        self.labels = pd.read_csv(split_csv)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        sid = row.get("FILENAME")
        isup = int(row.get("isup_grade"))
        with h5py.File(f"{self.encodings_root}/{sid}_{isup}.h5", "r") as file:
            bag_of_encodings = file["features"][...].squeeze(0)
        return torch.FloatTensor(bag_of_encodings), isup, sid


class ABMILHead(nn.Module):
    def __init__(self, in_dim: int = 512, emb_dim: int = 512, num_classes: int = 6):
        super().__init__()
        self.proj = nn.Linear(in_dim, emb_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, z):
        z = self.norm(self.act(self.proj(z)))
        return self.cls(z)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, proj_dim: int):
    model = create_model(
        "abmil.base.uni_v2.pc108-24k",
        from_pretrained=True,
        num_classes=num_classes,
    )
    if proj_dim != 512:
        model.model.classifier = ABMILHead(in_dim=512, emb_dim=proj_dim, num_classes=num_classes).to(device)
    model.to(device)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def evaluate(model: nn.Module, dataloader, num_classes: int):
    model.eval()
    y_true, probs = [], []
    with torch.no_grad():
        for bag, y, _ in dataloader:
            bag = bag.to(device)
            y = y.to(device).long()
            out, _ = model(bag)
            logits = out["logits"]
            prob = torch.softmax(logits, dim=-1)
            y_true.extend(to_numpy(y).tolist())
            probs.append(to_numpy(prob.squeeze(0)))
    return np.array(y_true), np.vstack(probs)


def compute_overall_metrics(y_true, y_prob, num_classes: int):
    y_true = np.asarray(y_true)
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class_sens, per_class_spec = [], []
    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        if TP + FN > 0:
            per_class_sens.append(TP / (TP + FN))
        if TN + FP > 0:
            per_class_spec.append(TN / (TN + FP))

    overall_auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    return {
        "overall_auroc": float(overall_auroc),
        "overall_sensitivity": float(np.mean(per_class_sens)) if per_class_sens else float("nan"),
        "overall_specificity": float(np.mean(per_class_spec)) if per_class_spec else float("nan"),
    }


def sensitivity_at_specificity(y_true_bin, scores, target_spec):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    specificity = 1 - fpr
    mask = specificity >= target_spec
    if not np.any(mask):
        return float("nan")
    return float(tpr[mask].max())


def macro_sens_at_spec(y_true, y_prob, num_classes: int, target_spec: float):
    values = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        sens = sensitivity_at_specificity(y_bin, y_prob[:, c], target_spec)
        if not np.isnan(sens):
            values.append(sens)
    return float(np.mean(values)) if values else float("nan")


def per_class_aurocs(y_true, y_prob, num_classes: int):
    metrics = {}
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            metrics[f"class_{c}_vs_rest_auroc"] = float("nan")
            continue
        metrics[f"class_{c}_vs_rest_auroc"] = float(roc_auc_score(y_bin, y_prob[:, c]))
    return metrics


def format_and_print(metrics: dict):
    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}" if np.isfinite(value) else f"{key}: nan")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ABMIL checkpoint on the test set.")
    parser.add_argument("--encodings_root", type=str, required=True)
    parser.add_argument("--split_csvs_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    set_seed()

    test_dataset = SlideBagOfEncodingsDataset(
        encodings_root=args.encodings_root,
        split_csv=f"{args.split_csvs_root}/test.csv",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=args.num_classes, proj_dim=args.proj_dim)
    load_checkpoint(model, args.ckpt_path)

    y_true, y_prob = evaluate(model, test_loader, args.num_classes)

    metrics = compute_overall_metrics(y_true, y_prob, args.num_classes)
    metrics["sens_at_spec40"] = macro_sens_at_spec(y_true, y_prob, args.num_classes, target_spec=0.40)
    metrics["sens_at_spec60"] = macro_sens_at_spec(y_true, y_prob, args.num_classes, target_spec=0.60)
    metrics.update(per_class_aurocs(y_true, y_prob, args.num_classes))

    format_and_print(metrics)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()


'''
python evaluate_checkpoint.py \
  --encodings_root /path/to/encodings \
  --split_csvs_root /Users/emma/Desktop/QUEENS/THESIS/MIL-Lab/panda_splits \
  --ckpt_path /Users/emma/Desktop/QUEENS/THESIS/MIL-Lab/results/default_512D/best_model.pth \
  --proj_dim 1024 \
  --output_json results/test_metrics.json
'''