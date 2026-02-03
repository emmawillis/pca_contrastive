#!/usr/bin/env python3
# make_histo_gaussian_stats.py
#
# Computes per-class Gaussian stats (mean + diagonal variance) over histopathology embeddings
# and saves them to a .pt file.
#
# Output format (torch.save):
# {
#   "mu":    FloatTensor [C, D],
#   "var":   FloatTensor [C, D],
#   "count": LongTensor  [C],
#   "dim":   int,
#   "num_classes": int,
#   "provider": str,
#   "encodings_dir": str,
#   "marksheet_csv": str,
# }

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch


def isup_to_3class(y: int) -> int:
    return 0 if y <= 1 else (1 if y <= 3 else 2)


def isup_to_binary(y: int) -> int:
    return 0 if y <= 1 else 1


def map_isup(isup: int, num_classes: int) -> int:
    if num_classes == 3:
        return isup_to_3class(isup)
    if num_classes == 2:
        return isup_to_binary(isup)
    return int(isup)  # e.g. 6-class (0..5)


@torch.no_grad()
def compute_histo_gaussian_stats(
    encodings_dir: str,
    marksheet_csv: str,
    num_classes: int,
    provider: str = "all",
    eps: float = 1e-6,
    dtype: torch.dtype = torch.float32,
):
    encodings_dir = Path(encodings_dir)
    if not encodings_dir.exists():
        raise FileNotFoundError(f"encodings_dir not found: {encodings_dir}")

    df = pd.read_csv(marksheet_csv)
    if provider != "all":
        if "data_provider" not in df.columns:
            raise ValueError("marksheet_csv missing 'data_provider' column, but provider != 'all'.")
        df = df[df["data_provider"] == provider]

    if "FILENAME" not in df.columns or "isup_grade" not in df.columns:
        raise ValueError("marksheet_csv must contain columns: FILENAME, isup_grade")

    fname2isup = dict(zip(df["FILENAME"].astype(str), df["isup_grade"].astype(int)))

    # accumulators per class: sum(x), sum(x^2), count
    sum_x = [None] * num_classes
    sum_x2 = [None] * num_classes
    cnt = torch.zeros(num_classes, dtype=torch.long)

    dim = None
    n_used = 0
    n_skipped_no_label = 0
    n_skipped_bad_class = 0

    npy_paths = list(encodings_dir.rglob("*.npy"))
    if len(npy_paths) == 0:
        raise RuntimeError(f"No .npy files found under {encodings_dir}")

    for path in npy_paths:
        filename = path.stem.split("_")[0]
        if filename not in fname2isup:
            n_skipped_no_label += 1
            continue

        isup = int(fname2isup[filename])
        c = map_isup(isup, num_classes)
        if c < 0 or c >= num_classes:
            n_skipped_bad_class += 1
            continue

        x_np = np.load(path)
        x = torch.as_tensor(x_np, dtype=dtype).flatten()  # [D]

        if dim is None:
            dim = int(x.numel())
        elif int(x.numel()) != dim:
            raise ValueError(f"Embedding dim mismatch at {path}: got {x.numel()}, expected {dim}")

        if sum_x[c] is None:
            sum_x[c] = torch.zeros(dim, dtype=dtype)
            sum_x2[c] = torch.zeros(dim, dtype=dtype)

        sum_x[c] += x
        sum_x2[c] += x * x
        cnt[c] += 1
        n_used += 1

    if dim is None or n_used == 0:
        raise RuntimeError(
            f"No usable histo encodings matched marksheet.\n"
            f"Found {len(npy_paths)} npy files, used {n_used}, skipped_no_label {n_skipped_no_label}."
        )

    mu = torch.zeros((num_classes, dim), dtype=dtype)
    var = torch.zeros((num_classes, dim), dtype=dtype)

    for c in range(num_classes):
        if cnt[c].item() == 0:
            # Avoid NaNs: leave mu=0, set var=1
            var[c].fill_(1.0)
            continue
        mu_c = sum_x[c] / float(cnt[c].item())
        ex2_c = sum_x2[c] / float(cnt[c].item())
        var_c = ex2_c - mu_c * mu_c
        var_c = torch.clamp(var_c, min=eps)
        mu[c] = mu_c
        var[c] = var_c

    meta = {
        "mu": mu,
        "var": var,
        "count": cnt,
        "dim": int(dim),
        "num_classes": int(num_classes),
        "provider": str(provider),
        "encodings_dir": str(encodings_dir),
        "marksheet_csv": str(marksheet_csv),
        "npy_total": int(len(npy_paths)),
        "n_used": int(n_used),
        "n_skipped_no_label": int(n_skipped_no_label),
        "n_skipped_bad_class": int(n_skipped_bad_class),
        "eps": float(eps),
        "dtype": str(dtype).replace("torch.", ""),
    }
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encodings_dir", required=True, help="Path containing histo .npy encodings (e.g. .../train)")
    p.add_argument("--marksheet_csv", required=True, help="CSV with FILENAME + isup_grade (e.g. .../train.csv)")
    p.add_argument("--num_classes", type=int, required=True, help="2, 3, or 6 (0..5)")
    p.add_argument("--provider", default="all", help="all | karolinska | radboud (must match marksheet column)")
    p.add_argument("--out", required=True, help="Output .pt file path")
    p.add_argument("--eps", type=float, default=1e-6, help="Min variance clamp for stability")
    args = p.parse_args()

    stats = compute_histo_gaussian_stats(
        encodings_dir=args.encodings_dir,
        marksheet_csv=args.marksheet_csv,
        num_classes=args.num_classes,
        provider=args.provider,
        eps=args.eps,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)

    print("Saved:", out)
    print("Provider:", stats["provider"])
    print("Num classes:", stats["num_classes"])
    print("Embedding dim:", stats["dim"])
    print("Counts per class:", stats["count"].tolist())
    print("Total .npy:", stats["npy_total"], "Used:", stats["n_used"],
          "Skipped(no label):", stats["n_skipped_no_label"], "Skipped(bad class):", stats["n_skipped_bad_class"])


if __name__ == "__main__":
    main()


'''

python /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/make_histo_gaussian_stats.py \
  --encodings_dir /project/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_512D/embeddings_512/val \
  --marksheet_csv /project/aip-medilab/shared/picai/histopathology_encodings/UNI2_splits/val.csv \
  --num_classes 3 \
  --provider all \
  --out /project/aip-medilab/shared/picai/histopathology_encodings/UNI2/kl_gaussian_stats/histo_stats_val.pt

'''