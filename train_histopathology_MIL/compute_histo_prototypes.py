#!/usr/bin/env python3
# compute_histo_prototypes.py

import argparse
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def isup_to_3class(y: int) -> int:
    return 0 if y <= 1 else (1 if y <= 3 else 2)


def isup_to_binary(y: int) -> int:
    return 0 if y <= 1 else 1


def map_label(isup: int, num_classes: int) -> int:
    if num_classes == 6:
        return isup
    elif num_classes == 3:
        return isup_to_3class(isup)
    elif num_classes == 2:
        return isup_to_binary(isup)
    else:
        raise ValueError(f"Unsupported num_classes={num_classes}. Use 2, 3, or 6.")


def load_h5_embedding(path: Path, key: str = "features") -> torch.Tensor:
    """
    Loads one .h5 embedding file.

    Expected common shapes:
      [N, D]      patch/bag embeddings
      [1, N, D]  bag embeddings with batch dimension
      [D]         already pooled slide embedding

    This function returns a single slide-level vector [D] by mean-pooling if needed.
    """
    with h5py.File(path, "r") as f:
        if key not in f:
            available = list(f.keys())
            raise KeyError(
                f"Key '{key}' not found in {path}. Available keys: {available}"
            )

        arr = f[key][...]

    x = torch.as_tensor(arr, dtype=torch.float32)

    # Remove singleton batch dimensions, e.g. [1, N, D] -> [N, D]
    x = x.squeeze()

    if x.ndim == 1:
        # Already [D]
        slide_emb = x
    elif x.ndim == 2:
        # Patch embeddings [N, D] -> slide prototype [D]
        slide_emb = x.mean(dim=0)
    else:
        raise ValueError(f"Unexpected embedding shape {tuple(x.shape)} in {path}")

    return slide_emb


def compute_prototypes(
    embeddings_dir: Path,
    csv_path: Path,
    output_path: Path,
    num_classes: int = 6,
    provider: str = "all",
    h5_key: str = "features",
    normalize_embeddings: bool = True,
    normalize_prototypes: bool = True,
):
    df = pd.read_csv(csv_path)

    required_cols = {"FILENAME", "isup_grade"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if provider != "all":
        if "data_provider" not in df.columns:
            raise ValueError("CSV has no 'data_provider' column, but provider filtering was requested.")
        df = df[df["data_provider"] == provider].copy()

    fname_to_isup = dict(
        zip(df["FILENAME"].astype(str), df["isup_grade"].astype(int))
    )

    buckets = defaultdict(list)

    h5_paths = sorted(embeddings_dir.rglob("*.h5"))
    print(f"Found {len(h5_paths)} .h5 files under: {embeddings_dir}")
    print(f"CSV rows after provider filter: {len(df)}")

    used = 0
    skipped_no_label = 0
    skipped_bad_file = 0

    for path in h5_paths:
        # Handles names like:
        #   000920ad0b612851f8e01bcc880d9b3d.h5
        #   000920ad0b612851f8e01bcc880d9b3d_x6736_y2832_crop.h5
        filename = path.stem.split("_")[0]

        if filename not in fname_to_isup:
            skipped_no_label += 1
            continue

        isup = int(fname_to_isup[filename])
        label = map_label(isup, num_classes=num_classes)

        try:
            emb = load_h5_embedding(path, key=h5_key)
        except Exception as e:
            print(f"[WARN] Skipping bad file: {path} | {repr(e)}")
            skipped_bad_file += 1
            continue

        if normalize_embeddings:
            emb = F.normalize(emb.view(1, -1), dim=1).squeeze(0)

        buckets[label].append(emb)
        used += 1

    print(f"Used files: {used}")
    print(f"Skipped because no CSV label: {skipped_no_label}")
    print(f"Skipped because unreadable/bad shape/key: {skipped_bad_file}")

    prototypes = []
    counts = []

    for c in range(num_classes):
        if len(buckets[c]) == 0:
            raise ValueError(
                f"No embeddings found for class {c}. "
                f"Check provider, num_classes, CSV labels, and filenames."
            )

        class_embs = torch.stack(buckets[c], dim=0)  # [N_c, D]
        proto = class_embs.mean(dim=0)

        if normalize_prototypes:
            proto = F.normalize(proto.view(1, -1), dim=1).squeeze(0)

        prototypes.append(proto)
        counts.append(len(buckets[c]))

        print(f"class {c}: n={len(buckets[c])}, dim={proto.numel()}")

    prototypes = torch.stack(prototypes, dim=0)  # [K, D]
    counts = torch.tensor(counts, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "prototypes": prototypes,          # [num_classes, D]
        "counts": counts,                  # [num_classes]
        "num_classes": num_classes,
        "provider": provider,
        "h5_key": h5_key,
        "normalize_embeddings": normalize_embeddings,
        "normalize_prototypes": normalize_prototypes,
        "csv_path": str(csv_path),
        "embeddings_dir": str(embeddings_dir),
    }

    torch.save(payload, output_path)

    print(f"\nSaved prototypes to: {output_path}")
    print(f"Prototype tensor shape: {tuple(prototypes.shape)}")
    print(f"Counts: {counts.tolist()}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        required=True,
        help="Folder containing histopathology .h5 embedding files.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to train.csv containing FILENAME,data_provider,isup_grade,...",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("histo_isup_prototypes.pt"),
        help="Where to save the prototype .pt file.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        choices=[2, 3, 6],
        default=6,
        help="Use 6 for ISUP0-5, 3 for ISUP01/23/45, or 2 for low/high.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        help="Use 'all' to disable provider filtering.",
    )
    parser.add_argument(
        "--h5_key",
        type=str,
        default="features",
        help="Dataset key inside each .h5 file.",
    )
    parser.add_argument(
        "--no_normalize_embeddings",
        action="store_true",
        help="Disable L2 normalization before averaging.",
    )
    parser.add_argument(
        "--no_normalize_prototypes",
        action="store_true",
        help="Disable L2 normalization after averaging.",
    )

    args = parser.parse_args()

    compute_prototypes(
        embeddings_dir=args.embeddings_dir,
        csv_path=args.csv_path,
        output_path=args.output_path,
        num_classes=args.num_classes,
        provider=args.provider,
        h5_key=args.h5_key,
        normalize_embeddings=not args.no_normalize_embeddings,
        normalize_prototypes=not args.no_normalize_prototypes,
    )


if __name__ == "__main__":
    main()

'''

python compute_histo_prototypes.py \
  --embeddings_dir /Users/emma/Desktop/QUEENS/THESIS/MIL-Lab/results/default_512D/embeddings_512/train \
  --csv_path /Users/emma/Desktop/QUEENS/THESIS/MIL-Lab/panda_splits/train.csv \
  --output_path /Users/emma/Desktop/QUEENS/THESIS/MIL-Lab/histo_isup6_prototypes.pt \
  --num_classes 6 

'''