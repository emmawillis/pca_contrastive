#!/usr/bin/env python3
# compute_histo_prototypes_npy.py

import argparse
from pathlib import Path
from collections import defaultdict

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


def get_slide_id_from_path(path: Path, suffix: str = "_abmil512") -> str:
    """
    Example:
      000920ad0b612851f8e01bcc880d9b3d_abmil512.npy
    becomes:
      000920ad0b612851f8e01bcc880d9b3d
    """
    stem = path.stem

    if stem.endswith(suffix):
        return stem[: -len(suffix)]

    # fallback: take everything before first underscore
    return stem.split("_")[0]


def load_npy_embedding(path: Path) -> torch.Tensor:
    """
    Loads one .npy embedding file.

    Expected shapes:
      [D]
      [1, D]
      [N, D]  if patch-level or bag-level embeddings accidentally appear

    Returns a single vector [D].
    """
    arr = np.load(path)
    x = torch.as_tensor(arr, dtype=torch.float32).squeeze()

    if x.ndim == 1:
        return x

    if x.ndim == 2:
        # If [1, D], this is effectively just x[0].
        # If [N, D], mean-pool to slide-level.
        return x.mean(dim=0)

    raise ValueError(f"Unexpected embedding shape {tuple(x.shape)} in {path}")


def compute_prototypes(
    embeddings_dir: Path,
    csv_path: Path,
    output_path: Path,
    num_classes: int = 6,
    provider: str = "all",
    filename_suffix: str = "_abmil512",
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

    npy_paths = sorted(embeddings_dir.rglob("*.npy"))

    print(f"Found {len(npy_paths)} .npy files under: {embeddings_dir}")
    print(f"CSV rows after provider filter: {len(df)}")
    print(f"num_classes={num_classes}, provider={provider}")

    buckets = defaultdict(list)

    used = 0
    skipped_no_label = 0
    skipped_bad_file = 0

    for path in npy_paths:
        slide_id = get_slide_id_from_path(path, suffix=filename_suffix)

        if slide_id not in fname_to_isup:
            skipped_no_label += 1
            continue

        isup = int(fname_to_isup[slide_id])
        label = map_label(isup, num_classes=num_classes)

        try:
            emb = load_npy_embedding(path)
        except Exception as e:
            print(f"[WARN] Skipping bad file: {path} | {repr(e)}")
            skipped_bad_file += 1
            continue

        if normalize_embeddings:
            emb = F.normalize(emb.view(1, -1), dim=1).squeeze(0)

        buckets[label].append(emb)
        used += 1

    print(f"\nUsed files: {used}")
    print(f"Skipped because no CSV label: {skipped_no_label}")
    print(f"Skipped because unreadable/bad shape: {skipped_bad_file}")

    prototypes = []
    counts = []

    for c in range(num_classes):
        if len(buckets[c]) == 0:
            raise ValueError(
                f"No embeddings found for class {c}. "
                f"Check CSV labels, provider, num_classes, and filename parsing."
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
        "prototypes": prototypes,
        "counts": counts,
        "num_classes": num_classes,
        "provider": provider,
        "filename_suffix": filename_suffix,
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
        help="Folder containing .npy histopathology embeddings.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to train.csv with FILENAME,data_provider,isup_grade,...",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Where to save prototype .pt file.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        choices=[2, 3, 6],
        default=6,
        help="6 = ISUP0-5, 3 = ISUP01/23/45, 2 = low/high.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        help="Use 'all' to disable provider filtering.",
    )
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="_abmil512",
        help="Suffix to strip from embedding filenames before matching CSV FILENAME.",
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
        filename_suffix=args.filename_suffix,
        normalize_embeddings=not args.no_normalize_embeddings,
        normalize_prototypes=not args.no_normalize_prototypes,
    )


if __name__ == "__main__":
    main()


'''
python /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/prototypes/compute_histo_prototypes.py \
  --embeddings_dir /project/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_512D/embeddings_512/train \
  --csv_path /project/aip-medilab/shared/picai/histopathology_encodings/UNI2_splits/train.csv \
  --output_path /project/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_512D/histo_isup6_prototypes.pt \
  --num_classes 6 \
  --provider all
'''