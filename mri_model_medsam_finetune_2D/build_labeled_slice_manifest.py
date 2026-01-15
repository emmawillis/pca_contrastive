#!/usr/bin/env python3
"""
build_labeled_slice_manifest.py

Create a new manifest that keeps only:
  - slices with has_lesion == 1 for MRIs that have lesion masks
  - a fraction of central slices for MRIs without masks, labeled with patient_ISUP

Usage:
  python build_labeled_slice_manifest.py \
      --in_manifest slices_manifest.csv \
      --out_manifest slices_manifest_labeled_only.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def isup_to_label3(isup: int) -> int:
    """
    Map 6-class ISUP (0–5) to 3-class label used in your work:
      - 0,1 -> 0   (ISUP 0–1)
      - 2,3 -> 1   (ISUP 2–3)
      - 4,5 -> 2   (ISUP 4–5)
    """
    if isup <= 1:
        return 0
    elif isup in (2, 3):
        return 1
    else:
        return 2


def build_labeled_manifest(
    in_manifest: Path,
    out_manifest: Path,
    min_slices_per_case: int = 1,
    fraction_override: float | None = None,
) -> None:
    df = pd.read_csv(in_manifest)

    # --- sanity checks on expected columns ---
    required_cols = [
        "case_id",
        "fold",
        "z",
        "label6",
        "label3",
        "has_lesion",
        "skip",
        "patient_ISUP",
        "merged_ISUP",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    # --- compute lesion slice fraction from cases that have masks ---
    grouped = df.groupby("case_id")
    case_n_slices = grouped["z"].count()
    case_n_lesion = grouped["has_lesion"].sum()

    cases_with_masks = case_n_lesion > 0
    if not cases_with_masks.any():
        raise ValueError("No cases with has_lesion > 0 found; cannot estimate lesion fraction.")

    lesion_fraction_per_case = case_n_lesion[cases_with_masks] / case_n_slices[cases_with_masks]

    if fraction_override is not None:
        lesion_fraction = float(fraction_override)
    else:
        lesion_fraction = lesion_fraction_per_case.mean()

    print("Number of cases with lesion masks:", cases_with_masks.sum())
    print("Mean lesion slice fraction over masked cases: "
          f"{lesion_fraction:.3f} (min={lesion_fraction_per_case.min():.3f}, "
          f"max={lesion_fraction_per_case.max():.3f})")

    # --- build new rows case by case ---
    new_rows = []

    for case_id, g in grouped:
        g = g.sort_values("z")  # ensure slices are ordered
        n_slices = len(g)
        n_lesion = int(g["has_lesion"].sum())

        if n_lesion > 0:
            # Case has a lesion mask: keep only slices that intersect the mask.
            selected = g[g["has_lesion"] == 1].copy()

            # Recompute labels from patient_ISUP to be safe.
            selected["label6"] = selected["patient_ISUP"]
            selected["label3"] = selected["patient_ISUP"].apply(isup_to_label3)
            selected["merged_ISUP"] = selected["label6"]

            # We are explicitly training on these slices, so skip=0
            selected["skip"] = 0

        else:
            # Case has NO lesion mask: approximate using a fraction of central slices.
            k = max(min_slices_per_case, int(round(lesion_fraction * n_slices)))
            k = max(1, min(k, n_slices))  # clamp

            start = (n_slices - k) // 2
            end = start + k

            selected = g.iloc[start:end].copy()

            # Use patient-level ISUP for these slices.
            selected["label6"] = selected["patient_ISUP"]
            selected["label3"] = selected["patient_ISUP"].apply(isup_to_label3)
            selected["merged_ISUP"] = selected["label6"]

            # We are now treating these as labeled slices, so skip=0.
            selected["skip"] = 0

            # "has_lesion" no longer literally means "intersects segmentation mask"
            # here, but it can still be useful as a flag that this slice is used
            # as a positive lesion slice when ISUP > 0.
            selected["has_lesion"] = 0 #(selected["patient_ISUP"] > 0).astype(int)

        new_rows.append(selected)

    new_df = pd.concat(new_rows, ignore_index=True)

    # Optional: sort for readability
    new_df = new_df.sort_values(["case_id", "z"]).reset_index(drop=True)

    print("Original #rows:", len(df))
    print("New #rows (kept slices only):", len(new_df))

    new_df.to_csv(out_manifest, index=False)
    print("Wrote new manifest to:", out_manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_manifest", type=Path, required=True,
                        help="Path to original slices_manifest.csv")
    parser.add_argument("--out_manifest", type=Path, required=True,
                        help="Path to write the new manifest")
    parser.add_argument("--min_slices_per_case", type=int, default=1,
                        help="Minimum # of slices to keep per MRI")
    parser.add_argument("--fraction_override", type=float, default=None,
                        help="Optional manual lesion-slice fraction "
                             "(if set, skip automatic estimation)")
    args = parser.parse_args()

    build_labeled_manifest(
        in_manifest=args.in_manifest,
        out_manifest=args.out_manifest,
        min_slices_per_case=args.min_slices_per_case,
        fraction_override=args.fraction_override,
    )


if __name__ == "__main__":
    main()

'''

python /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/build_labeled_slice_manifest.py \
--in_manifest /home/ewillis/projects/aip-medilab/shared/picai/manifests/slices_manifest.csv \
--out_manifest /home/ewillis/projects/aip-medilab/shared/picai/manifests/slices_manifest_olivia_lesion.csv
'''