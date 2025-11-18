#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
# If you later have patient IDs (groups), consider StratifiedGroupKFold (sklearn >=1.1)

def parse_args():
    p = argparse.ArgumentParser(
        description="Hold out 20% stratified TEST and split the remainder into 5 stratified CV folds from <image_id>_<ISUP>.h5 files."
    )
    p.add_argument("--encodings_dir", required=True, type=Path,
                   help="Directory containing *.h5 files named like <image_id>_<ISUP>.h5")
    p.add_argument("--out_csv", required=True, type=Path,
                   help="Path to write manifest CSV with split+fold assignments")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument("--make_symlinks", action="store_true",
                   help="If set, create symlinked dirs for test and cv5 folds under outputs/splits/")
    return p.parse_args()

def extract_id_isup(path: Path):
    """Extract image_id and ISUP from filename '<image_id>_<ISUP>.h5' (splitting on last underscore)."""
    stem = path.stem
    m = re.match(r"^(?P<image_id>.+)_(?P<isup>\d+)$", stem)
    if not m:
        return None, None
    return m.group("image_id"), int(m.group("isup"))

def summarize_dist(title, labels):
    c = dict(Counter(labels))
    print(f"{title}: n={len(labels)}  dist={c}")

def main():
    args = parse_args()
    enc_dir: Path = args.encodings_dir

    if not enc_dir.is_dir():
        print(f"ERROR: {enc_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in enc_dir.iterdir():
        if p.suffix.lower() != ".h5" or not p.is_file():
            continue
        image_id, isup = extract_id_isup(p)
        if image_id is None:
            print(f"Skipping non-matching file name: {p.name}")
            continue
        rows.append({"image_id": image_id, "isup_grade": isup, "path": str(p.resolve())})

    if not rows:
        print("No matching .h5 files found. Ensure files are named <image_id>_<ISUP>.h5")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["isup_grade"] = df["isup_grade"].astype(int)

    # Overall distribution
    summarize_dist("Overall ISUP distribution", df["isup_grade"])

    # 1) Stratified TEST split (20%)
    trainval_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["isup_grade"],
        random_state=args.seed,
        shuffle=True,
    )
    test_df = test_df.copy()
    test_df["split"] = "test"
    test_df["fold"] = -1

    summarize_dist("TEST ISUP distribution (20%)", test_df["isup_grade"])

    # 2) 5-fold stratified split on remaining (CV pool)
    cv_df = trainval_df.copy().reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(cv_df["image_id"], cv_df["isup_grade"])):
        cv_df.loc[val_idx, "fold"] = fold
    cv_df["split"] = "cv"

    # Sanity: each fold distribution
    print("\nPer-fold ISUP distributions (CV pool):")
    for f in range(5):
        sub = cv_df[cv_df["fold"] == f]
        summarize_dist(f"  Fold {f}", sub["isup_grade"])

    # Combine and save
    manifest = pd.concat([cv_df, test_df], ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_csv, index=False)
    print(f"\nWrote manifest to: {args.out_csv}")

    # Optional: create symlinked directories
    if args.make_symlinks:
        root = Path("outputs/splits")
        # Test symlinks
        test_dir = root / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        for _, r in test_df.iterrows():
            src = Path(r["path"])
            dst = test_dir / src.name
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)

        # CV folds: for fold k, val = fold k; train = all other folds
        cv_root = root / "cv5"
        for f in range(5):
            fold_dir = cv_root / f"fold{f}"
            train_dir = fold_dir / "train"
            val_dir = fold_dir / "val"
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

            val_df = cv_df[cv_df["fold"] == f]
            train_df = cv_df[cv_df["fold"] != f]

            def link_into(sub_df, target_dir: Path):
                for _, r in sub_df.iterrows():
                    src = Path(r["path"])
                    dst = target_dir / src.name
                    if not dst.exists():
                        try:
                            os.symlink(src, dst)
                        except OSError:
                            import shutil
                            shutil.copy2(src, dst)

            link_into(train_df, train_dir)
            link_into(val_df, val_dir)

        print(f"Symlinked directories under: {root.resolve()}")

if __name__ == "__main__":
    # Ensure deterministic numpy-based ops (for reproducibility of split only)
    np.random.seed(0)
    main()
