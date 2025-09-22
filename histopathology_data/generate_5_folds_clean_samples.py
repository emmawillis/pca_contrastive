import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

def main():
    ap = argparse.ArgumentParser("Build PANDA manifest: noise_ratio_10=1, 10% test, 5 stratified folds")
    ap.add_argument("--csv", required=True, help="Path to Kaggle train.csv (with noise_ratio_10 and isup_grade)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--test_size", type=float, default=0.10, help="Fraction for test split (default 0.10)")
    ap.add_argument("--n_folds", type=int, default=5, help="Number of CV folds (default 5)")
    args = ap.parse_args()

    # Load and filter to clean rows
    df = pd.read_csv(args.csv)
    if "noise_ratio_10" not in df.columns:
        raise SystemExit("ERROR: CSV is missing 'noise_ratio_10' column.")

    df = df[df["noise_ratio_10"].astype(int) == 1].copy()
    if "image_id" not in df.columns or "isup_grade" not in df.columns:
        raise SystemExit("ERROR: CSV must contain 'image_id' and 'isup_grade' columns.")

    df["isup_grade"] = df["isup_grade"].astype(int)

    total = len(df)
    print(f"After filtering noise_ratio_10==1: kept {total} rows")

    # 10% stratified TEST split
    trainval_df, test_df = train_test_split(
        df[["image_id", "isup_grade"]],
        test_size=args.test_size,
        stratify=df["isup_grade"],
        random_state=args.seed,
        shuffle=True,
    )
    test_df = test_df.copy()
    test_df["fold"] = -1
    test_df["split"] = "test"

    print(f"Test size: {len(test_df)} ({len(test_df)/total:.1%})")
    print("Test per-class counts:", test_df["isup_grade"].value_counts().sort_index().to_dict())

    # Stratified K-fold on remaining
    cv_df = trainval_df.reset_index(drop=True).copy()
    cv_df["fold"] = -1
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for fold, (_, val_idx) in enumerate(skf.split(cv_df["image_id"], cv_df["isup_grade"])):
        cv_df.loc[val_idx, "fold"] = fold
    cv_df["split"] = "cv"

    # Sanity prints
    print("\nCV per-fold per-class counts:")
    for f in range(args.n_folds):
        sub = cv_df[cv_df["fold"] == f]
        counts = sub["isup_grade"].value_counts().sort_index().to_dict()
        print(f"  fold {f}: n={len(sub)}  {counts}")

    # Assemble manifest with exact headers
    manifest = pd.concat([cv_df, test_df], ignore_index=True)[["image_id", "isup_grade", "fold", "split"]]
    manifest["fold"] = manifest["fold"].astype(int)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_csv, index=False)
    print(f"\nWrote manifest to: {args.out_csv}")
    print("Header preview:")
    print(manifest.head().to_string(index=False))

if __name__ == "__main__":
    # numpy seed here just for safety—sklearn uses random_state above
    np.random.seed(0)
    main()
