# dataloader_panda_mil.py
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def _path_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


class PandaPatchBagDataset(Dataset):
    """
    Loads pre-extracted UNI/UNI2 patch features for a single slide (bag).

    Expects a dataframe with at least columns:
        ['image_id', 'isup_grade', 'fold', 'split', 'path' (optional/incorrect)]

    Either the CSV 'path' must exist, or a valid --root must be provided to
    reconstruct '<root>/<image_id>_<isup>.h5'.

    Returns per item:
        X    : torch.FloatTensor [N, D]   (patch features; D=1536)
        y    : torch.LongTensor  []       (ISUP label 0..5)
        meta : dict with keys:
               'image_id', 'num_patches', 'path', 'fold', 'split', 'isup_grade'
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: Optional[str] = None,
        max_patches: Optional[int] = None,   # randomly cap #patches per slide
        feature_key: str = "features",
        dtype: torch.dtype = torch.float32,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.root = None if root is None else str(root)
        self.max_patches = max_patches
        self.feature_key = feature_key
        self.dtype = dtype

        # basic sanitation
        if "isup_grade" in self.df.columns:
            self.df["isup_grade"] = self.df["isup_grade"].astype(int)
        if "fold" in self.df.columns:
            self.df["fold"] = self.df["fold"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, row: pd.Series) -> str:
        # 1) try CSV path if it exists
        p = str(row.get("path", "") or "")
        if p and _path_exists(p):
            return p

        # 2) reconstruct from root + "<image_id>_<isup>.h5"
        if self.root is None:
            raise FileNotFoundError(
                f"CSV path missing/invalid for {row.get('image_id')} and no --root provided."
            )
        image_id = str(row["image_id"])
        isup = int(row["isup_grade"])
        fname = f"{image_id}_{isup}.h5"
        candidate = os.path.join(self.root, fname)
        if _path_exists(candidate):
            return candidate

        # try stripped variant
        alt = os.path.join(self.root, fname.strip())
        if _path_exists(alt):
            return alt

        raise FileNotFoundError(
            f"Could not find H5 for image_id={image_id} isup={isup}. "
            f"Tried: {candidate}"
        )

    @staticmethod
    def _to_2d(feats: np.ndarray) -> np.ndarray:
        """
        Ensure features are [N, D]. Handles shapes like (1, N, D) or (N, D).
        """
        arr = np.asarray(feats)
        # squeeze any leading singleton dims until 2D
        while arr.ndim > 2:
            arr = np.squeeze(arr, axis=0)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected feature shape after squeeze: {arr.shape}")
        return arr

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = int(row["isup_grade"])

        p = self._resolve_path(row)
        with h5py.File(p, "r") as f:
            if self.feature_key not in f:
                raise KeyError(
                    f"'{self.feature_key}' not found in {p}. Keys: {list(f.keys())}"
                )
            feats = self._to_2d(f[self.feature_key][:])  # [N, D]

        N, D = feats.shape
        # optional random subsample of patches for speed/regularization
        if self.max_patches is not None and N > self.max_patches:
            sel = np.random.choice(N, self.max_patches, replace=False)
            feats = feats[sel]
            N = feats.shape[0]

        X = torch.from_numpy(feats).to(self.dtype)  # [N, D]
        y = torch.tensor(y, dtype=torch.long)

        meta = {
            "image_id": str(row["image_id"]),
            "num_patches": int(N),
            "path": p,
            "fold": int(row.get("fold", -1)),
            "split": str(row.get("split", "")),
            "isup_grade": int(row["isup_grade"]),
        }
        return X, y, meta


def pad_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Create a batch with padding + mask for variable-length bags.

    Input (per sample):
      X: [N, D], y: [], meta: dict

    Returns:
      X_pad: [B, N_max, D]
      y    : [B]
      mask : [B, N_max]  (True for valid patches, False for padding)
      metas: list of meta dicts
    """
    Xs, ys, metas = zip(*batch)
    B = len(Xs)
    N_max = max(x.shape[0] for x in Xs)
    D = Xs[0].shape[1]

    X_pad = torch.zeros(B, N_max, D, dtype=Xs[0].dtype)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    y_out = torch.empty(B, dtype=torch.long)

    for i, (x, y, meta) in enumerate(batch):
        n = x.shape[0]
        X_pad[i, :n] = x
        mask[i, :n] = True
        y_out[i] = y

    return X_pad, y_out, mask, list(metas)


def make_loaders(
    csv_path: str,
    root: str,
    fold: int,
    batch_size: int = 8,
    num_workers: int = 4,
    max_patches_train: Optional[int] = None,  # e.g., 2000 for speed; None means all
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build train/val(/test) DataLoaders for ABMIL training with padding + masks.

    Splits:
      - train: split=='cv' and fold != <fold>
      - val  : split=='cv' and fold == <fold>
      - test : split=='test' (optional)

    Returns:
      Ltrain, Lval, Ltest, train_df, val_df, test_df
    """
    df = pd.read_csv(csv_path)
    if "split" not in df.columns or "fold" not in df.columns:
        raise ValueError("CSV must have 'split' and 'fold' columns.")

    df["fold"] = df["fold"].astype(int)
    df["isup_grade"] = df["isup_grade"].astype(int)

    train_df = df[(df["split"] == "cv") & (df["fold"] != int(fold))].copy()
    val_df   = df[(df["split"] == "cv") & (df["fold"] == int(fold))].copy()
    test_df  = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        test_df = None

    dtrain = PandaPatchBagDataset(train_df, root=root, max_patches=max_patches_train)
    dval   = PandaPatchBagDataset(val_df,   root=root, max_patches=None)
    dtest  = PandaPatchBagDataset(test_df,  root=root, max_patches=None) if test_df is not None else None

    # Optional: class-imbalance mitigation via per-sample weights (inverse freq)
    if use_weighted_sampler:
        counts = train_df["isup_grade"].value_counts().sort_index()
        class_weights = (1.0 / (counts + 1e-12)).to_dict()
        sample_weights = train_df["isup_grade"].map(class_weights).astype(float).values
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        Ltrain = DataLoader(
            dtrain,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=pad_collate,
            pin_memory=pin_memory,
        )
    else:
        Ltrain = DataLoader(
            dtrain,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=pad_collate,
            pin_memory=pin_memory,
        )

    Lval = DataLoader(
        dval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=pin_memory,
    )

    Ltest = None
    if dtest is not None:
        Ltest = DataLoader(
            dtest,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pad_collate,
            pin_memory=pin_memory,
        )

    return Ltrain, Lval, Ltest, train_df, val_df, test_df
