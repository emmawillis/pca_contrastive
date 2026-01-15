# dataset_picai_slices.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset

# --------------------------
# Helpers
# --------------------------

def _to_zhw(arr: np.ndarray) -> np.ndarray:
    """Ensure (Z,H,W). nnU-Net-style volumes are often (H,W,Z)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D, got {arr.shape}")
    # Assume last axis is Z (19-ish), move to front
    return np.moveaxis(arr, -1, 0)

def window_percentile(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip(x, lo, hi)
    if hi > lo:
        x = (x - lo) / (hi - lo)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x.astype(np.float32)

def map_isup3(y6: int) -> int:
    if y6 in (0, 1): return 0
    if y6 in (2, 3): return 1
    if y6 in (4, 5): return 2
    raise ValueError(f"bad label6={y6}")

def map_binary_low_high(y6: int) -> int:
    if y6 in (0, 1): return 0
    if y6 in (4, 5): return 1
    raise ValueError(f"bad label6={y6}")
 
def map_binary_all(y6: int) -> int:
    if y6 in (0, 1): return 0
    return 1

def map_isup0145(y6: int) -> int:
    if y6 == 0: return 0
    if y6 == 1: return 1
    if y6 == 4: return 2
    if y6 == 5: return 3
    raise ValueError(f"Unsupported label {y6} for target=isup0145")

def _clean_path(p: Union[str, Path]) -> Path:
    return Path(str(p).strip())

# --------------------------
# Dataset
# --------------------------

class PicaiSliceDataset(Dataset):
    """
    Loads PI-CAI slices from a manifest with columns:
      - case_id, fold, z, label6, label3, has_lesion, area_frac
      - path_T2, path_ADC, path_HBV
      - path_mask_lesion, path_mask_prostate
      - bbox_prostate_z0, bbox_prostate_z1, bbox_prostate_h0, bbox_prostate_h1, bbox_prostate_w0, bbox_prostate_w1
      - (optional) skip

    Features:
      - filter by folds (patient-level split)
      - crop slices to stored prostate bbox
      - stack available channels (HBV optional)
      - percentile windowing normalization per-channel
      - target = 'isup3' or 'isup6'
      - optional transform(image: Tensor CxHxW, target: int, meta: dict) -> same
    """
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        folds: Optional[Sequence[Union[int, str]]] = None,
        use_skip: bool = True,
        label6_column: str = 'label6',
        target: str = "isup3",               # 'isup3' or 'isup6'
        channels: Sequence[str] = ("path_T2", "path_ADC", "path_HBV"),
        missing_channel_mode: str = "zeros", # 'zeros' or 'repeat_t2'
        pct_lower: float = 0.5,
        pct_upper: float = 99.5,
        transform = None,                    # callable or None
        cache_size: int = 64,                # LRU cache for volumes (per-path)
    ):
        """
        Parameters
        ----------
        manifest_csv : str|Path
            Path to slices_manifest.csv.
        folds : list[int|str] | None
            Keep only rows whose 'fold' is in this set. If None, keep all folds.
        use_skip : bool
            If True and 'skip' column exists, drop rows with skip==1.
        target : 'isup3'|'isup6'
            Label scheme. For 'isup3', 0&1->0, 2&3->1, 4&5->2.
        channels : tuple[str,...]
            Which path columns to load, in order (e.g., ('path_T2','path_ADC','path_HBV')).
        missing_channel_mode : 'zeros'|'repeat_t2'
            If a channel path cell is empty, fill with zeros or repeat T2.
        pct_lower, pct_upper : float
            Percentile windowing per channel before [0,1] scaling.
        transform : callable or None
            Called as transform(torch.Tensor CxHxW, int label, dict meta) -> (image, label, meta) or similar.
        cache_size : int
            LRU cache size for loaded 3D volumes.
        """
        self.manifest_csv = _clean_path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)

        # normalize fold
        self.df["fold"] = self.df["fold"].astype(str).str.strip()
        self.df.loc[self.df["fold"].isin(["", "nan", "NaN"]), "fold"] = "NA"

        self.use_skip = use_skip
        if use_skip and "skip" in self.df.columns:
            self.df = self.df[self.df["skip"] == 0]

        if folds is not None:
            folds = [str(f) for f in folds]
            self.df = self.df[self.df["fold"].isin(folds)]

        # Keep only rows with at least T2/ADC paths (non-empty)
        for col in ("path_T2", "path_ADC"):
            if col not in self.df.columns:
                raise ValueError(f"Manifest missing required column: {col}")
        self.df["path_T2"] = self.df["path_T2"].astype(str)
        self.df["path_ADC"] = self.df["path_ADC"].astype(str)
        self.df = self.df[(self.df["path_T2"].str.len() > 0) & (self.df["path_ADC"].str.len() > 0)]

        # store config
        self.target = target
        assert self.target in ("isup3", "isup6", "binary_low_high", "binary_all", "isup0145")
        self.channels = tuple(channels)
        self.missing_channel_mode = missing_channel_mode
        assert self.missing_channel_mode in ("zeros", "repeat_t2")
        self.pct_lower = pct_lower
        self.pct_upper = pct_upper
        self.transform = transform

        # LRU cache setup (bind instance methods to cached functions)
        self._load_vol_cached = lru_cache(maxsize=cache_size)(self._load_vol_impl)

        # Keep only needed columns to save RAM in __getitem__
        needed = {
            "case_id","fold","z","label6","label3","patient_ISUP", "merged_ISUP", "has_lesion",
            "bbox_prostate_z0","bbox_prostate_z1","bbox_prostate_h0","bbox_prostate_h1","bbox_prostate_w0","bbox_prostate_w1",
            *self.channels
        }
        self.label6_column = label6_column
        missing = needed - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        self.df = self.df[list(needed)].reset_index(drop=True)

        if self.target == "binary_low_high":
            before = len(self.df)
            self.df = self.df[self.df["label3"].isin([0, 2])].reset_index(drop=True)
            dropped = before - len(self.df)
            print(f"[PicaiSliceDataset] binary_low_high: dropped {dropped} rows with label3==1; kept {len(self.df)}")

    # --------- volume IO ---------
    def _load_vol_impl(self, path: str) -> np.ndarray:
        """Load a 3D volume from disk and return (Z,H,W) float32 array in [raw units].
           NOTE: this is wrapped by an LRU cache in __init__.
        """
        if not path:
            raise FileNotFoundError("Empty path string")
        img = nib.load(path)  # use proxy/ lazy until get_fdata
        arr = img.get_fdata(dtype=np.float32)
        return _to_zhw(arr)

    def _get_channel_slice(self, path: str, z: int, bbox: Tuple[int,int,int,int,int,int], fallback_2d: Optional[np.ndarray]) -> np.ndarray:
        """
        Load a single z-slice from a channel path, crop to bbox -> (H',W') float32.
        If path is empty:
           - 'zeros': return zeros_like(fallback_2d)
           - 'repeat_t2': return fallback_2d (caller should pass T2 slice)
        """
        z0, z1, h0, h1, w0, w1 = bbox
        if path:
            vol = self._load_vol_cached(path)
            sl = vol[z]  # (H,W)
            return sl[h0:h1, w0:w1].astype(np.float32)
        else:
            if self.missing_channel_mode == "repeat_t2" and fallback_2d is not None:
                return fallback_2d.copy().astype(np.float32)
            if fallback_2d is None:
                raise ValueError("fallback_2d is None but missing_channel_mode needs it")
            return np.zeros_like(fallback_2d, dtype=np.float32)

    # --------- Dataset API ---------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # coords & bbox
        z = int(row["z"])
        bbox = (
            int(row["bbox_prostate_z0"]), int(row["bbox_prostate_z1"]),
            int(row["bbox_prostate_h0"]), int(row["bbox_prostate_h1"]),
            int(row["bbox_prostate_w0"]), int(row["bbox_prostate_w1"]),
        )

        # paths (strip!)
        paths = {k: str(row[k]).strip() for k in self.channels}
        p_t2 = paths.get("path_T2", "")

        # Load T2 slice first (used for shape & optional fallback)
        t2_full = self._load_vol_cached(p_t2)
        z0, z1, h0, h1, w0, w1 = bbox
        t2_slice = t2_full[z][h0:h1, w0:w1].astype(np.float32)

        # Other channels
        ch_slices: List[np.ndarray] = []
        for ch in self.channels:
            if ch == "path_T2":
                ch_slices.append(t2_slice)
            else:
                ch_slices.append(self._get_channel_slice(paths.get(ch, ""), z, bbox, t2_slice))

        # Normalize each channel (percentile windowing)
        ch_slices = [window_percentile(s, self.pct_lower, self.pct_upper) for s in ch_slices]

        # Stack to CxHxW torch tensor
        img = np.stack(ch_slices, axis=0)  # (C,H,W)
        img_t = torch.from_numpy(img)

        # Labels
        y6 = int(row[self.label6_column])
        # if not self.use_skip: 
        #     # if using the MRIs that don't have lesion masks, we don't have derived slice-level labels
        #     # using 'patient_ISUP' column falls back to patient-level isup for all slices
        #     # merged_ISUP falls back to patient-level isup only for slices where the MRI has no lesion mask
        #     y6 = int(row["patient_ISUP"])
    
        if self.target == "isup3":
            y = map_isup3(y6)
        elif self.target == "binary_low_high":
            y = map_binary_low_high(y6)
        elif self.target == "binary_all":
            y = map_binary_all(y6)
        elif self.target == "isup0145":
            y = map_isup0145(y6)
        else:
            y = y6

        meta = {
            "case_id": row["case_id"],
            "fold": row["fold"],
            "z": z,
            "has_lesion": int(row["has_lesion"]),
            "bbox": bbox,
            "channels": {k: paths.get(k, "") for k in self.channels},
        }

        if self.transform is not None:
            out = self.transform(img_t, y, meta)
            # Support transforms that either return (img,label) or (img,label,meta)
            if isinstance(out, tuple) and len(out) == 3:
                img_t, y, meta = out
            elif isinstance(out, tuple) and len(out) == 2:
                img_t, y = out
            else:
                img_t = out  # transform returned just the image

        return {
            "image": img_t,           # Tensor [C,H,W], float32 in [0,1]
            "label": int(y),          # int
            "label6": int(y6),        # int (original)
            "case_id": meta["case_id"],
            "fold": meta["fold"],
            "z": meta["z"],
            "has_lesion": meta["has_lesion"],
            "bbox": meta["bbox"],
            "channels": meta["channels"],
        }
