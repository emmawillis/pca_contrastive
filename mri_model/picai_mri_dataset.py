# picai_mri_dataset.py
# Dataset for PI-CAI MRI (T2W, HBV, ADC) with optional gland-ROI crop,
# split filtering, and ISUP labels for classification.
#
# Changes vs earlier version:
# - Adds `isup_map` (key -> int label) and returns `sample["isup"]`
# - Adds `allowed_keys` to restrict to a given train/val split
# - Returns a stable `sample["key"] = "<pid>_<study>"`
# - Keeps `pid`/`study` as strings (avoid losing leading zeros)
# - Makes lesion-mask loading optional (off by default)
# - Uses gland mask for ROI/normalization but does not return it by default
# - Pads volumes to multiple-of-N (default 16) for encoder stability
# - Asserts channel order matches pretrained checkpoint expectations

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.utils.data as tud

# New helpers for reference-based resampling
def _read_sitk(path: Path) -> "sitk.Image":
    return sitk.ReadImage(str(path))

def _make_ref_from_img(img: "sitk.Image", out_spacing_dhw: Tuple[float,float,float]) -> "sitk.Image":
    # convert (D,H,W) → (x,y,z)
    out_sp_xyz = (out_spacing_dhw[2], out_spacing_dhw[1], out_spacing_dhw[0])
    in_size = np.array(img.GetSize(), dtype=np.int32)      # (W,H,D)
    in_sp   = np.array(img.GetSpacing(), dtype=np.float32) # (x,y,z)
    out_sp  = np.array(out_sp_xyz, dtype=np.float32)
    out_size = np.maximum(np.round(in_size * (in_sp / out_sp)).astype(np.int32), 1)
    # resample T2 to target spacing, preserving origin & direction → use as reference grid
    ref = sitk.Resample(
        img,
        tuple(map(int, out_size)),
        sitk.Transform(),               # identity
        sitk.sitkLinear,
        img.GetOrigin(),
        tuple(map(float, out_sp)),
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )
    return ref

def _resample_like(img: "sitk.Image", ref: "sitk.Image", is_mask: bool=False) -> "sitk.Image":
    return sitk.Resample(
        img, ref, sitk.Transform(),
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear,
        0.0, img.GetPixelID()
    )



try:
    import SimpleITK as sitk
except ImportError as e:
    raise ImportError("Please `pip install SimpleITK` to read .mha/.nii(.gz) files") from e


# -------------------------
# I/O and geometry helpers
# -------------------------
def _sitk_load(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a 3D image with SimpleITK. Returns array [D,H,W] and spacing as (D,H,W) (i.e., (sz, sy, sx))."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]
    sx, sy, sz = img.GetSpacing()                         # (x,y,z)
    return arr, (sz, sy, sx)  # return spacing ordered as (D,H,W)


def load_any_3d_image(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return _sitk_load(path)


def resample_to_spacing(
    vol: np.ndarray,
    in_spacing: Tuple[float, float, float],
    out_spacing: Tuple[float, float, float],
    is_mask: bool = False,
) -> np.ndarray:
    """
    Resample [D,H,W] → target spacing using SimpleITK (nearest for masks, linear otherwise).
    Spacing tuples are in (D,H,W) order; SITK expects (x,y,z), handled internally.
    """
    img = sitk.GetImageFromArray(vol)
    # set incoming spacing back to (x,y,z)
    img.SetSpacing((in_spacing[2], in_spacing[1], in_spacing[0]))
    in_size = np.array(img.GetSize(), dtype=np.int32)              # (W,H,D)
    in_sp   = np.array(img.GetSpacing(), dtype=np.float32)         # (x,y,z)
    out_sp  = np.array((out_spacing[2], out_spacing[1], out_spacing[0]), dtype=np.float32)

    out_size = np.maximum(np.round(in_size * (in_sp / out_sp)).astype(np.int32), 1)

    res = sitk.ResampleImageFilter()
    res.SetSize(tuple(map(int, out_size)))
    res.SetOutputSpacing(tuple(map(float, out_sp)))
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    res.SetOutputOrigin((0.0, 0.0, 0.0))
    res.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out = res.Execute(img)
    arr = sitk.GetArrayFromImage(out)
    return (arr.astype(np.uint8) if is_mask else arr.astype(np.float32))


def robust_zscore(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Clip to 1–99th percentile and z-score (compute stats within mask if provided)."""
    if mask is not None and mask.sum() > 50:
        vals = x[mask > 0]
    else:
        vals = x
    lo, hi = np.percentile(vals, [1, 99]).astype(np.float32)
    x = np.clip(x, lo, hi)
    mu, sd = float(x.mean()), float(x.std() + 1e-6)
    return (x - mu) / sd


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    """Return tight bounding box (z, y, x) for a binary mask [D,H,W], or None if empty."""
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    zmin, ymin, xmin = idx.min(0)
    zmax, ymax, xmax = idx.max(0) + 1
    return slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax)


# -------------------------
# Dataset
# -------------------------
class PicaiMRIDataset(tud.Dataset):
    """
    PI-CAI MRI dataset that stacks requested series (default: axial T2W + HBV + ADC) as channels.
    Optionally uses a prostate gland mask to crop to ROI and to normalize intensities within-gland.

    Directory layout expected (per your screenshots):
      mri_data/
        images/
          10000/
            10000_1000000_t2w.mha
            10000_1000000_hbv.mha
            10000_1000000_adc.mha
            10000_1000000_sag.mha   # optional, ignored by default
            10000_1000000_cor.mha   # optional, ignored by default
          10001/
            ...
      mri_data/
        picai_labels/
          ... files containing patient & study ids; names vary across forks ...
            (we search by substrings; see gland_keywords)

    Args:
        images_root: path to `mri_data/images`
        labels_root: path to `mri_data/picai_labels` (optional but recommended for ROI)
        sequences:   subset of {"t2w","hbv","adc"} (pretrained expects this order)
        target_spacing: resampling spacing (D,H,W) in mm (e.g., (0.5, 0.5, 3.0))
        crop_to_gland: crop to gland bbox + margin if a gland mask is found
        gland_keywords: filename keywords to identify gland masks
        isup_map: mapping {"<pid>_<study>": int label 0..5}; required for classification
        allowed_keys: set of keys to include (train/val split like {"10000_1000000", ...})
        return_gland_mask: whether to include the gland mask tensor in the sample dict
        load_lesion_mask: whether to also load lesion masks (off by default for speed)
        pad_to_multiple_of: reflect-pad each spatial dim to multiple-of-N (default 16)
        require_isup: if True, raise if a key has no ISUP label in `isup_map`
    """

    def __init__(
        self,
        images_root: Path | str,
        labels_root: Optional[Path | str] = None,
        sequences: List[str] = ("t2w", "hbv", "adc"),
        target_spacing: Tuple[float, float, float] = (0.5, 0.5, 3.0),
        crop_to_gland: bool = True,
        gland_keywords: Tuple[str, ...] = ("whole_gland", "prostate", "wg"),
        # new/changed:
        isup_map: Optional[Dict[str, int]] = None,   # {"pid_study": 0..5}
        allowed_keys: Optional[Set[str]] = None,     # train/val split keys
        return_gland_mask: bool = False,
        load_lesion_mask: bool = False,
        pad_to_multiple_of: int = 32,
        require_isup: bool = True,
    ):
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root) if labels_root is not None else None
        self.sequences = list(sequences)
        self.target_spacing = target_spacing
        self.crop_to_gland = crop_to_gland
        self.gland_keywords = tuple(k.lower() for k in gland_keywords)

        self.isup_map = isup_map or {}
        self.allowed_keys = allowed_keys
        self.return_gland_mask = return_gland_mask
        self.load_lesion_mask = load_lesion_mask
        self.pad_m = int(pad_to_multiple_of)
        self.require_isup = require_isup

        # enforce the modality order used by your pretrained checkpoint
        if tuple(self.sequences) != ("t2w", "hbv", "adc"):
            raise ValueError(
                f"Pretrained encoder expects sequences ('t2w','hbv','adc') in this order, got {self.sequences}"
            )

        if not self.images_root.exists():
            raise FileNotFoundError(self.images_root)

        # Precompute voxel margin for ROI expansion (D,H,W) from mm
        # Defaults roughly (6mm, 10mm, 10mm) as earlier
        self._bbox_margin_mm = (6.0, 10.0, 10.0)
        self._bbox_margin_vox = tuple(
            int(round(m / s)) for m, s in zip(self._bbox_margin_mm, self.target_spacing)
        )

        # Build index from *_t2w.mha and filter by allowed_keys if provided
        self.index: List[Tuple[str, str]] = []  # [(pid, study)]
        for pid_dir in sorted(p for p in self.images_root.iterdir() if p.is_dir()):
            for t2 in sorted(pid_dir.glob(f"{pid_dir.name}_*_t2w.mha")):
                parts = t2.stem.split("_")
                if len(parts) >= 3:
                    pid, study = parts[0], parts[1]
                    key = f"{pid}_{study}"
                    if (self.allowed_keys is None) or (key in self.allowed_keys):
                        self.index.append((pid, study))

        if not self.index:
            raise RuntimeError(
                f"No studies found under {self.images_root} after applying split filter. "
                f"Expected files like <pid>_<study>_t2w.mha inside subfolders."
            )

    def __len__(self) -> int:
        return len(self.index)

    # ---------- label discovery (for masks) ----------
    def _find_label_file(self, pid: str, study: str, keywords: Tuple[str, ...]) -> Optional[Path]:
        if self.labels_root is None:
            return None
        # search deep; label filenames vary across forks
        for p in self.labels_root.rglob(f"*{pid}*{study}*"):
            if p.is_file():
                name = p.name.lower()
                if any(k in name for k in keywords) and (
                    name.endswith(".mha") or name.endswith(".nii.gz") or name.endswith(".nii")
                ):
                    return p
        return None

    # ---------- main loader ----------
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor | str]:
        pid, study = self.index[i]
        key = f"{pid}_{study}"
        base = self.images_root / pid

        # --- gland mask for ROI/normalization (not returned unless requested)
        # --- build a common reference grid from T2W ---
        t2_path = self.images_root / pid / f"{pid}_{study}_t2w.mha"
        if not t2_path.exists():
            raise FileNotFoundError(f"Missing T2W for {key}: {t2_path}")
        t2_img_in = _read_sitk(t2_path)
        ref_img   = _make_ref_from_img(t2_img_in, self.target_spacing)  # same size/origin/direction for all series
        t2_img    = _resample_like(t2_img_in, ref_img, is_mask=False)

        # --- optional gland mask for ROI/normalization (resampled to ref) ---
        gland_mask_resampled = None
        gland_path = self._find_label_file(pid, study, self.gland_keywords)
        if gland_path is not None:
            gland_img_in = _read_sitk(gland_path)
            gland_img    = _resample_like(gland_img_in, ref_img, is_mask=True)
            gland_mask_resampled = (sitk.GetArrayFromImage(gland_img) > 0).astype(np.uint8)  # [D,H,W]

        # --- load other sequences and resample them to the SAME ref grid ---
        vols = []
        for seq in self.sequences:
            f = self.images_root / pid / f"{pid}_{study}_{seq}.mha"
            if not f.exists():
                raise FileNotFoundError(f"Missing sequence '{seq}' for {key}: {f}")
            img_in = _read_sitk(f)
            img_rs = _resample_like(img_in, ref_img, is_mask=False)
            arr = sitk.GetArrayFromImage(img_rs).astype(np.float32)  # [D,H,W]
            arr = robust_zscore(arr, gland_mask_resampled)
            vols.append(arr)

        vol = np.stack(vols, axis=0)  # [C,D,H,W]  <-- shapes now match across series

        # --- crop to gland ROI (+ margin) if available
        crop_slices = None
        if self.crop_to_gland and gland_mask_resampled is not None:
            bbox = _bbox_from_mask(gland_mask_resampled)
            if bbox is not None:
                z, y, x = bbox
                dz, dy, dx = self._bbox_margin_vox
                z = slice(max(0, z.start - dz), min(gland_mask_resampled.shape[0], z.stop + dz))
                y = slice(max(0, y.start - dy), min(gland_mask_resampled.shape[1], y.stop + dy))
                x = slice(max(0, x.start - dx), min(gland_mask_resampled.shape[2], x.stop + dx))
                crop_slices = (z, y, x)
                vol = vol[:, z, y, x]
                gland_mask_resampled = gland_mask_resampled[z, y, x]

        # --- (optional) lesion mask – OFF by default for speed
        lesion_mask_resampled = None
        if self.load_lesion_mask:
            lesion_path = self._find_label_file(pid, study, ("lesion", "cspca", "cs_pca"))
            if lesion_path is not None:
                lesion_arr, lesion_sp = load_any_3d_image(lesion_path)
                lesion_mask_resampled = resample_to_spacing(
                    lesion_arr, lesion_sp, self.target_spacing, is_mask=True
                )
                lesion_mask_resampled = (lesion_mask_resampled > 0).astype(np.uint8)
                if crop_slices is not None:
                    z, y, x = crop_slices
                    lesion_mask_resampled = lesion_mask_resampled[z, y, x]

        # --- pad to multiple-of (e.g., 16) for nnU-Net encoder stability
        def _pad_to_m(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 4:  # [C,D,H,W]
                _, D, H, W = arr.shape
            else:              # [D,H,W]
                D, H, W = arr.shape

            def pad_len(sz: int) -> int:
                m = self.pad_m
                r = sz % m
                return 0 if r == 0 else (m - r)

            pd, ph, pw = pad_len(D), pad_len(H), pad_len(W)
            if arr.ndim == 4:
                return np.pad(arr, ((0, 0), (0, pd), (0, ph), (0, pw)), mode="reflect")
            else:
                return np.pad(arr, ((0, pd), (0, ph), (0, pw)), mode="reflect")

        vol = _pad_to_m(vol)
        if gland_mask_resampled is not None:
            gland_mask_resampled = _pad_to_m(gland_mask_resampled)

        # --- build sample dict
        sample: Dict[str, torch.Tensor | str] = {
            "key": key,                               # stable identifier "<pid>_<study>"
            "pid": pid,                               # keep as strings (no leading-zero loss)
            "study": study,
            "mri": torch.from_numpy(vol).float(),     # [C,D,H,W]
        }

        # ISUP label
        if key in self.isup_map:
            sample["isup"] = torch.tensor(int(self.isup_map[key]), dtype=torch.long)
        else:
            if self.require_isup:
                raise KeyError(f"ISUP label missing for key {key}. Add to `isup_map`.")
            # else: set -1 sentinel
            sample["isup"] = torch.tensor(-1, dtype=torch.long)

        # Optional returns
        if self.return_gland_mask and (gland_mask_resampled is not None):
            sample["gland_mask"] = torch.from_numpy(gland_mask_resampled[None]).float()  # [1,D,H,W]
        if self.load_lesion_mask and (lesion_mask_resampled is not None):
            sample["lesion_mask"] = torch.from_numpy(lesion_mask_resampled[None]).float()

        return sample
