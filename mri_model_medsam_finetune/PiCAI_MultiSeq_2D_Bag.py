from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import csv

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MEDSAM_INPUT_SIZE=512 #1024

# 1) deterministic (X,Y,Z) -> (Z,Y,X)
def load_nii(path):
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)      # [X, Y, Z]
    arr = np.transpose(arr, (2, 1, 0)).copy()  # -> [Z, Y, X]
    return arr

# 2) center pad to 1024x1024
def pad_to_square_1024(img):  # img: [3,H,W]
    _, H, W = img.shape
    pad_h = MEDSAM_INPUT_SIZE - H
    pad_w = MEDSAM_INPUT_SIZE - W
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(img, (left, right, top, bottom), mode="constant", value=0.0)


def percentile_clip_minmax(vol: np.ndarray,
                           p_low: float = 0.5,
                           p_high: float = 99.5) -> np.ndarray:
    """Robust min-max to [0,1] after percentile clipping."""
    lo = np.percentile(vol, p_low)
    hi = np.percentile(vol, p_high)
    vol = np.clip(vol, lo, hi)
    if hi > lo:
        vol = (vol - lo) / (hi - lo)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    return vol.astype(np.float32)


def resize_longest_to_1024(img: torch.Tensor) -> torch.Tensor:
    """img: [3,H,W] in [0,1]; resize longest side to 1024 (keep aspect)."""
    _, H, W = img.shape
    if H == MEDSAM_INPUT_SIZE and W == MEDSAM_INPUT_SIZE:
        return img
    scale = MEDSAM_INPUT_SIZE / max(H, W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    img = img.unsqueeze(0)  # [1,3,H,W]
    img = F.interpolate(img, size=(newH, newW), mode="bilinear", align_corners=False)
    return img.squeeze(0)   # [3,newH,newW]


def _load_label_map_from_marksheet(csv_path: Union[str, Path], include_folds: Optional[Tuple[str, ...]] = None) -> Dict[str, int]:
    """Read marksheet.csv and build { 'patientId_studyId': case_ISUP(int) }."""
    csv_path = Path(csv_path)
    label_map: Dict[str, int] = {}
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("patient_id", "")).strip()
            sid = str(row.get("study_id", "")).strip()
            fold = str(row.get("fold")).strip()
            if include_folds is not None and fold not in include_folds:
                continue
            isup_str = str(row.get("case_ISUP", "")).strip()
            if not pid or not sid or isup_str == "":
                continue
            case_id = f"{pid}_{sid}"
            try:
                label_map[case_id] = int(float(isup_str))
            except ValueError:
                # skip rows with non-numeric labels
                continue
    return label_map


class PiCAI_MultiSeq_2D_Bag(Dataset):
    """
    Dataset for PI-CAI nnU-Net 'imagesTr'/'imagesTs' that returns a bag of 2D slices:
      bag:   [N_slices, 3, 1024, 1024]
      label: int or None (loaded from marksheet.csv if provided)
      case_id: str

    Assumes each case has 3 channels:
      *_0000.nii.gz -> T2W
      *_0001.nii.gz -> ADC
      *_0002.nii.gz -> HBV
    """
    def __init__(self,
                 images_dir: Union[str, Path],
                 include_folds: Optional[Tuple[str, ...]] = None,
                 marksheet_csv: Optional[Union[str, Path]] = None,
                 channel_order: Tuple[int, int, int] = (0, 1, 2),
                 clip_percentiles: Tuple[float, float] = (0.5, 99.5)):
        self.images_dir = Path(images_dir)
        self.label_map = _load_label_map_from_marksheet(marksheet_csv, include_folds)
        self.channel_order = channel_order
        self.p_lo, self.p_hi = clip_percentiles

        # discover cases by *_0000.nii.gz
        self.cases = []
        self.isup_counts = [0] * 6
        for case_id, isup in self.label_map.items():
            p0 = self.images_dir / f"{case_id}_0000.nii.gz"
            p1 = self.images_dir / f"{case_id}_0001.nii.gz"
            p2 = self.images_dir / f"{case_id}_0002.nii.gz"
            if p0.exists() and p1.exists() and p2.exists():
                self.cases.append((case_id, p0, p1, p2, isup))
                self.isup_counts[isup] += 1

        if len(self.cases) == 0:
            raise RuntimeError(f"No cases found in {self.images_dir}")

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights inversely proportional to class frequencies."""
        total = sum(self.isup_counts)
        weights = [total / (count + 1e-6) for count in self.isup_counts]  # avoid div by zero
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum() * len(self.isup_counts)  # normalize to num classes
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_id, p0, p1, p2, isup = self.cases[idx]
        vols = [load_nii(p) for p in (p0, p1, p2)]
        # reorder channels if needed (e.g., (0,1,2) for T2W,ADC,HBV)
        vols = [vols[i] for i in self.channel_order]

        # per-sequence robust min-max to [0,1]
        vols = [percentile_clip_minmax(v, self.p_lo, self.p_hi) for v in vols]

        Z, H, W = vols[0].shape
        if not all(v.shape == (Z, H, W) for v in vols):
            raise ValueError("Modalities not aligned—nnU-Net export should already be co-registered.")

        bag = []
        for z in range(Z):
            # stack into [3,H,W] in [0,1]
            slice_3c = np.stack([vols[0][z], vols[1][z], vols[2][z]], axis=0)
            img = torch.from_numpy(slice_3c).float()  # [3,H,W], [0,1]

            # resize & pad to 1024x1024
            img = resize_longest_to_1024(img)
            img = pad_to_square_1024(img)             # [3,1024,1024]

            bag.append(img)

        bag = torch.stack(bag, dim=0)                 # [N_slices, 3, 1024, 1024]
        return bag, torch.tensor(isup, dtype=torch.long), case_id


def collate_one(batch):
    """Use when batch_size=1; unwrap (bag, label, case_id)."""
    assert len(batch) == 1
    return batch[0]


# def collate_pad(batch):
#     """
#     Pad variable-length bags so you can use batch_size>1.
#     Returns: x [B,maxN,3,1024,1024], labels [B], ids [B], mask [B,maxN]
#     """
#     bags, labels, ids = zip(*batch)
#     B = len(bags)
#     maxN = max(b.shape[0] for b in bags)

#     dtype = bags[0].dtype
#     x = torch.zeros(B, maxN, 3, 1024, 1024, dtype=dtype)
#     mask = torch.zeros(B, maxN, dtype=torch.bool)

#     for i, b in enumerate(bags):
#         n = b.shape[0]
#         x[i, :n] = b
#         mask[i, :n] = True

#     y = torch.tensor([int(l) for l in labels], dtype=torch.long)
#     return x, y, list(ids), mask


# --- quick sanity test ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from PIL import Image

    images_dir = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_data/nnUNet_raw_data/Task2203_picai_baseline/imagesTr")
    marksheet_csv = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_data/picai_labels/clinical_information/marksheet_folds.csv")
    include_folds = ['fold0']

    ds = PiCAI_MultiSeq_2D_Bag(images_dir, marksheet_csv=marksheet_csv, include_folds=include_folds)
    print(f"Found {len(ds)} cases in {include_folds}")
    print("ISUP counts:", ds.isup_counts)
    print("Class weights:", ds.get_class_weights())
    assert len(ds) == 300
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_one)

    bag, y, cid = next(iter(loader))
    print("First bag:", bag.shape, "label (case_ISUP):", y, "case:", cid)  # -> [N,3,1024,1024]
    assert bag.ndim == 4 and bag.shape[1:] == (3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE)

    # --- save middle slice (all 3 channels + composite RGB) for quick registration check ---
    if False:
        z = bag.shape[0] // 2            # middle slice index
        sl = bag[z].clone()                 # [3,1024,1024]
        vis = sl.clamp(0, 1)

        def to_uint8(x: torch.Tensor) -> np.ndarray:
            return (x.clamp(0, 1) * 255.0).byte().cpu().numpy()

        outdir = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_model_medsam_finetune/slice_preview") / cid
        outdir.mkdir(parents=True, exist_ok=True)

        t2 = to_uint8(vis[0])
        adc = to_uint8(vis[1])
        hbv = to_uint8(vis[2])

        Image.fromarray(t2).save(outdir / f"{cid}_z{z:03d}_t2w.png")
        Image.fromarray(adc).save(outdir / f"{cid}_z{z:03d}_adc.png")
        Image.fromarray(hbv).save(outdir / f"{cid}_z{z:03d}_hbv.png")

        rgb = np.stack([t2, adc, hbv], axis=-1)  # R=T2W, G=ADC, B=HBV
        Image.fromarray(rgb).save(outdir / f"{cid}_z{z:03d}_rgb.png")

        print(f"Saved middle-slice previews t   o: {outdir}")
                    
