# dataset_frozen_medsam.py
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_picai_slices import (
    map_isup3, map_binary_low_high, map_binary_all, map_isup0145
)


class PicaiSliceFrozenEncodingDataset(Dataset):
    """
    Loads MedSAM frozen encodings from files of the form:
        <case_id>_<slice>_<isup>.pt
    or:
        <patientID>_<sessionID>_<slice>_<isup>.pt

    The manifest is used ONLY for folds and identifying (case_id, z).
    ISUP is ALWAYS extracted from filenames, not from manifest.
    """

    def __init__(
        self,
        manifest_csv,
        encoding_dir,
        folds=None,
        target="isup3",
        label6_column=None,   # unused for frozen encodings
    ):
        self.encoding_dir = Path(encoding_dir)

        # -------------------------------
        # Load full manifest (keep all columns!)
        # -------------------------------
        df = pd.read_csv(manifest_csv)

        # Ensure fold exists and filter
        if folds is not None:
            folds = [str(f) for f in folds]
            if "fold" not in df.columns:
                raise ValueError("Manifest must contain a 'fold' column.")
            df["fold"] = df["fold"].astype(str)
            df = df[df["fold"].isin(folds)]

        # require case_id + z at least
        if "case_id" not in df.columns or "z" not in df.columns:
            raise ValueError("Manifest must contain 'case_id' and 'z' columns.")

        # -------------------------------
        # Extract isup from filenames
        # -------------------------------
        valid_rows = []
        isup_list = []

        for idx, row in df.iterrows():
            case_id = row["case_id"]
            z = int(row["z"])

            # match files of type <caseid>_*_z_*.pt or exactly <caseid>_z_*.pt
            patterns = [
                f"{case_id}_{z:03d}_*.pt",
                f"{case_id}_*_{z:03d}_*.pt",
            ]

            matches = []
            for p in patterns:
                matches = list(self.encoding_dir.glob(p))
                if matches:
                    break

            if not matches:
                continue  # skip rows with no encoding

            fname = matches[0].name
            try:
                # last part before .pt is ISUP
                isup = int(fname.split("_")[-1].split(".")[0])
            except:
                continue

            valid_rows.append(idx)
            isup_list.append(isup)

        # restrict df to rows that have encodings
        df = df.loc[valid_rows].copy()
        df["isup"] = isup_list

        # -------------------------------
        # Filter out isup 2 and 3 for isup0145
        # -------------------------------
        if target == "isup0145":
            before = len(df)
            df = df[df["isup"].isin([0, 1, 4, 5])]
            removed = before - len(df)
            print(f"[PicaiSliceFrozenEncodingDataset] Removed {removed} slices with ISUP 2 or 3 for target=isup0145")

        df = df.reset_index(drop=True)
        self.df = df
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        z = int(row["z"])
        isup = int(row["isup"])

        # locate file again
        patterns = [
            f"{case_id}_{z:03d}_{isup}.pt",
            f"{case_id}_*_{z:03d}_{isup}.pt",
        ]

        fpath = None
        for p in patterns:
            matches = list(self.encoding_dir.glob(p))
            if matches:
                fpath = matches[0]
                break

        if fpath is None or not fpath.exists():
            raise FileNotFoundError(f"Missing encoding for {case_id}, slice {z}, isup {isup}")

        encoding = torch.load(fpath).float()  # shape [256,64,64] (balanced slice encodings)
        encoding = encoding.squeeze()

        # Map label to target space
        if self.target == "isup3":
            y = map_isup3(isup)
        elif self.target == "binary_low_high":
            y = map_binary_low_high(isup)
        elif self.target == "binary_all":
            y = map_binary_all(isup)
        elif self.target == "isup0145":
            y = map_isup0145(isup)
        else:
            y = isup

        return {
            "image": encoding,   # [256,64,64]
            "label": y,          # mapped label
            "isup": isup,        # original label
            "case_id": case_id,
            "z": z,
            # keep ALL remaining manifest columns
            **{k: row[k] for k in row.index if k not in ["case_id", "z", "isup"]},
        }
