import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------
#            ISUP Mapping Helpers
# -------------------------------------------------------------------

def isup_to_3class(y: int) -> int:
    """0–1 → 0, 2–3 → 1, 4–5 → 2"""
    return 0 if y <= 1 else (1 if y <= 3 else 2)

def isup_to_binary(y: int) -> int:
    """0–1 → 0, 2–5 → 1"""
    return 0 if y <= 1 else 1

def contiguous_to_true_isup(c: int) -> int:
    """
    Maps histopathology contiguous classes (0,1,2,3) back to real ISUP labels (0,1,4,5).
    """
    mapping = {0: 0, 1: 1, 2: 4, 3: 5}
    if c not in mapping:
        raise ValueError(f"Invalid contiguous histo label {c}")
    return mapping[c]

# -------------------------------------------------------------------
#            Load Histopathology Encodings
# -------------------------------------------------------------------

def get_histo_by_isup(
    encodings_dir,
    marksheet_csv,
    num_classes=4,
    provider='all'
):
    """
    Load histopathology encodings and group them by class.

    Histopath files are named as:
        <patchID>_<contiguous0145label>.npy
    where contiguous0145label ∈ {0,1,2,3} → real ISUP {0,1,4,5}

    Marksheets store the true ISUP labels for each patch.

    Returns:
        histo_dict: list of lists, histo_dict[class] = [embedding tensors]
    """
    encodings_dir = Path(encodings_dir)
    df = pd.read_csv(marksheet_csv)

    # Optional filtering by provider
    if provider != 'all':
        df = df[df['data_provider'] == provider]

    # Map filename (patch ID) → true ISUP grade
    fname2isup = dict(zip(df["FILENAME"].astype(str), df["isup_grade"].astype(int)))

    # Prepare containers
    out = [[] for _ in range(num_classes)]

    # Iterate through all encoding files
    for path in encodings_dir.rglob("*.npy"):
        stem = path.stem  # e.g.  abcd1234_2
        parts = stem.split("_")

        if len(parts) < 2:
            continue  # Weird file, ignore

        patch_id = parts[0]
        try:
            contiguous_label = int(parts[1])  # 0/1/2/3
        except ValueError:
            continue

        # Map contiguous label → true ISUP label
        isup = contiguous_to_true_isup(contiguous_label)

        # Ensure the patch exists in the marksheet
        if patch_id not in fname2isup:
            continue

        # Optional consistency check:
        # marksheet_isup = fname2isup[patch_id]
        # assert marksheet_isup == isup, \
        #     f"Mismatch: file says {isup}, marksheet says {marksheet_isup}"

        # Load vector
        vector = torch.as_tensor(np.load(path))
        vector = F.normalize(vector, p=2, dim=0)

        # Determine output class
        if num_classes == 3:
            out_key = isup_to_3class(isup)
        elif num_classes == 2:
            out_key = isup_to_binary(isup)
        elif num_classes == 4:
            # Classes 0,1,4,5 → 0,1,2,3
            mapping = {0: 0, 1: 1, 4: 2, 5: 3}
            if isup not in mapping:
                continue
            out_key = mapping[isup]
        else:
            out_key = isup

        out[out_key].append(vector)

    return out

# -------------------------------------------------------------------
#            Negative Sampling Logic
# -------------------------------------------------------------------

def get_hard_negative_label(label, num_classes=3):
    """Pick an adjacent class as a hard negative."""
    if label == 0:
        return 1
    if label == num_classes - 1:
        return label - 1
    return random.choice([label - 1, label + 1])

def get_random_sample(label, histo_dict):
    return random.choice(histo_dict[label])

def get_triplet_samples(mri_gt_label, histo_dict, num_classes=3):
    """Return (positive, negative) histo embeddings for a given MRI label."""
    pos = get_random_sample(mri_gt_label, histo_dict)
    neg = get_random_sample(get_hard_negative_label(mri_gt_label, num_classes), histo_dict)
    return pos, neg

# -------------------------------------------------------------------
#            Loss Functions
# -------------------------------------------------------------------

def triplet_loss(
    mri_emb,
    mri_gt_label,
    histo_dict,
    margin: float = 0.2,
    reduction: str = "mean",
    num_classes=3
):
    pos, neg = get_triplet_samples(mri_gt_label, histo_dict, num_classes)
    pos = pos.to(device=mri_emb.device, dtype=mri_emb.dtype)
    neg = neg.to(device=mri_emb.device, dtype=mri_emb.dtype)

    loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2.0, reduction=reduction)
    return loss_fn(mri_emb, pos, neg)

def triplet_loss_batch(embeddings, labels, histo_dict,
                       margin: float = 0.2,
                       reduction: str = "mean",
                       num_classes=4):
    losses = []
    for i in range(embeddings.size(0)):
        losses.append(
            triplet_loss(
                embeddings[i],
                int(labels[i].item()),
                histo_dict,
                margin=margin,
                reduction=reduction,
                num_classes=num_classes
            )
        )
    return torch.stack(losses).mean()
