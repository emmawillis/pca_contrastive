import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

#triplet_loss_utils.py
def isup_to_3class(y: int) -> int:
    return 0 if y <= 1 else (1 if y <= 3 else 2)

def get_histo_by_isup(
    encodings_dir,
    marksheet_csv,
    num_classes = 3
):
    encodings_dir = Path(encodings_dir)
    df = pd.read_csv(marksheet_csv)

    df = df[df['data_provider'] == 'karolinska'] # only use one provider to avoid disjoint embedding spaces
    fname2isup = dict(zip(df["FILENAME"].astype(str), df["isup_grade"].astype(int)))

    out = [[] for i in range(num_classes)]

    for path in encodings_dir.rglob("*.npy"):
        filename = path.stem.split("_")[0]
        try:
            isup = fname2isup[filename]
        except KeyError:
            continue
        vector = torch.as_tensor(np.load(path))
        out[isup_to_3class(isup) if num_classes == 3 else isup].append(vector)

    return out

def get_hard_negative_label(label, num_classes=3):
    # to get challenging negative pairs, we pick a negative sample with a class label adjacent to the anchor
    if label == 0:
        return 1 # only neighbour of 0 is 1
    if label == num_classes - 1:
        return label - 1 # only neighbour of biggest class is second biggest
    
    else:
        return random.choice([label-1, label+1]) # otherwise randomly pick between previous and next label
    
def get_random_sample(label, histo_dict):
    return random.choice(histo_dict[label])


def get_triplet_samples(mri_gt_label, histo_dict, num_classes=3): 
    # positive sample = pick a random histo sample with same mri_gt_label
    pos = get_random_sample(mri_gt_label, histo_dict)
    # negative sample = pick a random histo sample with mri_gt_label += 1 
    # (for edges, don't wrap. ie class 0 should always be paired with class 1)
    neg = get_random_sample(get_hard_negative_label(mri_gt_label, num_classes), histo_dict)
    return pos, neg

def triplet_loss(
    mri_emb,
    mri_gt_label,
    histo_dict,
    margin: float = 0.2,
    reduction: str = "mean",
    num_classes=3
):
    pos, neg = get_triplet_samples(mri_gt_label, histo_dict, num_classes)
    # align to anchor’s device/dtype to avoid mismatch
    pos = pos.to(device=mri_emb.device, dtype=mri_emb.dtype)
    neg = neg.to(device=mri_emb.device, dtype=mri_emb.dtype)

    loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2.0, reduction=reduction)
    return loss_fn(mri_emb, pos, neg)

def triplet_loss_batch(embeddings, labels, histo_dict, margin: float = 0.2, reduction: str = "mean", num_classes=3):
    losses = []
    for i in range(embeddings.size(0)):
        losses.append(triplet_loss(embeddings[i], int(labels[i].item()), histo_dict,
                                   margin=margin, reduction=reduction, num_classes=num_classes))
    return torch.stack(losses).mean()
