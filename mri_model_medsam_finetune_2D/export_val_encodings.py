#!/usr/bin/env python3
# export_val_encodings.py
#
# Exports slice-level embeddings/logits from the validation folds
# using your trained MedSAMSliceSpatialAttn model.
#
# Alignment with train.py:
# - Uses PicaiSliceDataset (same channels/windowing/cropping logic)
# - Resizes to IMG_SIZE in a collate (same as train.py)
# - Builds SAM backbone by type (vit_b/l/h) and loads YOUR checkpoint's "model" state
# - pixel_mean_std=None (inputs already in [0,1])
# - Defaults: num_classes=3 (isup3), proj_dim=1024 (as in train.py)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# your model & dataset
from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from dataset_picai_slices import PicaiSliceDataset


def resize_to_img_size(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] == IMG_SIZE and x.shape[-1] == IMG_SIZE:
        return x
    return F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)


def parse_val_folds(s: str):
    parts = [p.strip() for p in s.split(",")]
    ints = [int(p) for p in parts if p != ""]
    return ints if len(ints) > 1 else ints[0]


def collate_and_resize(batch):
    # Resize EACH sample to IMG_SIZE before stacking (matches train.py)
    imgs, labels, ids, paths_joined = [], [], [], []
    for s in batch:
        x = s["image"].unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        imgs.append(x)
        labels.append(torch.tensor(int(s["label"]), dtype=torch.long))
        ids.append(f"{s['case_id']}_z{int(s['z'])}")
        paths_joined.append(";".join([v for v in s["channels"].values()]))

    return {
        "images": torch.stack(imgs, dim=0),   # [B,C,IMG_SIZE,IMG_SIZE]
        "labels": torch.stack(labels, dim=0), # [B]
        "ids": ids,
        "paths_joined": paths_joined,
    }


def build_sam_skeleton_cpu(sam_type: str):
    """Instantiate a SAM/MedSAM backbone by type on **CPU** (no external ckpt).
    Your training checkpoint will provide the actual weights later.
    Keeping it on CPU avoids device mismatch during ISUPMedSAM.__init__ probing.
    """
    from segment_anything import sam_model_registry
    try:
        sam = sam_model_registry[sam_type]()  # some builds allow no args
    except TypeError:
        sam = sam_model_registry[sam_type](checkpoint=None)  # others need checkpoint kwarg
    sam.eval()
    return sam  # stay on CPU for construction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to slices manifest CSV")
    ap.add_argument("--val_folds", required=True, help="Val folds: '0' or '0,2'")
    ap.add_argument("--out_dir", required=True, help="Where to save encodings + CSV")
    ap.add_argument("--checkpoint", required=True, help="Training checkpoint (.pt/.pth) containing encoder+head")
    ap.add_argument("--sam_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                    help="Backbone architecture only (weights come from --checkpoint)")

    # Must match your training config (defaults align with train.py)
    ap.add_argument("--proj_dim", type=int, default=1024)      # train.py uses 1024
    ap.add_argument("--attn_dim", type=int, default=256)
    ap.add_argument("--head_hidden", type=int, default=256)

    # DataLoader
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    # Dataset behavior (match training)
    ap.add_argument("--target", choices=["isup3","isup6","binary_low_high", "binary_all"], default="isup3")
    ap.add_argument("--channels", default="path_T2,path_ADC,path_HBV")
    ap.add_argument("--pct_lower", type=float, default=0.5)
    ap.add_argument("--pct_upper", type=float, default=99.5)
    ap.add_argument("--missing_channel_mode", default="zeros", choices=["zeros","repeat_t2"])
    ap.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True,
                help="If true, drop rows with skip==1. Use --no-use-skip to include them.")
    ap.add_argument("--label6_column", default="label6")
    ap.add_argument("--no_pre_neck", action="store_true")   # only if you trained with encoder.neck retained
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    num_classes = 2 if args.target == 'binary_low_high' or args.target == 'binary_all' else 3
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | IMG_SIZE={IMG_SIZE}")

    folds = parse_val_folds(args.val_folds)
    channels = tuple([c.strip() for c in args.channels.split(",") if c.strip()])

    # Dataset filtered to val folds
    ds = PicaiSliceDataset(
        manifest_csv=args.manifest,
        folds=folds if isinstance(folds, (list, tuple)) else [folds],
        use_skip=args.use_skip,
        label6_column=args.label6_column,
        target=args.target,
        channels=channels,
        missing_channel_mode=args.missing_channel_mode,
        pct_lower=args.pct_lower,
        pct_upper=args.pct_upper,
        transform=None,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_and_resize
    )
    print(f"Val rows: {len(ds)} | folds={folds} | channels={channels}")

    # ---- Model: build on CPU, then load weights, THEN move to GPU ----
    sam_cpu = build_sam_skeleton_cpu(args.sam_type)  # CPU
    model = MedSAMSliceSpatialAttn(
        sam_model=sam_cpu,
        num_classes=num_classes,
        proj_dim=args.proj_dim,
        attn_dim=args.attn_dim,
        head_hidden=args.head_hidden,
        head_dropout=0.0,
        use_pre_neck=not args.no_pre_neck,
        pixel_mean_std=None,  # inputs already in [0,1]
    )  # keep on CPU for construction

    # Load checkpoint (train.py saved {"model": model.state_dict(), "epoch": ...})
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."): nk = nk[7:]
        if nk.startswith("model."):  nk = nk[6:]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print("load_state_dict missing keys:", missing)
        print("load_state_dict unexpected keys:", unexpected)

    # NOW move the complete model to target device
    model.to(device)
    model.eval()

    # ---------- Inference ----------
    ids_all, paths_all = [], []
    labels_all, logits_all, embs_all = [], [], []

    use_amp = args.amp and torch.cuda.is_available()
    autocast_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                      else torch.float16)

    with torch.no_grad():
        for batch in dl:
            x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits, emb = model(x, return_attn=False)
            else:
                logits, emb = model(x, return_attn=False)

            # Cast AMP outputs to float32 on CPU BEFORE numpy
            logits = logits.detach().to(torch.float32).cpu()
            emb    = emb.detach().to(torch.float32).cpu()

            # Append exactly once per batch (no duplicates)
            ids_all.extend(batch["ids"])
            paths_all.extend(batch["paths_joined"])
            labels_all.append(y.cpu().numpy())
            logits_all.append(logits.numpy())
            embs_all.append(emb.numpy())

    labels = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0,), dtype=np.int64)
    logits = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, num_classes), dtype=np.float32)
    embs   = np.concatenate(embs_all,   axis=0) if embs_all   else np.empty((0, args.proj_dim), dtype=np.float32)
    probs  = torch.softmax(torch.from_numpy(logits), dim=1).numpy() if logits.size else np.empty((0, num_classes), dtype=np.float32)
    preds  = probs.argmax(axis=1) if probs.size else np.empty((0,), dtype=np.int64)

    # ---- Save arrays
    np.save(out_dir / "val_embeddings.npy", embs)
    np.save(out_dir / "val_logits.npy", logits)
    np.save(out_dir / "val_probs.npy", probs)
    np.save(out_dir / "val_labels.npy", labels)
    np.save(out_dir / "val_preds.npy", preds)

    # --- sanity check & align lengths before DataFrame (should already match now) ---
    n_ids   = len(ids_all)
    n_paths = len(paths_all)
    n_lab   = int(labels.shape[0])
    n_log   = int(logits.shape[0])
    n_emb   = int(embs.shape[0])
    n_pred  = int(preds.shape[0])

    N = min(n_ids, n_paths, n_lab, n_log, n_emb, n_pred)
    if not (n_ids == n_paths == n_lab == n_log == n_emb == n_pred):
        print(f"[warn] length mismatch before DataFrame: "
              f"ids={n_ids}, paths={n_paths}, labels={n_lab}, logits={n_log}, embs={n_emb}, preds={n_pred}. "
              f"Truncating all to N={N}.")
        ids_all   = ids_all[:N]
        paths_all = paths_all[:N]
        labels    = labels[:N]
        logits    = logits[:N]
        embs      = embs[:N]
        preds     = preds[:N]
        probs     = probs[:N]

    # ---- Metadata CSV (slice-level)
    meta = pd.DataFrame({
        "id": ids_all,                   # case_z
        "paths": paths_all,              # "T2;ADC;HBV"
        "label": labels.astype(int),
        "pred": preds.astype(int),
        "max_prob": probs.max(axis=1) if probs.size else np.array([]),
        "val_folds": str(folds),
        "channels": ",".join(channels),
        "target": args.target,
    })
    meta.to_csv(out_dir / "val_metadata.csv", index=False)

    print(f"Saved to {out_dir} | N={len(labels)} | emb_dim={embs.shape[1] if embs.size else 0}")
    for f in ["val_embeddings.npy","val_logits.npy","val_probs.npy","val_labels.npy","val_preds.npy","val_metadata.csv"]:
        print(" -", out_dir / f)


if __name__ == "__main__":
    main()
