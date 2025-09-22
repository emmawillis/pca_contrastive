import argparse, os, csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from abmil_isup import ABMILISUP
from dataloader_panda_mil import PandaPatchBagDataset, pad_collate

def dataloader_from_df(df: pd.DataFrame, root: str, batch_size: int, num_workers: int, device: torch.device):
    ds = PandaPatchBagDataset(df, root=root, max_patches=None)  # use ALL patches
    pin = (device.type == "cuda")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=pad_collate,
        pin_memory=pin, persistent_workers=(num_workers > 0)
    )

@torch.inference_mode()
def export_split(model, loader, out_dir: Path, manifest_writer):
    device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)

    total, correct = 0, 0

    for X, y, mask, metas in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Forward pass using the model's return_h option:
        # logits [B,6], attn [B,N], z [B,512], h_pre [B,384]
        logits, attn, z, h = model(X, mask, return_h=True)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        conf  = probs.max(dim=1).values

        total += y.numel()
        correct += (preds == y).sum().item()

        # Unpack batch and save per-slide .pt
        for i, meta in enumerate(metas):
            image_id = meta["image_id"]
            isup = int(meta["isup_grade"])
            n_valid = int(mask[i].sum().item())
            fname = f"{image_id}_{isup}.pt"
            fpath = out_dir / fname

            payload = {
                "image_id": image_id,
                "isup_grade": isup,
                "fold": int(meta.get("fold", -1)),
                "split": str(meta.get("split", "")),
                "num_patches": int(meta.get("num_patches", n_valid)),
                "source_path": meta.get("path", ""),

                # Embeddings and outputs
                "z": z[i].detach().cpu(),                 # [512]
                "h": h[i].detach().cpu(),                 # [384]  (ReLU(fc1(z)) pre-dropout from forward)
                "logits": logits[i].detach().cpu(),       # [6]
                "probs": probs[i].detach().cpu(),         # [6]
                "attn": attn[i, :n_valid].detach().cpu(), # [N_valid]

                # Predictions for manifest / QA
                "pred_isup": int(preds[i].item()),
                "pred_conf": float(conf[i].item()),
                "is_mislabel": int(preds[i].item() != isup),
            }
            torch.save(payload, fpath)

            manifest_writer.writerow({
                "image_id": image_id,
                "isup_grade": isup,
                "fold": payload["fold"],
                "split": payload["split"],
                "num_patches": payload["num_patches"],
                "embedding_path": str(fpath),
                "source_path": payload["source_path"],
                "pred_isup": payload["pred_isup"],
                "pred_conf": payload["pred_conf"],
                "is_mislabel": payload["is_mislabel"],
            })

    acc = correct / max(total, 1)
    return acc, total, correct

def main():
    ap = argparse.ArgumentParser("Export slide-level embeddings (z, h) and logits; flag mislabels; report accuracy")
    ap.add_argument("--csv", required=True, help="panda_folds.csv (with image_id,isup_grade,fold,split)")
    ap.add_argument("--root", required=True, help="dir with <image_id>_<isup>.h5")
    ap.add_argument("--ckpt", required=True, help="path to best .pth checkpoint")
    ap.add_argument("--out_dir", required=True, help="output directory for embeddings")
    ap.add_argument("--fold", type=int, required=True, help="which fold was val during training")
    ap.add_argument("--which", default="all", choices=["all","train","val","test"],
                    help="which split(s) to export")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load CSV and partition like the training script
    df = pd.read_csv(args.csv)
    df["fold"] = df["fold"].astype(int)
    df["isup_grade"] = df["isup_grade"].astype(int)

    train_df = df[(df["split"] == "cv") & (df["fold"] != int(args.fold))].copy()
    val_df   = df[(df["split"] == "cv") & (df["fold"] == int(args.fold))].copy()
    test_df  = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        test_df = None

    # Model
    model = ABMILISUP(
        input_dim=1536, num_classes=6,
        proj_dim=512, attn_hidden=384,
        p_input_dropout=0.10, p_mid_dropout=0.25
    ).to(device)

    # Load checkpoint (handle both raw SD and dict with 'model')
    state = torch.load(args.ckpt, map_location=device)
    sd = state.get("model", state)
    # strip 'module.' if present
    sd = { (k.replace("module.", "", 1) if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "embeddings_manifest.csv"

    # Write CSV header
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=[
            "image_id","isup_grade","fold","split","num_patches",
            "embedding_path","source_path","pred_isup","pred_conf","is_mislabel"
        ])
        writer.writeheader()

        # Export & report accuracy per split
        if args.which in ("all","train") and len(train_df):
            print(f"Exporting TRAIN ({len(train_df)}) …")
            Ltrain = dataloader_from_df(train_df, args.root, args.batch_size, args.num_workers, device)
            acc, tot, cor = export_split(model, Ltrain, out_dir / "train", writer)
            print(f"[TRAIN] n={tot}  acc={acc:.4f}")

        if args.which in ("all","val") and len(val_df):
            print(f"Exporting VAL ({len(val_df)}) …")
            Lval = dataloader_from_df(val_df, args.root, args.batch_size, args.num_workers, device)
            acc, tot, cor = export_split(model, Lval, out_dir / "val", writer)
            print(f"[VAL]   n={tot}  acc={acc:.4f}")

        if args.which in ("all","test") and test_df is not None and len(test_df):
            print(f"Exporting TEST ({len(test_df)}) …")
            Ltest = dataloader_from_df(test_df, args.root, args.batch_size, args.num_workers, device)
            acc, tot, cor = export_split(model, Ltest, out_dir / "test", writer)
            print(f"[TEST]  n={tot}  acc={acc:.4f}")

    print(f"Done. Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
