#!/usr/bin/env python3
# eval_test.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry
import train_utils

from train_utils import (
    build_datasets_and_loaders,
    evaluate_loader,
    format_perclass_acc_auc,
    format_sens_spec,
    print_operating_points_table,
)

def load_model(n_classes, proj_dim, model_ckpt_path, device="cuda"):
    # --- Build MedSAM backbone ---
    sam = sam_model_registry["vit_b"]()
    # sam.load_state_dict(torch.load(sam_checkpoint, map_location="cpu"), strict=True)

    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,
        proj_dim=proj_dim,      # keep consistent with training
        attn_dim=256,
        head_hidden=256,
        head_dropout=0.1,
        use_pre_neck=True,      # pre-neck + spatial attention
        pixel_mean_std=None,    # inputs already in [0,1]
    ).to(device)

    # --- Load classifier weights (supports either raw SD or {"model": SD}) ---
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    model.eval()
    return model

def collect_logits_and_labels(loader, model, device="cuda", n_classes=2):
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits, _ = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, n_classes)
    y = torch.cat(all_y, dim=0).numpy() if all_y else np.empty((0,), dtype=np.int64)
    probs = torch.softmax(logits, dim=1).numpy() if logits.numel() else np.empty((0, n_classes), dtype=np.float32)
    return logits, probs, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    # p.add_argument("--sam_checkpoint", required=True, help="Path to MedSAM ViT-B checkpoint.")
    p.add_argument("--model_ckpt", required=True, help="Trained model checkpoint to evaluate (e.g., ckpt_best.pt).")
    p.add_argument("--outdir", default="./runs/eval")
    p.add_argument("--target", choices=["isup3","isup6","binary_low_high","binary_all"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3",
                   help="Folds used for training (to compute class weights).")
    p.add_argument("--fold_test", default="4", help="Fold(s) to treat as TEST, comma-separated OK.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, drop rows with skip==1. Use --no-use-skip to include them.")
    p.add_argument("--label6_column", default="label6")
    p.add_argument("--spec_targets", default="0.8,0.9,0.95,0.975,0.99",
                   help="Comma-separated specificity targets for operating points table.")
    args = p.parse_args()

    print("ARGS:", args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.fold_test.split(",") if s.strip()]

    # -------- datasets / loaders (reuse the shared helper exactly as in training) --------
    # We will *ignore* the returned train loader and use the "val" loader as our TEST loader.
    train_ds, test_ds, train_loader, test_loader, w_ce, classes_present, n_classes = \
        build_datasets_and_loaders(
            manifest=args.manifest,
            folds_train=folds_train,
            folds_val=folds_test,
            target=args.target,
            use_skip=args.use_skip,
            label6_column=args.label6_column,
            batch_size=args.batch_size,
            pos_ratio=args.pos_ratio,
        )

    # -------- model (no training; just load + eval) --------
    model = load_model(
        n_classes=n_classes,
        proj_dim=args.proj_dim,
        model_ckpt_path=args.model_ckpt,
        device=device
    )

    # -------- One-shot evaluate on TEST loader (same metrics formatting as validation) --------
    val = evaluate_loader(test_loader, model, w_ce=w_ce, device=device, n_classes=n_classes)
    pcs, auc_part = format_perclass_acc_auc(val["per_acc"], val["per_auc"], val["macro_auc"], n_classes)
    extra2 = format_sens_spec(val["per_tpr"], val["per_tnr"], val["macro_tpr"], val["macro_tnr"], n_classes)

    print(f"[TEST] loss {val['loss']:.4f} acc {val['acc']:.4f} bacc {val['bacc']:.4f} f1 {val['f1_macro']:.4f} | "
          f"{pcs}{auc_part}{extra2}")
    print(train_utils.format_confusion_matrix(val["cm"], n_classes=n_classes))

    # -------- Sensitivity at fixed specificity (on TEST), same as the final block in train.py --------
    logits_test, probs_test, y_test = collect_logits_and_labels(
        test_loader, model, device=device, n_classes=n_classes
    )
    spec_targets = tuple(float(s) for s in args.spec_targets.split(",") if s.strip())
    per_cls, macro = train_utils.per_class_operating_points(y_test, probs_test, spec_targets)
    print_operating_points_table(per_cls, macro, spec_targets)

    # -------- Optional: save a small summary to disk --------
    summary = {
        "target": args.target,
        "folds_train": folds_train,
        "folds_test": folds_test,
        "n_classes": n_classes,
        "classes_present": classes_present,
        "metrics": {
            "loss": float(val["loss"]),
            "acc": float(val["acc"]),
            "bacc": float(val["bacc"]),
            "f1_macro": float(val["f1_macro"]),
            "macro_auc": float(val["macro_auc"]),
        }
    }
    # (outdir / f"eval_summary_test_fold_{'-'.join(folds_test)}.npz").write_bytes(
    #     np.savez_compressed(
    #         outdir / f"eval_preds_test_fold_{'-'.join(folds_test)}.npz",
    #         y=y_test, probs=probs_test
    #     ) or b""
    # )
    # # Write a JSON too (without bringing in an extra dep)
    # import json
    # with open(outdir / f"eval_summary_test_fold_{'-'.join(folds_test)}.json", "w") as f:
    #     json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
