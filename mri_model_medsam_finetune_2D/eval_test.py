#!/usr/bin/env python3
# eval_test.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd  # NEW
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

def load_model(n_classes, proj_dim, sam_checkpoint, model_ckpt_path, device="cuda"):
    # --- Build MedSAM backbone ---
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(sam_checkpoint, map_location="cpu"), strict=True)

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
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    model.eval()
    return model

def collect_logits_and_labels(loader, model, device="cuda", n_classes=2):
    """
    Collect logits, probabilities, labels, and per-sample metadata (patient_id, slice_idx)
    from the given loader.
    """
    all_logits, all_y = [], []
    all_pids, all_slices = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits, _ = model(x)

            all_logits.append(logits.cpu())
            all_y.append(y.cpu())

            # ---- Get patient ids ----
            if "patient_id" in batch:
                pids = batch["patient_id"]
            elif "pid" in batch:
                pids = batch["pid"]
            elif "case_id" in batch:
                pids = batch["case_id"]
            else:
                raise KeyError(
                    "Batch is missing a patient id key. Expected one of "
                    "['patient_id', 'pid', 'case_id']. Please update eval_test.py accordingly."
                )

            # ---- Get slice indices ----
            if "slice_idx" in batch:
                slices = batch["slice_idx"]
            elif "slice_index" in batch:
                slices = batch["slice_index"]
            elif "z" in batch:
                slices = batch["z"]
            else:
                raise KeyError(
                    "Batch is missing a slice index key. Expected one of "
                    "['slice_idx', 'slice_index', 'z']. Please update eval_test.py accordingly."
                )

            # DataLoader collates strings into list[str], tensors remain tensors, etc.
            # Normalize everything into Python scalars / strings.
            # pids: list of strings or tensor of ints -> cast to str
            # slices: list or tensor of ints -> cast to int
            if torch.is_tensor(pids):
                pids = [str(p.item()) for p in pids]
            else:
                pids = [str(p) for p in pids]

            if torch.is_tensor(slices):
                slices = [int(s.item()) for s in slices]
            else:
                slices = [int(s) for s in slices]

            all_pids.extend(pids)
            all_slices.extend(slices)

    if all_logits:
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0).numpy()
        probs = torch.softmax(logits, dim=1).numpy()
    else:
        logits = torch.empty(0, n_classes)
        y = np.empty((0,), dtype=np.int64)
        probs = np.empty((0, n_classes), dtype=np.float32)

    return logits, probs, y, all_pids, all_slices

def write_predictions_csv(
    csv_path: Path,
    patient_ids,
    slice_idxs,
    y_true,
    y_pred,
    pred_col_name: str,
):
    """
    Write per-sample predictions to a CSV.

    - Identified by (patient_id, slice_idx).
    - Columns: patient_id, slice_idx, true_class, <pred_col_name>
    - If CSV exists, updates/creates only the prediction column for matching rows,
      or appends new rows otherwise.
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["patient_id", "slice_idx", "true_class"])

    # Ensure prediction column exists
    if pred_col_name not in df.columns:
        df[pred_col_name] = np.nan

    # Make sure base columns exist
    for col in ["patient_id", "slice_idx", "true_class"]:
        if col not in df.columns:
            df[col] = np.nan

    for pid, sl, yt, yp in zip(patient_ids, slice_idxs, y_true, y_pred):
        # Normalize types
        pid_str = str(pid)
        sl_int = int(sl)
        yt_int = int(yt)
        yp_int = int(yp)

        mask = (df["patient_id"] == pid_str) & (df["slice_idx"] == sl_int)

        if mask.any():
            # Update existing row(s)
            # Keep true_class consistent; but if NaN, fill it.
            df.loc[mask & df["true_class"].isna(), "true_class"] = yt_int

            # If there is an existing non-NaN true_class that disagrees, we silently
            # overwrite here, but you could add a sanity check if you want.
            df.loc[mask, pred_col_name] = yp_int
        else:
            # Append new row
            new_row = {
                "patient_id": pid_str,
                "slice_idx": sl_int,
                "true_class": yt_int,
                pred_col_name: yp_int,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Saved per-sample predictions to: {csv_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True, help="Path to MedSAM ViT-B checkpoint.")
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
    # NEW ARGS
    p.add_argument("--csv_path", required=True,
                   help="Path to CSV file where per-sample predictions will be stored/updated.")
    p.add_argument("--pred_col_name", "--pred-col-name", default="baseline_prediction",
                   help="Column name to use for model predictions (e.g., 'baseline_prediction', 'histo_aligned').")

    args = p.parse_args()

    print("ARGS:", args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.fold_test.split(",") if s.strip()]

    # -------- datasets / loaders (reuse the shared helper exactly as in training) --------
    # We will *ignore* the returned train loader and use the "val" loader as our TEST loader.
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, w_ce, classes_present, n_classes = \
        build_datasets_and_loaders(
            manifest=args.manifest,
            folds_train=folds_train,
            folds_val=[0],
            folds_test=folds_test,
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
        sam_checkpoint=args.sam_checkpoint,
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
    logits_test, probs_test, y_test, patient_ids, slice_idxs = collect_logits_and_labels(
        test_loader, model, device=device, n_classes=n_classes
    )
    spec_targets = tuple(float(s) for s in args.spec_targets.split(",") if s.strip())
    per_cls, macro = train_utils.per_class_operating_points(y_test, probs_test, spec_targets)
    print_operating_points_table(per_cls, macro, spec_targets)

    # -------- Write per-sample predictions to CSV --------
    if logits_test.numel() > 0:
        y_pred = probs_test.argmax(axis=1)
        csv_path = Path(args.csv_path)
        write_predictions_csv(
            csv_path=csv_path,
            patient_ids=patient_ids,
            slice_idxs=slice_idxs,
            y_true=y_test,
            y_pred=y_pred,
            pred_col_name=args.pred_col_name,
        )
    else:
        print("No test samples found; skipping CSV write.")

    # Optional summary saving (left commented out as in your original)
    # summary = {
    #     "target": args.target,
    #     "folds_train": folds_train,
    #     "folds_test": folds_test,
    #     "n_classes": n_classes,
    #     "classes_present": classes_present,
    #     "metrics": {
    #         "loss": float(val["loss"]),
    #         "acc": float(val["acc"]),
    #         "bacc": float(val["bacc"]),
    #         "f1_macro": float(val["f1_macro"]),
    #         "macro_auc": float(val["macro_auc"]),
    #     }
    # }
    # import json
    # with open(outdir / f"eval_summary_test_fold_{'-'.join(folds_test)}.json", "w") as f:
    #     json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
