#!/usr/bin/env python3
# train_triplet_full.py
#
# Modes:
#   --train_mode baseline : End-to-end CE training (encoder+proj+head)
#   --train_mode triplet  : End-to-end training with (weighted CE + weighted triplet) loss (encoder+proj+head)
#
# Change vs your original:
#   - REMOVED: two-stage logic + stage2_scope + logistic-regression proxy selection + triplet-only phase
#   - ADDED: single-stage "triplet" mode that trains the full pipeline like baseline, but with a combined loss
#
# Still kept:
#   - patient overlap checks
#   - evaluate_loader() based validation/test + operating points + embedding saves
#   - baseline behavior and optimizer param-groups

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# W&B step control (we still use train_utils wrappers for init/log)
import wandb  # only for define_metric; safe when wandb disabled

from ISUPMedSAM import MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry

from triplet_loss_utils import (
    get_histo_by_isup,
    triplet_loss_batch,
)

import train_utils
from train_utils import (
    build_datasets_and_loaders,
    evaluate_loader,
    format_perclass_acc_auc,
    format_sens_spec,
    print_operating_points_table,
    EarlyStopper,
    set_seed,
    wandb_init, wandb_log, wandb_finish,
    save_embeddings,
)


# ---------------- Combined CE + Triplet helpers ----------------
def run_epoch_combined(
    loader,
    model,
    w_ce,
    *,
    ce_weight: float = 1.0,
    triplet_weight: float = 0.0,
    triplet_fn=None,  # callable(embeddings, labels)->loss
    optimizer=None,
    device="cuda",
):
    """
    Train/eval one epoch with total_loss = ce_weight * CE(logits,y) + triplet_weight * Triplet(emb,y).
    Returns:
      avg_total_loss, avg_ce_loss, avg_triplet_loss, acc, f1_macro, bacc
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score  # local import to keep top-level clean

    train_mode = optimizer is not None
    model.train(train_mode)

    ce = nn.CrossEntropyLoss(weight=w_ce)

    total_total, total_ce, total_tri = 0.0, 0.0, 0.0
    total_n, total_correct = 0, 0
    all_pred, all_true = [], []

    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            logits, emb = model(x)  # logits: [B,C], emb: [B,D]
            loss_ce = ce(logits, y)

            loss_tri = torch.tensor(0.0, device=device)
            if (triplet_weight != 0.0) and (triplet_fn is not None):
                loss_tri = triplet_fn(emb, y)

            loss = (ce_weight * loss_ce) + (triplet_weight * loss_tri)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = x.size(0)
            total_total += float(loss.item()) * bs
            total_ce += float(loss_ce.item()) * bs
            total_tri += float(loss_tri.item()) * bs
            total_n += bs

            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            all_pred.append(pred.detach().cpu())
            all_true.append(y.detach().cpu())

    avg_total = total_total / max(1, total_n)
    avg_ce = total_ce / max(1, total_n)
    avg_tri = total_tri / max(1, total_n)
    acc = total_correct / max(1, total_n)

    if all_pred:
        y_pred_np = torch.cat(all_pred).numpy()
        y_true_np = torch.cat(all_true).numpy()
        f1m = float(f1_score(y_true_np, y_pred_np, average="macro"))
        bacc = float(balanced_accuracy_score(y_true_np, y_pred_np))
    else:
        f1m, bacc = 0.0, 0.0

    return avg_total, avg_ce, avg_tri, acc, f1m, bacc


def run_eval_print(val_loader, model, w_ce, device, n_classes):
    val = evaluate_loader(
        val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=False
    )
    pcs, auc_part = format_perclass_acc_auc(val["per_acc"], val["per_auc"], val["macro_auc"], n_classes)
    extra2 = format_sens_spec(val["per_tpr"], val["per_tnr"], val["macro_tpr"], val["macro_tnr"], n_classes)
    return val, pcs, auc_part, extra2


# ---------------- Split / leakage checks ----------------
def _infer_patient_ids_from_df(df, *, patient_col="patient_id", case_col="case_id"):
    """
    Returns a list/array of patient IDs inferred from df.
    Priority:
      1) df[patient_col] if present
      2) parse df[case_col] by splitting on '_' and taking prefix
    """
    if df is None or len(df) == 0:
        return np.array([], dtype=str)

    cols = set(df.columns)
    if patient_col in cols:
        return df[patient_col].astype(str).to_numpy()

    if case_col in cols:
        return df[case_col].astype(str).map(lambda s: s.split("_")[0]).to_numpy()

    for c in ("patient", "pid", "case", "subject_id", "subject"):
        if c in cols:
            return df[c].astype(str).to_numpy()

    raise ValueError(
        f"Cannot infer patient IDs: none of [{patient_col}, {case_col}, patient, pid, case, subject_id, subject] "
        f"found in columns: {sorted(list(cols))}"
    )


def assert_no_patient_overlap(train_df, val_df, test_df=None, *, patient_col="patient_id", case_col="case_id"):
    """
    Enforces that NO patient ID appears across splits.
    Raises ValueError if overlaps are found.
    """
    tr_p = set(_infer_patient_ids_from_df(train_df, patient_col=patient_col, case_col=case_col).tolist())
    va_p = set(_infer_patient_ids_from_df(val_df,   patient_col=patient_col, case_col=case_col).tolist())
    te_p = set()
    if test_df is not None and len(test_df) > 0:
        te_p = set(_infer_patient_ids_from_df(test_df, patient_col=patient_col, case_col=case_col).tolist())

    tr_va = tr_p & va_p
    tr_te = tr_p & te_p
    va_te = va_p & te_p

    print("[split-check] #patients: train={}, val={}, test={}".format(len(tr_p), len(va_p), len(te_p)))
    print("[split-check] patient overlaps: train∩val={}, train∩test={}, val∩test={}".format(
        len(tr_va), len(tr_te), len(va_te)
    ))

    if len(tr_va) or len(tr_te) or len(va_te):
        def sample(s, k=10):
            s = sorted(list(s))
            return s[: min(k, len(s))]

        msg = (
            "DATA LEAK DETECTED: patient IDs overlap across splits.\n"
            f"  train∩val: {len(tr_va)} (sample: {sample(tr_va)})\n"
            f"  train∩test: {len(tr_te)} (sample: {sample(tr_te)})\n"
            f"  val∩test: {len(va_te)} (sample: {sample(va_te)})\n"
            "Fix your fold assignment at the patient level (not slice/case level) before trusting results."
        )
        raise ValueError(msg)


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()

    # Mode control
    p.add_argument(
        "--train_mode",
        choices=["baseline", "triplet"],
        default="baseline",
        help="baseline: end-to-end CE. triplet: end-to-end (weighted CE + weighted triplet).",
    )

    p.add_argument("--seed", type=int, default=42)

    # Data / splits
    p.add_argument("--manifest", required=True)
    p.add_argument("--target", choices=["isup3", "isup6", "binary_low_high", "binary_all"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val", default="0")
    p.add_argument("--folds_test", default="4")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label6_column", default="label6")

    # Model / base checkpoint
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--proj_dim", type=int, required=True)

    # Triplet resources (required only when train_mode=triplet)
    p.add_argument("--histo_dir", default=None)
    p.add_argument("--histo_marksheet_dir", default=None)
    p.add_argument("--provider", default="all")
    p.add_argument("--triplet_margin", type=float, default=0.2)

    # Epochs / patience (single-stage training)
    p.add_argument("--triplet_epochs", type=int, default=15, help="Kept for backward-compat; used in total_epochs calc.")
    p.add_argument("--head_epochs", type=int, default=15, help="Kept for backward-compat; used in total_epochs calc.")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience on val balanced accuracy.")

    # Loss weights for combined mode
    p.add_argument("--ce_weight", type=float, default=1.0, help="Weight for CE term in total loss.")
    p.add_argument("--triplet_weight", type=float, default=1.0, help="Weight for triplet term in total loss (triplet mode).")

    # Unified optimization hyperparams
    p.add_argument("--lr", type=float, default=3e-4, help="Unified learning rate.")
    p.add_argument("--wd", type=float, default=1e-4, help="Unified weight decay.")
    p.add_argument("--enc_lr_mult", type=float, default=0.1, help="Encoder LR multiplier relative to --lr.")

    # Leakage/split checks
    p.add_argument("--patient_col", default="patient_id", help="Column name for patient id (if present).")
    p.add_argument("--case_col", default="case_id", help="Column name for case id (used to infer patient id if needed).")

    # Misc
    p.add_argument("--outdir", default="./runs/combined_triplet_ce")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", default="MID_DEC_KILLARNEY_NEW_SPIE")
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    print("SCRIPT: train_triplet_full.py")
    print("ARGS:", args)

    # Enforce histo args only when needed
    if args.train_mode == "triplet":
        if args.histo_dir is None or args.histo_marksheet_dir is None:
            raise ValueError("--histo_dir and --histo_marksheet_dir are required when --train_mode=triplet")

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------- Datasets / loaders (with TEST) --------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    folds_test = [s.strip() for s in args.folds_test.split(",") if s.strip()]

    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader,
     w_ce, classes_present, n_classes) = build_datasets_and_loaders(
        manifest=args.manifest,
        folds_train=folds_train,
        folds_val=folds_val,
        folds_test=folds_test,
        target=args.target,
        use_skip=args.use_skip,
        label6_column=args.label6_column,
        batch_size=args.batch_size,
        pos_ratio=args.pos_ratio,
    )
    w_ce = w_ce.to(device)

    # -------- Leak check: no patient overlap across splits --------
    assert_no_patient_overlap(
        train_df=train_ds.df,
        val_df=val_ds.df,
        test_df=(test_ds.df if test_ds is not None else None),
        patient_col=args.patient_col,
        case_col=args.case_col,
    )

    # -------- Build model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam, num_classes=n_classes,
        proj_dim=args.proj_dim, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True, pixel_mean_std=None,
    ).to(device)

    # -------- Optional: triplet buckets + triplet_fn --------
    triplet_train_fn = None
    if args.train_mode == "triplet":
        train_histo_buckets = get_histo_by_isup(
            encodings_dir=str(Path(args.histo_dir) / "train"),
            marksheet_csv=str(Path(args.histo_marksheet_dir) / "train.csv"),
            num_classes=n_classes, provider=args.provider
        )

        def triplet_train_fn(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return triplet_loss_batch(
                embeddings, labels, train_histo_buckets, num_classes=n_classes, margin=args.triplet_margin
            )

    # -------- Optimizer (same structure as baseline) --------
    for p_ in model.parameters():
        p_.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": args.lr, "weight_decay": args.wd},
            {"params": model.proj.parameters(), "lr": args.lr, "weight_decay": args.wd},
            {"params": model.encoder.parameters(), "lr": args.lr * args.enc_lr_mult, "weight_decay": args.wd},
        ]
    )

    # W&B
    wb = wandb_init(bool(args.wandb), args.wandb_project, args.wandb_run_name, config=vars(args))
    if wb is not None:
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("aux/*", step_metric="epoch")

    # -------- Single-stage training loop --------
    total_epochs = int(args.triplet_epochs) + int(args.head_epochs)
    mode_str = "CE" if args.train_mode == "baseline" else f"{args.ce_weight:g}*CE + {args.triplet_weight:g}*Triplet"
    print(f"[train] mode={args.train_mode} ({mode_str}) total_epochs={total_epochs}")
    print(f"[train] lr={args.lr:g} wd={args.wd:g} enc_lr_mult={args.enc_lr_mult:g}")

    early = EarlyStopper(patience=args.patience)

    for epoch in range(1, total_epochs + 1):
        if args.train_mode == "baseline":
            tr_total, tr_ce, tr_tri, tr_acc, tr_f1, tr_bacc = run_epoch_combined(
                train_loader,
                model,
                w_ce,
                ce_weight=1.0,
                triplet_weight=0.0,
                triplet_fn=None,
                optimizer=optimizer,
                device=device,
            )
        else:
            tr_total, tr_ce, tr_tri, tr_acc, tr_f1, tr_bacc = run_epoch_combined(
                train_loader,
                model,
                w_ce,
                ce_weight=args.ce_weight,
                triplet_weight=args.triplet_weight,
                triplet_fn=triplet_train_fn,
                optimizer=optimizer,
                device=device,
            )

        val, pcs, auc_part, extra2 = run_eval_print(val_loader, model, w_ce, device, n_classes)

        tag = "BASELINE" if args.train_mode == "baseline" else "TRIPLET"
        print(f"[{tag} {epoch:03d}] "
              f"train: total {tr_total:.4f} (ce {tr_ce:.4f}, tri {tr_tri:.4f}) "
              f"bacc {tr_bacc:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} || "
              f"val: loss {val['loss']:.4f} acc {val['acc']:.4f} BAL-acc {val['bacc']:.4f} "
              f"f1 {val['f1_macro']:.4f} | {pcs}{auc_part}{extra2}")
        print(train_utils.format_confusion_matrix(val["cm"], n_classes=n_classes))

        if wb is not None:
            lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
            payload = {
                "epoch": epoch,
                "aux/lr_head": lrs[0] if len(lrs) > 0 else None,
                "aux/lr_proj": lrs[1] if len(lrs) > 1 else None,
                "aux/lr_enc": lrs[2] if len(lrs) > 2 else None,
                "train/total_loss": tr_total,
                "aux/train/ce_loss": tr_ce,
                "aux/train/triplet_loss": tr_tri,
                "train/bacc": tr_bacc,
                "aux/train/acc": tr_acc,
                "aux/train/f1_macro": tr_f1,
                "val/loss": val["loss"],
                "val/bacc": val["bacc"],
                "val/macro_auc": val["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"val/acc_c{c}"] = val["per_acc"][c]
                payload[f"val/auc_c{c}"] = val["per_auc"][c]
            payload["aux/val/macro_tpr"] = val["macro_tpr"]
            payload["aux/val/macro_tnr"] = val["macro_tnr"]
            wandb_log(wb, payload)

        ckpt_name = "ckpt_best.pt" if args.train_mode == "baseline" else "ckpt_triplet_best.pt"
        if early.update(val["bacc"], model, save_path=outdir / ckpt_name):
            print(f"  ↳ [train] new best (val BAL-acc={val['bacc']:.4f}) snapshot stored in memory")
        else:
            print(f"  ↳ [train] no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= early.patience:
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    if not early.load_best_into(model, strict=False):
        print("[train][warn] No improvement recorded; using last weights.")
    model.to(device).eval()

    # -------- Final VAL/TEST (best checkpoint; collect outputs for OP table) --------
    spec_targets = (0.4, 0.6, 0.8, 0.9, 0.95, 0.99)

    val_final = evaluate_loader(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=True)
    pcs_v, auc_v = format_perclass_acc_auc(val_final["per_acc"], val_final["per_auc"], val_final["macro_auc"], n_classes)
    extra_v = format_sens_spec(val_final["per_tpr"], val_final["per_tnr"], val_final["macro_tpr"], val_final["macro_tnr"], n_classes)
    print(f"[FINAL VAL] loss {val_final['loss']:.4f} acc {val_final['acc']:.4f} f1 {val_final['f1_macro']:.4f} | {pcs_v}{auc_v}{extra_v}")
    print(train_utils.format_confusion_matrix(val_final["cm"], n_classes=n_classes))
    if val_final["logits"].numel():
        probs_val = torch.softmax(val_final["logits"], dim=1).numpy()
        y_val = val_final["labels"].numpy()
        per_cls_val, macro_val = train_utils.per_class_operating_points(y_val, probs_val, spec_targets)
        print_operating_points_table(per_cls_val, macro_val, spec_targets)

    save_embeddings(outdir / "val_embeddings", "val.pt", val_final["embeddings"], val_final["labels"])

    if test_loader is not None:
        test_final = evaluate_loader(test_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=True)
        pcs_t, auc_t = format_perclass_acc_auc(test_final["per_acc"], test_final["per_auc"], test_final["macro_auc"], n_classes)
        extra_t = format_sens_spec(test_final["per_tpr"], test_final["per_tnr"], test_final["macro_tpr"], test_final["macro_tnr"], n_classes)
        print(f"[FINAL TEST] loss {test_final['loss']:.4f} acc {test_final['acc']:.4f} f1 {test_final['f1_macro']:.4f} | {pcs_t}{auc_t}{extra_t}")
        print(train_utils.format_confusion_matrix(test_final["cm"], n_classes=n_classes))
        if test_final["logits"].numel():
            probs_test = torch.softmax(test_final["logits"], dim=1).numpy()
            y_test = test_final["labels"].numpy()
            per_cls_test, macro_test = train_utils.per_class_operating_points(y_test, probs_test, spec_targets)
            print_operating_points_table(per_cls_test, macro_test, spec_targets)

        save_embeddings(outdir / "test_embeddings", "test.pt", test_final["embeddings"], test_final["labels"])

        if wb is not None:
            payload = {
                "epoch": total_epochs,
                "test/loss": test_final["loss"],
                "test/bacc": test_final["bacc"],
                "test/macro_auc": test_final["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"test/acc_c{c}"] = test_final["per_acc"][c]
                payload[f"test/auc_c{c}"] = test_final["per_auc"][c]
            payload["aux/test/macro_tpr"] = test_final["macro_tpr"]
            payload["aux/test/macro_tnr"] = test_final["macro_tnr"]
            wandb_log(wb, payload)
    else:
        print("[TEST] No test folds provided; skipping final test evaluation.")

    wandb_finish(wb)


if __name__ == "__main__":
    main()
