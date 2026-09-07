#!/usr/bin/env python3
# train_histo_head.py
#
# Modes:
#   --train_mode baseline   : end-to-end weighted CE training
#   --train_mode histo_head : replace MRI head with frozen histology head (Linear(512 -> 6))
#                             and train the rest of the MRI model with weighted CE
#
# Notes:
#   - This script is intended for target=isup6 only.
#   - The histology checkpoint is expected to contain:
#         model.classifier.weight
#         model.classifier.bias
#     from the ABMIL 512-dim run.

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import wandb

from ISUPMedSAM import MedSAMSliceSpatialAttn
from segment_anything import sam_model_registry

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


def run_epoch_ce(loader, model, w_ce, optimizer=None, device="cuda"):
    """Train/eval one epoch with weighted CE; returns (loss, acc, f1_macro, bacc)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    ce = nn.CrossEntropyLoss(weight=w_ce)
    total_loss, total_n, total_correct = 0.0, 0, 0
    all_pred, all_true = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits, _ = model(x)
        loss = ce(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        pred = logits.argmax(1)
        total_correct += (pred == y).sum().item()
        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)

    if all_pred:
        y_pred_np = torch.cat(all_pred).numpy()
        y_true_np = torch.cat(all_true).numpy()
        f1m = float(f1_score(y_true_np, y_pred_np, average="macro"))
        from sklearn.metrics import balanced_accuracy_score
        bacc = float(balanced_accuracy_score(y_true_np, y_pred_np))
    else:
        f1m, bacc = 0.0, 0.0

    return avg_loss, acc, f1m, bacc


def run_eval_print(val_loader, model, w_ce, device, n_classes):
    val = evaluate_loader(
        val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=False
    )
    pcs, auc_part = format_perclass_acc_auc(val["per_acc"], val["per_auc"], val["macro_auc"], n_classes)
    extra2 = format_sens_spec(val["per_tpr"], val["per_tnr"], val["macro_tpr"], val["macro_tnr"], n_classes)
    return val, pcs, auc_part, extra2


def _infer_patient_ids_from_df(df, *, patient_col="patient_id", case_col="case_id"):
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
    tr_p = set(_infer_patient_ids_from_df(train_df, patient_col=patient_col, case_col=case_col).tolist())
    va_p = set(_infer_patient_ids_from_df(val_df, patient_col=patient_col, case_col=case_col).tolist())
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
            "Fix your fold assignment at the patient level before trusting results."
        )
        raise ValueError(msg)


def replace_with_histo_head(model: nn.Module, histo_ckpt_path: str, device: str):
    """
    Replaces model.head with nn.Linear(512, 6), loads histology classifier weights,
    and freezes the head.
    """
    ckpt = torch.load(histo_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("model", ckpt))

    weight_key = "model.classifier.weight"
    bias_key = "model.classifier.bias"
    if weight_key not in sd or bias_key not in sd:
        raise KeyError(
            f"Could not find histology classifier keys '{weight_key}' and '{bias_key}' in checkpoint: {histo_ckpt_path}"
        )

    w = sd[weight_key]
    b = sd[bias_key]

    if tuple(w.shape) != (6, 512):
        raise ValueError(f"Expected histology classifier weight shape (6, 512), got {tuple(w.shape)}")
    if tuple(b.shape) != (6,):
        raise ValueError(f"Expected histology classifier bias shape (6,), got {tuple(b.shape)}")

    model.head = nn.Linear(512, 6)
    with torch.no_grad():
        model.head.weight.copy_(w)
        model.head.bias.copy_(b)
    model.head.to(device)

    for p in model.head.parameters():
        p.requires_grad = False

    print(f"[histo_head] Loaded and froze histology classifier from {histo_ckpt_path}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--train_mode", choices=["baseline", "histo_head", "random_head"], default="baseline")
    p.add_argument("--seed", type=int, default=42)

    # Data / splits
    p.add_argument("--manifest", required=True)
    p.add_argument("--target", choices=["isup6"], default="isup6")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val", default="0")
    p.add_argument("--folds_test", default="4")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label6_column", default="label6")

    # Model / checkpoints
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--histo_head_ckpt", type=str, default=None,
                   help="Required for --train_mode histo_head. Path to 512-dim histology checkpoint.")

    # Optimization
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--enc_lr_mult", type=float, default=0.1)

    # Leakage/split checks
    p.add_argument("--patient_col", default="patient_id")
    p.add_argument("--case_col", default="case_id")

    # Misc
    p.add_argument("--outdir", default="./runs/histo_head_mri")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", default="MID_DEC_KILLARNEY_NEW_SPIE")
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    print("SCRIPT: train_histo_head_mri.py")
    print("ARGS:", args)

    if args.target != "isup6":
        raise ValueError("This script is intended for target=isup6 only.")
    if args.proj_dim != 512:
        raise ValueError("This script expects --proj_dim 512 so the MRI embedding matches the histology head.")
    if args.train_mode == "histo_head" and not args.histo_head_ckpt:
        raise ValueError("--histo_head_ckpt is required when --train_mode histo_head")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    if n_classes != 6:
        raise ValueError(f"Expected n_classes=6 for isup6, got {n_classes}")

    assert_no_patient_overlap(
        train_df=train_ds.df,
        val_df=val_ds.df,
        test_df=(test_ds.df if test_ds is not None else None),
        patient_col=args.patient_col,
        case_col=args.case_col,
    )

    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)

    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=6,
        proj_dim=512,
        attn_dim=256,
        head_hidden=0,   # baseline head is Linear(512, 6)
        head_dropout=0.1,
        use_pre_neck=True,
        pixel_mean_std=None,
    ).to(device)

    if args.train_mode == "random_head":
        model.head = nn.Linear(512, 6)
        model.head.to(device)
        for p_ in model.parameters():
                p_.requires_grad = True
        for p_ in model.head.parameters():  #freeze the head with random weights
            p_.requires_grad = False
    elif args.train_mode == "histo_head":
        replace_with_histo_head(model, args.histo_head_ckpt, device=device)
        # Keep everything except the frozen head trainable
        for p_ in model.encoder.parameters():
            p_.requires_grad = True
        for p_ in model.proj.parameters():
            p_.requires_grad = True
        for p_ in model.pool.parameters():
            p_.requires_grad = True
    else:
        for p_ in model.parameters():
            p_.requires_grad = True

    # Optimizer excludes frozen params automatically
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd},
            {"params": [p for p in model.proj.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd},
            {"params": [p for p in model.pool.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.lr * args.enc_lr_mult, "weight_decay": args.wd},
        ]
    )

    wb = wandb_init(bool(args.wandb), args.wandb_project, args.wandb_run_name, config=vars(args))
    if wb is not None:
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("aux/*", step_metric="epoch")

    early = EarlyStopper(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_bacc = run_epoch_ce(
            train_loader, model, w_ce=w_ce, optimizer=optimizer, device=device
        )
        val, pcs, auc_part, extra2 = run_eval_print(val_loader, model, w_ce, device, n_classes)

        print(f"[{args.train_mode.upper()} {epoch:03d}] "
              f"train: loss {tr_loss:.4f} bacc {tr_bacc:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} || "
              f"val: loss {val['loss']:.4f} acc {val['acc']:.4f} BAL-acc {val['bacc']:.4f} "
              f"f1 {val['f1_macro']:.4f} | {pcs}{auc_part}{extra2}")
        print(train_utils.format_confusion_matrix(val["cm"], n_classes=n_classes))

        if wb is not None:
            lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
            payload = {
                "epoch": epoch,
                "aux/lr_head": lrs[0] if len(lrs) > 0 else None,
                "aux/lr_proj": lrs[1] if len(lrs) > 1 else None,
                "aux/lr_pool": lrs[2] if len(lrs) > 2 else None,
                "aux/lr_enc": lrs[3] if len(lrs) > 3 else None,
                "train/loss": tr_loss,
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

        if early.update(val["bacc"], model, save_path=outdir / f"ckpt_{args.train_mode}_best.pt"):
            print(f"  ↳ [{args.train_mode}] new best (val BAL-acc={val['bacc']:.4f}) snapshot stored in memory")
        else:
            print(f"  ↳ [{args.train_mode}] no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= early.patience:
                print(f"[{args.train_mode}] Early stopping at epoch {epoch}.")
                break

    if not early.load_best_into(model, strict=False):
        print(f"[{args.train_mode}][warn] No improvement recorded; using last weights.")
    model.to(device).eval()

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
                "epoch": args.epochs,
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
