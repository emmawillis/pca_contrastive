#!/usr/bin/env python3
# train.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score

import wandb  # only for define_metric; safe when wandb disabled

from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn
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
    unfreeze_and_add_param_group,
    save_embeddings,
    wandb_init, wandb_log, wandb_finish,
)

# ----------------- one epoch -----------------
def run_epoch(loader, model, loss_fn, optimizer=None, device="cuda",
              return_outputs=False, epoch_idx=None, probe_first_k=0, label_names=None):
    """Returns (loss, acc, f1_macro, bacc[, logits, y])."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_logits, all_y = [], []
    for bi, batch in enumerate(loader):
        if train_mode and probe_first_k and (epoch_idx == 1) and (bi < probe_first_k):
            y_np = batch["label"].cpu().numpy()
            hl = np.asarray(batch.get("has_lesion", []))
            if hl.size == 0:
                hl = np.zeros_like(y_np, dtype=np.int64)
            pos = int(hl.sum()); tot = int(hl.shape[0])
            uniq, cnts = np.unique(y_np, return_counts=True)
            if label_names is None:
                label_names = [f"c{i}" for i in range(int(y_np.max())+1)]
            lab_str = " | ".join(
                f"{(label_names[u] if 0 <= u < len(label_names) else f'c{u}')}={c} ({c/tot:.0%})"
                for u, c in zip(uniq, cnts)
            )
            print(f"[probe][epoch1 batch {bi:02d}] has_lesion: {pos}/{tot} ({pos/tot:.0%}) || labels: {lab_str}")

        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits, _ = model(x)
        loss = loss_fn(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
        all_logits.append(logits.detach())
        all_y.append(y.detach())

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)

    logits_cat = torch.cat(all_logits) if all_logits else torch.empty(0)
    y_cat = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)

    if logits_cat.ndim == 2 and logits_cat.numel() > 0:
        K = logits_cat.shape[1]
        y_pred = logits_cat.argmax(dim=1).cpu().numpy()
        y_true = y_cat.cpu().numpy()
        f1 = float(f1_score(
            y_true, y_pred,
            average="macro",
            labels=list(range(K)),
            zero_division=0
        ))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
    else:
        f1, bacc = 0.0, 0.0

    if return_outputs:
        return avg_loss, acc, f1, bacc, logits_cat, y_cat
    return avg_loss, acc, f1, bacc


# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--manifest", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--outdir", default="./runs/simple")
    p.add_argument("--target", choices=["isup3","isup6","binary_low_high", "binary_all"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val", default="0")
    p.add_argument("--folds_test", default="4", help="Folds evaluated as held-out test at the end.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, drop rows with skip==1. Use --no-use-skip to include them.")
    p.add_argument("--label6_column", default="label6")
    p.add_argument("--freeze_epochs", type=int, default=2,
                   help="Freeze MedSAM encoder for this many epochs, then unfreeze.")
    # W&B
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="Log epoch metrics to Weights & Biases.")
    p.add_argument("--wandb_project", default="mri-training")
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()

    print("SCRIPT: train.py")
    print("ARGS:", args)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.folds_test.split(",") if s.strip()]

    wb = wandb_init(bool(args.wandb), args.wandb_project, args.wandb_run_name, config=vars(args))
    if wb is not None:
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        wandb.define_metric("test/*",  step_metric="epoch")
        wandb.define_metric("aux/*",   step_metric="epoch")

    # -------- datasets / loaders (includes TEST) --------
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

    # -------- model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=n_classes,
        proj_dim=args.proj_dim, attn_dim=256,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,
        pixel_mean_std=None,
    ).to(device)

    # --- Freeze encoder for warmup ---
    for p_ in model.encoder.parameters():
        p_.requires_grad = False
    encoder_params = list(model.encoder.parameters())

    optimizer = torch.optim.AdamW(
        (p_ for p_ in model.parameters() if p_.requires_grad),
        lr=args.lr, weight_decay=args.wd
    )
    criterion = nn.CrossEntropyLoss(weight=w_ce)

    # ---- Early stopping ----
    early = EarlyStopper(patience=args.patience)
    best_path = outdir / "ckpt_best.pt"

    # -------- loop --------
    for epoch in range(1, args.epochs+1):
        if epoch == args.freeze_epochs + 1:
            unfreeze_and_add_param_group(encoder_params, optimizer, base_lr=args.lr, wd=args.wd, lr_mult=0.1)
            print(f"→ Unfroze encoder at epoch {epoch}; added to optimizer with lr={args.lr*0.1:g}")

        tr_loss, tr_acc, tr_f1, tr_bacc = run_epoch(
            train_loader, model, criterion, optimizer=optimizer, device=device,
            epoch_idx=epoch, probe_first_k=8, label_names=train_utils.get_label_names(args.target)
        )

        # normal val eval
        val = evaluate_loader(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=False)
        pcs, auc_part = format_perclass_acc_auc(val["per_acc"], val["per_auc"], val["macro_auc"], n_classes)
        extra2 = format_sens_spec(val["per_tpr"], val["per_tnr"], val["macro_tpr"], val["macro_tnr"], n_classes)

        print(f"[{epoch:03d}] "
              f"train: loss {tr_loss:.4f} bacc {tr_bacc:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"val: loss {val['loss']:.4f} acc {val['acc']:.4f} BAL-acc {val['bacc']:.4f} "
              f"f1 {val['f1_macro']:.4f} | {pcs}{auc_part}{extra2}")
        print(train_utils.format_confusion_matrix(val["cm"], n_classes=n_classes))

        # ---- wandb logging per epoch (unified names) ----
        if wb is not None:
            lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
            log_payload = {
                "epoch": epoch,
                "aux/lr": lrs[0] if lrs else None,
                "train/loss": tr_loss,
                "train/bacc": tr_bacc,
                # tuck extras away
                "aux/train/acc": tr_acc,
                "aux/train/f1_macro": tr_f1,
                "val/loss": val["loss"],
                "val/bacc": val["bacc"],
                "val/macro_auc": val["macro_auc"],
            }
            for c in range(n_classes):
                log_payload[f"val/acc_c{c}"] = val["per_acc"][c]
                log_payload[f"val/auc_c{c}"] = val["per_auc"][c]
            # extras
            log_payload["aux/val/macro_tpr"] = val["macro_tpr"]
            log_payload["aux/val/macro_tnr"] = val["macro_tnr"]
            wandb_log(wb, log_payload)

        if early.update(val["bacc"], model, best_path):
            print(f"  ↳ saved best to {best_path} (BAL-acc={val['bacc']:.4f})")
        else:
            print(f"  ↳ no improvement ({early.num_bad}/{early.patience})")
            if early.num_bad >= early.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # -------- Use in-memory best (no disk reload) --------
    loaded = early.load_best_into(model, strict=False)
    if not loaded:
        print("[warn] No validation improvement recorded; using last model state for final eval.")
    model.to(device)
    model.eval()

    # -------- Final VAL (collect embeddings) --------
    val_final = evaluate_loader(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=True)
    logits_val = val_final["logits"]; y_val_t = val_final["labels"]; val_embs = val_final["embeddings"]
    y_val = y_val_t.numpy() if y_val_t.numel() else np.empty((0,), dtype=np.int64)
    probs_val = torch.softmax(logits_val, dim=1).numpy() if logits_val.numel() else np.empty((0, n_classes), dtype=np.float32)
    spec_targets = (0.4, 0.6, 0.8, 0.9, 0.95, 0.99)
    per_cls_val, macro_val = train_utils.per_class_operating_points(y_val, probs_val, spec_targets)
    print_operating_points_table(per_cls_val, macro_val, spec_targets, dataset_label="validation")

    # -------- TEST (if provided) --------
    if test_loader is not None:
        test_final = evaluate_loader(test_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=True)
        pcs_t, auc_part_t = format_perclass_acc_auc(test_final["per_acc"], test_final["per_auc"], test_final["macro_auc"], n_classes)
        extra2_t = format_sens_spec(test_final["per_tpr"], test_final["per_tnr"], test_final["macro_tpr"], test_final["macro_tnr"], n_classes)
        print(f"[TEST] "
              f"loss {test_final['loss']:.4f} acc {test_final['acc']:.4f} f1 {test_final['f1_macro']:.4f} | "
              f"{pcs_t}{auc_part_t}{extra2_t}")
        print(train_utils.format_confusion_matrix(test_final["cm"], n_classes=n_classes))

        logits_test = test_final["logits"]; y_test_t = test_final["labels"]; test_embs = test_final["embeddings"]
        y_test = y_test_t.numpy() if y_test_t.numel() else np.empty((0,), dtype=np.int64)
        probs_test = torch.softmax(logits_test, dim=1).numpy() if logits_test.numel() else np.empty((0, n_classes), dtype=np.float32)
        per_cls_test, macro_test = train_utils.per_class_operating_points(y_test, probs_test, spec_targets)
        print_operating_points_table(per_cls_test, macro_test, spec_targets, dataset_label="test")

        # wandb final test summary (unified names)
        if wb is not None:
            payload = {
                "epoch": args.epochs,  # final step marker
                "test/loss": test_final["loss"],
                "test/bacc": test_final["bacc"],
                "test/macro_auc": test_final["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"test/acc_c{c}"] = test_final["per_acc"][c]
                payload[f"test/auc_c{c}"] = test_final["per_auc"][c]
            # extras tucked away
            payload["aux/test/macro_tpr"] = test_final["macro_tpr"]
            payload["aux/test/macro_tnr"] = test_final["macro_tnr"]
            wandb_log(wb, payload)

        # Save TEST embeddings (best ckpt)
        save_embeddings(outdir / "test_embeddings", "test.pt", test_embs, y_test_t)
    else:
        print("[TEST] No test folds provided; skipping test evaluation.")

    # Save VAL embeddings (best ckpt)
    save_embeddings(outdir / "val_embeddings", "val.pt", val_embs, y_val_t)

    wandb_finish(wb)


if __name__ == "__main__":
    main()
