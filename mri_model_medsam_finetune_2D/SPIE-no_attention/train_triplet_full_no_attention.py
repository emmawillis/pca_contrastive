#!/usr/bin/env python3
# train_triplet_full_no_attention_frozen_encoder.py
#
# Same as train_triplet_full.py, except:
#  1) Uses ISUPMedSAM_no_attention.py model (MedSAMSliceNoAttn)
#  2) MedSAM encoder is NEVER trained (frozen in ALL phases)
#     - Triplet phase trains: proj (+ pool p if GeM)
#     - Head phase trains: head (+ optionally proj)

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
)

# W&B step control (we still use train_utils wrappers for init/log)
import wandb  # only for define_metric; safe when wandb disabled

from ISUPMedSAM_no_attention import IMG_SIZE, MedSAMSliceNoAttn
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
from sklearn.linear_model import LogisticRegression


# ---------------- Triplet phase helpers ----------------
def run_epoch_triplet(loader, model, triplet_fn, optimizer=None, device="cuda"):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_n = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            _, emb = model(x)  # logits unused; emb: [B,D]
            loss = triplet_fn(emb, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs
    return total_loss / max(1, total_n)


@torch.no_grad()
def extract_embeddings(loader, model, device="cuda"):
    model.eval()
    embs, ys = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        _, emb = model(x)
        embs.append(emb.cpu())
        ys.append(y.cpu())
    X = torch.cat(embs, 0).numpy() if embs else np.empty((0, 0), dtype=np.float32)
    y = torch.cat(ys, 0).numpy() if ys else np.empty((0,), dtype=np.int64)
    return X, y


def eval_with_existing_clf(X, y, n_classes, clf, return_clf=False):
    y_pred = clf.predict(X)
    acc = float(accuracy_score(y, y_pred))
    bacc = float(balanced_accuracy_score(y, y_pred))
    f1_macro = float(f1_score(y, y_pred, average="macro"))

    per_acc = {}
    for c in range(n_classes):
        mask = (y == c)
        per_acc[c] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")

    per_auc = {c: float("nan") for c in range(n_classes)}
    macro_auc = float("nan")
    try:
        probs = clf.predict_proba(X)  # [N, K]
        auc_vals = []
        for c in range(n_classes):
            y_bin = (y == c).astype(np.int32)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                auc = roc_auc_score(y_bin, probs[:, c])
                per_auc[c] = float(auc)
                auc_vals.append(auc)
        if len(auc_vals) > 0:
            macro_auc = float(np.nanmean(auc_vals))
    except Exception:
        pass

    cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))
    out = {"acc": acc, "bacc": bacc, "f1_macro": f1_macro,
           "per_acc": per_acc, "per_auc": per_auc, "macro_auc": macro_auc, "cm": cm}
    if return_clf:
        out["clf"] = clf
    return out


def eval_with_logreg(X_train, y_train, X_val, y_val, n_classes, max_iter=5):
    clf = LogisticRegression(max_iter=max_iter, multi_class="auto", solver="lbfgs",
                             n_jobs=None, class_weight=None)
    clf.fit(X_train, y_train)
    return eval_with_existing_clf(X_val, y_val, n_classes, clf, return_clf=True)


# ---------------- Head phase helpers ----------------
def run_epoch_ce(loader, model, w_ce, optimizer=None, device="cuda"):
    """Train/eval one epoch with CE; returns (loss, acc, f1_macro, bacc)."""
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
        bacc = float(balanced_accuracy_score(y_true_np, y_pred_np))
    else:
        f1m, bacc = 0.0, 0.0
    return avg_loss, acc, f1m, bacc


def run_eval_print(val_loader, model, w_ce, device, n_classes):
    val = evaluate_loader(val_loader, model, w_ce=w_ce, device=device, n_classes=n_classes, collect_outputs=False)
    pcs, auc_part = format_perclass_acc_auc(val["per_acc"], val["per_auc"], val["macro_auc"], n_classes)
    extra2 = format_sens_spec(val["per_tpr"], val["per_tnr"], val["macro_tpr"], val["macro_tnr"], n_classes)
    return val, pcs, auc_part, extra2


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    # Data / splits
    p.add_argument("--manifest", required=True)
    p.add_argument("--target", choices=["isup3","isup6","binary_low_high","binary_all"], default="isup3")
    p.add_argument("--folds_train", default="1,2,3")
    p.add_argument("--folds_val",   default="0")
    p.add_argument("--folds_test",  default="4")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pos_ratio", type=float, default=0.33)
    p.add_argument("--use-skip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label6_column", default="label6")

    # Model / base checkpoint
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--proj_dim", type=int, required=True)

    # New model knobs (pooling)
    p.add_argument("--pool_mode", choices=["avg", "max", "avgmax", "gem"], default="avgmax")
    p.add_argument("--gem_p_init", type=float, default=3.0)

    # Triplet phase
    p.add_argument("--histo_dir", required=True)
    p.add_argument("--histo_marksheet_dir", required=True)
    p.add_argument("--provider", default="all")
    p.add_argument("--triplet_epochs", type=int, default=15)
    p.add_argument("--triplet_patience", type=int, default=10)
    p.add_argument("--triplet_lr", type=float, default=3e-4)
    p.add_argument("--triplet_wd", type=float, default=1e-4)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument("--lr_max_iter", type=int, default=5, help="LogReg max_iter during triplet selection")

    # Head phase
    p.add_argument("--head_epochs", type=int, default=15)
    p.add_argument("--head_patience", type=int, default=10)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--head_wd", type=float, default=1e-4)
    p.add_argument("--train_proj", action="store_true",
                   help="Also train projection MLP during head phase (default: only head).")

    # Misc
    p.add_argument("--outdir", default="./runs/triplet_then_head_noattn_frozenenc")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", default="mri-training")
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    print("SCRIPT: train_triplet_full_no_attention_frozen_encoder.py")
    print("ARGS:", args)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- Datasets / loaders (with TEST) --------
    folds_train = [s.strip() for s in args.folds_train.split(",") if s.strip()]
    folds_val   = [s.strip() for s in args.folds_val.split(",") if s.strip()]
    folds_test  = [s.strip() for s in args.folds_test.split(",") if s.strip()]
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

    # -------- Histo buckets for triplet --------
    train_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "train"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "train.csv"),
        num_classes=n_classes, provider=args.provider
    )
    val_histo_buckets = get_histo_by_isup(
        encodings_dir=str(Path(args.histo_dir) / "val"),
        marksheet_csv=str(Path(args.histo_marksheet_dir) / "val.csv"),
        num_classes=n_classes, provider=args.provider
    )

    # -------- Build model --------
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
    model = MedSAMSliceNoAttn(
        sam_model=sam, num_classes=n_classes,
        proj_dim=args.proj_dim,
        head_hidden=256, head_dropout=0.1,
        use_pre_neck=True,
        pool_mode=args.pool_mode,
        gem_p_init=args.gem_p_init,
        pixel_mean_std=None,
    ).to(device)

    # --- ALWAYS freeze encoder (never train MedSAM) ---
    for p_ in model.encoder.parameters():
        p_.requires_grad = False

    # W&B
    wb = wandb_init(bool(args.wandb), args.wandb_project, args.wandb_run_name, config=vars(args))
    if wb is not None:
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        wandb.define_metric("test/*",  step_metric="epoch")
        wandb.define_metric("aux/*",   step_metric="epoch")

    # =========================
    # Phase 1: Triplet (proj only; encoder frozen)
    # =========================
    # Freeze all; unfreeze proj (+ GeM p if present)
    for p_ in model.parameters():
        p_.requires_grad = False
    for p_ in model.encoder.parameters():  # keep frozen explicitly
        p_.requires_grad = False
    for p_ in model.proj.parameters():
        p_.requires_grad = True
    # If using GeM pooling, its exponent p is a Parameter
    if hasattr(model, "pool") and getattr(model.pool, "mode", "").lower() == "gem":
        if getattr(model.pool, "p", None) is not None:
            model.pool.p.requires_grad = True

    triplet_trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer_triplet = torch.optim.AdamW(triplet_trainable, lr=args.triplet_lr, weight_decay=args.triplet_wd)
    print(f"[triplet] lr={args.triplet_lr:g} | wd={args.triplet_wd:g} | margin={args.triplet_margin:g} | encoder_frozen=YES")

    def train_triplet_fn(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return triplet_loss_batch(embeddings, labels, train_histo_buckets, num_classes=n_classes, margin=args.triplet_margin)
    def val_triplet_fn(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return triplet_loss_batch(embeddings, labels, val_histo_buckets, num_classes=n_classes, margin=args.triplet_margin)

    early_triplet = EarlyStopper(patience=args.triplet_patience)
    best_lr_clf = None
    last_lr_clf = None

    for epoch in range(1, args.triplet_epochs + 1):
        tr_loss = run_epoch_triplet(train_loader, model, train_triplet_fn, optimizer=optimizer_triplet, device=device)
        va_loss = run_epoch_triplet(val_loader,   model, val_triplet_fn,   optimizer=None,            device=device)

        # Embed + LR proxy on VAL
        X_tr, y_tr = extract_embeddings(train_loader, model, device=device)
        X_va, y_va = extract_embeddings(val_loader,   model, device=device)
        lr_metrics = eval_with_logreg(X_tr, y_tr, X_va, y_va, n_classes=n_classes, max_iter=args.lr_max_iter)
        last_lr_clf = lr_metrics["clf"]

        if wb is not None:
            lrs = [pg.get("lr", None) for pg in optimizer_triplet.param_groups]
            payload = {
                "epoch": epoch,
                "aux/triplet/lr": lrs[0] if lrs else None,
                "aux/triplet/train_loss": tr_loss,
                "aux/triplet/val_loss": va_loss,
                "aux/lr_val/acc": lr_metrics["acc"],
                "aux/lr_val/bacc": lr_metrics["bacc"],
                "aux/lr_val/f1_macro": lr_metrics["f1_macro"],
                "aux/lr_val/macro_auc": lr_metrics["macro_auc"],
            }
            for c in range(n_classes):
                payload[f"aux/lr_val/acc_c{c}"] = lr_metrics["per_acc"][c]
                payload[f"aux/lr_val/auc_c{c}"] = lr_metrics["per_auc"][c]
            wandb_log(wb, payload)

        if early_triplet.update(lr_metrics["bacc"], model, save_path=outdir / "ckpt_triplet_best.pt"):
            best_lr_clf = lr_metrics["clf"]
            print(f"  ↳ [triplet] new best (val BAL-acc={early_triplet.best:.4f}) snapshot stored in memory")
        else:
            print(f"  ↳ [triplet] no improvement ({early_triplet.num_bad}/{early_triplet.patience})")
            if early_triplet.num_bad >= early_triplet.patience:
                print(f"[triplet] Early stopping at epoch {epoch}.")
                break

    if not early_triplet.load_best_into(model, strict=False):
        print("[triplet][warn] No improvement recorded; using last proj.")
        if best_lr_clf is None:
            best_lr_clf = last_lr_clf

    model.to(device).eval()

    # =========================
    # Phase 2: Head fine-tuning (head, optionally proj; encoder frozen)
    # =========================
    for p_ in model.parameters():
        p_.requires_grad = False
    for p_ in model.encoder.parameters():
        p_.requires_grad = False

    head_params = list(model.head.parameters())
    trainable = head_params
    if args.train_proj:
        trainable += list(model.proj.parameters())
        # If GeM, optionally keep p trainable during head phase too
        if hasattr(model, "pool") and getattr(model.pool, "mode", "").lower() == "gem":
            if getattr(model.pool, "p", None) is not None:
                model.pool.p.requires_grad = True
                trainable += [model.pool.p]

    for p_ in trainable:
        p_.requires_grad = True

    optimizer_head = torch.optim.AdamW(trainable, lr=args.head_lr, weight_decay=args.head_wd)
    early_head = EarlyStopper(patience=args.head_patience)

    for epoch in range(1, args.head_epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_bacc = run_epoch_ce(
            train_loader, model, w_ce=w_ce, optimizer=optimizer_head, device=device
        )
        val, pcs, auc_part, extra2 = run_eval_print(val_loader, model, w_ce, device, n_classes)

        print(f"[HEAD {epoch:03d}] "
              f"train: loss {tr_loss:.4f} bacc {tr_bacc:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} || "
              f"val: loss {val['loss']:.4f} acc {val['acc']:.4f} BAL-acc {val['bacc']:.4f} "
              f"f1 {val['f1_macro']:.4f} | {pcs}{auc_part}{extra2}")
        print(train_utils.format_confusion_matrix(val["cm"], n_classes=n_classes))

        if wb is not None:
            lrs = [pg.get("lr", None) for pg in optimizer_head.param_groups]
            payload = {
                "epoch": epoch,
                "aux/head/lr": lrs[0] if lrs else None,
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

        if early_head.update(val["bacc"], model, save_path=outdir / "ckpt_head_best.pt"):
            print(f"  ↳ [head] new best (val BAL-acc={val['bacc']:.4f}) snapshot stored in memory")
        else:
            print(f"  ↳ [head] no improvement ({early_head.num_bad}/{early_head.patience})")
            if early_head.num_bad >= early_head.patience:
                print(f"[head] Early stopping at epoch {epoch}.")
                break

    if not early_head.load_best_into(model, strict=False):
        print("[head][warn] No improvement recorded; using last head.")

    model.to(device).eval()

    # -------- Final VAL/TEST (using best head; collect outputs for OP table) --------
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
                "epoch": args.head_epochs,
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
