# train_isup_nnunet_v1.py
# Train the nnUNet-v1 encoder-only wrapper for ISUP (labels from marksheet.csv)
# Adds: fixed-size inputs, class-balanced sampler (works with bs=1), logit adjustment (LA),
# neutral head init, precise timing logs, device control, AMP, optional center-crop and depth stride,
# modality selection, and checkpointing by validation BALANCED ACCURACY.

import argparse, json, os, csv, time, contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score

from picai_mri_dataset import PicaiMRIDataset
from nnunet_v1_encoder_isup import NNUNetV1EncoderISUP


# ---------------------------- utilities ----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_split_keys(splits_path: str | Path, fold: int, split: str = "train") -> Set[str]:
    with open(splits_path) as f: S = json.load(f)
    if not (0 <= fold < len(S)):
        raise ValueError(f"Fold {fold} out of range (len={len(S)})")
    return set(S[fold][split])

def load_isup_map_from_marksheet(csv_path: str | Path) -> Dict[str, int]:
    """Read case_ISUP from marksheet.csv → {'pid_study': isup} (ints 0..5)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    mapping: Dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row["patient_id"]).strip()
            sid = str(row["study_id"]).strip()
            key = f"{pid}_{sid}"
            raw = str(row.get("case_ISUP", "")).strip()
            if raw == "" or raw.lower() == "nan":
                continue
            try:
                y = int(float(raw))  # handle "3" or "3.0"
            except Exception:
                continue
            if 0 <= y <= 5:
                mapping[key] = y
    if not mapping:
        raise ValueError(f"No valid case_ISUP labels parsed from {csv_path}")
    return mapping

def softmax_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()

def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: np.ndarray, num_classes: int) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    # per-class accuracy
    per_class_acc = []
    y_true_np = np.array(y_true, dtype=int); y_pred_np = np.array(y_pred, dtype=int)
    for c in range(num_classes):
        m = (y_true_np == c)
        per_class_acc.append(float("nan") if m.sum() == 0 else float((y_pred_np[m] == c).mean()))
    # macro AUROC (OvR)
    auroc = float("nan")
    try:
        y_oh = np.eye(num_classes, dtype=np.int32)[y_true_np]
        auroc = roc_auc_score(y_oh, y_prob, average="macro", multi_class="ovr")
    except Exception:
        pass
    balanced_acc = float(np.nanmean(per_class_acc))  # macro over classes present
    return {"acc": acc, "per_class_acc": per_class_acc, "auroc": auroc, "balanced_acc": balanced_acc}

def mps_sync():
    if torch.backends.mps.is_available():
        try: torch.mps.synchronize()
        except Exception: pass

def dev_autocast(device: str, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    if device == "mps" and torch.backends.mps.is_available():
        return torch.autocast("mps", dtype=torch.float16)
    return contextlib.nullcontext()

def dev_mem():
    try:
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB"
        if torch.backends.mps.is_available():
            get = getattr(torch.mps, "current_allocated_memory", None)
            if get:
                return f"{get() / (1024**2):.0f}MB"
        return "cpu"
    except Exception:
        return "n/a"

def pca_str(a):  # per-class acc pretty-print
    return ", ".join(f"c{c}:{(np.nan if np.isnan(v) else v):.3f}" for c, v in enumerate(a))

def maybe_parse_none(s: str | None):
    if s is None: return None
    if isinstance(s, str) and s.lower() == "none": return None
    return s

def _maybe_resize_batch(x: torch.Tensor, crop_hw: int, d_stride: int) -> torch.Tensor:
    # x: [B,C,D,H,W]
    if d_stride and d_stride > 1:
        x = x[:, :, ::d_stride, :, :]
    if crop_hw and crop_hw > 0:
        B, C, D, H, W = x.shape
        ch = min(crop_hw, H); cw = min(crop_hw, W)
        y0 = (H - ch) // 2; x0 = (W - cw) // 2
        x = x[:, :, :, y0:y0+ch, x0:x0+cw]
    return x

def _fix_size(x: torch.Tensor, Dfix: int, HWfix: int) -> torch.Tensor:
    """
    Center-crop or pad to a fixed (Dfix, HWfix, HWfix).
    F.pad expects (Wl,Wr, Hl,Hr, Dl,Dr).
    """
    if Dfix <= 0 and HWfix <= 0:
        return x
    import torch.nn.functional as F
    B,C,D,H,W = x.shape
    # Depth
    if Dfix > 0:
        if D >= Dfix:
            s = max(0, (D - Dfix)//2); x = x[:, :, s:s+Dfix]
        else:
            pad = Dfix - D; x = F.pad(x, (0,0,0,0, pad//2, pad - pad//2))
    # Height
    _,_,D,H,W = x.shape
    if HWfix > 0:
        if H >= HWfix:
            ys = (H - HWfix)//2; x = x[:, :, :, ys:ys+HWfix, :]
        else:
            pad = HWfix - H; x = F.pad(x, (0,0, pad//2, pad - pad//2, 0,0))
    # Width
    _,_,D,H,W = x.shape
    if HWfix > 0:
        if W >= HWfix:
            xs = (W - HWfix)//2; x = x[:, :, :, :, xs:xs+HWfix]
        else:
            pad = HWfix - W; x = F.pad(x, (pad//2, pad - pad//2, 0,0, 0,0))
    return x


# ---------------------------- train / eval loops with timing ----------------------------
def run_epoch_train(model, loader, criterion, optimizer, device, num_classes,
                    log_every=5, amp=False, crop_hw=0, d_stride=1, la=None,
                    fix_depth=32, fix_hw=192):
    pred_hist = torch.zeros(num_classes, dtype=torch.long)
    label_hist = torch.zeros(num_classes, dtype=torch.long)

    model.train(); model.net.train()
    tot, correct = 0, 0
    per_cls = torch.zeros(num_classes, dtype=torch.long)
    per_cls_correct = torch.zeros(num_classes, dtype=torch.long)
    all_probs, all_targets = [], []
    last_print = time.time()

    for step, batch in enumerate(loader):
        t0 = time.time()
        x = batch["mri"].to(device, non_blocking=False)
        y = batch["isup"].to(device, non_blocking=False)
        x = _maybe_resize_batch(x, crop_hw, d_stride)
        x = _fix_size(x, fix_depth, fix_hw)
        mps_sync()
        t_load = time.time() - t0

        # forward
        t1 = time.time()
        with dev_autocast(str(device), amp):
            z, logits = model(x)
            adj_logits = logits if la is None else (logits + la)
            loss = criterion(adj_logits, y)
        mps_sync()
        t_fwd = time.time() - t1

        # backward + step
        t2 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        mps_sync()
        t_bwd = time.time() - t2

        t3 = time.time()
        optimizer.step()
        mps_sync()
        t_opt = time.time() - t3

        # metrics
        with torch.no_grad():
            pred = (adj_logits if la is not None else logits).argmax(dim=1)
            correct += (pred == y).sum().item()
            tot += y.numel()
            for c in range(num_classes):
                mask = (y == c)
                per_cls[c] += mask.sum().item()
                if mask.any():
                    per_cls_correct[c] += (pred[mask] == c).sum().item()
            probs = torch.softmax(adj_logits if la is not None else logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.detach().cpu())
            for c in range(num_classes):
                pred_hist[c] += (pred == c).sum().item()
                label_hist[c] += (y == c).sum().item()

        if (step + 1) % max(1, log_every) == 0 or (time.time() - last_print) > 30:
            last_print = time.time()
            print(f"[train] {step+1}/{len(loader)} "
                  f"load {t_load:.3f}s fwd {t_fwd:.3f}s bwd {t_bwd:.3f}s opt {t_opt:.3f}s "
                  f"| x {tuple(x.shape)} dev={device} mem={dev_mem()} "
                  f"| loss {loss.item():.4f}")

    # aggregate AUROC
    probs_cat = np.concatenate(all_probs, 0) if all_probs else np.zeros((0, num_classes))
    y_cat = torch.cat(all_targets, 0).numpy() if all_targets else np.zeros((0,), dtype=np.int64)
    try:
        y_oh = np.eye(num_classes, dtype=np.int32)[y_cat]
        tr_auroc = roc_auc_score(y_oh, probs_cat, average="macro", multi_class="ovr") if len(y_cat) > 0 else float("nan")
    except Exception:
        tr_auroc = float("nan")

    per_cls_acc = (per_cls_correct.float() / per_cls.clamp(min=1).float()).tolist()
    tr_acc = correct / max(1, tot)

    print(f"[train] label_hist={label_hist.tolist()} pred_hist={pred_hist.tolist()}")

    # balanced acc
    bal_acc = float(np.nanmean(per_cls_acc))
    return float(loss.item()), {"acc": tr_acc, "auroc": tr_auroc, "per_class_acc": per_cls_acc, "balanced_acc": bal_acc}


@torch.no_grad()
def run_epoch_eval(model, loader, device, num_classes, log_every=999, amp=False,
                   crop_hw=0, d_stride=1, la=None, fix_depth=32, fix_hw=192):
    model.eval(); model.net.eval()
    tot, correct = 0, 0
    per_cls = torch.zeros(num_classes, dtype=torch.long)
    per_cls_correct = torch.zeros(num_classes, dtype=torch.long)
    all_probs, all_targets = [], []
    last_print = time.time()
    val_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction="mean")

    for step, batch in enumerate(loader):
        x = batch["mri"].to(device, non_blocking=False)
        y = batch["isup"].to(device, non_blocking=False)
        x = _maybe_resize_batch(x, crop_hw, d_stride)
        x = _fix_size(x, fix_depth, fix_hw)
        with dev_autocast(str(device), amp):
            z, logits = model(x)
            adj_logits = logits if la is None else (logits + la)
            loss = ce(adj_logits, y)
        mps_sync()
        val_loss += loss.item()

        pred = adj_logits.argmax(dim=1) if la is not None else logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        tot += y.numel()
        for c in range(num_classes):
            mask = (y == c)
            per_cls[c] += mask.sum().item()
            if mask.any():
                per_cls_correct[c] += (pred[mask] == c).sum().item()
        probs = torch.softmax(adj_logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.detach().cpu())

        if (step + 1) % log_every == 0 or (time.time() - last_print) > 30:
            last_print = time.time()
            print(f"[val] {step+1}/{len(loader)} x {tuple(x.shape)} dev={device} mem={dev_mem()}")

    probs_cat = np.concatenate(all_probs, 0) if all_probs else np.zeros((0, num_classes))
    y_cat = torch.cat(all_targets, 0).numpy() if all_targets else np.zeros((0,), dtype=np.int64)
    try:
        y_oh = np.eye(num_classes, dtype=np.int32)[y_cat]
        va_auroc = roc_auc_score(y_oh, probs_cat, average="macro", multi_class="ovr") if len(y_cat) > 0 else float("nan")
    except Exception:
        va_auroc = float("nan")

    per_cls_acc = (per_cls_correct.float() / per_cls.clamp(min=1).float()).tolist()
    va_acc = correct / max(1, tot)
    bal_acc = float(np.nanmean(per_cls_acc))
    return val_loss / max(1, len(loader)), {"acc": va_acc, "auroc": va_auroc, "per_class_acc": per_cls_acc, "balanced_acc": bal_acc}


# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser("Train NNUNetV1EncoderISUP with class-balanced sampling and logit adjustment.")
    ap.add_argument("--images_root", type=str, required=True)
    ap.add_argument("--labels_root", type=str, default=None)  # keep None for speed (no masks)
    ap.add_argument("--splits_path", type=str, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--marksheet_csv", type=str, required=True,
                    help="Path to clinical_information/marksheet.csv (uses case_ISUP)")
    ap.add_argument("--fold_dir", type=str, required=True,
                    help="Path to nnU-Net v1 fold dir or its parent (contains fold_0, ...)")
    ap.add_argument("--output_dir", type=str, default="runs_isup")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze_epochs", type=int, default=5)
    ap.add_argument("--unfreeze_top_k", type=int, default=2)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)  # lower default for stability on encoder
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)

    # perf / device knobs
    ap.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu"))
    ap.add_argument("--target_spacing", type=str, default="3.0,1.0,1.0",
                    help="(D,H,W) mm; try 3.0,1.2,1.2 for faster runs")
    ap.add_argument("--amp", action="store_true", help="enable autocast (fp16) if supported")
    ap.add_argument("--center_crop", type=int, default=0,
                    help="optional square HW center-crop size (e.g., 160/192). 0=off")
    ap.add_argument("--depth_stride", type=int, default=1,
                    help="optional depth subsampling stride (e.g., 2 keeps every other slice)")
    ap.add_argument("--fix_depth", type=int, default=32, help="fix D to this value (center-crop or pad). 0=off")
    ap.add_argument("--fix_hw", type=int, default=192, help="fix H and W to this value (center-crop or pad). 0=off")
    ap.add_argument("--log_every", type=int, default=5, help="print every N train steps")
    ap.add_argument("--preflight", action="store_true",
                    help="run and time one forward pass before training")
    ap.add_argument("--balance", action="store_true",
                    help="use class-balanced sampler for the train loader")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="temperature for logit adjustment; 0=off, 1=default")
    ap.add_argument("--modalities", type=str, default="t2w,hbv,adc",
                    help="comma-separated modalities to load (subset of: t2w,adc,hbv)")
    ap.add_argument("--crop_to_gland", action="store_true",
                    help="Crop to gland ROI and use gland mask for normalization when labels_root is set")

    args = ap.parse_args()

    args.labels_root = maybe_parse_none(args.labels_root)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # labels from marksheet.csv
    isup_map = load_isup_map_from_marksheet(args.marksheet_csv)

    # splits
    train_keys = load_split_keys(args.splits_path, fold=args.fold, split="train")
    val_keys   = load_split_keys(args.splits_path, fold=args.fold, split="val")

    # sanity: how many labeled in split?
    n_train_lab = sum(k in isup_map for k in train_keys)
    n_val_lab   = sum(k in isup_map for k in val_keys)
    print(f"[info] labeled: train {n_train_lab}/{len(train_keys)}, val {n_val_lab}/{len(val_keys)}")

    # modalities
    sequences = tuple(s.strip().lower() for s in args.modalities.split(",") if s.strip())
    assert len(sequences) >= 1, "At least one modality required (e.g., --modalities t2w,adc)"
    ts = tuple(float(x) for x in args.target_spacing.split(","))

    # datasets
    ds_tr = PicaiMRIDataset(
        images_root=args.images_root,
        labels_root=args.labels_root,          # set this to your picai_labels path or None
        sequences=sequences,
        target_spacing=ts,
        crop_to_gland=args.crop_to_gland,
        isup_map=isup_map,
        allowed_keys=train_keys,
        return_gland_mask=False,
        load_lesion_mask=False,
        pad_to_multiple_of=32,
        require_isup=True,
    )
    ds_va = PicaiMRIDataset(
        images_root=args.images_root,
        labels_root=args.labels_root,
        sequences=sequences,
        target_spacing=ts,
        crop_to_gland=args.crop_to_gland,
        isup_map=isup_map,
        allowed_keys=val_keys,
        return_gland_mask=False,
        load_lesion_mask=False,
        pad_to_multiple_of=32,
        require_isup=True,
    )

    # loaders (class-balanced sampler optional; works even with bs=1)
    use_pin = True if (args.device == "cuda" and torch.cuda.is_available()) else False

    def _dataset_keys(ds):
        if hasattr(ds, "keys"): return ds.keys
        if hasattr(ds, "index"): return [f"{pid}_{study}" for (pid, study) in ds.index]
        raise AttributeError("Dataset must expose .keys or .index")

    sampler = None
    cls_counts = None
    if args.balance:
        tr_keys_list = _dataset_keys(ds_tr)
        cls_counts = np.zeros(args.num_classes, dtype=np.int64)
        for k in tr_keys_list:
            y = isup_map.get(k, None)
            if y is not None and 0 <= y < args.num_classes:
                cls_counts[y] += 1
        weights = []
        for k in tr_keys_list:
            y = isup_map[k]
            w = 1.0 / max(1, cls_counts[y])
            weights.append(w)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size,
                       sampler=sampler, shuffle=(sampler is None),
                       num_workers=args.num_workers, pin_memory=use_pin)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=args.num_workers, pin_memory=use_pin)

    # model
    device = torch.device(args.device)
    model = NNUNetV1EncoderISUP(fold_dir=args.fold_dir, in_channels=len(sequences),
                                n_isup=args.num_classes).to(device)

    # neutral head init (avoid instant class-0 dominance)
    with torch.no_grad():
        if hasattr(model, "isup_head") and model.isup_head is not None:
            if isinstance(model.isup_head[-1], nn.Linear):
                model.isup_head[-1].weight.zero_()
                model.isup_head[-1].bias.zero_()

    # optional preflight
    if args.preflight:
        batch = next(iter(dl_tr))
        x = batch["mri"].to(device)
        y = batch["isup"].to(device)
        x = _maybe_resize_batch(x, args.center_crop, args.depth_stride)
        x = _fix_size(x, args.fix_depth, args.fix_hw)
        print("[preflight] device:", device, "x:", tuple(x.shape), "mem:", dev_mem())
        mps_sync()
        t0 = time.time()
        with dev_autocast(str(device), args.amp):
            z, logits = model(x)
        mps_sync()
        print(f"[preflight] forward {time.time()-t0:.3f}s; logits {tuple(logits.shape)}")
        return

    # Logit Adjustment vector (built from TRAIN class priors)
    if cls_counts is None:
        tr_keys_list = _dataset_keys(ds_tr)
        cls_counts = np.zeros(args.num_classes, dtype=np.int64)
        for k in tr_keys_list:
            y = isup_map.get(k, None)
            if y is not None and 0 <= y < args.num_classes:
                cls_counts[y] += 1
    pri = cls_counts / max(1, cls_counts.sum())
    eps = 1e-6
    tau = float(args.tau)
    la = None
    if tau > 0:
        la = (-tau) * torch.log(torch.tensor(pri + eps, dtype=torch.float32, device=device))  # shape [C]

    # optimizer / loss (no class weights when using LA)
    model.freeze_encoder(freeze=True)  # warmup heads only
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, head_lr=args.head_lr, enc_lr=None)

    best_val_bal = -np.inf
    ckpt_path = Path(args.output_dir) / f"best_fold{args.fold}.pt"

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            model.freeze_encoder(freeze=True, unfreeze_top_k=args.unfreeze_top_k)
            optimizer = make_optimizer(model, head_lr=args.head_lr, enc_lr=args.encoder_lr)
            print(f"[epoch {epoch}] unfroze top {args.unfreeze_top_k} encoder blocks")

        tr_loss, tr_metrics = run_epoch_train(
            model, dl_tr, criterion, optimizer, device, args.num_classes,
            log_every=args.log_every, amp=args.amp,
            crop_hw=args.center_crop, d_stride=args.depth_stride, la=la,
            fix_depth=args.fix_depth, fix_hw=args.fix_hw
        )
        va_loss, va_metrics = run_epoch_eval(
            model, dl_va, device, args.num_classes, log_every=999, amp=args.amp,
            crop_hw=args.center_crop, d_stride=args.depth_stride, la=la,
            fix_depth=args.fix_depth, fix_hw=args.fix_hw
        )

        print(
            f"[epoch {epoch:03d}] "
            f"loss={tr_loss:.4f} | "
            f"train acc={tr_metrics['acc']:.3f} bal_acc={tr_metrics['balanced_acc']:.3f} auroc={tr_metrics['auroc']:.3f} "
            f"per-class[{pca_str(tr_metrics['per_class_acc'])}] | "
            f"val acc={va_metrics['acc']:.3f} bal_acc={va_metrics['balanced_acc']:.3f} auroc={va_metrics['auroc']:.3f} "
            f"per-class[{pca_str(va_metrics['per_class_acc'])}]"
        )

        # save best by VALIDATION BALANCED ACCURACY
        val_bal = va_metrics["balanced_acc"]
        if not np.isnan(val_bal) and val_bal > best_val_bal:
            best_val_bal = val_bal
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "args": vars(args), "val_metrics": va_metrics},
                ckpt_path,
            )
            print(f"  ↳ saved best (balanced_acc={best_val_bal:.4f}) to {ckpt_path}")

        # also save 'last' each epoch
        last_path = Path(args.output_dir) / f"last_fold{args.fold}.pt"
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "args": vars(args), "val_metrics": va_metrics},
            last_path,
        )

    print(f"[done] best val balanced_acc = {best_val_bal:.4f}")


# ---------------------------- optimizer split ----------------------------
def make_optimizer(model: NNUNetV1EncoderISUP, head_lr: float, enc_lr: float | None = None):
    if enc_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=head_lr, weight_decay=1e-4)
    # split heads vs encoder
    head_params, enc_params = [], []
    heads = []
    if model.isup_head is not None:
        heads += list(model.isup_head.parameters())
    heads += list(model.proj.parameters()) + list(model.pool.parameters())
    head_ids = {id(p) for p in heads}
    for p in model.parameters():
        (head_params if id(p) in head_ids else enc_params).append(p)
    return optim.AdamW(
        [{"params": head_params, "lr": head_lr}, {"params": enc_params, "lr": enc_lr}],
        weight_decay=1e-4,
    )


if __name__ == "__main__":
    main()
