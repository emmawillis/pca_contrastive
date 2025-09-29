
from pathlib import Path
from typing import Tuple

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

from ISUPMedSAM_token_attention import ISUPMedSAM
from PiCAI_MultiSeq_2D_Bag import PiCAI_MultiSeq_2D_Bag, collate_one

from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import itertools

def pretty_confmat(cm: np.ndarray, labels: list[str] | None = None) -> str:
    """
    Nicely format a confusion matrix for logs.
    cm is (C, C). Optionally pass class labels.
    """
    C = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(C)]
    colw = max(6, max(len(s) for s in labels) + 2)
    header = " " * colw + "".join(f"{lbl:>{colw}}" for lbl in labels)
    rows = []
    for i in range(C):
        row = f"{labels[i]:>{colw}}" + "".join(f"{cm[i, j]:>{colw}.3f}" for j in range(C))
        rows.append(row)
    return header + "\n" + "\n".join(rows)

def build_weighted_sampler(train_dataset) -> WeightedRandomSampler:
    """
    Class-balanced sampler: one weight per dataset item.
    Uses train_dataset.cases which carries the label as index 4.
    """
    # labels per index
    labels = [isup for (_cid, _p0, _p1, _p2, isup) in train_dataset.cases]
    class_counts = np.bincount(labels, minlength=6)
    inv_freq = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [inv_freq[y] for y in labels]
    return WeightedRandomSampler(weights=sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True)

def stratified_indices(dataset, per_class: int | None = None, total: int | None = None):
    """
    Build a tiny subset for overfit tests.
    - If per_class is given: up to `per_class` per label (0..5)
    - Else if total is given: take ~uniform across classes up to total
    Returns: list of indices.
    """
    # gather indices by class
    by_cls = {c: [] for c in range(6)}
    for idx, (_cid, _p0, _p1, _p2, y) in enumerate(dataset.cases):
        by_cls[y].append(idx)

    for c in by_cls:
        np.random.shuffle(by_cls[c])

    if per_class is not None:
        chosen = list(itertools.chain.from_iterable(by_cls[c][:per_class] for c in range(6)))
        return chosen

    if total is not None:
        # round-robin pick until reaching total
        chosen = []
        ptr = {c: 0 for c in range(6)}
        while len(chosen) < min(total, len(dataset)):
            progressed = False
            for c in range(6):
                if ptr[c] < len(by_cls[c]) and len(chosen) < total:
                    chosen.append(by_cls[c][ptr[c]])
                    ptr[c] += 1
                    progressed = True
            if not progressed:
                break
        return chosen

    # no restriction
    return list(range(len(dataset)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM on PI-CAI with MIL pooling")
    parser.add_argument("--checkpoint", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth", help="Path to MedSAM checkpoint")
    parser.add_argument("--images-dir", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_data/nnUNet_raw_data/Task2203_picai_baseline/imagesTr", help="PI-CAI images directory")
    parser.add_argument("--marksheet-csv", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_data/picai_labels/clinical_information/marksheet_folds.csv", help="Marksheet CSV with ISUP labels")
    parser.add_argument("--train-folds", nargs="+", default=["fold1", "fold2", "fold3", "fold4"], help="Folds for training set")
    parser.add_argument("--val-folds", nargs="+", default=["fold0"], help="Folds for validation set")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--model-type", choices=["vit_b", "vit_l"], default="vit_b")
    parser.add_argument("--device", type=str, default=None, help="Override device string (cuda, mps, cpu)")
    parser.add_argument("--out_dir", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_model_medsam_finetune/results", help="Where to store best checkpoint")
    parser.add_argument("--max-lr-epochs", type=int, default=None, help="Override scheduler T_max (defaults to epochs)")
    parser.add_argument("--tiny-total", type=int, default=None,
                        help="Use only this many training cases total (stratified RR).")
    parser.add_argument("--tiny-per-class", type=int, default=None,
                        help="Use up to this many training cases per class.")
    parser.add_argument("--use-weighted-sampler", action="store_true",
                        help="Use class-balanced WeightedRandomSampler for training.")
    parser.add_argument("--confmat-normalize", choices=["true", "pred", "all", "none"], default="true",
                        help="Normalization for confusion matrix logging.")

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels, _case_id) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _, _, _ = model(images)
            loss = criterion(logits.float(), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: ISUPMedSAM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    confmat_normalize: str = "true",
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_buf = []
    labels_buf = []

    for images, labels, _case_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits, _, _, _ = model(images)
        loss = criterion(logits.float(), labels)

        running_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        logits_buf.append(logits.detach().cpu())
        labels_buf.append(labels.detach().cpu())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    if total == 0:
        return avg_loss, acc, float("nan"), {}, None, None, None

    logits_all = torch.cat(logits_buf, dim=0)
    labels_all = torch.cat(labels_buf, dim=0)
    probs = torch.softmax(logits_all, dim=1).numpy()
    targets = labels_all.numpy()
    preds = probs.argmax(axis=1)

    # Safe AUC (skip when invalid)
    try:
        auc = roc_auc_score(targets, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    # Per-class acc
    num_classes = probs.shape[1]
    per_class_acc = {}
    for cls in range(num_classes):
        cls_mask = targets == cls
        if cls_mask.any():
            per_class_acc[cls] = float((preds[cls_mask] == targets[cls_mask]).mean())
        else:
            per_class_acc[cls] = float("nan")

    # Confusion matrix (optionally normalized)
    norm = None if confmat_normalize == "none" else confmat_normalize
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)), normalize=norm)

    return avg_loss, acc, auc, per_class_acc, preds, targets, cm

################################################

args = parse_args()
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
if args.out_dir is not None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(args.out_dir / "training.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

logger.info(f"ARGS: {args}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if args.device is not None:
    device = args.device
logger.info(f"Using device: {device}")

model = ISUPMedSAM(
    checkpoint=args.checkpoint,
    proj_dim=args.proj_dim,
    device=device
).to(device)

logger.info(model)

pin_memory = device == "cuda"

# -- datasets --
train_dataset = PiCAI_MultiSeq_2D_Bag(
    images_dir=args.images_dir,
    marksheet_csv=args.marksheet_csv,
    include_folds=args.train_folds
)
val_dataset = PiCAI_MultiSeq_2D_Bag(
    images_dir=args.images_dir,
    marksheet_csv=args.marksheet_csv,
    include_folds=args.val_folds
)

# Tiny-subset overfit test (training set only)
if args.tiny_per_class is not None or args.tiny_total is not None:
    idxs = stratified_indices(train_dataset, per_class=args.tiny_per_class, total=args.tiny_total)
    train_dataset = Subset(train_dataset, idxs)
    logger.info(f"[Tiny-Subset] Using {len(train_dataset)} training cases "
                f"({'per_class=' + str(args.tiny_per_class) if args.tiny_per_class else 'total=' + str(args.tiny_total)})")

logger.info(f"Training cases: {len(train_dataset)}, Validation cases: {len(val_dataset)}")

# -- loaders --
if args.use_weighted_sampler and not isinstance(train_dataset, Subset):
    sampler = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate_one,
                              pin_memory=pin_memory)
else:
    # If we have a Subset, we still can use a sampler but we'd need to remap labels; keep it simple:
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_one,
                              pin_memory=pin_memory)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_one,
                        pin_memory=pin_memory)

# -- loss/opt/sched --
# (If you’re using a Subset for tiny-overfit, weights below still read from the full train set; that’s fine.)
full_train_for_weights = (train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset)
weights = full_train_for_weights.get_class_weights().to(device)
logger.info(f"Class weights: {weights}")

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=args.max_lr_epochs or args.epochs)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and torch.cuda.is_available()))

# -- training --
best_val_balanced_acc = 0.0
best_state = None
num_classes = 6
class_labels = [f"{i}" for i in range(num_classes)]

for epoch in range(args.epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch)

    val_loss, val_acc, val_auc, val_class_acc, preds, targets, cm = evaluate(
        model, val_loader, criterion, device, confmat_normalize=args.confmat_normalize
    )
    scheduler.step()

    val_balanced_acc = np.nanmean(list(val_class_acc.values()))
    logger.info(
        f"Epoch {epoch+1}/{args.epochs} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, "
        f"Val Balanced Acc: {val_balanced_acc:.4f} | Val Class Acc: {val_class_acc}"
    )

    # --- Confusion matrix logging ---
    if cm is not None:
        logger.info("Validation Confusion Matrix "
                    f"(normalize={args.confmat_normalize}):\n" + pretty_confmat(cm, class_labels))

    # --- Save best by balanced accuracy ---
    if val_balanced_acc > best_val_balanced_acc:
        best_val_balanced_acc = val_balanced_acc
        best_state = model.state_dict()
        if args.out_dir is not None:
            args.out_dir.mkdir(exist_ok=True, parents=True)
            torch.save(best_state, args.out_dir / "best_model.pth")
            logger.info(f"Saved best model to {args.out_dir}")

if best_state is None:
    logger.info("Training finished; no improvement over baseline balanced accuracy.")
else:
    logger.info(f"Best validation balanced accuracy: {best_val_balanced_acc:.4f}")
