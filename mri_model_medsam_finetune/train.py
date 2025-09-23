
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

from ISUPMedSAM import ISUPMedSAM
from PiCAI_MultiSeq_2D_Bag import PiCAI_MultiSeq_2D_Bag, collate_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM on PI-CAI with MIL pooling")
    parser.add_argument("--checkpoint", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth", help="Path to MedSAM checkpoint")
    parser.add_argument("--images-dir", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_data/nnUNet_raw_data/Task2203_picai_baseline/imagesTr", help="PI-CAI images directory")
    parser.add_argument("--marksheet-csv", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_data/picai_labels/clinical_information/marksheet_folds.csv", help="Marksheet CSV with ISUP labels")
    parser.add_argument("--train-folds", nargs="+", default=["fold1", "fold2", "fold3", "fold4"], help="Folds for training set")
    parser.add_argument("--val-folds", nargs="+", default=["fold0"], help="Folds for validation set")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--model-type", choices=["vit_b", "vit_l"], default="vit_b")
    parser.add_argument("--device", type=str, default=None, help="Override device string (cuda, mps, cpu)")
    parser.add_argument("--out_dir", type=Path, default="/Users/emma/Desktop/QUEENS/THESIS/contrastive/mri_model_medsam_finetune/results", help="Where to store best checkpoint")
    parser.add_argument("--max-lr-epochs", type=int, default=None, help="Override scheduler T_max (defaults to epochs)")
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_enabled = scaler.is_enabled()

    for step, (images, labels, _case_id) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
            logits, _, _, _ = model(images)
            loss = criterion(logits, labels)

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
) -> Tuple[float, float, float, dict]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_buf = []
    labels_buf = []

    for images, labels, _case_ids in loader:
        images = images.to(device, non_blocking=True)

        # Use autocast in eval to reduce memory/compute
        use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits, _, _, _ = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        logits_buf.append(logits.detach().cpu())
        labels_buf.append(labels.detach().cpu())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    if total == 0:
        return avg_loss, acc, float("nan"), {}

    logits_all = torch.cat(logits_buf, dim=0)
    labels_all = torch.cat(labels_buf, dim=0)
    probs = torch.softmax(logits_all, dim=1).numpy()
    targets = labels_all.numpy()

    try:
        auc = roc_auc_score(targets, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    preds = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    per_class_acc = {}
    for cls in range(num_classes):
        cls_mask = targets == cls
        if cls_mask.any():
            per_class_acc[cls] = float((preds[cls_mask] == targets[cls_mask]).mean())
        else:
            per_class_acc[cls] = float("nan")

    return avg_loss, acc, auc, per_class_acc


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
    proj_dim=128,
    device=device
).to(device)

logger.info(model)

pin_memory = device == "cuda"

train_dataset = PiCAI_MultiSeq_2D_Bag(images_dir=args.images_dir, marksheet_csv=args.marksheet_csv, include_folds=args.train_folds)
val_dataset = PiCAI_MultiSeq_2D_Bag(images_dir=args.images_dir, marksheet_csv=args.marksheet_csv, include_folds=args.val_folds)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_one, pin_memory=pin_memory)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_one, pin_memory=pin_memory)

logger.info(f"Training cases: {len(train_dataset)}, Validation cases: {len(val_dataset)}")
# assert(len(train_dataset) == 1199 and len(val_dataset) == 300)

weights = train_dataset.get_class_weights().to(device)
logger.info(f"Class weights: {weights}")

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=args.max_lr_epochs or args.epochs)
if device == "cuda" and torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = torch.cuda.amp.GradScaler(enabled=False)

best_val_balanced_acc = 0.0
best_state = None

for epoch in range(args.epochs):
    train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, criterion, device, scaler, epoch)
    val_loss, val_acc, val_auc, val_class_acc = evaluate(model, val_dataloader, criterion, device)
    scheduler.step()

    val_balanced_acc = np.nanmean(list(val_class_acc.values()))
    logger.info(f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f} | "
            f"Val Class Acc: {val_class_acc} ")
    
    if val_balanced_acc > best_val_balanced_acc:
        best_val_balanced_acc = val_balanced_acc
        best_state = model.state_dict()
        if args.out_dir is not None:
            args.out_dir.mkdir(exist_ok=True)

            torch.save(best_state, args.out_dir / "best_model.pth")
            logger.info(f"Saved best model to {args.out_dir}")

if best_state is not None and args.out_dir is None:
    logger.info("Best validation accuracy: {:.3f} at epoch {}".format(best_val_balanced_acc, best_state["epoch"]))

    
