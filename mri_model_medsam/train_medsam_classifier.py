#!/usr/bin/env python3
import os
import csv
import shutil
import argparse
import random
import logging
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

import wandb

# =========================
# Constants & Helpers
# =========================
ALLOWED_ISUPS = {0, 2, 3, 4, 5}
ISUP_TO_IDX = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
IDX_TO_ISUP = {v: k for k, v in ISUP_TO_IDX.items()}
NUM_CLASSES = 5

def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    # Clear existing handlers if re-run
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def parse_isup_from_name(fname: str) -> int:
    # file: <patient>_<study>_<patch>_<isup>.pt
    try:
        base = os.path.splitext(os.path.basename(fname))[0]
        isup = int(base.split("_")[-1])
        return isup
    except Exception:
        raise ValueError(f"Could not parse ISUP from filename: {fname}")

def parse_patient_study(fname: str):
    # returns (patient_id, study_id) from <patient>_<study>_...
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Filename missing patient/study: {fname}")
    return parts[0], parts[1]

def safe_auc(y_true_idx, y_probs, num_classes=NUM_CLASSES):
    """
    Macro one-vs-rest AUC. Handles missing classes gracefully.
    y_true_idx: (N,) integer class indices in [0..num_classes-1]
    y_probs: (N, num_classes) probabilities
    """
    try:
        y_true = np.asarray(y_true_idx)
        y_probs = np.asarray(y_probs)
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        aucs = []
        for i in range(num_classes):
            # Must have at least one positive and one negative sample
            pos = (y_true == i).sum()
            neg = (y_true != i).sum()
            if pos > 0 and neg > 0:
                aucs.append(roc_auc_score(y_true_bin[:, i], y_probs[:, i]))
        return float(np.mean(aucs)) if aucs else float("nan")
    except Exception as e:
        return float("nan")

def per_class_accuracy(y_true_idx, y_pred_idx, num_classes=NUM_CLASSES):
    acc = {}
    y_true = np.asarray(y_true_idx)
    y_pred = np.asarray(y_pred_idx)
    for c in range(num_classes):
        mask = (y_true == c)
        total = mask.sum()
        if total == 0:
            acc[c] = float("nan")
        else:
            acc[c] = float((y_pred[mask] == c).sum() / total)
    return acc

def count_isups_from_files(files):
    ctr = Counter()
    for f in files:
        ctr[parse_isup_from_name(f)] += 1
    # Only report allowed classes
    return {k: ctr.get(k, 0) for k in sorted(ALLOWED_ISUPS)}

def load_marksheet_folds(marksheet_csv, val_fold="fold0"):
    """
    Returns: dict mapping (patient_id, study_id) -> "val" or "train"
    """
    split_map = {}
    with open(marksheet_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row["patient_id"]).strip()
            sid = str(row["study_id"]).strip()
            fold = str(row["fold"]).strip()
            key = (pid, sid)
            split_map[key] = "val" if fold == val_fold else "train"
    return split_map

def build_splits_from_marksheet(embedding_dir, marksheet_csv, val_fold="fold0"):
    """
    Scans embedding_dir, filters to ALLOWED_ISUPS, then assigns each file
    to train/val based on (patient_id, study_id) fold in marksheet.
    Files whose (patient_id, study_id) are missing from marksheet are skipped.
    """
    split_map = load_marksheet_folds(marksheet_csv, val_fold=val_fold)
    all_files = [f for f in os.listdir(embedding_dir) if f.endswith(".pt")]
    # keep only allowed isups
    all_files = [f for f in all_files if parse_isup_from_name(f) in ALLOWED_ISUPS]

    train_files, val_files, skipped = [], [], 0
    for f in all_files:
        key = parse_patient_study(f)
        if key not in split_map:
            skipped += 1
            continue
        if split_map[key] == "val":
            val_files.append(f)
        else:
            train_files.append(f)
    return train_files, val_files, skipped

# =========================
# Dataset
# =========================
class SliceEmbeddingDataset(Dataset):
    """
    Expects .pt tensor shaped like (C,H,W) or (1,C,H,W); label from filename.
    Returns (embedding_tensor, class_index_in_[0,1,2])
    """
    def __init__(self, embedding_dir, files_list):
        self.embedding_dir = embedding_dir
        self.embedding_files = files_list

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        fname = self.embedding_files[idx]
        path = os.path.join(self.embedding_dir, fname)
        emb = torch.load(path)  # assumes saved tensor
        if emb.dim() == 4 and emb.size(0) == 1:
            emb = emb.squeeze(0)
        isup = parse_isup_from_name(fname)
        if isup not in ALLOWED_ISUPS:
            raise RuntimeError(f"Found disallowed ISUP in dataset: {isup} for file {fname}")
        label_idx = ISUP_TO_IDX[isup]
        return emb.float(), int(label_idx)

# =========================
# Model
# =========================
class MRIClassifierCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, projection_dim=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, projection_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        embedding = self.projection(x)
        logits = self.classifier(embedding)
        return logits, embedding

# =========================
# Save validation encodings (projected_vec only)
# =========================
@torch.no_grad()
def save_val_encodings(model, val_dataset, val_loader, embedding_dir, out_dir, device, logger):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    filenames = val_dataset.embedding_files  # stable order with shuffle=False
    cursor = 0

    model.eval()
    for xb, yb in val_loader:
        xb = xb.to(device)
        proj = model.projection(model.encoder(xb)).cpu()   # [B, D]
        bsz = xb.size(0)
        for i in range(bsz):
            fname = filenames[cursor]
            cursor += 1
            torch.save({
                "sample_id": os.path.splitext(fname)[0],
                "path": os.path.join(embedding_dir, fname),
                "label_idx": int(yb[i]),
                "isup": int(IDX_TO_ISUP[int(yb[i])]),
                "projected_vec": proj[i],
            }, os.path.join(out_dir, fname))
    logger.info(f"Wrote validation encodings to: {out_dir}")

# =========================
# Train
# =========================
def train(args):
    # ----- setup -----
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "mps" else "cpu")
    logger = setup_logger(args.log_file)
    wandb.init(project="picai-ggg-medsam-classifier", name=args.wandb_name, config=vars(args))

    # Reproducibility (torch)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----- split by marksheet -----
    train_files, val_files, skipped = build_splits_from_marksheet(
        args.embedding_dir, args.marksheet_csv, val_fold=args.val_fold
    )

    # counts
    train_counts = count_isups_from_files(train_files)
    val_counts = count_isups_from_files(val_files)

    logger.info("ARGS: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    logger.info(f"Skipped files not found in marksheet: {skipped}")
    logger.info(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    logger.info(f"Train counts by ISUP: {train_counts}")
    logger.info(f"Val   counts by ISUP: {val_counts}")

    # ----- data -----
    train_dataset = SliceEmbeddingDataset(args.embedding_dir, train_files)
    val_dataset   = SliceEmbeddingDataset(args.embedding_dir, val_files)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ----- model -----
    model = MRIClassifierCNN(num_classes=NUM_CLASSES, projection_dim=args.projection_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_bal_acc = -1.0
    ckpt_path = os.path.join(args.save_dir, "best_classifier.pth")
    enc_val_dir = os.path.join(args.save_dir, "best_val_encodings")
    os.makedirs(args.save_dir, exist_ok=True)

    # ----- training loop -----
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss, total = 0.0, 0
        all_logits_train, all_labels_train = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total += bs
            all_logits_train.append(logits.detach().cpu())
            all_labels_train.append(yb.detach().cpu())

        train_loss = total_loss / max(total, 1)
        train_logits = torch.cat(all_logits_train) if all_logits_train else torch.empty(0, NUM_CLASSES)
        train_labels = torch.cat(all_labels_train).numpy() if all_labels_train else np.empty((0,), dtype=int)

        if train_logits.nelement() > 0:
            train_probs = train_logits.softmax(dim=1).numpy()
            train_pred_idx = train_logits.argmax(dim=1).numpy()
            train_acc = float((train_pred_idx == train_labels).mean()) if train_labels.size else float("nan")
            train_bal_acc = balanced_accuracy_score(train_labels, train_pred_idx) if train_labels.size else float("nan")
            train_auc = safe_auc(train_labels, train_probs, num_classes=NUM_CLASSES)
        else:
            train_acc = train_bal_acc = train_auc = float("nan")

        # ---- Validate ----
        model.eval()
        val_total_loss, val_total = 0.0, 0
        all_logits_val, all_labels_val = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                bs = yb.size(0)
                val_total_loss += loss.item() * bs
                val_total += bs
                all_logits_val.append(logits.cpu())
                all_labels_val.append(yb.cpu())

        val_loss = val_total_loss / max(val_total, 1)
        val_logits = torch.cat(all_logits_val) if all_logits_val else torch.empty(0, NUM_CLASSES)
        val_labels = torch.cat(all_labels_val).numpy() if all_labels_val else np.empty((0,), dtype=int)

        if val_logits.nelement() > 0:
            val_probs = val_logits.softmax(dim=1).numpy()
            val_pred_idx = val_logits.argmax(dim=1).numpy()
            val_acc = float((val_pred_idx == val_labels).mean()) if val_labels.size else float("nan")
            val_bal_acc = balanced_accuracy_score(val_labels, val_pred_idx) if val_labels.size else float("nan")
            val_auc = safe_auc(val_labels, val_probs, num_classes=NUM_CLASSES)
            val_pc_acc = per_class_accuracy(val_labels, val_pred_idx, num_classes=NUM_CLASSES)  # idx keys 0,1,2
        else:
            val_acc = val_bal_acc = val_auc = float("nan")
            val_pc_acc = {0: float("nan"), 1: float("nan"), 2: float("nan")}

        # ---- Logging ----
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_balanced_accuracy": train_bal_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bal_acc,
            "val_auc": val_auc,
            # Per-ISUP accuracy on VAL (reported as real ISUP keys 0/4/5)
            "val_acc_isup_0": val_pc_acc.get(ISUP_TO_IDX[0], float("nan")),
            "val_acc_isup_2": val_pc_acc.get(ISUP_TO_IDX[2], float("nan")),
            "val_acc_isup_3": val_pc_acc.get(ISUP_TO_IDX[3], float("nan")),
            "val_acc_isup_4": val_pc_acc.get(ISUP_TO_IDX[4], float("nan")),
            "val_acc_isup_5": val_pc_acc.get(ISUP_TO_IDX[5], float("nan")),
        }
        wandb.log(log_row)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} balAcc={train_bal_acc:.4f} AUC={train_auc:.4f} || "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} balAcc={val_bal_acc:.4f} AUC={val_auc:.4f} || "
            f"Val per-ISUP acc: 0={log_row['val_acc_isup_0']:.4f},  2={log_row['val_acc_isup_2']:.4f}, 3={log_row['val_acc_isup_3']:.4f},"
            f"4={log_row['val_acc_isup_4']:.4f}, 5={log_row['val_acc_isup_5']:.4f}"
        )

        # ---- Save on best balanced accuracy (keep your criterion) ----
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_balanced_accuracy": best_val_bal_acc
            }, ckpt_path)
            logger.info(f"Saved BEST checkpoint (val_bal_acc={best_val_bal_acc:.4f}) -> {ckpt_path}")

            # Save validation encodings each time we get a new best
            save_val_encodings(
                model=model,
                val_dataset=val_dataset,
                val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers),
                embedding_dir=args.embedding_dir,
                out_dir=enc_val_dir,
                device=device,
                logger=logger
            )

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help="Directory with .pt slice embeddings named <patient>_<study>_<patch>_<isup>.pt")
    parser.add_argument('--marksheet_csv', type=str, required=True,
                        help="Path to marksheet.csv with 'patient_id,study_id,fold' columns (plus others).")
    parser.add_argument('--val_fold', type=str, default='fold0', help="Which fold to use as validation fold.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save checkpoints/encodings/logs.")
    parser.add_argument('--log_file', type=str, default=None, help="Log file path; default: <save_dir>/train.log")

    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--projection_dim', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda', help="cuda | cpu | mps")
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--wandb_name', type=str)

    args = parser.parse_args()
    if args.log_file is None:
        args.log_file = os.path.join(args.save_dir, "train.log")
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
