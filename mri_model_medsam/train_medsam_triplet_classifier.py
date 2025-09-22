#!/usr/bin/env python3
import os
import csv
import argparse
import logging
import random
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
# Constants & helpers
# =========================
ALLOWED_ISUPS = {0, 2, 3, 4, 5}
ISUP_TO_IDX = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
IDX_TO_ISUP = {v: k for k, v in ISUP_TO_IDX.items()}
NUM_CLASSES = 5

def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("aligned_clf")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_isup_from_name(fname: str) -> int:
    base = os.path.splitext(os.path.basename(fname))[0]
    return int(base.split("_")[-1])

def parse_patient_study(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Filename missing patient/study: {fname}")
    return parts[0], parts[1]

def load_marksheet_folds(marksheet_csv, val_fold="fold0"):
    split_map = {}
    with open(marksheet_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row["patient_id"]).strip()
            sid = str(row["study_id"]).strip()
            fold = str(row["fold"]).strip()
            split_map[(pid, sid)] = "val" if fold == val_fold else "train"
    return split_map

def build_splits_from_marksheet(embedding_dir, marksheet_csv, val_fold="fold0"):
    split_map = load_marksheet_folds(marksheet_csv, val_fold)
    all_files = [f for f in os.listdir(embedding_dir) if f.endswith(".pt")]
    # filter to allowed classes
    all_files = [f for f in all_files if parse_isup_from_name(f) in ALLOWED_ISUPS]

    train_files, val_files, skipped = [], [], 0
    for f in all_files:
        key = parse_patient_study(f)
        if key not in split_map:
            skipped += 1
            continue
        (val_files if split_map[key] == "val" else train_files).append(f)
    return train_files, val_files, skipped

def count_isups(files):
    c = Counter()
    for f in files:
        c[parse_isup_from_name(f)] += 1
    return {k: c.get(k, 0) for k in sorted(ALLOWED_ISUPS)}

def safe_auc(y_true_idx, y_probs, num_classes=NUM_CLASSES):
    try:
        y_true = np.asarray(y_true_idx)
        y_probs = np.asarray(y_probs)
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        aucs = []
        for i in range(num_classes):
            pos = (y_true == i).sum()
            neg = (y_true != i).sum()
            if pos > 0 and neg > 0:
                aucs.append(roc_auc_score(y_true_bin[:, i], y_probs[:, i]))
        return float(np.mean(aucs)) if aucs else float("nan")
    except Exception:
        return float("nan")

def per_class_accuracy(y_true_idx, y_pred_idx, num_classes=NUM_CLASSES):
    acc = {}
    y_true = np.asarray(y_true_idx)
    y_pred = np.asarray(y_pred_idx)
    for c in range(num_classes):
        mask = (y_true == c)
        n = mask.sum()
        acc[c] = float((y_pred[mask] == c).sum() / n) if n else float("nan")
    return acc

# =========================
# Dataset
# =========================
class SliceEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, files_list):
        self.dir = embedding_dir
        self.files = files_list

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        x = torch.load(os.path.join(self.dir, fname))
        if x.dim() == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        isup = parse_isup_from_name(fname)
        label_idx = ISUP_TO_IDX[isup]
        return x.float(), label_idx

# =========================
# Model (same encoder+projection as other scripts)
# =========================
class MRITripletEncoder(nn.Module):
    def __init__(self, projection_dim=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, projection_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.encoder(x)
        return self.projector(x)  # [B, 384]

class MRIClassifier(nn.Module):
    def __init__(self, projection_dim=384, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = MRITripletEncoder(projection_dim=projection_dim)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits

# =========================
# Train
# =========================
def train(args):
    device = torch.device(args.device if (args.device == "mps" or torch.cuda.is_available()) else "cpu")
    set_all_seeds(args.seed)

    logger = setup_logger(args.log_file)
    wandb.init(project="picai-medsam-triplet-classifier", name=args.wandb_name, config=vars(args))

    # ----- splits from marksheet -----
    train_files, val_files, skipped = build_splits_from_marksheet(
        args.embedding_dir, args.marksheet_csv, val_fold=args.val_fold
    )
    logger.info("ARGS: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    logger.info(f"Skipped files not in marksheet: {skipped}")
    logger.info(f"Train N={len(train_files)} | Val N={len(val_files)}")
    logger.info(f"Train counts (ISUP 0/4/5): {count_isups(train_files)}")
    logger.info(f"Val   counts (ISUP 0/4/5): {count_isups(val_files)}")

    # ----- data -----
    train_ds = SliceEmbeddingDataset(args.embedding_dir, train_files)
    val_ds   = SliceEmbeddingDataset(args.embedding_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ----- model -----
    model = MRIClassifier(projection_dim=args.projection_dim, num_classes=NUM_CLASSES).to(device)

    # Load frozen encoder from triplet checkpoint (robust to key names)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    loaded = False
    for key in ["mri_encoder_state_dict", "model_state_dict", "state_dict"]:
        if key in ckpt:
            try:
                model.encoder.load_state_dict(ckpt[key], strict=True)
                loaded = True
                break
            except Exception:
                pass
    if not loaded:
        # sometimes checkpoint itself is the encoder state_dict
        try:
            model.encoder.load_state_dict(ckpt, strict=True)
            loaded = True
        except Exception as e:
            raise RuntimeError(f"Could not load encoder weights from {args.checkpoint_path}: {e}")
    logger.info(f"Loaded frozen encoder from: {args.checkpoint_path}")

    for p in model.encoder.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_bal_acc = -1.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_aligned_classifier.pth")

    # ----- loop -----
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        total_loss, total, logits_list, labels_list = 0.0, 0, [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total += bs
            logits_list.append(logits.detach().cpu())
            labels_list.append(yb.detach().cpu())

        train_loss = total_loss / max(total, 1)
        train_logits = torch.cat(logits_list) if logits_list else torch.empty(0, NUM_CLASSES)
        train_labels = torch.cat(labels_list).numpy() if labels_list else np.empty((0,), dtype=int)
        if train_logits.nelement() > 0:
            train_probs = train_logits.softmax(dim=1).numpy()
            train_pred  = train_logits.argmax(dim=1).numpy()
            train_acc   = float((train_pred == train_labels).mean()) if train_labels.size else float("nan")
            train_bal   = balanced_accuracy_score(train_labels, train_pred) if train_labels.size else float("nan")
            train_auc   = safe_auc(train_labels, train_probs, num_classes=NUM_CLASSES)
        else:
            train_acc = train_bal = train_auc = float("nan")

        # validate
        model.eval()
        v_total_loss, v_total, v_logits_list, v_labels_list = 0.0, 0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                bs = yb.size(0)
                v_total_loss += loss.item() * bs
                v_total += bs
                v_logits_list.append(logits.cpu())
                v_labels_list.append(yb.cpu())

        val_loss = v_total_loss / max(v_total, 1)
        val_logits = torch.cat(v_logits_list) if v_logits_list else torch.empty(0, NUM_CLASSES)
        val_labels = torch.cat(v_labels_list).numpy() if v_labels_list else np.empty((0,), dtype=int)
        if val_logits.nelement() > 0:
            val_probs = val_logits.softmax(dim=1).numpy()
            val_pred  = val_logits.argmax(dim=1).numpy()
            val_acc   = float((val_pred == val_labels).mean()) if val_labels.size else float("nan")
            val_bal   = balanced_accuracy_score(val_labels, val_pred) if val_labels.size else float("nan")
            val_auc   = safe_auc(val_labels, val_probs, num_classes=NUM_CLASSES)
            per_cls   = per_class_accuracy(val_labels, val_pred, num_classes=NUM_CLASSES)
        else:
            val_acc = val_bal = val_auc = float("nan")
            per_cls = {0: float("nan"), 1: float("nan"), 2: float("nan")}

        # logging
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_balanced_accuracy": train_bal,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bal,
            "val_auc": val_auc,
            "val_acc_isup_0": per_cls.get(ISUP_TO_IDX[0], float("nan")),
            "val_acc_isup_2": per_cls.get(ISUP_TO_IDX[2], float("nan")),
            "val_acc_isup_3": per_cls.get(ISUP_TO_IDX[3], float("nan")),
            "val_acc_isup_4": per_cls.get(ISUP_TO_IDX[4], float("nan")),
            "val_acc_isup_5": per_cls.get(ISUP_TO_IDX[5], float("nan")),
        }
        wandb.log(log_row)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} balAcc={train_bal:.4f} AUC={train_auc:.4f} || "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} balAcc={val_bal:.4f} AUC={val_auc:.4f} || "
            f"Val per-ISUP acc: 0={log_row['val_acc_isup_0']:.4f}, "
            f"4={log_row['val_acc_isup_4']:.4f}, 5={log_row['val_acc_isup_5']:.4f}"
        )

        # save on best val balanced accuracy
        if val_bal > best_val_bal_acc:
            best_val_bal_acc = val_bal
            torch.save({
                "epoch": epoch,
                "encoder_frozen": True,
                "model_state_dict": model.state_dict(),
                "best_val_balanced_accuracy": best_val_bal_acc
            }, ckpt_path)
            logger.info(f"Saved BEST checkpoint (val_bal_acc={best_val_bal_acc:.4f}) -> {ckpt_path}")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--embedding_dir', type=str, required=True,
                   help="MRI .pt encodings named <patient>_<study>_<patch>_<isup>.pt")
    p.add_argument('--marksheet_csv', type=str, required=True,
                   help="marksheet.csv with patient_id,study_id,fold")
    p.add_argument('--val_fold', type=str, default='fold0')
    # pretrained
    p.add_argument('--checkpoint_path', type=str, required=True,
                   help="Triplet checkpoint containing MRI encoder weights")
    # training
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--log_file', type=str, default=None)
    p.add_argument('--projection_dim', type=int, default=384)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')   # cuda | mps | cpu
    p.add_argument('--num_workers', type=int, default=4)
    # wandb
    p.add_argument('--wandb_name', type=str)

    args = p.parse_args()
    if args.log_file is None:
        args.log_file = os.path.join(args.save_dir, "aligned_classifier_train.log")
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
