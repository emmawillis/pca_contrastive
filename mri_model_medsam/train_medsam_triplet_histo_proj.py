#!/usr/bin/env python3
import os, csv, argparse, random, logging
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

# =========================
# Constants & helpers
# =========================
ALLOWED_ISUPS = {0, 2, 3, 4, 5}
SEED_DEFAULT  = 42

def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("triplet")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_isup_from_mri_fname(fname: str) -> int:
    # <patient_id>_<study_id>_<patch#>_<isup>.pt
    base = os.path.splitext(os.path.basename(fname))[0]
    return int(base.split("_")[-1])

def parse_patient_study(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    toks = base.split("_")
    return toks[0], toks[1]  # patient, study

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

def build_mri_splits_from_marksheet(embedding_dir, marksheet_csv, val_fold="fold0"):
    split_map = load_marksheet_folds(marksheet_csv, val_fold)
    all_files = [f for f in os.listdir(embedding_dir) if f.endswith(".pt")]
    all_files = [f for f in all_files if parse_isup_from_mri_fname(f) in ALLOWED_ISUPS]
    train_files, val_files, skipped = [], [], 0
    for f in all_files:
        key = parse_patient_study(f)
        if key not in split_map:
            skipped += 1
            continue
        (val_files if split_map[key] == "val" else train_files).append(f)
    return train_files, val_files, skipped

def count_isups_from_files(files):
    c = Counter()
    for f in files:
        c[parse_isup_from_mri_fname(f)] += 1
    return {k: c.get(k, 0) for k in sorted(ALLOWED_ISUPS)}

def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# =========================
# Datasets
# =========================
class MRISliceDataset(Dataset):
    """Loads MRI slice encodings (tensor[256,H,W]) and returns (x, isup_int)."""
    def __init__(self, embedding_dir, files_list):
        self.dir = embedding_dir
        self.files = files_list

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        x = torch.load(os.path.join(self.dir, fname))
        if x.dim() == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        isup = parse_isup_from_mri_fname(fname)
        return x.float(), int(isup)

# =========================
# Histo helpers (folder-based, direct 384-D 'h' vectors)
# =========================
def _choose_split_dir(root: str, split: str) -> str:
    """
    Returns the directory to use for a given split.
    - For 'train' -> prefer 'train/'
    - For 'val'   -> prefer 'val/', else fallback to 'test/'
    """
    candidates = [split]
    if split == "val":
        candidates.append("test")
    for c in candidates:
        d = os.path.join(root, c)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"No subdir found for split='{split}' under {root} (looked for {candidates}).")

def _parse_isup_from_histo_fname(fname: str) -> int:
    # <image_id>_<isup>.pt  (suffix label)
    base = os.path.splitext(os.path.basename(fname))[0]
    tok = base.split("_")[-1]
    return int(tok) if tok.isdigit() else None

def load_histo_bank(histo_root, device, split: str):
    """
    Build histo bank as dict[isup] -> list[tensor (384,)].
    Reads from subfolder (train/, val/ or test/). Each file holds a dict with 'h' (384-D).
    Label is parsed from filename suffix _<isup>.pt. Keeps only ALLOWED_ISUPS.
    """
    bank = defaultdict(list)
    subdir = _choose_split_dir(histo_root, split)
    files = [f for f in os.listdir(subdir) if f.endswith(".pt")]

    for f in files:
        isup = _parse_isup_from_histo_fname(f)
        if isup is None or isup not in ALLOWED_ISUPS:
            continue
        obj = torch.load(os.path.join(subdir, f), map_location="cpu")

        # Expect a dict with key 'h' (384-D). If not, try best-effort fallback.
        if isinstance(obj, dict) and 'h' in obj:
            vec = obj['h']
        else:
            # fallback: any 1-D tensor
            vec = obj

        v = torch.as_tensor(vec).float().view(-1)  # [384]
        if v.numel() != 384:
            raise ValueError(f"Expected 384-D histo vectors, got {v.numel()}D in file: {f}")
        bank[isup].append(v)  # keep on CPU; move later per batch
    return bank

def bank_counts(bank):  # pretty print
    return {k: len(v) for k, v in sorted(bank.items())}

# =========================
# Models
# =========================
class MRITripletEncoder(nn.Module):
    """Small conv encoder + projection -> 384-D (to match histo 'h')."""
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
        return self.projector(self.encoder(x))  # [B, projection_dim]

# =========================
# Metrics
# =========================
@torch.no_grad()
def proto_metrics(mri_encoder, val_loader, histo_bank_val, device):
    if not histo_bank_val:
        blank = {k: float("nan") for k in ALLOWED_ISUPS}
        return float("nan"), blank, {k: 0 for k in ALLOWED_ISUPS}

    protos = {}
    for k, vecs in histo_bank_val.items():
        if len(vecs) == 0: continue
        M = torch.stack(vecs, dim=0).float()
        protos[k] = l2_normalize(M.mean(dim=0))

    if len(protos) == 0:
        blank = {k: float("nan") for k in ALLOWED_ISUPS}
        return float("nan"), blank, {k: 0 for k in ALLOWED_ISUPS}

    proto_labels = sorted(protos.keys())
    proto_mat = torch.stack([protos[k] for k in proto_labels], dim=0)  # [K,384]

    total, correct = 0, 0
    per_cls = {k: {"n": 0, "c": 0} for k in ALLOWED_ISUPS}

    mri_encoder.eval()
    for xb, y_isup in val_loader:
        xb = xb.to(device)
        y_isup = y_isup.numpy().tolist()
        z = l2_normalize(mri_encoder(xb).detach())  # [B,384]
        sims = torch.matmul(z.cpu(), proto_mat.t())
        pred_idx = sims.argmax(dim=1).numpy()
        preds = [proto_labels[i] for i in pred_idx]

        for yt, yp in zip(y_isup, preds):
            if yt not in ALLOWED_ISUPS: continue
            total += 1; per_cls[yt]["n"] += 1
            if yp == yt:
                correct += 1; per_cls[yt]["c"] += 1

    overall = float(correct / total) if total else float("nan")
    per_class_acc = {k: (float(v["c"] / v["n"]) if v["n"] else float("nan")) for k, v in per_cls.items()}
    counts = {k: per_cls[k]["n"] for k in per_cls}
    return overall, per_class_acc, counts

def cosine_margin(anchor, pos, neg):
    a = l2_normalize(anchor); p = l2_normalize(pos); n = l2_normalize(neg)
    return float(((a*p).sum(dim=-1) - (a*n).sum(dim=-1)).mean().item())

# =========================
# Train
# =========================
def train(args):
    device = torch.device(args.device if (args.device == "mps" or torch.cuda.is_available()) else "cpu")
    set_all_seeds(args.seed)

    logger = setup_logger(args.log_file)
    wandb.init(project="picai-medsam-contrastive", name=args.wandb_name, config=vars(args))

    # ------- MRI splits from marksheet -------
    mri_train_files, mri_val_files, skipped = build_mri_splits_from_marksheet(
        args.embedding_dir, args.marksheet_csv, val_fold=args.val_fold
    )
    mri_train_counts = count_isups_from_files(mri_train_files)
    mri_val_counts   = count_isups_from_files(mri_val_files)

    logger.info("ARGS: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))
    logger.info(f"Skipped MRI files not present in marksheet: {skipped}")
    logger.info(f"MRI Train N={len(mri_train_files)} | MRI Val N={len(mri_val_files)}")
    logger.info(f"MRI Train counts {mri_train_counts}")
    logger.info(f"MRI Val   counts {mri_val_counts}")

    # ------- Data loaders -------
    train_ds = MRISliceDataset(args.embedding_dir, mri_train_files)
    val_ds   = MRISliceDataset(args.embedding_dir, mri_val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ------- Models (MRI -> 384 by default) -------
    if args.projection_dim != 384:
        logger.info(f"NOTE: projection_dim={args.projection_dim} (not 384). Make sure your histo vectors match!")
    mri_encoder = MRITripletEncoder(projection_dim=args.projection_dim).to(device)

    # ------- Triplet loss / optimizer -------
    triplet   = nn.TripletMarginLoss(margin=1.0, p=2)
    optim_all = optim.Adam(mri_encoder.parameters(), lr=args.lr, weight_decay=1e-5)

    # ------- Histo banks (folder-based, direct 384-D) -------
    histo_bank_train = load_histo_bank(args.histo_root, device, split="train")
    histo_bank_val   = load_histo_bank(args.histo_root, device, split="val")
    logger.info(f"Histo TRAIN bank counts: {bank_counts(histo_bank_train)}")
    logger.info(f"Histo VAL   bank counts: {bank_counts(histo_bank_val)}")

    train_classes = sorted([k for k, v in histo_bank_train.items() if len(v) > 0])
    val_classes   = sorted([k for k, v in histo_bank_val.items() if len(v) > 0])
    if len(train_classes) < 2:
        logger.info("Insufficient histo classes in TRAIN bank for triplet training. Exiting.")
        return

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_triplet_model.pth")

    # ------- Training loop -------
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        mri_encoder.train()
        running_loss, N = 0.0, 0
        running_margin_terms = []

        for xb, y_isup in train_loader:
            xb = xb.to(device)
            y_isup = y_isup.numpy().tolist()
            anchor = mri_encoder(xb)  # [B, 384]

            pos_list, neg_list, keep_idx = [], [], []
            for i, isup in enumerate(y_isup):
                if isup not in histo_bank_train or len(histo_bank_train[isup]) == 0:
                    continue
                neg_pool = [c for c in train_classes if c != isup and len(histo_bank_train[c]) > 0]
                if not neg_pool:
                    continue
                pos = random.choice(histo_bank_train[isup])           # CPU 384
                neg = random.choice(histo_bank_train[random.choice(neg_pool)])
                pos_list.append(pos); neg_list.append(neg); keep_idx.append(i)

            if not keep_idx:
                continue

            anchor_sel = anchor[torch.tensor(keep_idx, device=anchor.device)]
            pos_tensor = torch.stack([p.view(-1) for p in pos_list], dim=0).to(device)
            neg_tensor = torch.stack([n.view(-1) for n in neg_list], dim=0).to(device)

            loss = triplet(anchor_sel, pos_tensor, neg_tensor)
            optim_all.zero_grad(); loss.backward(); optim_all.step()

            running_loss += loss.item() * anchor_sel.size(0)
            N += anchor_sel.size(0)
            running_margin_terms.append(cosine_margin(anchor_sel.detach(), pos_tensor, neg_tensor))

        train_loss = running_loss / max(N, 1)
        train_margin = float(np.mean(running_margin_terms)) if running_margin_terms else float("nan")

        # ---- Validation ----
        mri_encoder.eval()
        val_running, Nval = 0.0, 0
        val_margins = []

        with torch.no_grad():
            for xb, y_isup in val_loader:
                xb = xb.to(device)
                y_isup = y_isup.numpy().tolist()
                anchor = mri_encoder(xb)

                pos_list, neg_list, keep_idx = [], [], []
                for i, isup in enumerate(y_isup):
                    if isup not in histo_bank_val or len(histo_bank_val[isup]) == 0:
                        continue
                    neg_pool = [c for c in val_classes if c != isup and len(histo_bank_val[c]) > 0]
                    if not neg_pool:
                        continue
                    pos = random.choice(histo_bank_val[isup])
                    neg = random.choice(histo_bank_val[random.choice(neg_pool)])
                    pos_list.append(pos); neg_list.append(neg); keep_idx.append(i)

                if not keep_idx:
                    continue

                anchor_sel = anchor[torch.tensor(keep_idx, device=anchor.device)]
                pos_tensor = torch.stack([p.view(-1) for p in pos_list], dim=0).to(device)
                neg_tensor = torch.stack([n.view(-1) for n in neg_list], dim=0).to(device)

                loss = triplet(anchor_sel, pos_tensor, neg_tensor)
                val_running += loss.item() * anchor_sel.size(0)
                Nval += anchor_sel.size(0)
                val_margins.append(cosine_margin(anchor_sel, pos_tensor, neg_tensor))

        val_loss = val_running / max(Nval, 1)
        val_margin = float(np.mean(val_margins)) if val_margins else float("nan")

        # ---- Prototype top-1 on VAL ----
        proto_top1, proto_per_cls, proto_counts = proto_metrics(mri_encoder, val_loader, histo_bank_val, device)

        # ---- Logging ----
        log_row = {
            "epoch": epoch,
            "train_triplet_loss": train_loss,
            "val_triplet_loss": val_loss,
            "train_cosine_margin": train_margin,
            "val_cosine_margin": val_margin,
            "proto_top1_acc": proto_top1,
        }
        for k in sorted(ALLOWED_ISUPS):
            log_row[f"proto_acc_isup_{k}"]   = proto_per_cls.get(k, float("nan"))
            log_row[f"proto_count_isup_{k}"] = proto_counts.get(k, 0)

        wandb.log(log_row)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} | "
            f"TrainMargin={train_margin:.4f} ValMargin={val_margin:.4f} | "
            f"ProtoTop1={proto_top1:.4f} | Per-ISUP Acc="
            + ", ".join(f"{k}:{log_row[f'proto_acc_isup_{k}']:.4f}" for k in sorted(ALLOWED_ISUPS))
        )

        # ---- Checkpoint on best val triplet loss ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "mri_encoder_state_dict": mri_encoder.state_dict(),
                "best_val_triplet_loss": best_val_loss,
                "projection_dim": args.projection_dim,
            }, ckpt_path)
            logger.info(f"Saved BEST checkpoint (val_triplet_loss={best_val_loss:.4f}) -> {ckpt_path}")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # MRI data
    p.add_argument('--embedding_dir', type=str, required=True,
                   help="MRI .pt encodings named <patient>_<study>_<patch>_<isup>.pt (ISUP in {0,2,3,4,5})")
    p.add_argument('--marksheet_csv', type=str, required=True,
                   help="marksheet.csv with patient_id,study_id,fold")
    p.add_argument('--val_fold', type=str, default='fold0')

    # Histo data (folder-based)
    p.add_argument('--histo_root', type=str, required=True,
                   help="Root folder with subdirs train/, val/ (or test/) containing slide .pt files with key 'h' (384-D).")

    # training
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--log_file', type=str, default=None)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--projection_dim', type=int, default=384,  # match histo 'h'
                   help="MRI projection dim; should be 384 to match histo 'h'.")
    p.add_argument('--seed', type=int, default=SEED_DEFAULT)
    p.add_argument('--device', type=str, default='cuda')   # cuda | mps | cpu
    p.add_argument('--num_workers', type=int, default=4)

    # logging
    p.add_argument('--wandb_name', type=str, default=None)

    args = p.parse_args()
    if args.log_file is None:
        args.log_file = os.path.join(args.save_dir, "triplet_train.log")

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
