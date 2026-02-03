# kl_align_utils.py
#
# Utilities for classwise Gaussian KL alignment (diagonal covariance) between
# MRI embeddings and precomputed histopathology embedding distributions.
#
# Saved stats are produced by make_histo_gaussian_stats.py and include:
#   stats["mu"]    : [C, D]
#   stats["var"]   : [C, D]
#   stats["count"] : [C]
#
# Typical usage in training:
#   histo = HistoGaussianStats.load("histo_stats_train.pt", device=device)
#   loss = classwise_diag_gaussian_kl_loss(mri_emb, labels, histo.mu, histo.var, reduction="mean")
#
# Notes:
# - This is NOT KL between probabilities over classes; it's KL between two Gaussians in embedding space.
# - With diagonal covariance, KL is cheap and stable.
# - By default, we estimate an MRI Gaussian per class from the current batch and KL it to histo per class:
#     KL( N(mu_mri_c, var_mri_c) || N(mu_histo_c, var_histo_c) )
#   You can swap direction if you prefer.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch


# -----------------------------
# Loading / holding saved stats
# -----------------------------
@dataclass
class HistoGaussianStats:
    mu: torch.Tensor        # [C, D]
    var: torch.Tensor       # [C, D]  (diagonal variances)
    count: torch.Tensor     # [C]
    num_classes: int
    dim: int
    provider: str = "all"
    eps: float = 1e-6

    @staticmethod
    def load(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> "HistoGaussianStats":
        d = torch.load(path, map_location="cpu")
        mu = d["mu"].to(device=device, dtype=dtype)
        var = d["var"].to(device=device, dtype=dtype)
        count = d["count"].to(device=device)
        eps = float(d.get("eps", 1e-6))
        return HistoGaussianStats(
            mu=mu,
            var=torch.clamp(var, min=eps),
            count=count,
            num_classes=int(d["num_classes"]),
            dim=int(d["dim"]),
            provider=str(d.get("provider", "all")),
            eps=eps,
        )


# -----------------------------
# Core math: KL of diag Gaussians
# -----------------------------
def kl_diag_gaussians(
    mu_q: torch.Tensor,
    var_q: torch.Tensor,
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    KL( q || p ) where
      q = N(mu_q, diag(var_q)), p = N(mu_p, diag(var_p))
    Shapes:
      mu_*  : [..., D]
      var_* : [..., D]
    Returns:
      kl : [...]  (scalar per leading index)
    """
    var_q = torch.clamp(var_q, min=eps)
    var_p = torch.clamp(var_p, min=eps)

    # 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1 )
    log_term = torch.log(var_p) - torch.log(var_q)
    quad = (var_q + (mu_q - mu_p) ** 2) / var_p
    kl = 0.5 * (log_term + quad - 1.0).sum(dim=-1)
    return kl


# -----------------------------
# Batch -> per-class Gaussian estimate
# -----------------------------
def estimate_classwise_diag_gaussian(
    embeddings: torch.Tensor,  # [B, D]
    labels: torch.Tensor,      # [B]
    num_classes: int,
    eps: float = 1e-6,
    unbiased: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate per-class mean and diag variance from a batch.
    Returns:
      mu  : [C, D]
      var : [C, D]
      cnt : [C] (long)
    For classes with cnt=0, mu=0 and var=1.
    """
    assert embeddings.dim() == 2, f"embeddings must be [B,D], got {tuple(embeddings.shape)}"
    B, D = embeddings.shape
    labels = labels.view(-1).long()
    if labels.numel() != B:
        raise ValueError(f"labels must have B elements, got {labels.numel()} vs B={B}")

    device = embeddings.device
    dtype = embeddings.dtype

    mu = torch.zeros((num_classes, D), device=device, dtype=dtype)
    var = torch.ones((num_classes, D), device=device, dtype=dtype)
    cnt = torch.zeros((num_classes,), device=device, dtype=torch.long)

    for c in range(num_classes):
        mask = labels == c
        n = int(mask.sum().item())
        cnt[c] = n
        if n == 0:
            continue
        x = embeddings[mask]  # [n, D]
        mu_c = x.mean(dim=0)
        if n == 1:
            # variance undefined; keep small/stable
            var_c = torch.full((D,), eps, device=device, dtype=dtype)
        else:
            var_c = x.var(dim=0, unbiased=unbiased)
            var_c = torch.clamp(var_c, min=eps)
        mu[c] = mu_c
        var[c] = var_c

    return mu, var, cnt


# -----------------------------
# Loss: classwise Gaussian KL
# -----------------------------
def classwise_diag_gaussian_kl_loss(
    embeddings: torch.Tensor,              # [B, D]
    labels: torch.Tensor,                  # [B]
    histo_mu: torch.Tensor,                # [C, D]
    histo_var: torch.Tensor,               # [C, D]
    *,
    reduction: Literal["mean", "sum"] = "mean",
    direction: Literal["mri_to_histo", "histo_to_mri"] = "mri_to_histo",
    eps: float = 1e-6,
    min_count_per_class: int = 1,
    weight_by_batch_count: bool = True,
    unbiased_batch_var: bool = False,
) -> torch.Tensor:
    """
    Computes a classwise KL loss for the current batch:
      - Estimate per-class (mu_mri_c, var_mri_c) from the batch.
      - Compute KL_c between MRI Gaussian and Histo Gaussian for each class present.
      - Aggregate across classes.

    Args:
      direction:
        "mri_to_histo": KL( N_mri || N_histo )
        "histo_to_mri": KL( N_histo || N_mri )  (sometimes more stable if MRI var is tiny early)
      min_count_per_class:
        Only include classes with at least this many samples in the batch.
      weight_by_batch_count:
        If True, class contribution is weighted by cnt_c (like a micro-average).
        If False, average equally across included classes.
    """
    assert histo_mu.dim() == 2 and histo_var.dim() == 2, "histo_mu/var must be [C,D]"
    C, D = histo_mu.shape
    if embeddings.shape[1] != D:
        raise ValueError(f"Embedding dim mismatch: embeddings D={embeddings.shape[1]} vs histo D={D}")

    mu_mri, var_mri, cnt = estimate_classwise_diag_gaussian(
        embeddings, labels, num_classes=C, eps=eps, unbiased=unbiased_batch_var
    )

    # Select classes present enough in this batch
    keep = cnt >= int(min_count_per_class)
    if keep.sum().item() == 0:
        # nothing to align this step
        return embeddings.new_tensor(0.0)

    mu_h = histo_mu.to(device=embeddings.device, dtype=embeddings.dtype)
    mu_h = torch.nan_to_num(mu_h, nan=0.0, posinf=0.0, neginf=0.0)

    var_h = histo_var.to(device=embeddings.device, dtype=embeddings.dtype)
    var_h = torch.nan_to_num(var_h, nan=eps, posinf=eps, neginf=eps)
    var_h = torch.clamp(var_h, min=eps)

    if direction == "mri_to_histo":
        kl_c = kl_diag_gaussians(mu_mri, var_mri, mu_h, var_h, eps=eps)  # [C]
    elif direction == "histo_to_mri":
        kl_c = kl_diag_gaussians(mu_h, var_h, mu_mri, var_mri, eps=eps)  # [C]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    kl_c = kl_c[keep]
    cnt_keep = cnt[keep].to(dtype=embeddings.dtype)

    if weight_by_batch_count:
        loss = (kl_c * cnt_keep).sum() / torch.clamp(cnt_keep.sum(), min=1.0)
    else:
        loss = kl_c.mean()

    if reduction == "sum":
        return loss * float(keep.sum().item())
    if reduction == "mean":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")


# -----------------------------
# Optional helper: combined loss
# -----------------------------
def combined_ce_and_kl(
    logits: torch.Tensor,
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    w_ce: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    histo: HistoGaussianStats,
    direction: Literal["mri_to_histo", "histo_to_mri"] = "mri_to_histo",
    min_count_per_class: int = 1,
    weight_by_batch_count: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, dict]:
    """
    Convenience for stage-2 experiments where you want CE + KL alignment together.
    Returns: (total_loss, logs_dict)
    """
    ce_fn = torch.nn.CrossEntropyLoss(weight=w_ce)
    ce_loss = ce_fn(logits, labels)

    kl_loss = classwise_diag_gaussian_kl_loss(
        embeddings, labels, histo.mu, histo.var,
        direction=direction,
        min_count_per_class=min_count_per_class,
        weight_by_batch_count=weight_by_batch_count,
        eps=max(eps, histo.eps),
    )

    total = ce_loss + float(kl_weight) * kl_loss
    logs = {"ce": float(ce_loss.item()), "kl": float(kl_loss.item()), "kl_weight": float(kl_weight)}
    return total, logs
