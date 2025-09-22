# nnunet_v1_encoder_isup.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Allowlist a legacy NumPy scalar for old nnU-Net checkpoints (PyTorch >= 2.6)
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray.scalar])
except Exception:
    pass


class TopKPool3D(nn.Module):
    """
    Score voxels with a 1x1x1 conv and average top-K features.
    This reduces background dominance vs global average/attention.
    """
    def __init__(self, c_in: int, k: int = 256):
        super().__init__()
        self.score = nn.Conv3d(c_in, 1, 1)
        self.k = k

    def forward(self, Fmap: torch.Tensor) -> torch.Tensor:
        # Fmap: [B,C,D,H,W]
        B, C, D, H, W = Fmap.shape
        s = self.score(Fmap).view(B, -1)                 # [B, DHW]
        k = min(self.k, s.size(1))
        _, idx = torch.topk(s, k=k, dim=1)               # [B, k]
        flat = Fmap.view(B, C, -1)                       # [B, C, DHW]
        idx = idx.unsqueeze(1).expand(-1, C, -1)         # [B, C, k]
        gathered = torch.gather(flat, 2, idx)            # [B, C, k]
        return gathered.mean(dim=2)                      # [B, C]


class Projection384(nn.Module):
    def __init__(self, c_in: int, d_out: int = 384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(c_in), nn.GELU(),
            nn.Linear(c_in, 2 * c_in), nn.GELU(),
            nn.Linear(2 * c_in, d_out),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


def _resolve_model_root_and_fold(p: Path) -> Tuple[Path, int]:
    p = Path(p).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir() and p.name.startswith("fold_"):
        return p.parent, int(p.name.split("_")[-1])
    if p.is_dir() and any((p / f"fold_{i}").exists() for i in range(5)):
        return p, 0
    raise ValueError(f"Path {p} is not a nnU-Net v1 fold dir or its parent.")


class NNUNetV1EncoderISUP(nn.Module):
    """
    Load an nnU-Net v1 Generic_UNet, but run **encoder-only** to avoid ConvTranspose3d.
    Exposes:
      - 384-D embedding via Top-K pooling over the bottleneck
      - ISUP logits from a small classifier head
    """
    def __init__(self, fold_dir: str | Path, in_channels: int = 3, n_isup: int = 6):
        super().__init__()
        from nnunet.training.model_restore import load_model_and_checkpoint_files

        model_root, fold_idx = _resolve_model_root_and_fold(fold_dir)

        # Temporarily force weights_only=False for legacy checkpoints
        _orig_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load
        try:
            try:
                trainer, _ = load_model_and_checkpoint_files(
                    str(model_root), folds=[fold_idx], mixed_precision=False, checkpoint_name="model_best"
                )
            except Exception:
                trainer, _ = load_model_and_checkpoint_files(
                    str(model_root), folds=[fold_idx], mixed_precision=False, checkpoint_name="model_final_checkpoint"
                )
        finally:
            torch.load = _orig_torch_load

        self.net = trainer.network  # Generic_UNet with weights
        # Precompute downsample multiples (Z,Y,X)
        self._factors = self._compute_required_factors()

        # Infer bottleneck channel count without a forward
        try:
            c_in = int(self.net.conv_blocks_context[-1].output_channels)
        except Exception:
            c_in = None
            for m in reversed(list(self.net.conv_blocks_context[-1].modules())):
                if isinstance(m, nn.Conv3d):
                    c_in = m.out_channels; break
            if c_in is None:
                raise RuntimeError("Could not infer bottleneck channels")

        # Heads
        self.pool = TopKPool3D(c_in, k=256)
        self.proj = Projection384(c_in, 384)
        self.isup_head: Optional[nn.Module] = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, n_isup))

    # -------- encoder-only path (no ConvTranspose3d) --------
    def _encode_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only forward that works with or without separate td[]
        (handles convolutional pooling configs where td is empty).
        """
        # number of context stages EXCLUDING bottleneck
        num_context = len(self.net.conv_blocks_context) - 1

        has_td = hasattr(self.net, "td") and isinstance(self.net.td, (nn.ModuleList, list))
        td_len = len(self.net.td) if has_td else 0

        for d in range(num_context):
            # conv path (may include strided conv if convolutional_pooling=True)
            x = self.net.conv_blocks_context[d](x)
            # only apply explicit downsample if td exists for this level
            if has_td and d < td_len:
                x = self.net.td[d](x)

        # bottleneck
        x = self.net.conv_blocks_context[-1](x)
        return x

    def _compute_required_factors(self) -> tuple[int, int, int]:
        pks = getattr(self.net, "pool_op_kernel_sizes", None)
        if pks is None:
            return (8, 32, 32)
        dz = int(np.prod([lvl[0] for lvl in pks])); dy = int(np.prod([lvl[1] for lvl in pks])); dx = int(np.prod([lvl[2] for lvl in pks]))
        return (max(dz, 1), max(dy, 1), max(dx, 1))

    def _pad_to_factors(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = x.shape
        fz, fy, fx = self._factors
        pad_d = (fz - (D % fz)) % fz; pad_h = (fy - (H % fy)) % fy; pad_w = (fx - (W % fx)) % fx
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
        return x

    # -------- public API --------
    def disable_classifier(self): self.isup_head = None
    def reinit_classifier(self, n_isup: int = 6):
        self.isup_head = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, n_isup))

    def freeze_encoder(self, freeze: bool = True, unfreeze_top_k: int = 0):
        num = len(self.net.conv_blocks_context)
        for i, blk in enumerate(self.net.conv_blocks_context):
            train_this = (not freeze) or (i >= num - 1 - unfreeze_top_k)
            for p in blk.parameters(): p.requires_grad = train_this

    def _bottleneck_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_factors(x)
        return self._encode_only(x)   # encoder only → no ConvTranspose3d

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        Fmap = self._bottleneck_feat(x)
        f = self.pool(Fmap)
        z = self.proj(f)
        return z  # [B, 384]

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.get_embedding(x)
        if return_embedding or self.isup_head is None:
            return z
        logits = self.isup_head(z)
        return z, logits
