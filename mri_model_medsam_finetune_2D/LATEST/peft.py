# peft.py
#
# Minimal PEFT utilities for SAM/MedSAM ViT encoder.
# Provides:
#   - LoRALinear: wraps nn.Linear with a low-rank update (base frozen)
#   - apply_adapters_to_sam_encoder: in-place LoRA injection (train.py expects this)
#   - get_adapter_params: returns ONLY adapter params (train.py expects this)
#   - freeze_encoder_except_adapters: optional helper
#
# This file is designed to work with your train.py "adaptor" stage2_scope.

import math
from typing import Iterable, Sequence, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a trainable low-rank update:
      y = xW^T + (alpha/r) * ( (x A^T) B^T )
    Base W is frozen; A and B are trainable.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        # LoRA params
        self.A = nn.Parameter(torch.empty(self.r, in_features))    # [r, in]
        self.B = nn.Parameter(torch.zeros(out_features, self.r))   # [out, r]

        # Init: A ~ Kaiming, B = 0 => starts as exact base layer
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # self.B already zeros

        # Freeze base weights/bias
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        x_d = self.drop(x)
        # (x @ A.T) -> [*, r]; then @ B.T -> [*, out]
        y_lora = (x_d @ self.A.t()) @ self.B.t()
        return y0 + self.scaling * y_lora


def _iter_vit_blocks(encoder: nn.Module):
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise AttributeError("Encoder has no .blocks (expected SAM/MedSAM ViT image_encoder).")
    return blocks


def apply_lora_to_sam_encoder(
    encoder: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target: Sequence[str] = ("qkv", "proj"),
    last_n_blocks: Optional[int] = None,
) -> int:
    """
    Mutates encoder in-place: replaces selected nn.Linear layers with LoRALinear.
    Designed for SAM/MedSAM ViT encoder blocks.
    Returns number of layers replaced.
    """
    blocks = _iter_vit_blocks(encoder)
    n = len(blocks)
    start = 0 if last_n_blocks is None else max(0, n - int(last_n_blocks))

    replaced = 0
    for i in range(start, n):
        blk = blocks[i]

        # Attention
        if hasattr(blk, "attn"):
            attn = blk.attn
            if "qkv" in target and hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = LoRALinear(attn.qkv, r=r, alpha=alpha, dropout=dropout)
                replaced += 1
            if "proj" in target and hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                attn.proj = LoRALinear(attn.proj, r=r, alpha=alpha, dropout=dropout)
                replaced += 1

        # Optional MLP
        if hasattr(blk, "mlp") and "mlp" in target:
            mlp = blk.mlp
            for name in ("lin1", "lin2", "fc1", "fc2"):
                if hasattr(mlp, name) and isinstance(getattr(mlp, name), nn.Linear):
                    setattr(mlp, name, LoRALinear(getattr(mlp, name), r=r, alpha=alpha, dropout=dropout))
                    replaced += 1

    return replaced


def freeze_encoder_except_lora(encoder: nn.Module) -> None:
    """
    Freeze *everything* in encoder, then unfreeze only LoRA params (A and B).
    """
    for p in encoder.parameters():
        p.requires_grad = False
    for m in encoder.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True


# -------------------------------------------------------------------------
# train.py compatibility (expects "adapters" naming)
# -------------------------------------------------------------------------

def apply_adapters_to_sam_encoder(
    encoder: nn.Module,
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target: Sequence[str] = ("qkv", "proj"),
    last_n_blocks: Optional[int] = None,
) -> int:
    """
    train.py hook. Applies LoRA adapters to the SAM/MedSAM ViT encoder in-place.

    You can call this at model build time; if you only actually train adapters in stage2,
    that's handled by optimizer selection + requires_grad in train.py.
    """
    return apply_lora_to_sam_encoder(
        encoder=encoder,
        r=r,
        alpha=alpha,
        dropout=dropout,
        target=tuple(target),
        last_n_blocks=last_n_blocks,
    )


def get_adapter_params(encoder: nn.Module) -> Iterable[nn.Parameter]:
    """
    train.py hook. Returns ONLY adapter parameters (LoRA A and B matrices),
    suitable for passing as a param group to the optimizer.
    """
    for m in encoder.modules():
        if isinstance(m, LoRALinear):
            # Yield actual nn.Parameters (not tensors)
            yield m.A
            yield m.B


def freeze_encoder_except_adapters(encoder: nn.Module) -> None:
    """
    Optional convenience: same semantics as freeze_encoder_except_lora,
    but matches the 'adapter' wording used in train.py.
    """
    freeze_encoder_except_lora(encoder)
