import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from PiCAI_MultiSeq_2D_Bag import MEDSAM_INPUT_SIZE
from segment_anything import sam_model_registry
import torch.nn.functional as F

class ABMILProjector(nn.Module):
    def __init__(self, in_dim=1024, attn_dim=256, emb_dim=128):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
        )
        self.attn_score = nn.Linear(attn_dim, 1)
        self.embed_head = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x):
        # x: [B, N, in_dim]
        attn_h = self.attn_fc(x)                      # [B, N, attn_dim]
        scores = self.attn_score(attn_h).squeeze(-1)  # [B, N]
        weights = torch.softmax(scores, dim=1)        # [B, N]
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, in_dim]
        embedding = self.embed_head(pooled)           # [B, emb_dim]
        return embedding, weights


class ISUPMedSAM(nn.Module):
    """
    MedSAM image encoder + MIL projector + classifier.

    - Supports arbitrary square inputs whose side is a multiple of 16 (e.g., 512, 768, 1024)
      by resizing the encoder’s absolute 2D positional embeddings each forward.
    - Pools the encoder’s spatial feature maps to one vector per slice before MIL.
    """
    def __init__(
        self,
        checkpoint,
        proj_dim=128,
        model_type="vit_b",
        device="cuda"
    ):
        super().__init__()
        # Build SAM/MedSAM and load weights
        sam_model = sam_model_registry[model_type]()
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        sam_model.load_state_dict(state, strict=True)

        self.image_encoder = sam_model.image_encoder
        # Remove the neck so each slice is represented by a single feature vector
        self.image_encoder.neck = nn.Identity()

        # Cache native pos_embed to re-interpolate from (don’t train it)
        with torch.no_grad():
            native = self.image_encoder.pos_embed.detach().clone()
        # keep dtype/device flexible; store as a buffer so it follows .to(device/dtype)
        self.register_buffer("_pos_native", native, persistent=False)

        self.proj_dim = proj_dim
        self.in_dim = 1024 if model_type == 'vit_l' else 768
        self.projector = ABMILProjector(in_dim=self.in_dim, emb_dim=proj_dim)
        self.classifier = nn.Linear(proj_dim, 6)  # 6 ISUP grades (0–5)

    # ---- helpers ----
    def _ensure_pos_embed(self, H: int, W: int):
        """
        Resize absolute 2D positional embeddings to match current patch grid (H/16, W/16).
        Works for inputs like 512, 768, 1024 (multiples of 16).
        """
        enc = self.image_encoder
        # SAM uses 16x16 patch embed for ViT-B/L
        patch = 16
        gh, gw = H // patch, W // patch

        pe = self._pos_native.to(device=enc.pos_embed.device, dtype=enc.pos_embed.dtype)
        gh0, gw0 = pe.shape[1], pe.shape[2]
        if gh0 == gh and gw0 == gw:
            # already matches
            enc.pos_embed = nn.Parameter(pe, requires_grad=False)
            return

        # interpolate channels-last [1, Gh0, Gw0, C] -> [1, gh, gw, C]
        pe_chlast = pe.permute(0, 3, 1, 2)  # [1, C, Gh0, Gw0]
        pe_resized = F.interpolate(pe_chlast, size=(gh, gw), mode="bicubic", align_corners=False)
        pe_resized = pe_resized.permute(0, 2, 3, 1).contiguous()  # [1, gh, gw, C]
        enc.pos_embed = nn.Parameter(pe_resized, requires_grad=False)

    @staticmethod
    def _pool_slice_features(feat: torch.Tensor) -> torch.Tensor:
        """
        Convert encoder output feature map to a single vector per slice.
        Supports:
          - channels-last: [B, gh, gw, C]  -> mean over (gh, gw) -> [B, C]
          - channels-first: [B, C, gh, gw] -> mean over (gh, gw) -> [B, C]
        """
        if feat.ndim != 4:
            raise RuntimeError(f"Unexpected encoder output shape: {feat.shape}")

        # channels-last (SAM pre-neck)
        if feat.shape[-1] in (768, 1024) and feat.shape[1] != feat.shape[-1]:
            return feat.mean(dim=(1, 2))  # [B, C]

        # channels-first (SAM post-neck or if neck is Identity but still called on permuted tensor)
        if feat.shape[1] in (768, 1024):
            return feat.mean(dim=(2, 3))  # [B, C]

        raise RuntimeError(f"Cannot infer layout from encoder output shape {feat.shape}")

    # ---- forward ----

    def forward(
        self,
        image: torch.Tensor,
        chunk_size: int = 0,
    ):
        """
        Encode one case (bag of slices) and aggregate to a single embedding.

        Args:
            image: [N, 3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE]
            chunk_size: how many slices to encode per forward pass to reduce memory.

        Returns:
            logits:           [6]
            pooled_embedding: [proj_dim]
            slice_embeddings: [N, in_dim]
            attn:             [N]  (attention weights over slices)
        """
        assert image.ndim == 4, f"expected [N, 3, {MEDSAM_INPUT_SIZE}, {MEDSAM_INPUT_SIZE}]"
        N, C, H, W = image.shape
        assert (C, H, W) == (3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE), \
            f"got {(C, H, W)}, expected (3, {MEDSAM_INPUT_SIZE}, {MEDSAM_INPUT_SIZE})"

        # Make encoder's positional embeddings match current size (e.g., 512 -> 32x32 grid)
        self._ensure_pos_embed(H, W)

        # Encode in chunks to save memory, then spatially pool to slice vectors
        vecs = []
        if chunk_size <= 0:
            chunk_size = N
        for start in range(0, N, chunk_size):
            chunk = image[start:start + chunk_size]        # [k, 3, H, W]
            feat_map = self.image_encoder(chunk)           # [k, C, gh, gw] or [k, gh, gw, C]
            vecs.append(self._pool_slice_features(feat_map))  # [k, in_dim]
        slice_embeddings = torch.cat(vecs, dim=0)          # [N, in_dim]

        # MIL projector + classifier
        bag_embeddings = slice_embeddings.unsqueeze(0)     # [1, N, in_dim]
        pooled_embedding, attn = self.projector(bag_embeddings)  # [1, proj_dim], [1, N]
        logits = self.classifier(pooled_embedding)         # [1, 6]

        return (
            logits.squeeze(0),                 # [6]
            pooled_embedding.squeeze(0),       # [proj_dim]
            slice_embeddings,                  # [N, in_dim]
            attn.squeeze(0)                    # [N]
        )
