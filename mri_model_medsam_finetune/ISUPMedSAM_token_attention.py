import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from PiCAI_MultiSeq_2D_Bag import MEDSAM_INPUT_SIZE


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
        """
        x: [B, N, in_dim] where N = #slices
        Returns:
          embedding: [B, emb_dim]
          weights:   [B, N]   (slice attention)
        """
        attn_h = self.attn_fc(x)                       # [B,N,attn_dim]
        scores = self.attn_score(attn_h).squeeze(-1)   # [B,N]
        weights = torch.softmax(scores, dim=1)         # [B,N]
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B,in_dim]
        embedding = self.embed_head(pooled)            # [B,emb_dim]
        return embedding, weights


class TokenAttnPool(nn.Module):
    """
    Attention over tokens WITHIN a slice (gated attention):
      tokens: [B, T, C]  ->  pooled: [B, C], weights: [B, T]

    This preserves small, localized signals (tiny lesions) better than mean pooling.
    """
    def __init__(self, in_dim, attn_dim=128, gated=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, attn_dim)
        self.gated = gated
        self.gate = nn.Linear(in_dim, attn_dim) if gated else None
        self.a = nn.Linear(attn_dim, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens):  # tokens: [B, T, C]
        h = self.tanh(self.fc(tokens))                # [B,T,attn_dim]
        if self.gated:
            h = h * self.sigmoid(self.gate(tokens))  # gated-tanh
        w = torch.softmax(self.a(h).squeeze(-1), dim=1)  # [B,T]
        z = torch.bmm(w.unsqueeze(1), tokens).squeeze(1)  # [B,C]
        return z, w


class ISUPMedSAM(nn.Module):
    """
    MedSAM image encoder + token-level attention (within-slice) + MIL over slices + classifier.

    - Supports arbitrary square inputs whose side is a multiple of 16 (e.g., 256, 512, 768, 1024)
      by resizing the encoder’s absolute 2D positional embeddings per forward.
    - Per-slice descriptor is obtained by attention over patch tokens (not mean pooling).
    - ABMIL then attends over slice descriptors to form a case embedding.
    """
    def __init__(self, checkpoint, proj_dim=128, model_type="vit_b", device="cuda"):
        super().__init__()

        # Build SAM/MedSAM and load weights
        sam_model = sam_model_registry[model_type]()
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        sam_model.load_state_dict(state, strict=True)

        self.image_encoder = sam_model.image_encoder
        # Remove neck (keep raw encoder map); encoder outputs a spatial grid of tokens
        self.image_encoder.neck = nn.Identity()

        # Cache native 2D pos_embed grid (stored as buffer so it tracks .to())
        with torch.no_grad():
            native = self.image_encoder.pos_embed.detach().clone()
        self.register_buffer("_pos_native", native, persistent=False)

        self.in_dim = 1024 if model_type == "vit_l" else 768
        self.proj_dim = proj_dim

        # Token-level (intra-slice) attention pooling
        self.token_pool = TokenAttnPool(self.in_dim, attn_dim=128, gated=True)

        # Slice-level MIL
        self.projector = ABMILProjector(in_dim=self.in_dim, emb_dim=proj_dim)
        self.classifier = nn.Linear(proj_dim, 6)  # ISUP 0..5

        # For optional inspection (list of [k, T] per processed chunk)
        self._last_token_weights = []

    # ---- helpers ----

    def _ensure_pos_embed(self, H: int, W: int):
        """
        Resize absolute 2D positional embeddings to match current patch grid (H/16, W/16).
        - If the target shape equals current pos_embed shape, copy in-place (no new Parameter).
        - If shapes differ, REASSIGN a new nn.Parameter with the resized tensor.
        """
        enc = self.image_encoder
        patch = 16
        gh, gw = H // patch, W // patch

        # Native grid (buffer)
        pe_native = self._pos_native.to(device=enc.pos_embed.device, dtype=enc.pos_embed.dtype)

        def _resize_to(pe_src, gh, gw):
            # pe_src: [1, Gh0, Gw0, C]  ->  [1, gh, gw, C]
            pe_chlast = pe_src.permute(0, 3, 1, 2)  # [1,C,Gh0,Gw0]
            pe_resized = F.interpolate(pe_chlast, size=(gh, gw), mode="bicubic", align_corners=False)
            return pe_resized.permute(0, 2, 3, 1).contiguous()  # [1,gh,gw,C]

        # Compute desired target grid
        gh0, gw0 = pe_native.shape[1], pe_native.shape[2]
        pe_target = pe_native if (gh0 == gh and gw0 == gw) else _resize_to(pe_native, gh, gw)

        # If shape matches current parameter, copy; otherwise reassign a new Parameter
        if tuple(enc.pos_embed.shape) == tuple(pe_target.shape):
            with torch.no_grad():
                enc.pos_embed.data.copy_(pe_target)
        else:
            enc.pos_embed = nn.Parameter(pe_target, requires_grad=False)

    def _features_to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert encoder output to tokens [B, T, C].
        Supports:
          - channels-first: [B, C, gh, gw]
          - channels-last:  [B, gh, gw, C]
        """
        if feat.ndim != 4:
            raise RuntimeError(f"Unexpected encoder output shape: {feat.shape}")

        # to [B, C, gh, gw]
        if feat.shape[1] in (768, 1024):
            x = feat
        elif feat.shape[-1] in (768, 1024):
            x = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f"Cannot infer layout from encoder output shape {feat.shape}")

        B, C, gh, gw = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B, T=gh*gw, C]
        return tokens

    # ---- forward ----

    def forward(self, image: torch.Tensor, chunk_size: int = 0):
        """
        Encode one case (bag of slices) and aggregate to a single embedding.

        Args:
            image: [N, 3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE]
            chunk_size: how many slices to encode per forward pass to reduce memory.
                        If <= 0, processes all N slices at once.

        Returns:
            logits:           [1, 6]
            pooled_embedding: [proj_dim]
            slice_embeddings: [N, in_dim]     (after token-attention pooling per slice)
            attn:             [N]             (ABMIL attention over slices)
        """
        assert image.ndim == 4, f"expected [N, 3, {MEDSAM_INPUT_SIZE}, {MEDSAM_INPUT_SIZE}]"
        N, C, H, W = image.shape
        assert (C, H, W) == (3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE), \
            f"got {(C, H, W)}, expected (3, {MEDSAM_INPUT_SIZE}, {MEDSAM_INPUT_SIZE})"

        # Match positional embeddings to current size (e.g., 256 -> 16x16 grid)
        self._ensure_pos_embed(H, W)

        # Encode in chunks (optional), do token-attention within each slice
        if chunk_size <= 0:
            chunk_size = N

        self._last_token_weights = []  # clear debug buffer
        slice_vecs = []

        for start in range(0, N, chunk_size):
            chunk = image[start:start + chunk_size]        # [k, 3, H, W]
            feat_map = self.image_encoder(chunk)           # [k,C,gh,gw] or [k,gh,gw,C]
            tokens = self._features_to_tokens(feat_map)    # [k, T, C]
            z, w_tok = self.token_pool(tokens)             # z: [k,C], w_tok: [k,T]
            slice_vecs.append(z)
            # store token weights for optional inspection
            self._last_token_weights.append(w_tok.detach())

        slice_embeddings = torch.cat(slice_vecs, dim=0)     # [N, in_dim]

        # MIL projector over slices + classifier
        bag_embeddings = slice_embeddings.unsqueeze(0)      # [1, N, in_dim]
        pooled_embedding, attn_slices = self.projector(bag_embeddings)  # [1,proj_dim], [1,N]
        logits = self.classifier(pooled_embedding)          # [1, 6]

        return (
            logits,                        # [1,6]
            pooled_embedding.squeeze(0),   # [proj_dim]
            slice_embeddings,              # [N, in_dim]
            attn_slices.squeeze(0),        # [N]
        )
