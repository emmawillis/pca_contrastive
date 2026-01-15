import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# Hard-coded training input size
# (must be a multiple of the ViT patch size; ViT-B/L uses 16)
# ==============================
IMG_SIZE = 256  # e.g., 512, 768, 1024

def resize_to_img_size(x: torch.Tensor) -> torch.Tensor:
    """Resize a batch of images to [B, C, IMG_SIZE, IMG_SIZE]."""
    if x.shape[-2] == IMG_SIZE and x.shape[-1] == IMG_SIZE:
        return x
    return F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

class GlobalPool2d(nn.Module):
    """
    Non-attention pooling over spatial grid (patch map).
    Input:  feats [B, C, H, W]
    Output: pooled [B, D]  where:
      - mode='avg'    -> D = C
      - mode='max'    -> D = C
      - mode='avgmax' -> D = 2C (concat avg + max)
      - mode='gem'    -> D = C (learnable generalized mean pooling)
    """
    def __init__(self, mode: str = "avgmax", gem_p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.mode = mode.lower()
        self.eps = eps

        if self.mode == "gem":
            # Learnable p: larger p -> closer to max; p=1 -> avg
            self.p = nn.Parameter(torch.tensor(float(gem_p_init)))
        else:
            self.p = None

        if self.mode not in {"avg", "max", "avgmax", "gem"}:
            raise ValueError(f"Unknown pool mode: {mode}. Use one of: avg, max, avgmax, gem")

    def out_dim(self, C: int) -> int:
        return 2 * C if self.mode == "avgmax" else C

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, C, H, W]
        if self.mode == "avg":
            pooled = F.adaptive_avg_pool2d(feats, output_size=1).flatten(1)  # [B, C]
            return pooled

        if self.mode == "max":
            pooled = F.adaptive_max_pool2d(feats, output_size=1).flatten(1)  # [B, C]
            return pooled

        if self.mode == "avgmax":
            avg = F.adaptive_avg_pool2d(feats, output_size=1).flatten(1)     # [B, C]
            mx  = F.adaptive_max_pool2d(feats, output_size=1).flatten(1)     # [B, C]
            return torch.cat([avg, mx], dim=1)                                # [B, 2C]

        # mode == "gem"
        # Generalized mean pooling: (mean(x^p))^(1/p)
        p = torch.clamp(self.p, min=1.0, max=10.0)
        x = feats.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return x.pow(1.0 / p)

class MedSAMSliceNoAttn(nn.Module):
    """
    MedSAM/SAM encoder (optionally pre-neck) -> GlobalPool2d -> Linear(proj_dim) -> Classifier.

    This implementation forces a fixed input resolution of IMG_SIZE×IMG_SIZE and
    resizes the encoder's absolute 2D positional embeddings to the corresponding patch grid.
    """
    def __init__(self,
                 sam_model,                 # loaded SAM/MedSAM model
                 num_classes: int = 3,
                 proj_dim: int = 128,
                 head_hidden: int = 256,
                 head_dropout: float = 0.1,
                 use_pre_neck: bool = True,
                 pool_mode: str = "avgmax",  # "avg" | "max" | "avgmax" | "gem"
                 gem_p_init: float = 3.0,
                 pixel_mean_std=None):
        super().__init__()
        assert IMG_SIZE % 16 == 0, "IMG_SIZE must be a multiple of 16 for ViT-B/L patching."
        self.encoder = sam_model.image_encoder
        if use_pre_neck and hasattr(self.encoder, "neck"):
            self.encoder.neck = nn.Identity()

        # Cache native absolute positional embeddings ([1, Gh0, Gw0, C] for SAM ViT).
        with torch.no_grad():
            pe = getattr(self.encoder, "pos_embed", None)
            if pe is None:
                raise AttributeError("Expected self.encoder.pos_embed to exist for SAM/MedSAM ViT.")
            if pe.dim() != 4:
                raise RuntimeError(f"Expected pos_embed to be 4D [1,Gh,Gw,C], got {tuple(pe.shape)}")
            self._pos_native = pe.detach().clone().float().cpu()   # [1,Gh0,Gw0,C]

        # Set positional embedding to match IMG_SIZE before probing output channels
        self._set_pos_embed_for_img_size()

        with torch.no_grad():
            feats = self.encoder(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
            if feats.dim() != 4:
                raise RuntimeError(f"unexpected encoder output: {tuple(feats.shape)}")
            # SAM/MedSAM is channels-first: [B,C,H,W]
            self.C = feats.shape[1]

        self.pool = GlobalPool2d(mode=pool_mode, gem_p_init=gem_p_init)
        pooled_dim = self.pool.out_dim(self.C)

        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, proj_dim),
            nn.GELU(),
        )

        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Linear(proj_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(proj_dim, num_classes)

        self.pixel_mean_std = pixel_mean_std

    # ---- Positional embedding resize to fixed IMG_SIZE ----
    def _set_pos_embed_for_img_size(self):
        """
        Resize absolute 2D positional embeddings to match the patch grid of IMG_SIZE.
        Assumes ViT-B/L 16x16 patch size for SAM/MedSAM encoders.
        """
        enc = self.encoder
        patch = 16
        gh, gw = IMG_SIZE // patch, IMG_SIZE // patch  # target grid

        pe = self._pos_native.to(device=enc.pos_embed.device, dtype=enc.pos_embed.dtype)  # [1,Gh0,Gw0,C]
        gh0, gw0 = pe.shape[1], pe.shape[2]
        if gh0 == gh and gw0 == gw:
            enc.pos_embed = nn.Parameter(pe, requires_grad=False)
            return

        # [1, Gh0, Gw0, C] -> [1, C, Gh0, Gw0] -> interpolate -> [1, gh, gw, C]
        pe_chlast = pe.permute(0, 3, 1, 2)  # [1,C,Gh0,Gw0]
        pe_resized = F.interpolate(pe_chlast, size=(gh, gw), mode="bicubic", align_corners=False)
        pe_resized = pe_resized.permute(0, 2, 3, 1).contiguous()  # [1,gh,gw,C]
        enc.pos_embed = nn.Parameter(pe_resized, requires_grad=False)

    def _apply_pixel_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.pixel_mean_std is None:
            return x
        mean, std = self.pixel_mean_std
        mean = x.new_tensor(mean)[None, :, None, None]
        std  = x.new_tensor(std)[None, :, None, None]
        return (x - mean) / std

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: [B,3,H,W] normalized to [0,1]
        -> logits [B,K], emb [B,proj_dim], (attn is always None now), feats [B,C,Hf,Wf]
        """
        x = resize_to_img_size(x)
        self._set_pos_embed_for_img_size()
        x = self._apply_pixel_norm(x)

        feats = self.encoder(x)           # [B,C,Hf,Wf]
        pooled = self.pool(feats)         # [B,C] or [B,2C]
        emb = self.proj(pooled)           # [B,proj_dim]
        logits = self.head(emb)           # [B,num_classes]

        if return_attn:
            return logits, emb, None, feats
        return logits, emb
