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

class SpatialAttnPool2d(nn.Module):
    """
    Attention over a feature map (no flattening required).
    Input:  feats [B, C, H, W]
    Output: pooled [B, C], attn [B, 1, H, W] (softmax over H*W)
    """
    def __init__(self, C: int, attn_dim: int = 256, gated: bool = True, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=C)
        self.gated = gated
        self.theta = nn.Conv2d(C, attn_dim, kernel_size=1, bias=True)
        self.gate  = nn.Conv2d(C, attn_dim, kernel_size=1, bias=True) if gated else None
        self.score = nn.Conv2d(attn_dim, 1, kernel_size=1, bias=True)
        self.drop  = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor):
        x = self.norm(feats)                               # [B,C,H,W]
        h = torch.tanh(self.theta(x))                      # [B,A,H,W]
        if self.gated:
            h = h * torch.sigmoid(self.gate(x))           # gated tanh
        h = self.drop(h)
        logits = self.score(h)                             # [B,1,H,W]
        attn = torch.softmax(logits.flatten(2), dim=-1)    # [B,1,H*W]
        attn = attn.view(logits.shape)                     # [B,1,H,W]
        pooled = (feats * attn).sum(dim=(2,3))             # [B,C]
        return pooled, attn

class MedSAMSliceSpatialAttn(nn.Module):
    """
    MedSAM/SAM encoder (optionally pre-neck) -> SpatialAttnPool2d -> Linear(proj_dim) -> Classifier.
    Returns a single proj_dim vector per slice (and logits).

    This implementation **forces a fixed input resolution** of IMG_SIZE×IMG_SIZE and
    resizes the encoder's absolute 2D positional embeddings to the corresponding patch grid.
    """
    def __init__(self,
                 sam_model,                 # loaded SAM/MedSAM model
                 num_classes: int = 3,
                 proj_dim: int = 128,
                 attn_dim: int = 256,
                 head_hidden: int = 256,
                 head_dropout: float = 0.1,
                 use_pre_neck: bool = True,
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

        # Set PE to match our fixed IMG_SIZE before probing channels
        self._set_pos_embed_for_img_size()

        # Probe channel dim C using a dummy at IMG_SIZE
        with torch.no_grad():
            feats = self.encoder(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
            if feats.dim() != 4:
                raise RuntimeError(f"unexpected encoder output: {tuple(feats.shape)}")
            # Handle both [B,C,H,W] and channels-last [B,H,W,C] just in case
            C_guess = feats.shape[1] if feats.shape[1] < feats.shape[-1] else feats.shape[-1]
        self.C = C_guess

        self.pool = SpatialAttnPool2d(C=self.C, attn_dim=attn_dim, gated=True, dropout=head_dropout)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.C),
            nn.Linear(self.C, proj_dim),
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
            # Exact match, set and return
            enc.pos_embed = nn.Parameter(pe, requires_grad=False)
            return

        # [1, Gh0, Gw0, C] -> [1, C, Gh0, Gw0] -> interpolate -> [1, gh, gw, C]
        pe_chlast = pe.permute(0, 3, 1, 2)  # [1,C,Gh0,Gw0]
        pe_resized = F.interpolate(pe_chlast, size=(gh, gw), mode="bicubic", align_corners=False)
        pe_resized = pe_resized.permute(0, 2, 3, 1).contiguous()  # [1,gh,gw,C]
        enc.pos_embed = nn.Parameter(pe_resized, requires_grad=False)

    def _apply_pixel_norm(self, x):
        if self.pixel_mean_std is None:
            return x
        mean, std = self.pixel_mean_std
        mean = x.new_tensor(mean)[None,:,None,None]
        std  = x.new_tensor(std)[None,:,None,None]
        return (x - mean) / std

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: [B,3,H,W] normalized to [0,1]
        -> logits [B,K], emb [B,proj_dim], (opt) attn [B,1,Hf,Wf]
        """
        # Force inputs to IMG_SIZE × IMG_SIZE and ensure matching PE
        x = resize_to_img_size(x)
        self._set_pos_embed_for_img_size()

        x = self._apply_pixel_norm(x)

        feats = self.encoder(x)                 # [B,C,Hf,Wf] (pre- or post-neck)
        if feats.shape[1] > feats.shape[-1]:    # in case it's channels-last
            feats = feats.permute(0,3,1,2).contiguous()

        pooled, attn = self.pool(feats)         # [B,C], [B,1,Hf,Wf]
        emb = self.proj(pooled)                 # [B,proj_dim]
        logits = self.head(emb)                 # [B,num_classes]
        return (logits, emb, attn) if return_attn else (logits, emb)
