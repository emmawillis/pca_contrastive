import torch
import torch.nn as nn
import torch.nn.functional as F

def resize_and_pad_to_1024(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    if H == 1024 and W == 1024:
        return x
    s = 1024.0 / max(H, W)
    nh, nw = int(round(H * s)), int(round(W * s))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    ph, pw = 1024 - nh, 1024 - nw
    return F.pad(x, (0, pw, 0, ph), value=0.0)

class SpatialAttnPool2d(nn.Module):
    """
    Attention over a feature map (no flattening required).
    Input:  feats [B, C, H, W]
    Output: pooled [B, C], attn [B, 1, H, W] (softmax over H*W)
    """
    def __init__(self, C: int, attn_dim: int = 256, gated: bool = True, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=C)  # LayerNorm over channels is OK too
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
        # weighted average over spatial dims
        pooled = (feats * attn).sum(dim=(2,3))             # [B,C]
        return pooled, attn

class MedSAMSliceSpatialAttn(nn.Module):
    """
    MedSAM/SAM encoder (optionally pre-neck) -> SpatialAttnPool2d -> Linear(proj_dim) -> Classifier.
    Returns a single proj_dim vector per slice (and logits).
    """
    def __init__(self,
                 sam_model,                 # loaded SAM/MedSAM model
                 num_classes: int = 3,
                 proj_dim: int = 128,
                 attn_dim: int = 256,
                 head_hidden: int = 256,
                 head_dropout: float = 0.1,
                 use_pre_neck: bool = True,
                 allow_var_size: bool = False,  # if False, we pad/resize to 1024
                 pixel_mean_std=None):
        super().__init__()
        self.encoder = sam_model.image_encoder
        if use_pre_neck and hasattr(self.encoder, "neck"):
            self.encoder.neck = nn.Identity()

        # probe channel dim C from a dummy
        with torch.no_grad():
            feats = self.encoder(torch.zeros(1,3,1024,1024))
            if feats.dim() != 4:
                raise RuntimeError(f"unexpected encoder output: {feats.shape}")
            C = feats.shape[1] if feats.shape[1] < feats.shape[-1] else feats.shape[-1]
        self.C = C

        self.pool = SpatialAttnPool2d(C=C, attn_dim=attn_dim, gated=True, dropout=head_dropout)

        self.proj = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, proj_dim),
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

        self.allow_var_size = allow_var_size
        self.pixel_mean_std = pixel_mean_std

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
        if not self.allow_var_size:
            x = resize_and_pad_to_1024(x)
        x = self._apply_pixel_norm(x)

        feats = self.encoder(x)                 # [B,C,Hf,Wf] (pre- or post-neck)
        # If channels-last, permute to channels-first
        if feats.shape[1] > feats.shape[-1]:    # rare
            feats = feats.permute(0,3,1,2).contiguous()

        pooled, attn = self.pool(feats)         # [B,C], [B,1,Hf,Wf]
        emb = self.proj(pooled)                 # [B,proj_dim]
        logits = self.head(emb)                 # [B,num_classes]
        return (logits, emb, attn) if return_attn else (logits, emb)
