import torch
import torch.nn.functional as F
import torch.nn as nn

class SymmetricInfoNCELoss(nn.Module):
    """
    Expects a dict with:
      - data["mri_feats"] or data["image_feats_needle"] : Tensor [B, D]
      - data["positive_hist"]                            : Tensor [B, D]
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, mri_feats, positive_hist):
        # unify device & dtype to the anchor (MRI) tensor
        device = mri_feats.device
        dtype = mri_feats.dtype
        hist = positive_hist.to(device=device, dtype=dtype, non_blocking=True)

        logits_us_to_hist = (mri_feats @ hist.t()) / self.tau
        logits_hist_to_us = (hist @ mri_feats.t()) / self.tau
        targets = torch.arange(mri_feats.size(0), device=device)
        loss_us = F.cross_entropy(logits_us_to_hist, targets)
        loss_hist = F.cross_entropy(logits_hist_to_us, targets)
        return 0.5 * (loss_us + loss_hist)
