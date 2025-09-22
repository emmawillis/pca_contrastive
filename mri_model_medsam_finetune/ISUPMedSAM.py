import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
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
        attn_h = self.attn_fc(x)                  # [B, N, attn_dim]
        scores = self.attn_score(attn_h).squeeze(-1)  # [B, N]
        weights = torch.softmax(scores, dim=1)    # [B, N]
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, in_dim]
        embedding = self.embed_head(pooled)       # [B, emb_dim]
        return embedding, weights


class ISUPMedSAM(nn.Module):
    def __init__(
        self,
        checkpoint,
        proj_dim=128,
        model_type="vit_b",
        device="cuda"
    ):
        super().__init__()
        sam_model = sam_model_registry[model_type]()
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        sam_model.load_state_dict(state, strict=True)

        self.image_encoder = sam_model.image_encoder
        # Remove the neck so each slice is represented by a single feature vector
        self.image_encoder.neck = nn.Identity()
        self.proj_dim = proj_dim
        self.in_dim = 1024 if model_type == 'vit_l' else 768
        self.projector = ABMILProjector(in_dim=self.in_dim, emb_dim=proj_dim)
        self.classifier = nn.Linear(proj_dim, 6)  # 6 ISUP grades (0-5)

    # def forward(self, image):
    #     """Encode a bag of slices and aggregate to one feature per case."""
    #     # old shape (batch size, slices, 3 sequences, 1024, 1024)
    #     assert image.ndim == 4
    #     N, C, H, W = image.shape  # (slices, 3 sequences, 1024, 1024)
    #     assert (C, H, W) == (3, 1024, 1024)

    #     # flattened_batch = image.reshape(B * N, 3, 1024, 1024) # => (B*N, feat_dim)
    #     slice_embeddings = self.image_encoder(image)  # (N, feat_dim)
    #     pooled_embedding, attn = self.projector(slice_embeddings)
    #     logits = self.classifier(pooled_embedding)
    #     return logits, pooled_embedding, slice_embeddings, attn

    def forward(
        self,
        image: torch.Tensor,
        chunk_size: int = 1,
    ):
        """Encode one case (bag of slices) and aggregate to a single embedding."""
        assert image.ndim == 4, "expected [N, 3, 1024, 1024]"
        N, C, H, W = image.shape
        assert (C, H, W) == (3, 1024, 1024)


        if chunk_size > 0:
            slice_feats = []
            for start in range(0, N, chunk_size):
                chunk = image[start:start + chunk_size]            # [chunk, 3, 1024, 1024]
                slice_feats.append(self.image_encoder(chunk))
            slice_embeddings = torch.cat(slice_feats, dim=0)        # [N, feat_dim]
            bag_embeddings = slice_embeddings.unsqueeze(0)          # [1, N, feat_dim]
            pooled_embedding, attn = self.projector(bag_embeddings) # [1, proj_dim], [1, N]
            pooled_embedding = pooled_embedding.squeeze(0)          # [proj_dim]
            logits = self.classifier(pooled_embedding.unsqueeze(0)) # [1, 6]

            return (
                logits.squeeze(0),                 # [6]
                pooled_embedding,                  # [proj_dim]
                slice_embeddings,                  # [N, feat_dim]
                attn.squeeze(0)                    # [N]
            )

        else:
            slice_embeddings = self.image_encoder(image)  # (N, feat_dim)
            pooled_embedding, attn = self.projector(slice_embeddings)
            logits = self.classifier(pooled_embedding)
            return logits, pooled_embedding, slice_embeddings, attn


