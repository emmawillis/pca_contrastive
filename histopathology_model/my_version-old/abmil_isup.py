import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMILISUP(nn.Module):
    """
    Gated ABMIL (two-layer gated variant) for slide-level ISUP classification
    using pre-extracted patch features (e.g., UNI/UNI2 1536-D embeddings).

    See https://www.nature.com/articles/s41591-024-02857-3#Sec13 -> evaluation on PANDA WSI-level ISUP classification.

    Matches the UNI Methods config:
      - Input dropout: p = 0.10 on the *input embeddings* X
      - First FC maps input_dim -> 512
      - Gated attention hidden dims = 384 (V and U branches)
      - Dropout p = 0.25 *after each intermediate layer* (V and U branches, and
        the classifier hidden layer)
      - Classifier: 512 -> 384 -> num_classes

    Inputs
    ------
    X    : Tensor [B, N, D] or [N, D]
    mask : Optional[BoolTensor] [B, N], True for valid patches (handles padding)

    Outputs
    -------
    logits : [B, num_classes]
    attn   : [B, N]  (attention weights over patches, sum to 1 over valid N)
    z      : [B, 512] (slide embeddings)
    """
    def __init__(
        self,
        input_dim: int = 1536,   # your UNI features are 1536-D
        num_classes: int = 6,    # ISUP 0-5
        proj_dim: int = 512,     # fixed by paper
        attn_hidden: int = 384,  # fixed by paper
        p_input_dropout: float = 0.10,
        p_mid_dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(p_input_dropout)

        # First FC: input_dim -> 512
        self.proj = nn.Linear(input_dim, proj_dim)

        # Gated attention:
        # a_i ∝ exp( w^T [tanh(V h_i) ⊙ σ(U h_i)] )
        self.attn_V = nn.Linear(proj_dim, attn_hidden)
        self.attn_U = nn.Linear(proj_dim, attn_hidden)
        self.attn_w = nn.Linear(attn_hidden, 1)

        # Dropout p=0.25 after each intermediate layer (here: after V and U branches)
        self.attn_do = nn.Dropout(p_mid_dropout)

        # Classifier: 512 -> 384 -> C   (dropout after hidden)
        self.cls_fc1 = nn.Linear(proj_dim, attn_hidden)
        self.cls_do = nn.Dropout(p_mid_dropout)
        self.cls_fc2 = nn.Linear(attn_hidden, num_classes)

        # Nonlinearities
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None, return_h: bool = False):
        squeeze_back = False
        if X.dim() == 2:  # [N, D] -> [1, N, D]
            X = X.unsqueeze(0)
            squeeze_back = True

        B, N, D = X.shape
        device = X.device
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Dropout on *input embeddings* (p=0.10)
        X = self.input_dropout(X)

        # Project to 512
        H = self.relu(self.proj(X))   # [B, N, 512]

        # Gated attention with hidden=384
        Vh = torch.tanh(self.attn_V(H))           # [B, N, 384]
        Uh = torch.sigmoid(self.attn_U(H))        # [B, N, 384]

        # Dropout after each intermediate layer (V and U branches)
        Vh = self.attn_do(Vh)
        Uh = self.attn_do(Uh)

        A = self.attn_w(Vh * Uh).squeeze(-1)      # [B, N]
        A = A.masked_fill(~mask, torch.finfo(A.dtype).min)  # mask pads
        A = F.softmax(A, dim=1)                   # attention weights over patches

        # Slide embedding z = Σ_i a_i * h_i
        z = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [B, 512]

        # Classifier 512 -> 384 -> C with dropout after hidden
        h_pre = self.relu(self.cls_fc1(z))           # [B, 384]
        h = self.cls_do(h_pre)                        # dropout p=0.25
        logits = self.cls_fc2(h)                  # [B, C]

        if squeeze_back:
            logits = logits.squeeze(0)
            A = A.squeeze(0)
            z = z.squeeze(0)

        if return_h:
            return logits, A, z, h_pre


        return logits, A, z

    @torch.no_grad()
    def embed(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Return only the slide embedding z (useful for downstream analysis/UMAP).
        """
        self.eval()
        logits, attn, z = self.forward(X, mask)
        return z
