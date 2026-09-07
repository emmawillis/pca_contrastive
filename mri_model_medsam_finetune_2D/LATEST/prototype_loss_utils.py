# prototype_loss_utils.py

from pathlib import Path
import torch
import torch.nn.functional as F


def load_histo_prototypes(
    prototype_path: str | Path,
    device: str | torch.device = "cuda",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads prototype tensor from a .pt file produced by compute_histo_prototypes_npy.py.

    Expected payload:
      {
        "prototypes": Tensor [K, D],
        "counts": Tensor [K],
        ...
      }

    Returns:
      prototypes: Tensor [K, D]
    """
    payload = torch.load(prototype_path, map_location="cpu")

    if isinstance(payload, dict):
        if "prototypes" not in payload:
            raise KeyError(f"'prototypes' not found in {prototype_path}. Keys: {list(payload.keys())}")
        prototypes = payload["prototypes"]
    elif isinstance(payload, torch.Tensor):
        prototypes = payload
    else:
        raise TypeError(f"Unsupported prototype file format: {type(payload)}")

    prototypes = prototypes.float().to(device)

    if normalize:
        prototypes = F.normalize(prototypes, dim=1)

    return prototypes


def prototype_alignment_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
    normalize_embeddings: bool = True,
    normalize_prototypes: bool = True,
) -> torch.Tensor:
    """
    Align MRI embeddings to histology class prototypes.

    embeddings: [B, D]
    labels:     [B]
    prototypes: [K, D]

    Uses cosine-similarity logits:
      logits[b, c] = cos(mri_embedding_b, prototype_c) / temperature

    Then applies cross-entropy so each MRI embedding is pulled toward its
    grade-matched histology prototype and pushed away from other prototypes.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [B, D], got {tuple(embeddings.shape)}")

    if prototypes.ndim != 2:
        raise ValueError(f"Expected prototypes [K, D], got {tuple(prototypes.shape)}")

    if embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: MRI embeddings have D={embeddings.shape[1]}, "
            f"but prototypes have D={prototypes.shape[1]}"
        )

    labels = labels.long()

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, dim=1)

    if normalize_prototypes:
        prototypes = F.normalize(prototypes, dim=1)

    logits = embeddings @ prototypes.T
    logits = logits / temperature

    return F.cross_entropy(logits, labels)


@torch.no_grad()
def prototype_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
    normalize_embeddings: bool = True,
    normalize_prototypes: bool = True,
) -> float:
    """
    Optional helper for debugging: classifies each embedding by nearest prototype.
    """
    labels = labels.long()

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, dim=1)

    if normalize_prototypes:
        prototypes = F.normalize(prototypes, dim=1)

    logits = embeddings @ prototypes.T
    logits = logits / temperature
    preds = logits.argmax(dim=1)

    return float((preds == labels).float().mean().item())