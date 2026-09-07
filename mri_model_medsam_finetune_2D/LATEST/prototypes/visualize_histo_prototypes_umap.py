# visualize_histo_prototypes_umap.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap


def get_slide_id_from_path(path: Path, suffix: str = "_abmil512") -> str:
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem.split("_")[0]


def isup_to_3class(y: int) -> int:
    return 0 if y <= 1 else (1 if y <= 3 else 2)


def isup_to_binary(y: int) -> int:
    return 0 if y <= 1 else 1


def map_label(isup: int, num_classes: int) -> int:
    if num_classes == 6:
        return isup
    elif num_classes == 3:
        return isup_to_3class(isup)
    elif num_classes == 2:
        return isup_to_binary(isup)
    else:
        raise ValueError(f"Unsupported num_classes={num_classes}")


def load_embeddings(embeddings_dir, csv_path, num_classes=6, provider="all", filename_suffix="_abmil512"):
    df = pd.read_csv(csv_path)

    if provider != "all":
        df = df[df["data_provider"] == provider].copy()

    fname_to_isup = dict(zip(df["FILENAME"].astype(str), df["isup_grade"].astype(int)))

    X = []
    y = []
    slide_ids = []

    for path in sorted(Path(embeddings_dir).rglob("*.npy")):
        slide_id = get_slide_id_from_path(path, filename_suffix)

        if slide_id not in fname_to_isup:
            continue

        isup = fname_to_isup[slide_id]
        label = map_label(isup, num_classes)

        emb = torch.as_tensor(np.load(path), dtype=torch.float32).squeeze()

        if emb.ndim == 2:
            emb = emb.mean(dim=0)

        emb = F.normalize(emb.view(1, -1), dim=1).squeeze(0)

        X.append(emb.numpy())
        y.append(label)
        slide_ids.append(slide_id)

    return np.stack(X), np.array(y), slide_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--prototypes_path", required=True)
    parser.add_argument("--output_png", default="histo_prototypes_umap.png")
    parser.add_argument("--num_classes", type=int, choices=[2, 3, 6], default=6)
    parser.add_argument("--provider", default="all")
    parser.add_argument("--filename_suffix", default="_abmil512")
    args = parser.parse_args()

    X, y, slide_ids = load_embeddings(
        args.embeddings_dir,
        args.csv_path,
        num_classes=args.num_classes,
        provider=args.provider,
        filename_suffix=args.filename_suffix,
    )

    proto_payload = torch.load(args.prototypes_path, map_location="cpu")
    prototypes = proto_payload["prototypes"].float()
    prototypes = F.normalize(prototypes, dim=1).numpy()

    print("Embeddings:", X.shape)
    print("Prototypes:", prototypes.shape)

    # Fit UMAP on real histology embeddings + prototypes together
    combined = np.concatenate([X, prototypes], axis=0)

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    combined_2d = reducer.fit_transform(combined)

    X_2d = combined_2d[: len(X)]
    proto_2d = combined_2d[len(X):]

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        s=8,
        alpha=0.35,
        cmap="tab10",
    )

    plt.scatter(
        proto_2d[:, 0],
        proto_2d[:, 1],
        c=np.arange(args.num_classes),
        s=250,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        cmap="tab10",
        label="Prototypes",
    )

    for c in range(args.num_classes):
        plt.text(
            proto_2d[c, 0],
            proto_2d[c, 1],
            f" P{c}",
            fontsize=12,
            weight="bold",
        )

    plt.colorbar(scatter, label="ISUP class")
    plt.title("Histopathology embedding space with ISUP prototypes")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300)
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()

'''

python /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/prototypes/visualize_histo_prototypes_umap.py \
  --embeddings_dir /project/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_512D/embeddings_512/train \
  --csv_path /project/aip-medilab/shared/picai/histopathology_encodings/UNI2_splits/train.csv \
  --prototypes_path /project/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_512D/histo_isup6_prototypes.pt \
  --output_png /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/prototypes/histo_isup6_prototypes_umap.png \
  --num_classes 6 \
  --provider all
  
  '''