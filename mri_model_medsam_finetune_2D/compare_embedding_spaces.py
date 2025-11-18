#!/usr/bin/env python3
# compare_embedding_spaces.py
# Quantitative alignment report for histopathology vs MRI encodings.

import argparse
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import sqrtm, orthogonal_procrustes
from scipy.stats import spearmanr

# ---------------------- Utilities ----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)

def load_pt(path, x_key=None, y_key=None):
    """
    Load a .pt that may be:
      - a dict with keys for embeddings and labels
      - a plain tensor for embeddings (then labels must be present via y_key or error)
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # Try common defaults if keys not specified
        if x_key is None:
            for cand in ["embeddings", "features", "X", "feats"]:
                if cand in obj:
                    x_key = cand
                    break
        if y_key is None:
            for cand in ["labels", "y", "label", "targets"]:
                if cand in obj:
                    y_key = cand
                    break
        if x_key is None:
            raise KeyError(f"Couldn't infer embeddings key in dict from {path}. "
                           f"Pass --histo-x-key/--mri-x-key.")
        X = obj[x_key]
        y = obj[y_key] if y_key is not None else None
    elif torch.is_tensor(obj):
        X = obj
        y = None
        if y_key is not None:
            raise ValueError("y_key provided but file is a raw tensor; labels must be inside the .pt dict.")
    else:
        raise TypeError(f"Unsupported .pt content type: {type(obj)}")

    X = X.detach().cpu().numpy().astype(np.float64)
    if y is not None:
        y = np.array(y, dtype=np.int64)
    return X, y

def ensure_same_class_set(y_h, y_m):
    s_h, s_m = set(np.unique(y_h)), set(np.unique(y_m))
    if s_h != s_m:
        inter = sorted(list(s_h.intersection(s_m)))
        if len(inter) == 0:
            raise ValueError(f"No overlapping class labels between sets. histo={s_h}, mri={s_m}")
        print(f"[warn] Class sets differ; restricting to intersection {inter}")
        mask_h = np.isin(y_h, inter)
        mask_m = np.isin(y_m, inter)
        return mask_h, mask_m, np.array(inter)
    else:
        return np.ones_like(y_h, dtype=bool), np.ones_like(y_m, dtype=bool), np.unique(y_h)

def class_centroids(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    cents = []
    for c in classes:
        Xi = X[y == c]
        if len(Xi) == 0:
            raise ValueError(f"No samples for class {c}")
        cents.append(Xi.mean(0))
    return np.stack(cents)

def centroid_cosines(c_h: np.ndarray, c_m: np.ndarray) -> np.ndarray:
    num = (c_h * c_m).sum(1)
    den = np.linalg.norm(c_h, axis=1) * np.linalg.norm(c_m, axis=1)
    return num / np.clip(den, 1e-12, None)

def interclass_structure_corr(c_h: np.ndarray, c_m: np.ndarray, metric="cosine") -> float:
    Dh = squareform(pdist(c_h, metric=metric))
    Dm = squareform(pdist(c_m, metric=metric))
    mask = ~np.eye(Dh.shape[0], dtype=bool)
    rho, _ = spearmanr(Dh[mask], Dm[mask])
    return float(rho)

def procrustes_disparity(c_h: np.ndarray, c_m: np.ndarray) -> float:
    R, scale = orthogonal_procrustes(c_h, c_m)
    num = np.linalg.norm(c_h @ R - c_m, 'fro')**2
    den = np.linalg.norm(c_h, 'fro')**2 + 1e-12
    return float(num / den)

def gaussian_fit(X: np.ndarray, shrink: float = 1e-6):
    mu = X.mean(0)
    cov = np.cov(X.T)
    # add shrinkage on diagonal for stability
    cov = cov + shrink * np.eye(cov.shape[0], dtype=cov.dtype)
    return mu, cov

def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6) -> float:
    diff = mu1 - mu2
    cov_prod_sqrt = sqrtm(cov1 @ cov2 + eps * np.eye(cov1.shape[0]))
    cov_prod_sqrt = cov_prod_sqrt.real if np.iscomplexobj(cov_prod_sqrt) else cov_prod_sqrt
    d = diff @ diff + np.trace(cov1 + cov2 - 2 * cov_prod_sqrt)
    return float(np.real(d))

def _median_bandwidth(Z: np.ndarray) -> float:
    # Use pooled sample pairwise distances (subsample if huge)
    n = len(Z)
    max_pairs = 200000
    if n*(n-1)//2 <= max_pairs:
        d2 = squareform(pdist(Z, 'sqeuclidean'))
        med = np.median(d2[d2 > 0])
    else:
        # Monte Carlo sample of pairs
        idx_i = np.random.randint(0, n, size=max_pairs)
        idx_j = np.random.randint(0, n, size=max_pairs)
        mask = idx_i != idx_j
        diffs = Z[idx_i[mask]] - Z[idx_j[mask]]
        d2s = np.sum(diffs * diffs, axis=1)
        d2s = d2s[d2s > 0]
        med = np.median(d2s)
    return 1.0 / (med + 1e-12)

def mmd2_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None, approx_pairs: int = 300000) -> float:
    """
    Unbiased Maximum Mean Discrepancy (MMD) with an RBF kernel.
    Uses exact computation if small; else Monte Carlo approximation using random pairs.
    """
    n, m = len(X), len(Y)
    if gamma is None:
        gamma = _median_bandwidth(np.vstack([X, Y]))

    def k_dot(a, b):
        # returns exp(-gamma * ||a-b||^2)
        d2 = np.sum((a - b) ** 2, axis=1)
        return np.exp(-gamma * d2)

    # If small, compute exact (O(n^2) memory/time)
    if n <= 2000 and m <= 2000:
        from itertools import combinations
        # XX
        s_xx = []
        for i, j in combinations(range(n), 2):
            s_xx.append(math.exp(-gamma * np.sum((X[i] - X[j])**2)))
        # YY
        s_yy = []
        for i, j in combinations(range(m), 2):
            s_yy.append(math.exp(-gamma * np.sum((Y[i] - Y[j])**2)))
        # XY
        XY = cdist(X, Y, metric='sqeuclidean')
        s_xy = np.exp(-gamma * XY).mean()
        mmd2 = (2 * np.mean(s_xx)) + (2 * np.mean(s_yy)) - 2 * s_xy
        return float(mmd2)

    # Approximate for large sets
    pairs_xx = min(approx_pairs // 2, n*(n-1)//2)
    pairs_yy = min(approx_pairs // 2, m*(m-1)//2)
    idx_x1 = np.random.randint(0, n, size=pairs_xx)
    idx_x2 = np.random.randint(0, n, size=pairs_xx)
    mask = idx_x1 != idx_x2
    s_xx = k_dot(X[idx_x1[mask]], X[idx_x2[mask]]).mean()

    idx_y1 = np.random.randint(0, m, size=pairs_yy)
    idx_y2 = np.random.randint(0, m, size=pairs_yy)
    mask = idx_y1 != idx_y2
    s_yy = k_dot(Y[idx_y1[mask]], Y[idx_y2[mask]]).mean()

    # XY
    pairs_xy = approx_pairs
    idx_x = np.random.randint(0, n, size=pairs_xy)
    idx_y = np.random.randint(0, m, size=pairs_xy)
    s_xy = k_dot(X[idx_x], Y[idx_y]).mean()

    mmd2 = (s_xx + s_yy) - 2 * s_xy
    return float(mmd2)

def knn_transfer(X_train, y_train, X_test, y_test, k=5, metric='cosine'):
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    return float(accuracy_score(y_test, yhat)), float(balanced_accuracy_score(y_test, yhat))

def prototype_transfer(X_src, y_src, X_tgt, y_tgt, classes, metric='cosine'):
    cents = class_centroids(X_src, y_src, classes)
    if metric == 'cosine':
        # cosine similarity -> argmax
        Xn = l2_normalize(X_tgt)
        Cn = l2_normalize(cents)
        sims = Xn @ Cn.T
        yhat = classes[np.argmax(sims, axis=1)]
    else:
        D = cdist(X_tgt, cents, metric=metric)
        yhat = classes[np.argmin(D, axis=1)]
    return float(accuracy_score(y_tgt, yhat)), float(balanced_accuracy_score(y_tgt, yhat))

# ---------------------- Main report ----------------------

def build_report(X_h, y_h, X_m, y_m, normalize=True, metric="cosine",
                 per_class_metrics=False, k=5, seed=42):
    set_seed(seed)
    if normalize:
        X_h = l2_normalize(X_h)
        X_m = l2_normalize(X_m)

    mask_h, mask_m, classes = ensure_same_class_set(y_h, y_m)
    X_h, y_h = X_h[mask_h], y_h[mask_h]
    X_m, y_m = X_m[mask_m], y_m[mask_m]

    # Class centroids & geometric measures
    C_h = class_centroids(X_h, y_h, classes)
    C_m = class_centroids(X_m, y_m, classes)
    centroid_cos = centroid_cosines(C_h, C_m)
    interclass_rho = interclass_structure_corr(C_h, C_m, metric="cosine")
    proc_disp = procrustes_disparity(C_h, C_m)

    # Distribution-level similarity (overall)
    mu_h, cov_h = gaussian_fit(X_h)
    mu_m, cov_m = gaussian_fit(X_m)
    frechet_all = frechet_distance(mu_h, cov_h, mu_m, cov_m)
    mmd2_all = mmd2_rbf(X_h, X_m)

    # Transfer tests
    acc_knn_h2m, bacc_knn_h2m = knn_transfer(X_h, y_h, X_m, y_m, k=k, metric=metric)
    acc_knn_m2h, bacc_knn_m2h = knn_transfer(X_m, y_m, X_h, y_h, k=k, metric=metric)

    acc_proto_h2m, bacc_proto_h2m = prototype_transfer(X_h, y_h, X_m, y_m, classes, metric=metric)
    acc_proto_m2h, bacc_proto_m2h = prototype_transfer(X_m, y_m, X_h, y_h, classes, metric=metric)

    report = {
        "notes": {
            "normalization": "L2 unit-length per vector" if normalize else "none",
            "distance_metric": metric,
            "seed": seed,
            "acronyms": {
                "kNN": "k-Nearest Neighbors",
                "MMD": "Maximum Mean Discrepancy",
                "RBF": "Radial Basis Function",
            }
        },
        "classes": classes.tolist(),
        "geometry": {
            "centroid_cosine_per_class": dict(zip([int(c) for c in classes], [float(v) for v in centroid_cos])),
            "interclass_spearman_rho": float(interclass_rho),
            "procrustes_disparity": float(proc_disp)
        },
        "distribution": {
            "frechet_distance_overall": float(frechet_all),
            "mmd2_rbf_overall": float(mmd2_all)
        },
        "transfer": {
            "knn_histopathology_to_mri": {"accuracy": acc_knn_h2m, "balanced_accuracy": bacc_knn_h2m},
            "knn_mri_to_histopathology": {"accuracy": acc_knn_m2h, "balanced_accuracy": bacc_knn_m2h},
            "prototype_histopathology_to_mri": {"accuracy": acc_proto_h2m, "balanced_accuracy": bacc_proto_h2m},
            "prototype_mri_to_histopathology": {"accuracy": acc_proto_m2h, "balanced_accuracy": bacc_proto_m2h},
        }
    }

    # Optional per-class FID-style Fréchet and MMD
    if per_class_metrics:
        percls = {}
        for c in classes:
            Xh_c = X_h[y_h == c]
            Xm_c = X_m[y_m == c]
            if len(Xh_c) < 2 or len(Xm_c) < 2:
                percls[int(c)] = {"frechet": None, "mmd2_rbf": None, "n_histo": int(len(Xh_c)), "n_mri": int(len(Xm_c))}
                continue
            mu_hc, cov_hc = gaussian_fit(Xh_c)
            mu_mc, cov_mc = gaussian_fit(Xm_c)
            fre_c = frechet_distance(mu_hc, cov_hc, mu_mc, cov_mc)
            mmd_c = mmd2_rbf(Xh_c, Xm_c)
            percls[int(c)] = {
                "frechet": float(fre_c),
                "mmd2_rbf": float(mmd_c),
                "n_histo": int(len(Xh_c)),
                "n_mri": int(len(Xm_c))
            }
        report["distribution_per_class"] = percls

    return report

# ---------------------- CLI ----------------------

def main():
    p = argparse.ArgumentParser(description="Compare similarity between histopathology and MRI embedding spaces.")
    p.add_argument("--histo-pt", required=True, help=".pt file for histopathology encodings (dict or tensor)")
    p.add_argument("--mri-pt", required=True, help=".pt file for MRI encodings (dict or tensor)")
    p.add_argument("--histo-x-key", default=None, help="Key for embeddings in histopathology .pt dict")
    p.add_argument("--histo-y-key", default=None, help="Key for labels in histopathology .pt dict")
    p.add_argument("--mri-x-key", default=None, help="Key for embeddings in MRI .pt dict")
    p.add_argument("--mri-y-key", default=None, help="Key for labels in MRI .pt dict")
    p.add_argument("--normalize", action="store_true", help="L2-normalize vectors before metrics")
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"], help="Distance metric for kNN/prototypes")
    p.add_argument("--k", type=int, default=5, help="k for k-Nearest Neighbors (kNN)")
    p.add_argument("--per-class-metrics", action="store_true", help="Also compute per-class Fréchet and MMD")
    p.add_argument("--json-out", default=None, help="Path to save JSON report")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X_h, y_h = load_pt(args.histo_pt, x_key=args.histo_x_key, y_key=args.histo_y_key)
    X_m, y_m = load_pt(args.mri_pt, x_key=args.mri_x_key, y_key=args.mri_y_key)
    if y_h is None or y_m is None:
        raise ValueError("Labels were not found. Provide .pt dicts that include labels or specify the label keys.")

    report = build_report(
        X_h, y_h, X_m, y_m,
        normalize=args.normalize,
        metric=args.metric,
        per_class_metrics=args.per_class_metrics,
        k=args.k,
        seed=args.seed
    )

    # Pretty print
    print("\n=== Alignment Report (Histopathology vs MRI) ===")
    print(f"- Normalization: {'L2' if args.normalize else 'none'}")
    print(f"- Distance metric: {args.metric}")
    print("\n[Geometry]")
    cc = report["geometry"]["centroid_cosine_per_class"]
    for k,v in cc.items():
        print(f"  Centroid cosine (class {k}): {v:.4f}")
    print(f"  Inter-class Spearman rank correlation (centroid distances): {report['geometry']['interclass_spearman_rho']:.4f}")
    print(f"  Orthogonal Procrustes disparity (centroids): {report['geometry']['procrustes_disparity']:.6f}")

    print("\n[Distribution]")
    print(f"  Fréchet distance (overall): {report['distribution']['frechet_distance_overall']:.6f}")
    print(f"  Maximum Mean Discrepancy MMD^2 (RBF, overall): {report['distribution']['mmd2_rbf_overall']:.6f}")

    if args.per_class_metrics and "distribution_per_class" in report:
        print("  Per-class Fréchet and MMD^2:")
        for c, d in report["distribution_per_class"].items():
            fre_str = "None" if d["frechet"] is None else f"{d['frechet']:.6f}"
            mmd_str = "None" if d["mmd2_rbf"] is None else f"{d['mmd2_rbf']:.6f}"
            print(
                f"    class {c} (n_histo={d['n_histo']}, n_mri={d['n_mri']}): "
                f"Fréchet={fre_str}  MMD^2={mmd_str}"
            )

    print("\n[Cross-domain transfer]")
    t = report["transfer"]
    print(f"  k-Nearest Neighbors (kNN) histopathology→MRI: "
          f"accuracy={t['knn_histopathology_to_mri']['accuracy']:.4f}, "
          f"balanced_accuracy={t['knn_histopathology_to_mri']['balanced_accuracy']:.4f}")
    print(f"  k-Nearest Neighbors (kNN) MRI→histopathology: "
          f"accuracy={t['knn_mri_to_histopathology']['accuracy']:.4f}, "
          f"balanced_accuracy={t['knn_mri_to_histopathology']['balanced_accuracy']:.4f}")
    print(f"  Prototype (nearest-centroid) histopathology→MRI: "
          f"accuracy={t['prototype_histopathology_to_mri']['accuracy']:.4f}, "
          f"balanced_accuracy={t['prototype_histopathology_to_mri']['balanced_accuracy']:.4f}")
    print(f"  Prototype (nearest-centroid) MRI→histopathology: "
          f"accuracy={t['prototype_mri_to_histopathology']['accuracy']:.4f}, "
          f"balanced_accuracy={t['prototype_mri_to_histopathology']['balanced_accuracy']:.4f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to {args.json_out}")

if __name__ == "__main__":
    main()
