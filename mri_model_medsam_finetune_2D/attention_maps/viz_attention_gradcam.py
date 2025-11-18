#!/usr/bin/env python3
# Visualize attention (class-agnostic) and Grad-CAM (class-specific)
# for BASELINE vs ALIGNED checkpoints, with diagnostics and masks.
#
# Updates in this version:
#   1) Overlay style switched to a faint **white glow** so the strongest areas pop.
#      Tunable with --overlay-alpha, --overlay-gamma, --overlay-pmin.
#   2) Added --ids to pick exact slices by ID. Accepts:
#        --ids 10000_1000000,10001_1000001:37
#      Where tokens are:
#        • "<case_id>"               -> first slice found for that case_id
#        • "<case_id>:<z_index>"     -> specific axial slice index
#      (If an ID isn't found, a warning is printed and it's skipped.)
#   3) Keeps multi-channel (T2/ADC/HBV) layout, lesion mask loading, and 90° CCW rotation.

import argparse
from pathlib import Path
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib  # for NIfTI masks

from segment_anything import sam_model_registry
from ISUPMedSAM import IMG_SIZE, MedSAMSliceSpatialAttn, resize_to_img_size
from dataset_picai_slices import PicaiSliceDataset

# -------------------------- Collate (resize to IMG_SIZE) --------------------------
def collate_resize_to_imgsize(batch):
    imgs, labels, metas = [], [], []
    for s in batch:
        x = s["image"].unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        imgs.append(x)
        labels.append(torch.as_tensor(s["label"], dtype=torch.long))
        metas.append(s)  # keep the raw sample to extract paths/slice indices
    return {"image": torch.stack(imgs, 0), "label": torch.stack(labels, 0), "meta": metas}

# -------------------------- Utils --------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def to_uint8_gray(channel_tensor: torch.Tensor) -> np.ndarray:
    """1×H×W (or H×W) in [0,1] -> H×W×3 uint8 grayscale."""
    if channel_tensor.ndim == 3 and channel_tensor.shape[0] == 1:
        ch = channel_tensor[0]
    else:
        ch = channel_tensor
    x = ch.detach().cpu().clamp(0, 1).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return np.stack([x, x, x], axis=-1)

def normalize01(arr: torch.Tensor | np.ndarray, eps=1e-12):
    a = arr.detach().cpu().numpy() if torch.is_tensor(arr) else arr
    mn, mx = float(a.min()), float(a.max())
    if mx <= mn + eps:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)

def overlay_white_glow(
    img_uint8_hwc: np.ndarray,
    heat01_hw: np.ndarray,
    alpha: float = 0.25,
    gamma: float = 3.0,
    pmin: float = 90.0,
):
    """
    White 'glow' overlay that makes only the strongest regions pop.

    Steps:
      * Percentile-threshold low activations (pmin, e.g., 90th percentile)
      * Re-normalize 0..1 and apply gamma (>1 compresses mid values)
      * Blend toward white proportional to activation
    """
    base = img_uint8_hwc.astype(np.float32) / 255.0
    h = np.clip(heat01_hw, 0.0, 1.0).astype(np.float32)

    # threshold at percentile to suppress low activations
    thr = np.percentile(h, pmin)
    if thr >= 1.0:  # degenerate case
        thr = 0.999
    h = np.clip((h - thr) / (1.0 - thr + 1e-8), 0.0, 1.0)

    # gamma to make only top hotspots visible
    if gamma is not None and gamma > 0:
        h = h ** gamma

    h3 = h[..., None]  # H×W×1
    white = np.ones_like(base, dtype=np.float32)

    # blend toward white: base*(1 - alpha*h) + white*(alpha*h)
    out = base * (1.0 - alpha * h3) + white * (alpha * h3)
    return (out * 255.0).clip(0, 255).astype(np.uint8)

def rotate_ccw(arr: np.ndarray) -> np.ndarray:
    """Rotate HxW or HxWx3 90° CCW for display."""
    return np.rot90(arr, k=1, axes=(0, 1))

# -------------------------- Debug helpers --------------------------
def dbg_stats(name: str, hm: torch.Tensor):
    """
    hm: [B,H,W] (or [B,1,H,W]) in any range; we normalize elsewhere for display.
    Prints per-image min/max/std and the mass in the top 5% pixels (uniform ≈ 0.05).
    """
    if hm.ndim == 4:
        hm = hm.squeeze(1)
    flat = hm.reshape(hm.shape[0], -1)
    mins = flat.min(dim=1).values[:8]
    maxs = flat.max(dim=1).values[:8]
    stds = flat.std(dim=1)[:8]
    k = max(1, int(flat.shape[1] * 0.05))
    topk_mass = torch.topk(flat, k, dim=1).values.sum(dim=1) / flat.sum(dim=1).clamp(min=1e-12)
    print(f"[{name}] min[:8]={mins.tolist()}")
    print(f"[{name}] max[:8]={maxs.tolist()}")
    print(f"[{name}] std[:8]={stds.tolist()}")
    print(f"[{name}] top5% mass (uniform≈0.05)[:8]={topk_mass.tolist()}")

# -------------------------- Models --------------------------
def build_model(sam_type, sam_ckpt, num_classes=3, proj_dim=512, attn_dim=256, head_hidden=256, head_dropout=0.1, use_pre_neck=True, device="cuda"):
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    model = MedSAMSliceSpatialAttn(
        sam_model=sam,
        num_classes=num_classes,
        proj_dim=proj_dim,
        attn_dim=attn_dim,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        use_pre_neck=use_pre_neck,
    )
    return model.to(device).eval()

def load_ckpt_strict_filtered(model, ckpt_path: str):
    """Load a checkpoint but drop any keys whose shapes don't match."""
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)
    own = model.state_dict()
    filtered, dropped = {}, []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[load] loaded {ckpt_path}")
    if dropped:
        print(f"[load] dropped {len(dropped)} incompatible keys (e.g., {dropped[:6]})")
    if missing:
        print(f"[load] missing: {missing[:6]}")
    if unexpected:
        print(f"[load] unexpected: {unexpected[:6]}")

# -------------------------- Grad-CAM --------------------------
def gradcam_from_feats(model: MedSAMSliceSpatialAttn, feats: torch.Tensor, imgs_hw: tuple[int, int], class_idx: int | None = None):
    """
    Compute Grad-CAM from pre-pool feature map (feats: [B,C,Hf,Wf]).
    Returns (cam01 [B,H,W], target_classes [B]).
    """
    feats_req = feats.detach().requires_grad_(True)
    pooled, _ = model.pool(feats_req)            # [B,C]
    emb2 = model.proj(pooled)                    # [B,D]
    logits2 = model.head(emb2)                   # [B,K]

    target = logits2.argmax(1) if class_idx is None else torch.full(
        (logits2.size(0),), int(class_idx), device=logits2.device, dtype=torch.long
    )

    score = logits2.gather(1, target.view(-1, 1)).sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    grads = feats_req.grad                       # [B,C,Hf,Wf]
    if grads is None:
        raise RuntimeError("Gradients are None. Ensure no torch.no_grad() wraps this block.")

    print("[gradcam] |grads| mean:", float(grads.abs().mean().item()))

    weights = grads.mean(dim=(2, 3), keepdim=True)    # [B,C,1,1]
    cam = torch.relu((weights * feats_req).sum(dim=1))  # [B,Hf,Wf]
    cam = F.interpolate(cam.unsqueeze(1), size=imgs_hw, mode="bilinear", align_corners=False).squeeze(1)
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (cam.amax(dim=(1, 2), keepdim=True) - cam.amin(dim=(1, 2), keepdim=True) + 1e-12)
    return cam, target.detach().cpu()

# -------------------------- Mask helpers --------------------------
SUFFIX_RE = re.compile(r"(_000\d)?(\.nii(\.gz)?)$")

def path_root_from_channel_path(p: str) -> str:
    """Strip modality suffix like '_0000.nii.gz' → '10000_1000000'."""
    name = Path(p).name
    name = SUFFIX_RE.sub("", name)  # remove _000X and extension
    return name

def extract_slice_index(meta_sample: dict) -> int | None:
    """Best-effort slice index extraction from the sample dict."""
    for k in ("slice_idx", "slice", "z", "z_idx", "index_z"):
        if k in meta_sample and meta_sample[k] is not None:
            try:
                return int(meta_sample[k])
            except Exception:
                pass
    # Sometimes nested under 'meta'
    if "meta" in meta_sample and isinstance(meta_sample["meta"], dict):
        for k in ("slice_idx", "slice", "z", "z_idx", "index_z"):
            if k in meta_sample["meta"]:
                try:
                    return int(meta_sample["meta"][k])
                except Exception:
                    pass
    return None

def load_mask_slice(mask_dir: Path, root_name: str, slice_idx: int | None) -> np.ndarray | None:
    """Load a 2D mask slice (H×W float32 in {0,1}); choose middle slice if slice_idx is None."""
    f = mask_dir / f"{root_name}.nii.gz"
    if not f.exists():
        f = mask_dir / f"{root_name}.nii"
        if not f.exists():
            print(f"[mask] missing for root={root_name}")
            return None
    try:
        vol = nib.load(str(f))
        m = vol.get_fdata().astype(np.float32)  # H×W×Z or similar
        z = slice_idx if (slice_idx is not None and 0 <= slice_idx < m.shape[-1]) else m.shape[-1] // 2
        sl = m[..., z]
        return (sl > 0).astype(np.float32)
    except Exception as e:
        print(f"[mask] failed to load {f}: {e}")
        return None

# -------------------------- Plotting --------------------------
def make_grid_pngs_multi_channel(
    imgs,                 # [B,3,H,W] in [0,1]
    attn_b, cam_b,        # [B,H,W] each
    attn_a, cam_a,        # [B,H,W] each
    chan_names,           # tuple/list of len 3, e.g. ("T2","ADC","HBV")
    masks_hw,             # list length B of H×W np arrays or None
    out_dir, tag,
    overlay_alpha=0.25, overlay_gamma=3.0, overlay_pmin=90.0,
    show_hm=True,
):
    """
    Layout per row (for each channel independently):
      [Chan] |
      Base—Attn(HM) | Base—Attn(Overlay) | Base—Grad-CAM(HM) | Base—Grad-CAM(Overlay) |
      Align—Attn(HM) | Align—Attn(Overlay) | Align—Grad-CAM(HM) | Align—Grad-CAM(Overlay)
    Repeated for T2, ADC, HBV. Two extra columns at the end:
      Mask (bin) | T2 + Mask (Overlay)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    B = imgs.size(0)
    per_chan_cols = 9
    extra_cols = 2
    cols = per_chan_cols * 3 + extra_cols

    fig, axes = plt.subplots(B, cols, figsize=(cols * 1.6, B * 1.6))
    if B == 1:
        axes = np.expand_dims(axes, 0)

    chan_titles = []
    for c, cn in enumerate(chan_names):
        chan_titles += [
            f"{cn}",
            "Base—Attn (HM)", "Base—Attn (Overlay)",
            "Base—Grad-CAM (HM)", "Base—Grad-CAM (Overlay)",
            "Align—Attn (HM)", "Align—Attn (Overlay)",
            "Align—Grad-CAM (HM)", "Align—Grad-CAM (Overlay)",
        ]
    chan_titles += ["Mask (bin)", "T2 + Mask (Overlay)"]

    for i in range(B):
        # channel images (grayscale)
        t2 = to_uint8_gray(imgs[i, 0:1])
        adc = to_uint8_gray(imgs[i, 1:2])
        hbv = to_uint8_gray(imgs[i, 2:3])

        # heatmaps (0..1) as numpy
        a_b = normalize01(attn_b[i]); c_b = normalize01(cam_b[i])
        a_a = normalize01(attn_a[i]); c_a = normalize01(cam_a[i])

        # rotate all displays 90° CCW
        t2r, adcr, hbvr = rotate_ccw(t2), rotate_ccw(adc), rotate_ccw(hbv)
        a_b_r, c_b_r = rotate_ccw(a_b), rotate_ccw(c_b)
        a_a_r, c_a_r = rotate_ccw(a_a), rotate_ccw(c_a)

        panels = []

        # helper to append a 9-panel block for a given channel base
        def blk(base_rgb):
            hm_gray = lambda h: (np.stack([h, h, h], axis=-1) * 255).astype(np.uint8)
            return [
                base_rgb,
                hm_gray(a_b_r), overlay_white_glow(base_rgb, a_b_r, alpha=overlay_alpha, gamma=overlay_gamma, pmin=overlay_pmin),
                hm_gray(c_b_r), overlay_white_glow(base_rgb, c_b_r, alpha=overlay_alpha, gamma=overlay_gamma, pmin=overlay_pmin),
                hm_gray(a_a_r), overlay_white_glow(base_rgb, a_a_r, alpha=overlay_alpha, gamma=overlay_gamma, pmin=overlay_pmin),
                hm_gray(c_a_r), overlay_white_glow(base_rgb, c_a_r, alpha=overlay_alpha, gamma=overlay_gamma, pmin=overlay_pmin),
            ]

        panels += blk(t2r)
        panels += blk(adcr)
        panels += blk(hbvr)

        # Mask panels (mask overlay stays red for contrast with white attention)
        mask = masks_hw[i] if masks_hw is not None else None
        if mask is not None:
            mask_r = rotate_ccw(mask)
            mask_bin_rgb = (np.stack([mask_r, mask_r, mask_r], axis=-1) * 255).astype(np.uint8)
            # red mask overlay for T2
            red = np.zeros_like(t2r, dtype=np.float32)
            red[..., 0] = (mask_r * 255.0)
            t2_mask_overlay = (0.6 * t2r.astype(np.float32) + 0.4 * red).clip(0, 255).astype(np.uint8)
        else:
            H, W = t2r.shape[:2]
            mask_bin_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            t2_mask_overlay = t2r.copy()

        panels += [mask_bin_rgb, t2_mask_overlay]

        # draw row
        assert len(panels) == cols
        for j in range(cols):
            # optionally hide pure HM columns if desired
            if not show_hm and (j % per_chan_cols) in (1, 3, 5, 7):  # the HM columns within each 9-block
                axes[i, j].imshow(np.zeros_like(panels[j]))
            else:
                axes[i, j].imshow(panels[j])
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(chan_titles[j], fontsize=9)

        # save strip per row
        strip = np.concatenate(panels, axis=1)
        plt.imsave(str(out_dir / f"{tag}_row{i:02d}.png"), strip)

    plt.tight_layout()
    fig.savefig(str(out_dir / f"{tag}_grid.pdf"), bbox_inches="tight")
    fig.savefig(str(out_dir / f"{tag}_grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_dir / f'{tag}_grid.pdf'} and per-row PNGs")

# -------------------------- Data helpers --------------------------
def sample_batch(ds, n, seed=42):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    batch = [ds[i] for i in idxs]
    coll = collate_resize_to_imgsize(batch)
    return coll["image"], coll["label"], coll["meta"], idxs

def parse_ids_arg(ids_str: str):
    """
    Parse --ids like "10000_1000000,10001_1000001:37".
    Returns list of tuples: [(case_id:str, z:int|None), ...]
    """
    out = []
    for tok in [t.strip() for t in ids_str.split(",") if t.strip()]:
        if ":" in tok:
            cid, z = tok.split(":", 1)
            try:
                z = int(z)
            except Exception:
                z = None
            out.append((cid, z))
        else:
            out.append((tok, None))
    return out

def sample_by_ids(ds, id_specs):
    """
    id_specs: [(case_id, z|None), ...]
    Scans the dataset and picks the first slice that matches each spec.
    Returns tensors and metas via collate.
    """
    wanted = list(id_specs)
    picked = []
    found = [False] * len(wanted)

    for i in range(len(ds)):
        s = ds[i]  # sample dict
        cid = str(s.get("case_id", ""))
        z = None
        # try to infer z index from sample
        for k in ("slice_idx", "slice", "z", "z_idx", "index_z"):
            if k in s and s[k] is not None:
                try:
                    z = int(s[k])
                except Exception:
                    pass
                break

        for idx, (want_cid, want_z) in enumerate(wanted):
            if found[idx]:
                continue
            if cid == want_cid and (want_z is None or want_z == z):
                picked.append(s)
                found[idx] = True
                break

        if all(found):
            break

    # warn for missing ones
    for idx, ok in enumerate(found):
        if not ok:
            wcid, wz = wanted[idx]
            print(f"[ids] WARNING: not found case_id='{wcid}' z={wz}")

    coll = collate_resize_to_imgsize(picked)
    idxs = []  # not used downstream
    return coll["image"], coll["label"], coll["meta"], idxs

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser("Visualize attention, Grad-CAM, and lesion masks with per-channel inputs")
    # Data
    ap.add_argument("--manifest", required=True, help="Path to slices manifest CSV.")
    ap.add_argument("--folds-test", default="4", help="Comma-separated folds for test set, e.g. '4' or '3,4'.")
    ap.add_argument("--target", default="isup3", choices=["isup3", "binary_all", "binary_low_high", "raw"])
    ap.add_argument("--channels", default="path_T2,path_ADC,path_HBV", help="Comma-separated channel column names.")
    ap.add_argument("--num", type=int, default=8, help="Number of random test slices to visualize (ignored if --ids is set).")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated IDs to plot. Forms: '<case_id>' or '<case_id>:<z>'. Example: '10000_1000000,10001_1000001:37'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask-dir", default="/home/ewillis/projects/aip-medilab/shared/picai/picai_prepped_registered/labelsTr_lesion",
                    help="Directory containing lesion masks named <root>.nii.gz")

    # Models
    ap.add_argument("--sam-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--aligned-ckpt", required=True)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--attn-dim", type=int, default=256)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--use-pre-neck", action="store_true")

    # Projection sizes (you said both are 512)
    ap.add_argument("--proj-dim", type=int, default=512,
                    help="Fallback proj_dim for BOTH models if per-model args not given.")
    ap.add_argument("--baseline-proj-dim", type=int, default=None,
                    help="Projection dim for BASELINE (overrides --proj-dim).")
    ap.add_argument("--aligned-proj-dim", type=int, default=None,
                    help="Projection dim for ALIGNED (overrides --proj-dim).")

    # Output & overlay style
    ap.add_argument("--out-dir", default="viz_attn_gradcam")
    ap.add_argument("--overlay-alpha", type=float, default=0.25, help="Blend strength toward white (0..1).")
    ap.add_argument("--overlay-gamma", type=float, default=3.0, help=">1 compresses mid values so hotspots pop.")
    ap.add_argument("--overlay-pmin", type=float, default=90.0, help="Percentile threshold; lower values show more area.")
    ap.add_argument("--hide-hm", action="store_true", help="Hide standalone gray HM columns; show overlays only.")

    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chan_keys = tuple(args.channels.split(","))
    chan_names = ("T2", "ADC", "HBV")  # display names (order must match your channel order)

    # Dataset
    folds = [int(x.strip()) for x in args.folds_test.split(",") if x.strip()]
    ds_test = PicaiSliceDataset(
        manifest_csv=args.manifest,
        folds=folds,
        use_skip=True,
        label6_column="label6",
        target=args.target,
        channels=chan_keys,
        missing_channel_mode="zeros",
        pct_lower=0.5, pct_upper=99.5,
        cache_size=64,
    )

    # Choose samples: by --ids if provided, else random
    if args.ids:
        id_specs = parse_ids_arg(args.ids)
        imgs, labels, metas, _ = sample_by_ids(ds_test, id_specs)
    else:
        imgs, labels, metas, _ = sample_batch(ds_test, n=args.num, seed=args.seed)

    imgs = imgs.to(device)
    H, W = imgs.shape[-2], imgs.shape[-1]

    # Per-model proj dims
    pd_base  = args.baseline_proj_dim or args.proj_dim
    pd_align = args.aligned_proj_dim or args.proj_dim

    # Build + load
    m_base = build_model(args.sam_type, args.sam_ckpt, num_classes=args.num_classes,
                         proj_dim=pd_base, attn_dim=args.attn_dim,
                         head_hidden=args.head_hidden, head_dropout=args.head_dropout,
                         use_pre_neck=args.use_pre_neck, device=device)
    load_ckpt_strict_filtered(m_base, args.baseline_ckpt)

    m_align = build_model(args.sam_type, args.sam_ckpt, num_classes=args.num_classes,
                          proj_dim=pd_align, attn_dim=args.attn_dim,
                          head_hidden=args.head_hidden, head_dropout=args.head_dropout,
                          use_pre_neck=args.use_pre_neck, device=device)
    load_ckpt_strict_filtered(m_align, args.aligned_ckpt)

    # Forward once to get attention + feats (no grads needed here)
    with torch.no_grad():
        logits_b, emb_b, attn_b, feats_b = m_base(
            imgs, return_attn=True, return_feats=True, attn_upsample_to=(H, W)
        )
        logits_a, emb_a, attn_a, feats_a = m_align(
            imgs, return_attn=True, return_feats=True, attn_upsample_to=(H, W)
        )

    # Diagnostic summaries for attention maps
    dbg_stats("ATTN_base", attn_b.squeeze(1).detach().cpu())
    dbg_stats("ATTN_align", attn_a.squeeze(1).detach().cpu())

    # Grad-CAM (needs grads through pool→proj→head)
    cam_b, pred_b = gradcam_from_feats(m_base, feats_b, (H, W), class_idx=None)
    cam_a, pred_a = gradcam_from_feats(m_align, feats_a, (H, W), class_idx=None)

    # Diagnostic summaries for Grad-CAM maps
    dbg_stats("CAM_base", cam_b.detach().cpu())
    dbg_stats("CAM_align", cam_a.detach().cpu())

    # --- Load masks for each sampled slice ---
    mask_dir = Path(args.mask_dir)
    mask_list = []
    for meta in metas:
        # try to locate any channel path
        sample_path = None
        for k in chan_keys:
            if k in meta and meta[k] is not None:
                sample_path = meta[k]
                break
            if "paths" in meta and isinstance(meta["paths"], dict) and k in meta["paths"]:
                sample_path = meta["paths"][k]
                break

        root = path_root_from_channel_path(str(sample_path)) if sample_path is not None else None
        z = extract_slice_index(meta)
        m2d = load_mask_slice(mask_dir, root, z) if root is not None else None
        mask_list.append(m2d)

    # Write grid (per-channel layout + masks)
    tag_core = f"folds{args.folds_test.replace(',', '')}"
    tag = f"ids_{args.ids.replace(',','_')}" if args.ids else f"n{imgs.size(0)}_{tag_core}"
    make_grid_pngs_multi_channel(
        imgs,
        attn_b.squeeze(1), cam_b,
        attn_a.squeeze(1), cam_a,
        chan_names=chan_names,
        masks_hw=mask_list,
        out_dir=args.out_dir, tag=tag,
        overlay_alpha=args.overlay_alpha,
        overlay_gamma=args.overlay_gamma,
        overlay_pmin=args.overlay_pmin,
        show_hm=not args.hide_hm,
    )

    # Small text dump of true/pred classes
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "pred_classes.txt", "w") as f:
        for i, (y, pb, pa) in enumerate(zip(labels.tolist(), pred_b.tolist(), pred_a.tolist())):
            f.write(f"idx{i:02d}\ttrue={y}\tbaseline_pred={pb}\taligned_pred={pa}\n")
    print(f"[ok] wrote {Path(args.out_dir) / 'pred_classes.txt'}")

if __name__ == "__main__":
    main()
