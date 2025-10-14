PROJ_DIM = 128
BASELINE_CKPT_PATH = "/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results/baseline/ckpt_best.pt"
ALIGNED_CKPT_PATH = "/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results/head_only/ckpt_head_best.pt"

MRI_DIR = "/project/aip-medilab/shared/picai/picai_prepped_registered/imagesTr"
MRI_MANIFEST_PATH = "/project/aip-medilab/shared/picai/manifests/slices_manifest_filtered.csv"

sam.load_state_dict(torch.load(args.sam_checkpoint, map_location="cpu"), strict=True)
model = MedSAMSliceSpatialAttn(
    sam_model=sam,
    num_classes=n_classes,
    proj_dim=1024, attn_dim=256,
    head_hidden=256, head_dropout=0.1,
    use_pre_neck=True,              # pre-neck + spatial attention
    pixel_mean_std=None,            # inputs already in [0,1]
).to(device)