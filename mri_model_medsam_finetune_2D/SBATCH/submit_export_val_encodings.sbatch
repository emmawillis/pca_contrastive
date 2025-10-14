#!/bin/bash
#SBATCH -J 128-export-umap-encs
#SBATCH -A aip-medilab
#SBATCH -t 0-12:00:00
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results/head_only/encodings/slurm-%x-%j.out
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

# --- env ---
source /project/aip-medilab/ewillis/pca_contrastive/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Host: $HOSTNAME"
which python
python --version
nvidia-smi || true

# --- paths (EDIT ME) ---
SCRIPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/export_val_encodings.py
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest_filtered.csv

# Choose checkpoint & outdir for this run (checkpoint contains encoder+head weights)
CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results/head_only/ckpt_head_best.pt
OUTDIR=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results/head_only/encodings/encodings_val_f0

# Validation folds (e.g., "0" or "0,2")
VAL_FOLDS=0

# Backbone architecture ONLY (weights are loaded from CKPT)
SAM_TYPE=vit_b   # vit_b | vit_l | vit_h  (match training)

# Dataset/label config (match training)
TARGET=isup3
CHANNELS=path_T2,path_ADC,path_HBV
PCT_LOWER=0.5
PCT_UPPER=99.5
MISSING_MODE=zeros   # zeros | repeat_t2

# Model head dims (must match checkpoint)
NUM_CLASSES=3
PROJ_DIM=128
ATTN_DIM=256
HEAD_HIDDEN=256

# Loader
BATCH_SIZE=32
NUM_WORKERS=${SLURM_CPUS_PER_TASK}

EXTRA_FLAGS=""  # add --no_pre_neck if you kept encoder.neck during training

# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --val_folds "$VAL_FOLDS" \
  --out_dir "$OUTDIR" \
  --checkpoint "$CKPT" \
  --sam_type "$SAM_TYPE" \
  --num_classes "$NUM_CLASSES" \
  --proj_dim "$PROJ_DIM" \
  --attn_dim "$ATTN_DIM" \
  --head_hidden "$HEAD_HIDDEN" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --target "$TARGET" \
  --channels "$CHANNELS" \
  --pct_lower "$PCT_LOWER" \
  --pct_upper "$PCT_UPPER" \
  --missing_channel_mode "$MISSING_MODE" \
  --amp \
  $EXTRA_FLAGS
