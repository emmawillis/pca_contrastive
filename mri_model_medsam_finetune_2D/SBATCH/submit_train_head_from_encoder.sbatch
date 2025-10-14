#!/bin/bash
#SBATCH -J 1024-head-only-ce
#SBATCH -A aip-medilab
#SBATCH -t 1-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results1024/head_only/slurm-%x-%j.out
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

# --- paths ---
SCRIPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/train_head_from_encoder.py
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest_filtered.csv
BASE_SAM=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth

# encoder checkpoint saved by the triplet+LR script
ENCODER_CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results1024/triplet_logreg/ckpt_best.pt

OUTDIR=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results1024/head_only

# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --sam_checkpoint "$BASE_SAM" \
  --encoder_ckpt "$ENCODER_CKPT" \
  --outdir "$OUTDIR" \
  --target isup3 \
  --folds_train 1,2,3 \
  --folds_val 0 \
  --batch_size 32 \
  --epochs 20 \
  --lr_head 1e-3 \
  --wd 1e-4
  # add --train_proj if you want to fine-tune the projection MLP too
