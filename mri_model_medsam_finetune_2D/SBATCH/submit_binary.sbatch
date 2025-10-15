#!/bin/bash
#SBATCH -J BINARY-1024-baseline
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai 
#SBATCH -o /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/RESULTS/binary_low_high/results1024/baseline/slurm-%x-%j.out
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
SCRIPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/train.py
# SCRIPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/train_coral_loss.py
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest_filtered.csv
CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth
OUTDIR=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/RESULTS/binary_low_high/results1024/baseline

# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --sam_checkpoint "$CKPT" \
  --target binary_low_high \
  --folds_train 1,2,3 \
  --folds_val 0 \
  --batch_size 16 \
  --epochs 100 \
  --pos_ratio 0.33 \
  --outdir "$OUTDIR" \
  --proj_dim 1024
