#!/bin/bash
#SBATCH -J head_binary
#SBATCH -A aip-medilab
#SBATCH -t 1-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/INFOLOSS/all_providers/head_only/slurm-%x-%j.out
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
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv
BASE_SAM=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth

# encoder checkpoint saved by the triplet+LR script
ENCODER_CKPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/triplet_infoloss_all/ckpt_best.pt

OUTDIR=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/INFOLOSS/all_providers/head_only
# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --sam_checkpoint "$BASE_SAM" \
  --encoder_ckpt "$ENCODER_CKPT" \
  --outdir "$OUTDIR" \
  --target binary_all \
  --folds_train 1,2,3 \
  --folds_val 0 \
  --batch_size 16 \
  --epochs 100 \
  --proj_dim 512 \
  --lr 1e-5 \
  --wd 0 \
  --no-use-skip \
  --label6_column merged_ISUP
  # add --train_proj if you want to fine-tune the projection MLP too
