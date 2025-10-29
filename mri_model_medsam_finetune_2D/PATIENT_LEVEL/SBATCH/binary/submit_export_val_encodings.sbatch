#!/bin/bash
#SBATCH -J export-binary
#SBATCH -A aip-medilab
#SBATCH -t 0-12:00:00
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/baseline/export-encodings-%x-%j.out
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
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv

# Choose checkpoint & outdir for this run (checkpoint contains encoder+head weights)
CKPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/baseline/ckpt_best.pt
OUTDIR=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/baseline/encodings_test_f4

# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --val_folds 4 \
  --out_dir "$OUTDIR" \
  --checkpoint "$CKPT" \
  --proj_dim 512 \
  --batch_size 16 \
  --target binary_all \
  --no-use-skip \
  --label6_column merged_ISUP 
