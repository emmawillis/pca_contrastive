#!/bin/bash
#SBATCH -J merged_binary_baseline
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai 
#SBATCH -o /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/baseline/slurm-%x-%j.out
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
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv
CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth
OUTDIR=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/baseline

# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --sam_checkpoint "$CKPT" \
  --target binary_all \
  --folds_train 1,2,3 \
  --folds_val 0 \
  --batch_size 16 \
  --epochs 100 \
  --pos_ratio 0.5 \
  --outdir "$OUTDIR" \
  --proj_dim 512 \
  --lr 1e-5 \
  --wd 0 \
  --no-use-skip \
  --label6_column merged_ISUP
