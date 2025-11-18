#!/bin/bash
#SBATCH -J TEST-isup3-baseline
#SBATCH -A aip-medilab
#SBATCH -t 08:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai 
#SBATCH -o /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/CHECKING/final/test/BASELINE_WITH_SAM_CKPT-%j.out
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
SCRIPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/eval_test.py
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv
OUTDIR=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/isup3/results512/TRIPLET/all_providers/head_only
MODEL_CKPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/isup3/results512/baseline/ckpt_best.pt
SAM_CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth

# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --model_ckpt "$MODEL_CKPT" \
  --sam_checkpoint "$SAM_CKPT" \
  --target isup3 \
  --folds_train 1,2,3 \
  --fold_test 4 \
  --batch_size 16 \
  --pos_ratio 0.33 \
  --outdir "$OUTDIR" \
  --proj_dim 512 \
  --no-use-skip \
  --label6_column merged_ISUP
