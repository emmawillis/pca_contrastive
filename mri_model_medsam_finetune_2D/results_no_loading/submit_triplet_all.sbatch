#!/bin/bash
#SBATCH -J all-isup3-triplet
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai 
#SBATCH -o /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/triplet/triplet-%j.out
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

SCRIPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/train_triplet_full.py
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv
CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth
OUTDIR=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/triplet
HISTO_DIR=/project/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_512D/embeddings_512
HISTO_MARKSHEET_DIR=/project/aip-medilab/shared/picai/histopathology_encodings/UNI2_splits

# --- run ---
srun -u python -u "$SCRIPT" \
  --manifest "$MANIFEST" \
  --sam_checkpoint "$CKPT" \
  --target isup3 \
  --folds_train 1,2,3 \
  --folds_val 0 \
  --folds_test 4 \
  --batch_size 16 \
  --pos_ratio 0.33 \
  --outdir "$OUTDIR" \
  --histo_dir "$HISTO_DIR" \
  --histo_marksheet_dir "$HISTO_MARKSHEET_DIR" \
  --proj_dim 512 \
  --provider "all" \
  --triplet_epochs 40 \
  --triplet_patience 10 \
  --triplet_lr 1e-5 \
  --triplet_wd 0 \
  --head_epochs 40 \
  --head_patience 10 \
  --head_lr 1e-5 \
  --head_wd 0 \
  --no-use-skip \
  --label6_column merged_ISUP \
  --wandb_run_name triplet-isup3

