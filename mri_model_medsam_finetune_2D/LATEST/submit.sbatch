#!/bin/bash
#SBATCH -J isup3-proj-triplet
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/results_isup3/triplet_head_and_proj-%j.out
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

SCRIPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/train.py
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv
CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth

OUTDIR=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/results_isup3

HISTO_DIR=/project/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_512D/embeddings_512
HISTO_MARKSHEET_DIR=/project/aip-medilab/shared/picai/histopathology_encodings/UNI2_splits

#   TRAIN_MODE=two_stage   -> histopathology alignment (triplet -> CE head)
#   TRAIN_MODE=baseline    -> baseline end-to-end CE (no histo args needed)

TRAIN_MODE=two_stage
# TRAIN_MODE=baseline

# STAGE2_SCOPE=all
STAGE2_SCOPE=head_and_proj

# Make outputs separable per mode (avoids overwriting checkpoints)
OUTDIR_MODE="${OUTDIR}/${TRAIN_MODE}-head_and_proj"

# --- common args ---
COMMON_ARGS=(
  --train_mode "$TRAIN_MODE"
  --manifest "$MANIFEST"
  --sam_checkpoint "$CKPT"
  --target isup3
  --folds_train 1,2,3
  --folds_val 0
  --folds_test 4
  --batch_size 16
  --pos_ratio 0.33
  --outdir "$OUTDIR_MODE"
  --proj_dim 512
  --provider "all"
  --triplet_epochs 40
  --triplet_patience 10
  --head_epochs 40
  --head_patience 10
  --lr 1e-5
  --wd 0
  --enc_lr_mult 0.1
  --no-use-skip
  --label6_column merged_ISUP
  --wandb_run_name "isup3-${TRAIN_MODE}-scope-${STAGE2_SCOPE}"
  --stage2_scope "$STAGE2_SCOPE"
)

# --- add histo args only for two_stage ---
if [[ "$TRAIN_MODE" == "two_stage" ]]; then
  COMMON_ARGS+=(
    --histo_dir "$HISTO_DIR"
    --histo_marksheet_dir "$HISTO_MARKSHEET_DIR"
  )
fi

echo "TRAIN_MODE=$TRAIN_MODE"
echo "OUTDIR_MODE=$OUTDIR_MODE"

# --- run ---
srun -u python -u "$SCRIPT" "${COMMON_ARGS[@]}"
