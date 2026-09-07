#!/bin/bash
#SBATCH -J random_head
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o '/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/random_head/random_head-%j.out'
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

SCRIPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/train_histo_head.py
MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv
CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth
HISTO_HEAD_CKPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/histo_head/histo_512_best_model.pth

OUTDIR='/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/LATEST/random_head'

# TRAIN_MODE=baseline
# TRAIN_MODE=histo_head
TRAIN_MODE=random_head
OUTDIR_MODE="${OUTDIR}/${TRAIN_MODE}"

# --- common args ---
COMMON_ARGS=(
  --train_mode "$TRAIN_MODE"
  --manifest "$MANIFEST"
  --sam_checkpoint "$CKPT"
  --target isup6
  --folds_train 1,2,3
  --folds_val 0
  --folds_test 4
  --batch_size 16
  --pos_ratio 0.33
  --outdir "$OUTDIR_MODE"
  --proj_dim 512
  --epochs 40
  --patience 10
  --lr 1e-5
  --wd 0
  --enc_lr_mult 0.1
  --no-use-skip
  --label6_column merged_ISUP
  --wandb_run_name "isup6-${TRAIN_MODE}"
)

if [[ "$TRAIN_MODE" == "histo_head" ]]; then
  COMMON_ARGS+=(
    --histo_head_ckpt "$HISTO_HEAD_CKPT"
  )
fi

echo "TRAIN_MODE=$TRAIN_MODE"
echo "OUTDIR_MODE=$OUTDIR_MODE"

# --- run ---
srun -u python -u "$SCRIPT" "${COMMON_ARGS[@]}"
