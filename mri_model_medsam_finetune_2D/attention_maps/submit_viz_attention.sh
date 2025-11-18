#!/bin/bash
#SBATCH -J viz_attention
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/figs/viz_attn_gradcam/%j.out
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

# --- env ---
source /project/aip-medilab/ewillis/pca_contrastive/venv/bin/activate
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Host: $HOSTNAME"
which python
python --version
nvidia-smi || true

OUTDIR="/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/figs/viz_attn_gradcam"
mkdir -p "$OUTDIR"

# ----- Option A: random sample (keep --num) -----
# srun --unbuffered python /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/viz_attention_gradcam.py \
#   --manifest /project/aip-medilab/shared/picai/manifests/slices_manifest.csv \
#   --folds-test 4 \
#   --target isup3 \
#   --sam-type vit_b \
#   --sam-ckpt /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth \
#   --baseline-ckpt /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/baseline/ckpt_best.pt \
#   --aligned-ckpt  /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/triplet/ckpt_head_best.pt \
#   --num 8 \
#   --use-pre-neck \
#   --out-dir "$OUTDIR" \
#   --mask-dir /home/ewillis/projects/aip-medilab/shared/picai/picai_prepped_registered/labelsTr_lesion \
#   --overlay-alpha 0.22 \
#   --overlay-gamma 3.5 \
#   --overlay-pmin 92 \
#   --hide-hm

# ----- Option B: specific IDs (uncomment and edit; comment Option A if you use this) -----
srun --unbuffered python /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/viz_attention_gradcam.py \
  --manifest /project/aip-medilab/shared/picai/manifests/slices_manifest.csv \
  --folds-test 4 \
  --target isup3 \
  --sam-type vit_b \
  --sam-ckpt /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth \
  --baseline-ckpt /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/baseline/ckpt_best.pt \
  --aligned-ckpt  /home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/results_no_loading/triplet/ckpt_head_best.pt \
  --use-pre-neck \
  --out-dir "$OUTDIR" \
  --mask-dir /home/ewillis/projects/aip-medilab/shared/picai/picai_prepped_registered/labelsTr_lesion \
  --ids 10023_1004321:37,10045_1006789,10101_1010101:12 \
  --overlay-alpha 0.22 \
  --overlay-gamma 3.5 \
  --overlay-pmin 92 \
  --hide-hm
