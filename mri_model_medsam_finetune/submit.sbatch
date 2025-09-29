#!/bin/bash
#SBATCH -J medsam-finetune
#SBATCH -A aip-medilab                             # e.g., aip-medilab (your Slurm/CCDB project)
#SBATCH -t 012:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/ewillis/pca_contrastive
#SBATCH -o /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/results/medsam-finetune-slurm-%x-%j.out
#SBATCH --gres=gpu:l40s:1                      # (only if you actually need a GPU)

set -euo pipefail

# Activate your venv (keep it under /project, not /home)
source /project/aip-medilab/ewillis/pca_contrastive/venv/bin/activate

# Help PyTorch reduce fragmentation on large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $HOSTNAME"
which python
python --version

# Run your command
srun -u python -u /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/train.py \
  --checkpoint /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth \
  --images-dir /project/aip-medilab/ewillis/picai_baseline/picai_baseline/workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr \
  --marksheet-csv /project/aip-medilab/ewillis/pca_contrastive/mri_data/marksheet_folds.csv \
  --out_dir /project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/results/token_sttention_200_epochs \
  --use-weighted-sampler --confmat-normalize true 