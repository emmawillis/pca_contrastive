#!/bin/bash
# This script SUBMITS two separate SLURM jobs (baseline + triplet)
# They run in PARALLEL.

set -euo pipefail

###########################################
# COMMON PATHS
###########################################

LOGDIR=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/new_SPIE/RESULTS
mkdir -p "$LOGDIR"

BASELINE_SCRIPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/new_SPIE/train_baseline_oldSPIE.py
TRIPLET_SCRIPT=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/new_SPIE/train_triplet_oldSPIE.py

MANIFEST=/project/aip-medilab/shared/picai/manifests/slices_manifest.csv
CKPT=/project/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune/work_dir/MedSAM/medsam_vit_b.pth

OUTDIR_BASE=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/new_SPIE/RESULTS/baseline
OUTDIR_TRIP=/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/new_SPIE/RESULTS/triplet

HISTO_DIR=/home/ewillis/projects/aip-medilab/shared/picai/histopathology_encodings/UNI2/projected_128D/embeddings_128
HISTO_MARKSHEET=/home/ewillis/projects/aip-medilab/shared/picai/histopathology_encodings/UNI2_splits

mkdir -p "$OUTDIR_BASE" "$OUTDIR_TRIP"

###########################################
# SUBMIT BASELINE JOB
###########################################
BASELINE_JOB=$(sbatch <<EOF
#!/bin/bash
#SBATCH -J newSPIE-baseline-isup6
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH --gres=gpu:l40s:1
#SBATCH -o $LOGDIR/baseline-%j.out

set -euo pipefail

source /project/aip-medilab/ewillis/pca_contrastive/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

echo "Running BASELINE job"
which python
python --version
nvidia-smi || true

srun -u python -u "$BASELINE_SCRIPT" \
  --manifest "$MANIFEST" \
  --sam_checkpoint "$CKPT" \
  --target isup6 \
  --folds_train 1,2,3 \
  --folds_val 0 \
  --folds_test 4 \
  --batch_size 16 \
  --epochs 40 \
  --patience 10 \
  --pos_ratio 0.33 \
  --proj_dim 128 \
  --lr 1e-5 \
  --wd 0 \
  --no-use-skip \
  --label6_column merged_ISUP \
  --outdir "$OUTDIR_BASE" \
  --wandb_run_name newSPIE-baseline-isup6-128D
EOF
)
echo "Submitted BASELINE job: $BASELINE_JOB"


###########################################
# SUBMIT TRIPLET JOB
###########################################
TRIPLET_JOB=$(sbatch <<EOF
#!/bin/bash
#SBATCH -J newSPIE-triplet-isup6
#SBATCH -A aip-medilab
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH --gres=gpu:l40s:1
#SBATCH -o $LOGDIR/triplet-%j.out

set -euo pipefail

source /project/aip-medilab/ewillis/pca_contrastive/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

echo "Running TRIPLET job"
which python
python --version
nvidia-smi || true

srun -u python -u "$TRIPLET_SCRIPT" \
  --manifest "$MANIFEST" \
  --sam_checkpoint "$CKPT" \
  --target isup6 \
  --folds_train 1,2,3 \
  --folds_val 0 \
  --folds_test 4 \
  --batch_size 16 \
  --triplet_epochs 40 \
  --triplet_patience 10 \
  --triplet_lr 1e-5 \
  --triplet_wd 0 \
  --triplet_margin 0.2 \
  --lr_max_iter 5 \
  --head_epochs 40 \
  --head_patience 10 \
  --head_lr 1e-5 \
  --head_wd 0 \
  --proj_dim 128 \
  --no-use-skip \
  --label6_column merged_ISUP \
  --histo_dir "$HISTO_DIR" \
  --histo_marksheet_dir "$HISTO_MARKSHEET" \
  --provider all \
  --outdir "$OUTDIR_TRIP" \
  --wandb_run_name newSPIE-triplet-isup6-128D
EOF
)
echo "Submitted TRIPLET job: $TRIPLET_JOB"

echo ""
echo "Both jobs submitted in parallel!"
