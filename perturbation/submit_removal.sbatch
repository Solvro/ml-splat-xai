#!/bin/bash -l
# Deletion experiment over all default ShapeSplat seeds x {splat, voxel}: 6 array tasks, <=6 at once.
# Submit from ~/ml-splat-xai:   sbatch perturbation/submit_removal.sbatch
#SBATCH --job-name=xsplain-removal
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:hopper:1
#SBATCH --time=06:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%A_%a.err

source /etc/profile
module load Python/3.12.3-GCCcore-13.3.0 || { echo 'module load failed' >&2; exit 3; }
set -eo pipefail
source /lustre/pd03/hpc-danbor2008-1756464546/julka/envs/xsplain/bin/activate
set -u

SEEDS=(777 778 779)
MODES=(splat voxel)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 2))]}
MODE=${MODES[$((SLURM_ARRAY_TASK_ID % 2))]}

CKPT_ROOT=/lustre/pd03/hpc-danbor2008-1756464546/xsplain_revision/checkpoints/shapesplat/default
OUT_ROOT=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/shapesplat/default
OUT=$OUT_ROOT/seed_${SEED}/${MODE}

cd "${SLURM_SUBMIT_DIR:-$HOME/ml-splat-xai}"
echo "task=$SLURM_ARRAY_TASK_ID seed=$SEED mode=$MODE node=$(hostname) account=$SLURM_JOB_ACCOUNT"
python perturbation/run_removal.py \
    --ckpt "$CKPT_ROOT/seed_${SEED}/stage2_compensated.pt" --ckpt_seed "$SEED" \
    --modes "$MODE" --out "$OUT" --check "$@"
