#!/bin/bash -l
# Deletion experiment for MACGS and toys: 2 datasets x 3 seeds x {splat, voxel} = 12 array tasks, <=6 at once.
# Submit from ~/ml-splat-xai:   sbatch perturbation/submit_removal_datasets.sbatch [extra run_removal.py args]
#SBATCH --job-name=xsplain-removal-ds
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:hopper:1
#SBATCH --time=06:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%A_%a.err

source /etc/profile
module load Python/3.12.3-GCCcore-13.3.0 || { echo 'module load failed' >&2; exit 3; }
set -eo pipefail
source /lustre/pd03/hpc-danbor2008-1756464546/julka/envs/xsplain/bin/activate
set -u

PD=/lustre/pd03/hpc-danbor2008-1756464546
DATASETS=(macgs toys)
declare -A DATA_DIRS=([macgs]=$PD/datasets/MACGS/MACGS_ply [toys]=$PD/datasets/toys/data)
SEEDS=(777 778 779)
MODES=(splat voxel)
T=$SLURM_ARRAY_TASK_ID
DS=${DATASETS[$((T / 6))]}
SEED=${SEEDS[$(((T % 6) / 2))]}
MODE=${MODES[$((T % 2))]}
CONFIG=default_90ep

CKPT=$PD/xsplain_revision/checkpoints/$DS/$CONFIG/seed_${SEED}/stage2_compensated.pt
OUT=$PD/julka/xsplain_removal/$DS/$CONFIG/seed_${SEED}/${MODE}

cd "${SLURM_SUBMIT_DIR:-$HOME/ml-splat-xai}"
echo "task=$T dataset=$DS seed=$SEED mode=$MODE node=$(hostname) account=$SLURM_JOB_ACCOUNT"
python perturbation/run_removal.py \
    --ckpt "$CKPT" --ckpt_seed "$SEED" --data_dir "${DATA_DIRS[$DS]}" \
    --modes "$MODE" --out "$OUT" --check "$@"
