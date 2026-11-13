#!/bin/bash -l
# Voxel removal where the level = % of each object's NON-EMPTY voxels (branch voxel-percent-nonempty).
# One task per checkpoint seed; point (splat) removal is unchanged and reused from shapesplat/default.
#   sbatch perturbation/submit_removal_voxpct.sbatch            # full run
#   sbatch --array=0 perturbation/submit_removal_voxpct.sbatch --limit 20   # smoke (extra args -> run_removal.py)
#SBATCH --job-name=xsplain-voxpct
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%6
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%A_%a.err
source /etc/profile
module load Python/3.12.3-GCCcore-13.3.0 || exit 3
set -eo pipefail
source /lustre/pd03/hpc-danbor2008-1756464546/julka/envs/xsplain/bin/activate
set -u
cd ~/ml-splat-xai
SEEDS=(777 778 779)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
PD=/lustre/pd03/hpc-danbor2008-1756464546
CKPT=$PD/xsplain_revision/checkpoints/shapesplat/default/seed_${SEED}/stage2_compensated.pt
OUT_ROOT=${OUT_ROOT:-$PD/julka/xsplain_removal/shapesplat/voxel_pct_nonempty}
OUT=$OUT_ROOT/seed_${SEED}/voxel
echo "task=$SLURM_ARRAY_TASK_ID seed=$SEED node=$(hostname) account=$SLURM_JOB_ACCOUNT out=$OUT args=$*"
python perturbation/run_removal.py --ckpt "$CKPT" --ckpt_seed "$SEED" --modes voxel \
    --voxel_unit percent --out "$OUT" --check "$@"
