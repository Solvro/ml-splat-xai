#!/bin/bash -l
# CPU smoke test (20 samples, seed 777) for submit_removal_voxpct.sbatch.
#SBATCH --job-name=xsplain-voxpct-smoke
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0
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
OUT_ROOT=$PD/julka/xsplain_removal/smoke_voxpct
OUT=$OUT_ROOT/seed_${SEED}/voxel
echo "task=$SLURM_ARRAY_TASK_ID seed=$SEED node=$(hostname) account=$SLURM_JOB_ACCOUNT out=$OUT args=$*"
python perturbation/run_removal.py --ckpt "$CKPT" --ckpt_seed "$SEED" --modes voxel \
    --voxel_unit percent --out "$OUT" --check --limit 20 "$@"
