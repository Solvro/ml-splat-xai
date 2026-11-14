#!/bin/bash -l
# CPU smoke test of run_removal.py on MACGS and toys (seed 777, 10 samples, both modes).
#SBATCH --job-name=xsplain-smoke-ds
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%j.out
source /etc/profile
module load Python/3.12.3-GCCcore-13.3.0 || exit 3
set -eo pipefail
source /lustre/pd03/hpc-danbor2008-1756464546/julka/envs/xsplain/bin/activate
cd ~/ml-splat-xai
PD=/lustre/pd03/hpc-danbor2008-1756464546
for pair in macgs:$PD/datasets/MACGS/MACGS_ply toys:$PD/datasets/toys/data; do
  DS=${pair%%:*}; DD=${pair#*:}
  python perturbation/run_removal.py --ckpt $PD/xsplain_revision/checkpoints/$DS/default_90ep/seed_777/stage2_compensated.pt \
      --ckpt_seed 777 --data_dir "$DD" --limit 10 --n_random 2 --check --out $PD/julka/xsplain_removal/smoke_$DS
done
