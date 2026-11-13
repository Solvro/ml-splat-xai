#!/bin/bash -l
#SBATCH --job-name=xsplain-merge
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%j.out
source /etc/profile
module load Python/3.12.3-GCCcore-13.3.0 || exit 3
set -eo pipefail
source /lustre/pd03/hpc-danbor2008-1756464546/julka/envs/xsplain/bin/activate
cd ~/ml-splat-xai
R=${1:-/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/shapesplat/default}
python perturbation/run_removal.py --merge "$R"
for m in acc p_true; do python perturbation/plot_removal.py "$R/summary_all_seeds.csv" --metric $m --out "$R/removal_curves_$m"; done
echo "=== SUMMARY"
cat "$R/summary_all_seeds.csv"
