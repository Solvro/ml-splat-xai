#!/bin/bash -l
# One-off: build the persistent venv used by submit_removal.sbatch (same pins as ml-splat-xai-revision).
#SBATCH --job-name=xsplain-venv
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/julka/xsplain_removal/slurm_logs/%x_%j.out
source /etc/profile
module load Python/3.12.3-GCCcore-13.3.0 || { echo 'module load failed' >&2; exit 3; }
set -eo pipefail
VENV=/lustre/pd03/hpc-danbor2008-1756464546/julka/envs/xsplain
python3 -m venv "$VENV"
source "$VENV/bin/activate"
set -u
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/pip-cache"
pip install --upgrade pip
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r /home/domgal7481/ml-splat-xai-revision/requirements.txt pandas
python -c "import torch, pytorch_lightning, torchmetrics, einops, plyfile, sklearn, pandas, matplotlib; print('OK', torch.__version__)"
