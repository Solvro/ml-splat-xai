#!/bin/bash
#SBATCH --job-name=sanity_test           # nowa nazwa joba
#SBATCH --partition=lem-gpu-short       # zmień jeśli potrzebujesz innej partycji
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=output_sanity_%j.txt   # nowy plik output
#SBATCH --error=error_sanity_%j.txt     # nowy plik error

source /etc/profile
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

python sanity_check.py                # nowy skrypt lub ten sam, jeśli chcesz

deactivate
