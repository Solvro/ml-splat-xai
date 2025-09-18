#!/bin/bash
#SBATCH --job-name=training_pointnet
#SBATCH --partition=lem-gpu-short   # zmień na lem-gpu-normal jeśli masz dostęp
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10 
#SBATCH --mem=96GB
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

source /etc/profile
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

python train_epic3_3.py

deactivate
