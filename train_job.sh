#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=multipie_train_new
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate prj2

# Change to project directory
cd /home/gu.yunp/cs7180/prj2/mywork/DPR

# Start training
python train_multipie_dpr.py

