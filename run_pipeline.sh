#!/bin/bash
#SBATCH --job-name=interformer_1tqn
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=dock_%j.out
#SBATCH --error=dock_%j.err
#SBATCH -p aichem
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate interformer

cd /mnt/tank/scratch/ikarpushkina/Interformer_new/Interformer_bench

python master_pipeline.py -p 1tqn -w vs -d energy_VS -l CYP3a4_ic50_df -o 32,32
