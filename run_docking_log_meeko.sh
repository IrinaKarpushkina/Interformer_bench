#!/bin/bash
#SBATCH --job-name=interformer_1tqn
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=dock_%j.out
#SBATCH --error=dock_%j.err
#SBATCH -p aichem

# 1. Активация окружения
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate interformer

# 2. Переход в папку проекта (где лежат наши скрипты)
cd /mnt/tank/scratch/ikarpushkina/Interformer_new/Interformer_bench

# Явно задаем переменную PYTHONPATH, чтобы питон видел текущую папку
export PYTHONPATH=$PYTHONPATH:.

# 3. Запуск пайплайна
# Аргументы:
# -p: PDB код (например, 1tqn). Скрипт будет искать data/1tqn_ligand.pdb
# -w: Рабочая директория (например, vs)
# -d: Папка для вывода SDF (например, energy_VS)
# -l: Имя CSV файла без .csv (лежит в data/)

/mnt/tank/scratch/ikarpushkina/miniconda3/envs/interformer/bin/python master_pipeline_log_meeko.py -p 4f65 -w vs -d energy_VS -l FGFR1_Ki_WT_ChEMBL_134_nodubl -o 32,32
