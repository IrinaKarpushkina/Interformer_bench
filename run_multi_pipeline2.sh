#!/bin/bash

# SLURM настройки (как в твоём run_pipeline_log.sh)
#SBATCH --job-name=interformer_multi
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=dock_multi_%j.out
#SBATCH --error=dock_multi_%j.err
#SBATCH -p aichem


# Массивы из твоего списка
PROTEINS=("1g5m" "7awe" "5mo4" "7kk3" "2z5x" "3lxk" "3jy9" "3eyg" "1g5m" "3mjg" "4tz4" "5jkv" "4ase" "6jok" "6gqj" "4zau")
LIGANDS=("BCL2_Ki_WT_ChEMBL_252_nodubl.csv" "PSMB5_Ki_WT_ChEMBL_88_nodubl.csv" "ABL1_BCR-ABL_Ki_WT_ChEMBL_693_nodubl.csv" "PARP1_Ki_WT_ChEMBL_1075_nodubl.csv" "MAO-B_Ki_WT_ChEMBL_246_nodubl.csv" "JAK3_Ki_WT_ChEMBL_786_nodubl.csv" "JAK2_Ki_WT_ChEMBL_2027_nodubl.csv" "JAK1_Ki_WT_ChEMBL_2255_nodubl.csv" "BCL2L1_BCL-XL_Ki_WT_ChEMBL_287_nodubl.csv" "PDGFRB_Ki_WT_ChEMBL_275_nodubl.csv" "CRBN_Ki_WT_ChEMBL_127_nodubl.csv" "CYP19A1_Aromatase_Ki_WT_ChEMBL_548_nodubl.csv" "VEGFR2_Ki_WT_ChEMBL_875_nodubl.csv" "PDGFRA_Ki_WT_curated_250_nodubl.csv" "KIT_Ki_WT_curated_1298_nodubl.csv" "EGFR_Ki_WT_curated_251_nodubl.csv")
REF_LIGANDS=("BAK" "ONX" "ASC+NIL" "TLZ" "RAG" "TOF" "STU" "STU" "BAK" "STI" "LEN" "LZ0" "TIV" "STI" "SCF" "OSM")
SAFE_CHAINS=("A" "L" "A" "C" "A" "A" "A" "A" "A" "B" "C" "A" "A" "A" "A" "A")

# Если аргументы переданы — используем их как список PDB, иначе все
if [ $# -gt 0 ]; then
    selected_pdbs=("$@")
else
    selected_pdbs=("${PROTEINS[@]}")
fi

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate interformer
cd /mnt/tank/scratch/ikarpushkina/Interformer_new/Interformer_bench

# Последовательный цикл по выбранным PDB
for pdb in "${selected_pdbs[@]}"; do
    # Найти индекс PDB в массиве
    index=-1
    for i in "${!PROTEINS[@]}"; do
        if [[ "${PROTEINS[$i]}" == "$pdb" ]]; then
            index=$i
            break
        fi
    done

    if [[ $index -eq -1 ]]; then
        echo "ERROR: PDB $pdb not found in PROTEINS list. Skipping."
        continue
    fi

    ligand_csv="${LIGANDS[$index]}"
    ref_ligand="${REF_LIGANDS[$index]}"
    chain="${SAFE_CHAINS[$index]}"
    ligand_name="${ligand_csv%.csv}"  # Без .csv для -l

    echo "Processing $pdb with ligand $ligand_csv, ref $ref_ligand, chain $chain"

    # Шаг подготовки
    python prepare_protein_no_meeko.py -p "$pdb" -w vs -c "$chain" -r "$ref_ligand"

conda deactivate
conda activate interformer2

    # Основной пайплайн
    python master_pipeline_log_meeko_centering.py -p "$pdb" -w vs -d "energy_VS_$pdb" -l "$ligand_name" -o 32,32
done
