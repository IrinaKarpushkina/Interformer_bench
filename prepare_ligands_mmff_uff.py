import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os
import sys

def prepare_ligands_for_interformer(input_csv_path, output_sdf_path, smiles_column='ligand', max_mol_weight=500):
    """
    Гибкая функция для подготовки лигандов.
    Принимает пути в качестве аргументов.
    """
    print("--- Запуск гибкого скрипта подготовки лигандов ---")

    output_dir = os.path.dirname(output_sdf_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория: {output_dir}")

    # Извлечение PDB из пути output_sdf_path (например, '1tqn' из 'energy_VS/uff/1tqn_uff.sdf')
    pdb_id = os.path.basename(output_sdf_path).split('_')[0]  # Берем первый элемент до '_'
    trouble_csv = os.path.join('result', f'trouble_ligands_{pdb_id}.csv')

    if not os.path.exists('result'):
        os.makedirs('result')
        print(f"Создана директория: result")

    try:
        df = pd.read_csv(input_csv_path, delimiter=',')
        if smiles_column not in df.columns:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Колонка '{smiles_column}' не найдена в файле {input_csv_path}.")
            print(f"Найденные колонки: {list(df.columns)}")
            return

    except FileNotFoundError:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл не найден: {input_csv_path}")
        return
    except Exception as e:
        print(f"Произошла ошибка при чтении CSV: {e}")
        return

    writer = Chem.SDWriter(output_sdf_path)
    print(f"Начинаем обработку {len(df)} лигандов из колонки '{smiles_column}'...")

    success_count, filtered_out_count, error_count = 0, 0, 0
    trouble_data = []

    for index, row in df.iterrows():
        smiles = row.get(smiles_column)
        if pd.isna(smiles):
            error_count += 1
            continue

        mol = Chem.MolFromSmiles(str(smiles).strip())

        if mol is None:
            error_count += 1
            trouble_data.append({'SMILES': smiles, 'ligand_source_row': f"ligand_source_row_{index+2}", 'error': 'Invalid SMILES'})
            continue

        if Descriptors.MolWt(mol) > max_mol_weight:
            filtered_out_count += 1
            continue

        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            AllChem.EmbedMolecule(mol_h, params)

            # Попытка оптимизации с UFF
            uff_result = AllChem.UFFOptimizeMolecule(mol_h)
            if uff_result == 0:
                mol_h.SetProp("_Name", f"ligand_source_row_{index+2}")
                writer.write(mol_h)
                success_count += 1
            else:
                # Попытка оптимизации с MMFF, если UFF не сработал
                mmff_result = AllChem.MMFFOptimizeMolecule(mol_h)
                if mmff_result == 0:
                    mol_h.SetProp("_Name", f"ligand_source_row_{index+2}")
                    writer.write(mol_h)
                    success_count += 1
                else:
                    error_count += 1
                    trouble_data.append({'SMILES': smiles, 'ligand_source_row': f"ligand_source_row_{index+2}", 'error': f'UFF failed (code {uff_result}), MMFF failed (code {mmff_result})'})
        except (RuntimeError, ValueError) as e:
            error_count += 1
            trouble_data.append({'SMILES': smiles, 'ligand_source_row': f"ligand_source_row_{index+2}", 'error': str(e)})

    writer.close()

    # Сохранение проблемных молекул
    if trouble_data:
        trouble_df = pd.DataFrame(trouble_data)
        trouble_df.to_csv(trouble_csv, index=False)
        print(f"Проблемные молекулы сохранены в: {trouble_csv}")

    print("\n--- Отчет ---")
    print(f"Успешно сохранено: {success_count} молекул")
    print(f"Отфильтровано по массе (> {max_mol_weight} Da): {filtered_out_count} молекул")
    print(f"Пропущено из-за ошибок: {error_count} молекул")
    print(f"Результат сохранен в файл: {output_sdf_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Ошибка: неверное количество аргументов.")
        print("Пример использования: python prepare_ligands_flexible.py <путь_к_входному_csv> <путь_к_выходному_sdf>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_sdf = sys.argv[2]
    prepare_ligands_for_interformer(input_csv, output_sdf)
