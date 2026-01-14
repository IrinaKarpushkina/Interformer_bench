                                                                                                                                                                                                                                                                                 prepare_ligands_new.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os
import sys

def prepare_ligands_for_interformer(input_csv_path, output_sdf_path, smiles_column='ligand', max_mol_weight=800):
    """
    Готовит лиганды для докинга.
    Автоматически определяет разделитель (CSV/TSV/Semicolon).
    """
    print("--- Starting flexible ligand preparation script (AUTO-DELIMITER) ---")

    output_dir = os.path.dirname(output_sdf_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdb_id = os.path.basename(output_sdf_path).split('_')[0]

    if not os.path.exists('result'):
        os.makedirs('result')

    report_csv_path = os.path.join('result', f'ligand_preparation_report_{pdb_id}.csv')

    # --- АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ РАЗДЕЛИТЕЛЯ ---
    detected_sep = ',' # По умолчанию
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if ';' in first_line:
                detected_sep = ';'
            elif '\t' in first_line:
                detected_sep = '\t'
    except Exception as e:
        print(f"Warning: Could not sniff separator: {e}")

    print(f"Detected separator: '{detected_sep}'")

    # --- ЧТЕНИЕ CSV ---
    df = None
    try:
        # Попытка 1: С определенным разделителем
        try:
            df = pd.read_csv(input_csv_path, sep=detected_sep, on_bad_lines='skip')
        except TypeError:
            df = pd.read_csv(input_csv_path, sep=detected_sep, error_bad_lines=False)

    except Exception as e:
        print(f"CRITICAL: Failed to read CSV: {e}")
        return [], [("Global Error", "CSV file is unreadable")]

    # Очистка имен колонок
    df.columns = [str(c).strip() for c in df.columns]

    # Поиск колонки SMILES
    if smiles_column not in df.columns:
        candidates = [c for c in df.columns if 'smi' in str(c).lower()]
        if candidates:
            smiles_column = candidates[0]
            print(f"Auto-selected SMILES column: {smiles_column}")
        else:
            print(f"ERROR: Could not find SMILES column. Available: {list(df.columns)}")
            return [], [("Global Error", f"Column '{smiles_column}' missing")]

    writer = Chem.SDWriter(output_sdf_path)
    print(f"Processing {len(df)} ligands...")

    successful_ligands = []
    failed_ligands = []
    report_data = []

    for index, row in df.iterrows():
        try:
            smiles = row.get(smiles_column)

            # Попытка найти имя
            ligand_name = f"ligand_row_{index}"
            for col in ['Molecule ID', 'ID', 'Name', 'name', 'id', 'chembl_id', 'molecule_chembl_id']:
                if col in df.columns:
                    val = row.get(col)
                    if not pd.isna(val):
                        ligand_name = str(val)
                        break
        except Exception as e:
            failed_ligands.append((f"Row {index}", f"Row Access Error: {e}"))
            continue

        # 1. Проверка SMILES
        if pd.isna(smiles) or str(smiles).strip() == "":
            failed_ligands.append((ligand_name, "Missing SMILES"))
            report_data.append({'ligand_name': ligand_name, 'status': 'error', 'details': "Missing SMILES"})
            continue

        smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(smiles)

        # 2. Валидность
        if mol is None:
            failed_ligands.append((ligand_name, "Invalid SMILES structure"))
            report_data.append({'ligand_name': ligand_name, 'status': 'error', 'details': "Invalid SMILES"})
            continue

        # 3. Фильтр массы
        try:
            mw = Descriptors.MolWt(mol)
            if mw > max_mol_weight:
                reason = f"Mass too high ({mw:.1f} > {max_mol_weight})"
                failed_ligands.append((ligand_name, reason))
                report_data.append({'ligand_name': ligand_name, 'status': 'filtered', 'details': reason})
                continue
        except: pass

        # 4. 3D Генерация
        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            embed_res = AllChem.EmbedMolecule(mol_h, params)

            if embed_res == -1:
                params.useRandomCoords = True
                embed_res = AllChem.EmbedMolecule(mol_h, params)
                if embed_res == -1:
                    failed_ligands.append((ligand_name, "Embedding 3D failed"))
                    report_data.append({'ligand_name': ligand_name, 'status': 'error', 'details': "Embedding 3D failed"})
                    continue

            # Оптимизация
            try:
                if AllChem.UFFOptimizeMolecule(mol_h) == 0:
                    status = 'UFF OK'
                elif AllChem.MMFFOptimizeMolecule(mol_h) == 0:
                    status = 'MMFF OK'
                else:
                    status = 'Optimization Failed (Saved Raw 3D)'
            except:
                status = 'Optimization Error'

            mol_h.SetProp("_Name", ligand_name)
            writer.write(mol_h)
            successful_ligands.append(ligand_name)
            report_data.append({'ligand_name': ligand_name, 'status': 'success', 'details': status})

        except Exception as e:
            failed_ligands.append((ligand_name, str(e)))
            report_data.append({'ligand_name': ligand_name, 'status': 'error', 'details': str(e)})

    writer.close()

    try:
        pd.DataFrame(report_data).to_csv(report_csv_path, index=False)
    except: pass

    print(f"Preparation complete. Success: {len(successful_ligands)}, Failed/Filtered: {len(failed_ligands)}")
    return successful_ligands, failed_ligands

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_ligands_new.py <input_csv> <output_sdf>")
        sys.exit(1)
    prepare_ligands_for_interformer(sys.argv[1], sys.argv[2])
