import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os
import sys

def prepare_ligands_for_interformer(input_csv_path, output_sdf_path, smiles_column='ligand', max_mol_weight=500):
    """
    Prepares ligands for docking, with fallback to MMFF if UFF fails.
    Returns lists of successful and failed ligands.
    """
    print("--- Starting flexible ligand preparation script ---")

    output_dir = os.path.dirname(output_sdf_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    pdb_id = os.path.basename(output_sdf_path).split('_')[0]
    report_csv_path = os.path.join('result', f'ligand_preparation_report_{pdb_id}.csv')

    if not os.path.exists('result'):
        os.makedirs('result')
        print("Created directory: result")

    try:
        df = pd.read_csv(input_csv_path, delimiter=',')
        if smiles_column not in df.columns:
            print(f"CRITICAL ERROR: Column '{smiles_column}' not found in {input_csv_path}.")
            print(f"Found columns: {list(df.columns)}")
            return [], []
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File not found: {input_csv_path}")
        return [], []
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return [], []

    writer = Chem.SDWriter(output_sdf_path)
    print(f"Processing {len(df)} ligands from column '{smiles_column}'...")

    successful_ligands = []
    failed_ligands = []
    report_data = []

    for index, row in df.iterrows():
        smiles = row.get(smiles_column)
        ligand_name = f"ligand_source_row_{index+2}"

        if pd.isna(smiles):
            failed_ligands.append(ligand_name)
            report_data.append({'ligand_name': ligand_name, 'SMILES': '', 'status': 'error', 'details': 'Missing SMILES'})
            continue

        mol = Chem.MolFromSmiles(str(smiles).strip())

        if mol is None:
            failed_ligands.append(ligand_name)
            report_data.append({'ligand_name': ligand_name, 'SMILES': smiles, 'status': 'error', 'details': 'Invalid SMILES'})
            continue

        if Descriptors.MolWt(mol) > max_mol_weight:
            report_data.append({'ligand_name': ligand_name, 'SMILES': smiles, 'status': 'filtered', 'details': f'Exceeds max weight ({max_mol_weight} Da)'})
            continue

        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            AllChem.EmbedMolecule(mol_h, params)

            uff_result = AllChem.UFFOptimizeMolecule(mol_h)
            if uff_result == 0:
                mol_h.SetProp("_Name", ligand_name)
                writer.write(mol_h)
                successful_ligands.append(ligand_name)
                report_data.append({'ligand_name': ligand_name, 'SMILES': smiles, 'status': 'success', 'details': 'UFF optimization'})
            else:
                mmff_result = AllChem.MMFFOptimizeMolecule(mol_h)
                if mmff_result == 0:
                    mol_h.SetProp("_Name", ligand_name)
                    writer.write(mol_h)
                    successful_ligands.append(ligand_name)
                    report_data.append({'ligand_name': ligand_name, 'SMILES': smiles, 'status': 'success', 'details': 'MMFF optimization (UFF failed)'})
                else:
                    failed_ligands.append(ligand_name)
                    report_data.append({'ligand_name': ligand_name, 'SMILES': smiles, 'status': 'error', 'details': f'UFF failed (code {uff_result}), MMFF failed (code {mmff_result})'})
        except (RuntimeError, ValueError) as e:
            failed_ligands.append(ligand_name)
            report_data.append({'ligand_name': ligand_name, 'SMILES': smiles, 'status': 'error', 'details': str(e)})

    writer.close()

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_csv_path, index=False)
    print(f"Ligand preparation report saved to: {report_csv_path}")

    print("\n--- Report ---")
    print(f"Successfully saved: {len(successful_ligands)} molecules")
    print(f"Filtered by mass (> {max_mol_weight} Da): {len(report_df[report_df['status'] == 'filtered'])} molecules")
    print(f"Skipped due to errors: {len(failed_ligands)} molecules")
    print(f"Output SDF file: {output_sdf_path}")

    return successful_ligands, failed_ligands

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Error: Invalid number of arguments.")
        print("Usage: python prepare_ligands.py <input_csv_path> <output_sdf_path>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_sdf = sys.argv[2]
    prepare_ligands_for_interformer(input_csv, output_sdf)
