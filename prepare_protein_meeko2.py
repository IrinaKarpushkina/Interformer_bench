"""Script for preparing reference ligand and Meeko-cleaned protein — version 2 with debug logging."""

import os
import shutil
import sys
import argparse
import logging
from pathlib import Path
from typing import List

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    from rdkit import Chem
    from rdkit.Chem import SplitMolByPDBResidues
except ImportError as e:
    print(f"ERROR: Required packages not installed: {e}", file=sys.stderr)
    print("Install with: pip install meeko rdkit", file=sys.stderr)
    raise

# Patch for RDKit compatibility
if not hasattr(Chem.Mol, 'HasQuery'):
    def _has_query(self):
        return False
    Chem.Mol.HasQuery = _has_query

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prepare_meeko2.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def _generate_clean_pdb(source: Path, destination: Path, chain_id: str,
                        include_het: bool = False, include_cofactors: bool = False,
                        include_waters: bool = False) -> None:
    chain_id = chain_id.strip() or "A"
    selected_lines = []
    chain_col = 21
    record_whitelist = {"ATOM"}
    if include_het or include_cofactors:
        record_whitelist.add("HETATM")

    logging.debug(f"Generating cleaned PDB for chain {chain_id}, waters={include_waters}")

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line.startswith("HETATM") and "HETATM" not in record_whitelist:
                continue
            if not include_waters and line[17:20].strip() == "HOH":
                continue
            if line[chain_col].strip() != chain_id:
                continue
            selected_lines.append(line)

    if not selected_lines:
        raise RuntimeError(f"No atoms selected for {source.name} with chain '{chain_id}'.")
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.writelines(selected_lines)
    logging.info(f"Cleaned PDB saved: {destination} ({len(selected_lines)} lines)")


def _prepare_receptor_meeko(cleaned_pdb: Path, pdbqt_path: Path, pdb_path_out: Path) -> None:
    logging.info(f"Starting Meeko preparation from {cleaned_pdb}")
    try:
        templates = ResidueChemTemplates.create_from_defaults()
        mk_prep = MoleculePreparation.from_config({})
        with open(cleaned_pdb, 'r') as f:
            pdb_string = f.read()
        
        polymer = Polymer.from_pdb_string(
            pdb_string, templates, mk_prep, allow_bad_res=True, default_altloc='A'
        )
        
        pdbqt_tuple = PDBQTWriterLegacy.write_from_polymer(polymer)
        rigid_pdbqt, flex_pdbqt_dict = pdbqt_tuple
        
        with open(pdbqt_path, 'w') as f:
            f.write(rigid_pdbqt)
        
        pdb_string = polymer.to_pdb()
        with open(pdb_path_out, 'w') as f:
            f.write(pdb_string)
        
        if flex_pdbqt_dict:
            flex_out = pdbqt_path.parent / (pdbqt_path.stem + "_flex.pdbqt")
            with open(flex_out, 'w') as f:
                f.write("".join(flex_pdbqt_dict.values()))
            logging.info(f"Flexible residues saved: {flex_out} ({len(flex_pdbqt_dict)} entries)")
        
        logging.info(f"Meeko preparation complete: rigid PDBQT {pdbqt_path}, cleaned PDB {pdb_path_out}")
    except Exception as e:
        logging.exception(f"Meeko preparation failed for {cleaned_pdb}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Prepare reference ligand and Meeko-cleaned protein (v2)")
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-w", "--work_path", default="vs")
    parser.add_argument("-c", "--chain", default="A")
    parser.add_argument("-r", "--ref_ligand", required=True)
    args = parser.parse_args()

    data_dir = Path(args.work_path) / "data"
    raw_pdb = data_dir / f"{args.pdb}.pdb"
    ligand_pdb = data_dir / f"{args.pdb}_ligand.pdb"
    meeko_pdb = data_dir / f"{args.pdb}_meeko.pdb"

    logging.info(f"STARTING PREPARATION v2 for PDB={args.pdb}, chain={args.chain}, ref_ligand={args.ref_ligand}")
    logging.info(f"Input raw PDB: {raw_pdb.resolve()}")

    if not raw_pdb.exists():
        logging.critical(f"RAW PDB NOT FOUND: {raw_pdb}")
        sys.exit(1)

    if ligand_pdb.exists() and meeko_pdb.exists():
        logging.info(f"Files already exist → skipping: {ligand_pdb} and {meeko_pdb}")
        return

    # Загрузка PDB
    logging.debug("Loading raw PDB with RDKit...")
    mol = Chem.MolFromPDBFile(str(raw_pdb), sanitize=False, removeHs=False)
    if mol is None:
        logging.critical(f"RDKit failed to load PDB: {raw_pdb}")
        sys.exit(1)
    logging.info(f"Raw PDB loaded: {mol.GetNumAtoms()} atoms")

    # Поиск лиганда
    res_mols = SplitMolByPDBResidues(mol)
    ligand_parts: List[Chem.Mol] = []
    standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}

    logging.info(f"Scanning {len(res_mols)} residues for ref ligand '{args.ref_ligand}' in chain '{args.chain}'")

    for res_key, res_mol in res_mols.items():
        if res_mol.GetNumAtoms() == 0:
            continue
        atom = res_mol.GetAtomWithIdx(0)
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        res_name = info.GetResidueName().strip()
        chain = info.GetChainId().strip()
        if chain != args.chain:
            continue
        if res_name == 'HOH':
            continue
        if res_name in standard_aa:
            continue
        if res_name != args.ref_ligand:
            continue
        has_carbon = any(a.GetAtomicNum() == 6 for a in res_mol.GetAtoms())
        if not has_carbon:
            continue
        ligand_parts.append(res_mol)
        logging.info(f"Found ligand candidate: {res_key}, resname={res_name}, atoms={res_mol.GetNumAtoms()}")

    if not ligand_parts:
        logging.error(f"NO REFERENCE LIGAND '{args.ref_ligand}' FOUND in chain {args.chain}")
        logging.info("Dumping first 30 HETATM lines from PDB for diagnostics:")
        with open(raw_pdb) as f:
            hetatm_lines = [line.strip() for line in f if line.startswith("HETATM")][:30]
        logging.info("\n".join(hetatm_lines))
        sys.exit(1)

    ligand_mol = max(ligand_parts, key=lambda m: m.GetNumAtoms())
    logging.info(f"Selected ligand: {ligand_mol.GetNumAtoms()} atoms")

    Chem.MolToPDBFile(ligand_mol, str(ligand_pdb))
    logging.info(f"Reference ligand saved: {ligand_pdb}")

    # Нормализация resname на LIG
    for atom in ligand_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info:
            info.SetResidueName("LIG")
    ligand_lig = data_dir / f"{args.pdb}_ligand_LIG.pdb"
    Chem.MolToPDBFile(ligand_mol, str(ligand_lig))
    logging.info(f"Normalized ligand (resname LIG) saved: {ligand_lig}")

    # Подготовка белка
    temp_dir = Path(f"temp_prep_{args.pdb}")
    temp_dir.mkdir(exist_ok=True)
    cleaned_pdb = temp_dir / f"{args.pdb}_cleaned.pdb"
    temp_pdbqt = temp_dir / f"{args.pdb}_temp.pdbqt"

    _generate_clean_pdb(raw_pdb, cleaned_pdb, args.chain)
    _prepare_receptor_meeko(cleaned_pdb, temp_pdbqt, meeko_pdb)

    shutil.rmtree(temp_dir, ignore_errors=True)
    logging.info("PREPARATION FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
