"""Script for preparing reference ligand and cleaned protein using Meeko."""

import os
import shutil
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Any, List

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    from rdkit import Chem
    from rdkit.Chem import SplitMolByPDBResidues
except ImportError as e:
    print(f"ERROR: Required packages not installed: {e}", file=sys.stderr)
    print("Install with: pip install meeko rdkit", file=sys.stderr)
    raise

# Patch for RDKit compatibility: add HasQuery method if missing
if not hasattr(Chem.Mol, 'HasQuery'):
    def _has_query(self):
        """Compatibility method for older Meeko versions."""
        return False
    Chem.Mol.HasQuery = _has_query

def _generate_clean_pdb(
    source: Path,
    destination: Path,
    chain_id: str,
    include_het: bool,
    include_cofactors: bool,
    include_waters: bool,
) -> None:
    """Filter the original PDB to the requested chain and optional records."""
    chain_id = chain_id.strip() or "A"
    selected_lines = []
    chain_col = 21
    record_whitelist = {"ATOM"}

    if include_het or include_cofactors:
        record_whitelist.add("HETATM")

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
        raise RuntimeError(
            f"No atoms selected for {source.name} with chain '{chain_id}'."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.writelines(selected_lines)

def _prepare_receptor_meeko(cleaned_pdb: Path, pdbqt_path: Path, pdb_path_out: Path) -> None:
    """Prepare receptor using Meeko Python API."""
    try:
        # Create templates and molecule preparation
        templates = ResidueChemTemplates.create_from_defaults()
        mk_prep = MoleculePreparation.from_config({})

        # Read PDB file and create Polymer
        with open(cleaned_pdb, 'r') as f:
            pdb_string = f.read()
        
        polymer = Polymer.from_pdb_string(
            pdb_string,
            templates,
            mk_prep,
            allow_bad_res=True,  # Automatically remove residues that don't match templates
            default_altloc='A'   # Use first alternative location by default
        )
        
        # Get PDBQT through PDBQTWriterLegacy
        pdbqt_tuple = PDBQTWriterLegacy.write_from_polymer(polymer)
        rigid_pdbqt, flex_pdbqt_dict = pdbqt_tuple
        
        # Save rigid part (main receptor) in PDBQT
        with open(pdbqt_path, 'w') as f:
            f.write(rigid_pdbqt)
        
        # Save cleaned and prepared PDB file
        pdb_string = polymer.to_pdb()
        with open(pdb_path_out, 'w') as f:
            f.write(pdb_string)
        
        # If there are flexible residues, save them separately
        if flex_pdbqt_dict:
            flex_out_path = pdbqt_path.parent / (pdbqt_path.stem + "_flex.pdbqt")
            all_flex_pdbqt = "".join(flex_pdbqt_dict.values())
            with open(flex_out_path, 'w') as f:
                f.write(all_flex_pdbqt)
        
        print(f"Protein {pdbqt_path.stem}: prepared using Meeko Python API")
        
    except Exception as e:
        raise RuntimeError(f"Meeko Python API receptor preparation failed: {e}") from e

def main():
    parser = argparse.ArgumentParser(description="Prepare reference ligand and Meeko-cleaned protein.")
    parser.add_argument("-p", "--pdb", required=True, help="PDB ID (e.g., 5mo4)")
    parser.add_argument("-w", "--work_path", default="vs", help="Work path (default: vs)")
    parser.add_argument("-c", "--chain", default="A", help="Chain ID (default: A)")
    parser.add_argument("-r", "--ref_ligand", required=True, help="Reference ligand RESNAME (e.g., 0S9)")
    args = parser.parse_args()

    data_dir = os.path.join(args.work_path, "data")
    raw_pdb = os.path.join(data_dir, f"{args.pdb}.pdb")
    if not os.path.exists(raw_pdb):
        raise FileNotFoundError(f"Raw PDB file not found: {raw_pdb}")

    ligand_pdb = os.path.join(data_dir, f"{args.pdb}_ligand.pdb")
    meeko_pdb = os.path.join(data_dir, f"{args.pdb}_meeko.pdb")

    if os.path.exists(ligand_pdb) and os.path.exists(meeko_pdb):
        print(f"Files for {args.pdb} already exist, skipping preparation.")
        return

    # Extract reference ligand by RESNAME
    mol = Chem.MolFromPDBFile(raw_pdb, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to load PDB: {raw_pdb}")

    res_mols = SplitMolByPDBResidues(mol)
    ligand_parts = []
    standard_aa = set([
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ])

    for res_key, res_mol in res_mols.items():
        if res_mol.GetNumAtoms() < 1: continue
        atom = res_mol.GetAtomWithIdx(0)
        res_info = atom.GetPDBResidueInfo()
        if res_info is None: continue
        res_name = res_info.GetResidueName().strip()
        chain = res_info.GetChainId().strip()
        if chain != args.chain: continue
        if res_name == 'HOH': continue
        if res_name in standard_aa: continue
        if res_name != args.ref_ligand: continue  # Новый фильтр по REF_LIGAND
        has_carbon = any(a.GetAtomicNum() == 6 for a in res_mol.GetAtoms())
        if not has_carbon: continue
        ligand_parts.append(res_mol)

    if not ligand_parts:
        raise ValueError(f"No reference ligand '{args.ref_ligand}' found in chain {args.chain} of {raw_pdb}")

    # Take the first matching ligand (or max by size if multiple)
    ligand_mol = ligand_parts[0]  # Или max(ligand_parts, key=lambda m: m.GetNumAtoms())
    Chem.MolToPDBFile(ligand_mol, ligand_pdb)
    print(f"Saved reference ligand '{args.ref_ligand}' to {ligand_pdb}")

    # Prepare cleaned protein
    temp_dir = f"temp_prep_{args.pdb}"  # Уникальный temp для каждого PDB
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    cleaned_pdb = os.path.join(temp_dir, f"{args.pdb}_cleaned.pdb")
    temp_pdbqt = os.path.join(temp_dir, f"{args.pdb}_temp.pdbqt")

    _generate_clean_pdb(
        Path(raw_pdb),
        Path(cleaned_pdb),
        args.chain,
        include_het=False,
        include_cofactors=False,
        include_waters=False,
    )

    _prepare_receptor_meeko(Path(cleaned_pdb), Path(temp_pdbqt), Path(meeko_pdb))
    print(f"Saved Meeko-processed protein to {meeko_pdb}")

    # Cleanup temp files
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
