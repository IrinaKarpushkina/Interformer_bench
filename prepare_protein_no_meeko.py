"""Simple PDB cleaner."""
import os
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SplitMolByPDBResidues

def _generate_clean_pdb(source: Path, destination: Path, chain_id: str):
    chain_id = chain_id.strip() or "A"
    selected_lines = []
    found_atoms = 0
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                if len(line) > 21 and line[21] == chain_id:
                    selected_lines.append(line)
                    found_atoms += 1
            elif line.startswith(("TER", "END")):
                selected_lines.append(line)
    
    # Fallback: Если цепь не нашли, берем всё (бывает в PDB без цепей)
    if found_atoms == 0:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("ATOM"):
                    selected_lines.append(line)
                    found_atoms += 1
    
    if found_atoms == 0:
        raise RuntimeError(f"No atoms found in {source}")
        
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.writelines(selected_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-w", "--work_path", default="vs")
    parser.add_argument("-c", "--chain", default="A")
    parser.add_argument("-r", "--ref_ligand", required=True)
    args = parser.parse_args()

    data_dir = os.path.join(args.work_path, "data")
    raw_pdb = os.path.join(data_dir, f"{args.pdb}.pdb")
    if not os.path.exists(raw_pdb):
        raise FileNotFoundError(f"Raw PDB not found: {raw_pdb}")

    # Output files
    ligand_pdb = os.path.join(data_dir, f"{args.pdb}_ligand.pdb")
    meeko_pdb = os.path.join(data_dir, f"{args.pdb}_meeko.pdb")

    # 1. Extract Ligand
    mol = Chem.MolFromPDBFile(raw_pdb, sanitize=False, removeHs=False)
    res_mols = SplitMolByPDBResidues(mol)
    
    # Find specific ligand
    candidates = []
    for m in res_mols.values():
        if m.GetNumAtoms() < 1: continue
        info = m.GetAtomWithIdx(0).GetPDBResidueInfo()
        if info.GetResidueName().strip() == args.ref_ligand:
            if info.GetChainId().strip() == args.chain:
                candidates.append(m)
    
    if not candidates:
        print(f"Warning: Ref ligand {args.ref_ligand} not found in chain {args.chain}. Checking other chains...")
        # Fallback search without chain
        for m in res_mols.values():
            if m.GetNumAtoms() < 1: continue
            info = m.GetAtomWithIdx(0).GetPDBResidueInfo()
            if info.GetResidueName().strip() == args.ref_ligand:
                candidates.append(m)
    
    if candidates:
        # Take largest (in case of alt locs)
        best_lig = max(candidates, key=lambda x: x.GetNumAtoms())
        Chem.MolToPDBFile(best_lig, ligand_pdb)
        print(f"Saved ligand: {ligand_pdb}")
    else:
        raise ValueError(f"Ligand {args.ref_ligand} not found anywhere in {raw_pdb}")

    # 2. Clean Protein
    _generate_clean_pdb(Path(raw_pdb), Path(meeko_pdb), args.chain)
    print(f"Saved cleaned protein: {meeko_pdb}")

if __name__ == "__main__":
    main()
