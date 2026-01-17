import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import pandas as pd
from rdkit import Chem
from prepare_ligands_new import prepare_ligands_for_interformer

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("master_pipeline_meeko2_debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

FINAL_RESULTS_DIR = ""
TEMP_ROOT = ""
MAX_LIGAND_DURATION = 1200

class PipelineStepFailed(Exception):
    pass

class LigandTimeout(Exception):
    pass

def run_command(command, custom_env=None, timeout=None):
    cmd_str = ' '.join(command)
    env = os.environ.copy()
    if custom_env:
        env.update(custom_env)
    logging.debug(f"RUN COMMAND: {cmd_str}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout
        )
        logging.debug(f"Command OK, stdout len={len(result.stdout)}")
        return result.stdout
    except Exception as e:
        logging.exception(f"Command failed: {cmd_str}")
        raise

def check_timeout(start_time, stage_name):
    elapsed = time.time() - start_time
    logging.debug(f"Elapsed at {stage_name}: {elapsed:.1f}s")
    if elapsed > MAX_LIGAND_DURATION:
        raise LigandTimeout(f"Timeout at {stage_name}")

def prepare_global_resources(work_path, pdb):
    logging.info(">>> Preparing global resources")
    assets_dir = os.path.join(work_path, f"global_assets_{pdb}")
    shutil.rmtree(assets_dir, ignore_errors=True)
    os.makedirs(assets_dir, exist_ok=True)

    raw_dir = os.path.join(work_path, f"raw_{pdb}")
    raw_pocket = os.path.join(raw_dir, "pocket")

    ref_ligand_sdf = os.path.join(assets_dir, "reference_ligand.sdf")
    ligand_file = os.path.join(raw_dir, f"{pdb}_ligand_LIG.pdb") if os.path.exists(os.path.join(raw_dir, f"{pdb}_ligand_LIG.pdb")) else os.path.join(raw_dir, f"{pdb}_ligand.pdb")
    logging.info(f"Obabel input: {ligand_file}")
    run_command(["obabel", ligand_file, "-p", "7.4", "-O", ref_ligand_sdf])

    temp_extract = os.path.join(work_path, f"temp_extract_{pdb}")
    shutil.rmtree(temp_extract, ignore_errors=True)
    os.makedirs(temp_extract, exist_ok=True)
    shutil.copy(ref_ligand_sdf, os.path.join(temp_extract, f"{pdb}_docked.sdf"))

    logging.info("Running pocket extraction...")
    run_command(["python", "tools/extract_pocket_by_ligand.py", raw_pocket + "/", temp_extract + "/", "0"])

    gen_pocket = os.path.join(raw_pocket, "output", f"{pdb}_pocket.pdb")
    if not os.path.exists(gen_pocket):
        logging.error(f"Pocket failed: {gen_pocket}")
        os.system(f"ls -la {os.path.join(raw_pocket, 'output')}")
        raise Exception("Pocket not generated")
    logging.info(f"Pocket: {gen_pocket}, size={os.path.getsize(gen_pocket)}")

    shutil.copy(gen_pocket, os.path.join(assets_dir, "pocket.pdb"))
    shutil.rmtree(temp_extract)
    return assets_dir

def run_isolated_docking(idx, ligand_name, mol, original_pdb, assets_dir, docking_path):
    start_time = time.time()
    fake_pdb = f"L{idx:03d}"
    iso_dir = os.path.join(TEMP_ROOT, fake_pdb)
    logging.info(f"[{ligand_name}] START isolated docking → {iso_dir}")

    try:
        current_stage = "Setup"
        logging.info(f"[{ligand_name}] {current_stage}")
        check_timeout(start_time, current_stage)
        shutil.rmtree(iso_dir, ignore_errors=True)
        for d in ["ligand", "pocket", "uff", "infer", "complex", "result"]:
            os.makedirs(os.path.join(iso_dir, d), exist_ok=True)

        pocket_dst = os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb")
        shutil.copy(os.path.join(assets_dir, "pocket.pdb"), pocket_dst)
        logging.info(f"Pocket copied: size={os.path.getsize(pocket_dst)}")

        ref_dst = os.path.join(iso_dir, "ligand", f"{fake_pdb}_docked.sdf")
        shutil.copy(os.path.join(assets_dir, "reference_ligand.sdf"), ref_dst)
        logging.info(f"Ref SDF copied: size={os.path.getsize(ref_dst)}")

        uff_path = os.path.join(iso_dir, "uff", f"{fake_pdb}_uff.sdf")
        if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
            raise ValueError(f"Bad mol {ligand_name}: atoms={mol.GetNumAtoms() if mol else 0}, confs={mol.GetNumConformers() if mol else 0}")
        
        logging.info(f"Writing UFF {ligand_name}: {mol.GetNumAtoms()} atoms")
        with Chem.SDWriter(uff_path) as w:
            w.write(mol)

        if not os.path.exists(uff_path) or os.path.getsize(uff_path) == 0:
            raise FileNotFoundError(f"UFF not created: {uff_path}")

        logging.info(f"UFF OK: {uff_path}, size={os.path.getsize(uff_path)}")

        input_csv = os.path.join(iso_dir, f"{fake_pdb}_input.csv")
        with open(input_csv, 'w') as f:
            f.write(f"Target,Molecule ID,pose_rank\n{fake_pdb},{ligand_name},0\n")

        env = {"PYTHONPATH": "interformer/"}

        current_stage = "Energy Prediction"
        logging.info(f"[{ligand_name}] {current_stage}")
        check_timeout(start_time, current_stage)
        cmd = [
            "python", "inference.py", "-test_csv", input_csv, "-work_path", iso_dir,
            "-ensemble", "checkpoints/v0.2_energy_model", "-gpus", "1", "-batch_size", "1",
            "-posfix", "*val_loss*", "-energy_output_folder", iso_dir, "-uff_as_ligand", "-debug", "-reload"
        ]
        run_command(cmd, custom_env=env, timeout=900)

        current_stage = "Pocket Restore"
        pocket_root = os.path.join(iso_dir, f"{fake_pdb}_pocket.pdb")
        if not os.path.exists(pocket_root):
            shutil.copy(os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb"), pocket_root)

        current_stage = "Reconstruction"
        logging.info(f"[{ligand_name}] {current_stage}")
        check_timeout(start_time, current_stage)
        run_command([
            "python", "docking/reconstruct_ligands.py", "-y", "--cwd", iso_dir,
            "--find_all", "--uff_folder", "uff", "find"
        ], timeout=900)

        recon_sdf = os.path.join(iso_dir, "ligand_reconstructing", f"{fake_pdb}_docked.sdf")
        if not os.path.exists(recon_sdf):
            raise PipelineStepFailed("No recon SDF")

        current_stage = "Scoring"
        logging.info(f"[{ligand_name}] {current_stage}")
        check_timeout(start_time, current_stage)
        score_sdf = os.path.join(iso_dir, "infer", f"{fake_pdb}_docked.sdf")
        shutil.copy(recon_sdf, score_sdf)
        run_command(["python", "tools/inference/inter_sdf2csv.py", score_sdf, "0"])

        run_command([
            "python", "inference.py", "-test_csv", score_sdf.replace(".sdf", "_infer.csv"),
            "-work_path", iso_dir, "-ligand_folder", "infer",
            "-ensemble", "checkpoints/v0.2_affinity_model/model*",
            "-use_ff_ligands", "''", "-vs", "-gpus", "1", "-batch_size", "20",
            "-posfix", "*val_loss*", "--pose_sel", "True"
        ], custom_env=env, timeout=600)

        current_stage = "Collection"
        gen_res = os.path.join(iso_dir, "result", f"{fake_pdb}_docked_infer_ensemble.csv")
        if not os.path.exists(gen_res) and os.path.exists(f"result/{fake_pdb}_docked_infer_ensemble.csv"):
            shutil.move(f"result/{fake_pdb}_docked_infer_ensemble.csv", gen_res)

        if not os.path.exists(gen_res):
            raise PipelineStepFailed("No result CSV")

        df = pd.read_csv(gen_res)
        if df.empty:
            raise PipelineStepFailed("Result CSV empty")

        df['Target'] = original_pdb
        df['Molecule ID'] = ligand_name
        df['ligand_name'] = ligand_name

        final_dir = os.path.join(FINAL_RESULTS_DIR, ligand_name)
        os.makedirs(final_dir, exist_ok=True)
        df.to_csv(os.path.join(final_dir, f"{ligand_name}_ensemble.csv"), index=False)

        final_sdf = os.path.join(docking_path, ligand_name, f"{original_pdb}_docked.sdf")
        os.makedirs(os.path.dirname(final_sdf), exist_ok=True)
        shutil.copy(recon_sdf, final_sdf)

        elapsed = round(time.time() - start_time, 2)
        logging.info(f"[{ligand_name}] SUCCESS. Time: {elapsed}s")

        return {
            "ligand_name": ligand_name, "status": "success",
            "failed_stage": "", "error_details": "", "duration_sec": elapsed
        }
    except Exception as e:
        logging.exception(f"[{ligand_name}] FAILED at {current_stage}")
        return {
            "ligand_name": ligand_name, "status": "failed",
            "failed_stage": current_stage, "error_details": str(e),
            "duration_sec": round(time.time() - start_time, 2)
        }
    finally:
        shutil.rmtree(iso_dir, ignore_errors=True)


def collect_final_csv(pdb):
    logging.info("Collecting final CSV...")
    all_dfs = []
    for root, _, files in os.walk(FINAL_RESULTS_DIR):
        for f in files:
            if f.endswith("_ensemble.csv") and not f.startswith(pdb):
                try:
                    all_dfs.append(pd.read_csv(os.path.join(root, f)))
                except:
                    pass
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(
            os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_infer_ensemble.csv"), index=False
        )


def collect_final_sdf(pdb, docking_path):
    logging.info("Collecting final SDF...")
    outfile = os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_ALL.sdf")
    with Chem.SDWriter(outfile) as w:
        for root, _, files in os.walk(docking_path):
            for file in files:
                if file.endswith("_docked.sdf"):
                    try:
                        for m in Chem.SDMolSupplier(os.path.join(root, file)):
                            if m: w.write(m)
                    except:
                        pass


def main():
    parser = argparse.ArgumentParser(description="Interformer Pipeline v2")
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-w", "--work_path", required=True)
    parser.add_argument("-d", "--docking_path", required=True)
    parser.add_argument("-l", "--df_lig", required=True)
    parser.add_argument("-o", "--omp_num_threads", default="32,32")
    parser.add_argument("-c", "--chain", default="A")
    parser.add_argument("-r", "--ref_ligand", default=None)
    args = parser.parse_args()

    global FINAL_RESULTS_DIR, TEMP_ROOT
    FINAL_RESULTS_DIR = f"results_{args.pdb}"
    TEMP_ROOT = f"temp_isolation_work_{args.pdb}"

    logging.info(f"START PIPELINE v2 PDB={args.pdb}")
    logging.info(f"Results: {FINAL_RESULTS_DIR}")
    logging.info(f"Temp: {TEMP_ROOT}")

    if os.path.exists(FINAL_RESULTS_DIR): shutil.rmtree(FINAL_RESULTS_DIR)
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    if os.path.exists(TEMP_ROOT): shutil.rmtree(TEMP_ROOT)
    os.makedirs(TEMP_ROOT, exist_ok=True)

    raw_dir = os.path.join(args.work_path, f"raw_{args.pdb}")
    if os.path.exists(raw_dir): shutil.rmtree(raw_dir)
    os.makedirs(os.path.join(raw_dir, "pocket"), exist_ok=True)

    original_csv_path = os.path.join(raw_dir, f"{args.df_lig}.csv")
    data_dir = os.path.join(args.work_path, "data")
    ligand_pdb = os.path.join(data_dir, f"{args.pdb}_ligand.pdb")
    meeko_pdb = os.path.join(data_dir, f"{args.pdb}_meeko.pdb")

    # Пропускаем Meeko, если файлы есть
    if not (os.path.exists(ligand_pdb) and os.path.exists(meeko_pdb)):
        logging.info(f"Meeko prep needed for {args.pdb}")
        prep_cmd = [
            "python", "prepare_protein_meeko2.py",
            "-p", args.pdb, "-w", args.work_path, "-c", args.chain
        ]
        if args.ref_ligand:
            prep_cmd += ["-r", args.ref_ligand]
        result = subprocess.run(prep_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.critical(f"Meeko failed:\n{result.stderr}")
            sys.exit(1)
    else:
        logging.info("Skipping Meeko — files exist")

    shutil.copy(ligand_pdb, os.path.join(raw_dir, f"{args.pdb}_ligand.pdb"))
    shutil.copy(os.path.join(data_dir, f"{args.df_lig}.csv"), original_csv_path)
    shutil.copy(meeko_pdb, os.path.join(raw_dir, "pocket", f"{args.pdb}_meeko.pdb"))

    uff_src = os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf")
    uff_dst = os.path.join(raw_dir, f"{args.pdb}_uff.sdf")
    if os.path.exists(uff_src):
        shutil.copy(uff_src, uff_dst)
        logging.info(f"UFF copied: {uff_dst}, size={os.path.getsize(uff_dst)}")
    else:
        logging.critical(f"UFF missing: {uff_src}")
        sys.exit(1)

    try:
        assets_dir = prepare_global_resources(args.work_path, args.pdb)
    except Exception as e:
        logging.critical(f"Global prep failed: {e}")
        sys.exit(1)

    successful_names, failed_names = prepare_ligands_for_interformer(
        original_csv_path,
        os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf")
    )

    logging.info("Loading UFF molecules...")
    suppl = Chem.SDMolSupplier(uff_dst)
    mol_dict = {}
    for i, mol in enumerate(suppl):
        if mol is None:
            logging.warning(f"Mol {i} is None")
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"ligand_row_{i}"
        if mol.GetNumAtoms() > 0 and mol.GetNumConformers() > 0:
            mol_dict[name] = mol
            logging.debug(f"Loaded: {name} ({mol.GetNumAtoms()} atoms)")
        else:
            logging.warning(f"Skipped {name}: atoms={mol.GetNumAtoms()}, confs={mol.GetNumConformers()}")

    logging.info(f"Loaded {len(mol_dict)} valid mols. Keys: {list(mol_dict.keys())[:5]}...")

    all_report_data = []
    report_file = os.path.join(FINAL_RESULTS_DIR, f"pipeline_summary_report_{args.pdb}.csv")

    for item in failed_names:
        name = item[0] if isinstance(item, (list, tuple)) else str(item)
        reason = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else "Unknown"
        all_report_data.append({
            "ligand_name": name,
            "status": "failed",
            "failed_stage": "Preparation",
            "error_details": reason,
            "duration_sec": 0.0
        })
    pd.DataFrame(all_report_data).to_csv(report_file, index=False)

    logging.info(f"Docking {len(successful_names)} ligands...")

    for idx, lig_name in enumerate(successful_names):
        lig_name_key = lig_name.strip()
        mol = mol_dict.get(lig_name_key)

        if not mol:
            # Пробуем по индексу
            try:
                row_idx = lig_name_key.split('_')[-1]
                mol = mol_dict.get(f"ligand_row_{row_idx}")
            except:
                mol = None

        if not mol:
            res = {
                "ligand_name": lig_name,
                "status": "failed",
                "failed_stage": "Mol load",
                "error_details": f"Not found '{lig_name_key}' (keys: {list(mol_dict.keys())[:5]}...)",
                "duration_sec": 0
            }
        else:
            res = run_isolated_docking(idx+1, lig_name, mol, args.pdb, assets_dir, args.docking_path)

        all_report_data.append(res)
        pd.DataFrame(all_report_data).to_csv(report_file, index=False)

    logging.info("Collecting finals...")
    collect_final_csv(args.pdb)
    collect_final_sdf(args.pdb, args.docking_path)

    if os.path.exists("analyze.py") and len(successful_names) > 0:
        run_command([
            "python", "analyze.py",
            "--results-csv", os.path.join(FINAL_RESULTS_DIR, f"{args.pdb}_docked_infer_ensemble.csv"),
            "--original-csv", original_csv_path,
            "--experimental-col", "pValue", "--output-folder", FINAL_RESULTS_DIR
        ])

    logging.info(f"FINISHED. Report: {report_file}")


def collect_final_csv(pdb):
    all_dfs = []
    for root, _, files in os.walk(FINAL_RESULTS_DIR):
        for f in files:
            if f.endswith("_ensemble.csv") and not f.startswith(pdb):
                try:
                    all_dfs.append(pd.read_csv(os.path.join(root, f)))
                except:
                    pass
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(
            os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_infer_ensemble.csv"), index=False
        )


def collect_final_sdf(pdb, docking_path):
    outfile = os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_ALL.sdf")
    with Chem.SDWriter(outfile) as w:
        for root, _, files in os.walk(docking_path):
            for file in files:
                if file.endswith("_docked.sdf"):
                    try:
                        for m in Chem.SDMolSupplier(os.path.join(root, file)):
                            if m: w.write(m)
                    except:
                        pass


if __name__ == "__main__":
    main()
