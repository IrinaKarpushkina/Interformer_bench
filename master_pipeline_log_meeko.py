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

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
FINAL_RESULTS_DIR_SUFFIX = "results"          # будет results_{pdb}
TEMP_ROOT_SUFFIX = "temp_isolation_work"      # будет temp_isolation_work_{pdb}
MAX_LIGAND_DURATION = 1200  # 20 минут на лиганд (Hard Limit)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("master_pipeline_meeko.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# ИСКЛЮЧЕНИЯ И УТИЛИТЫ
# ==============================================================================
class PipelineStepFailed(Exception):
    pass

class LigandTimeout(Exception):
    pass

def run_command(command, custom_env=None, timeout=None):
    cmd_str = ' '.join(command)
    env = os.environ.copy()
    if custom_env:
        env.update(custom_env)
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        raise PipelineStepFailed(f"TIMEOUT EXPIRED ({timeout}s): {cmd_str}")
    except subprocess.CalledProcessError as e:
        error_snippet = e.stderr.strip()[-300:] if e.stderr else "No stderr"
        logging.error(f"CMD ERROR: {cmd_str}\nSTDERR: ...{error_snippet}")
        raise PipelineStepFailed(f"Exit code {e.returncode} | {error_snippet}")

def check_timeout(start_time, stage_name):
    elapsed = time.time() - start_time
    if elapsed > MAX_LIGAND_DURATION:
        raise LigandTimeout(f"Лимит времени ({MAX_LIGAND_DURATION}с) истек на этапе: {stage_name}")

# ==============================================================================
# ПОДГОТОВКА РЕСУРСОВ
# ==============================================================================
def prepare_global_resources(work_path, pdb):
    logging.info(">>> STEP: Подготовка глобальных ресурсов...")
    assets_dir = os.path.join(work_path, f"global_assets_{pdb}")
    if os.path.exists(assets_dir):
        shutil.rmtree(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)
   
    raw_dir = os.path.join(work_path, f"raw_{pdb}")
    raw_pocket = os.path.join(raw_dir, "pocket")
   
    ref_ligand_sdf = os.path.join(assets_dir, "reference_ligand.sdf")
    ligand_file = os.path.join(raw_dir, f"{pdb}_ligand_LIG.pdb") if os.path.exists(os.path.join(raw_dir, f"{pdb}_ligand_LIG.pdb")) else os.path.join(raw_dir, f"{pdb}_ligand.pdb")
    run_command(["obabel", ligand_file, "-p", "7.4", "-O", ref_ligand_sdf])
   
    temp_extract = os.path.join(work_path, f"temp_extract_ligand_{pdb}")
    os.makedirs(temp_extract, exist_ok=True)
    shutil.copy(ref_ligand_sdf, os.path.join(temp_extract, f"{pdb}_docked.sdf"))
   
    run_command(["python", "tools/extract_pocket_by_ligand.py", raw_pocket + "/", temp_extract + "/", "0"])
   
    gen_pocket = os.path.join(raw_pocket, "output", f"{pdb}_pocket.pdb")
    if not os.path.exists(gen_pocket):
        logging.error(f"Pocket not generated! Check output dir: {os.path.join(raw_pocket, 'output')}")
        os.system(f"ls -l {os.path.join(raw_pocket, 'output')}")
        raise Exception("Failed to generate pocket.pdb")
       
    shutil.copy(gen_pocket, os.path.join(assets_dir, "pocket.pdb"))
    shutil.rmtree(temp_extract)
    return assets_dir

# ==============================================================================
# ИЗОЛИРОВАННЫЙ ДОКИНГ (ЯДРО)
# ==============================================================================
def run_isolated_docking(idx, ligand_name, mol, original_pdb, assets_dir, docking_path):
    start_time = time.time()
    fake_pdb = f"L{idx:03d}"
    iso_dir = os.path.join(TEMP_ROOT, fake_pdb)
    current_stage = "Init"
   
    try:
        logging.info(f"[{ligand_name}] Processing in {iso_dir}...")
       
        # --- STAGE 1: SETUP ---
        current_stage = "Setup"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        check_timeout(start_time, current_stage)
        if os.path.exists(iso_dir): shutil.rmtree(iso_dir)
        for d in ["ligand", "pocket", "uff", "infer", "complex", "result"]:
            os.makedirs(os.path.join(iso_dir, d), exist_ok=True)
           
        shutil.copy(os.path.join(assets_dir, "pocket.pdb"), os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb"))
        shutil.copy(os.path.join(assets_dir, "reference_ligand.sdf"), os.path.join(iso_dir, "ligand", f"{fake_pdb}_docked.sdf"))
       
        uff_path = os.path.join(iso_dir, "uff", f"{fake_pdb}_uff.sdf")
        
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid molecule for {ligand_name}: atoms = {mol.GetNumAtoms() if mol else 0}")
        
        logging.info(f"Writing UFF SDF for {ligand_name}: {mol.GetNumAtoms()} atoms → {uff_path}")
        with Chem.SDWriter(uff_path) as w:
            w.write(mol)
        
        input_csv = os.path.join(iso_dir, f"{fake_pdb}_input.csv")
        with open(input_csv, 'w') as f:
            f.write(f"Target,Molecule ID,pose_rank\n{fake_pdb},{ligand_name},0\n")
           
        env = {"PYTHONPATH": "interformer/"}
        # --- STAGE 2: ENERGY PREDICTION ---
        current_stage = "Energy Prediction"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        check_timeout(start_time, current_stage)
        run_command([
            "python", "inference.py", "-test_csv", input_csv, "-work_path", iso_dir,
            "-ensemble", "checkpoints/v0.2_energy_model", "-gpus", "1", "-batch_size", "1",
            "-posfix", "*val_loss*", "-energy_output_folder", iso_dir, "-uff_as_ligand", "-debug", "-reload"
        ], custom_env=env, timeout=900)
        # ... (остальные стадии без изменений)
        # --- STAGE 3: POCKET RESTORE ---
        current_stage = "Pocket Restore"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        pocket_root = os.path.join(iso_dir, f"{fake_pdb}_pocket.pdb")
        if not os.path.exists(pocket_root):
            shutil.copy(os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb"), pocket_root)
        # --- STAGE 4: RECONSTRUCTION ---
        current_stage = "Reconstruction"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        check_timeout(start_time, current_stage)
        run_command([
            "python", "docking/reconstruct_ligands.py", "-y", "--cwd", iso_dir,
            "--find_all", "--uff_folder", "uff", "find"
        ], timeout=900)
       
        recon_sdf = os.path.join(iso_dir, "ligand_reconstructing", f"{fake_pdb}_docked.sdf")
        if not os.path.exists(recon_sdf):
            raise PipelineStepFailed("Reconstruction failed (no SDF output)")
        # --- STAGE 5: SCORING ---
        current_stage = "Scoring"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
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
        # --- STAGE 6: RESULT COLLECTION ---
        current_stage = "Collection"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        gen_res = os.path.join(iso_dir, "result", f"{fake_pdb}_docked_infer_ensemble.csv")
       
        if not os.path.exists(gen_res) and os.path.exists(f"result/{fake_pdb}_docked_infer_ensemble.csv"):
            shutil.move(f"result/{fake_pdb}_docked_infer_ensemble.csv", gen_res)
           
        if not os.path.exists(gen_res):
            raise PipelineStepFailed("Result CSV missing")
        df = pd.read_csv(gen_res)
        if df.empty: raise PipelineStepFailed("Result CSV empty")
       
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
    except LigandTimeout as e:
        elapsed = round(time.time() - start_time, 2)
        logging.error(f"[{ligand_name}] TIMEOUT at stage {current_stage}: {e}")
        return {
            "ligand_name": ligand_name, "status": "failed",
            "failed_stage": current_stage, "error_details": str(e), "duration_sec": elapsed
        }
    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        logging.error(f"[{ligand_name}] FAILED at stage {current_stage}: {e}")
        return {
            "ligand_name": ligand_name, "status": "failed",
            "failed_stage": current_stage, "error_details": str(e), "duration_sec": elapsed
        }
    finally:
        if os.path.exists(iso_dir): shutil.rmtree(iso_dir)

# ==============================================================================
# ФИНАЛИЗАЦИЯ И ОТЧЕТЫ
# ==============================================================================
def collect_final_csv(pdb):
    logging.info("Сборка финального CSV...")
    all_dfs = []
    for root, _, files in os.walk(FINAL_RESULTS_DIR):
        for f in files:
            if f.endswith("_ensemble.csv") and not f.startswith(pdb):
                try:
                    all_dfs.append(pd.read_csv(os.path.join(root, f)))
                except: pass
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(
            os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_infer_ensemble.csv"), index=False
        )

def collect_final_sdf(pdb, docking_path):
    logging.info("Сборка финального SDF...")
    outfile = os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_ALL.sdf")
    with Chem.SDWriter(outfile) as w:
        for root, dirs, files in os.walk(docking_path):
            for file in files:
                if file.endswith("_docked.sdf"):
                    try:
                        for m in Chem.SDMolSupplier(os.path.join(root, file)):
                            if m: w.write(m)
                    except: pass

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Interformer Pipeline (with Meeko prep)")
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-w", "--work_path", required=True)
    parser.add_argument("-d", "--docking_path", required=True)
    parser.add_argument("-l", "--df_lig", required=True)
    parser.add_argument("-o", "--omp_num_threads", default="32,32")
    parser.add_argument("-c", "--chain", default="A")
    parser.add_argument("-r", "--ref_ligand", default=None)
    args = parser.parse_args()

    # Уникальные директории
    global FINAL_RESULTS_DIR, TEMP_ROOT
    FINAL_RESULTS_DIR = f"results_{args.pdb}"
    TEMP_ROOT = f"temp_isolation_work_{args.pdb}"

    # Очистка
    if os.path.exists(FINAL_RESULTS_DIR): shutil.rmtree(FINAL_RESULTS_DIR)
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    if os.path.exists(TEMP_ROOT): shutil.rmtree(TEMP_ROOT)
    os.makedirs(TEMP_ROOT, exist_ok=True)
   
    # Копирование исходников
    raw_dir = os.path.join(args.work_path, f"raw_{args.pdb}")
    if os.path.exists(raw_dir): shutil.rmtree(raw_dir)
    os.makedirs(os.path.join(raw_dir, "pocket"), exist_ok=True)
   
    original_csv_path = os.path.join(raw_dir, f"{args.df_lig}.csv")
    data_dir = os.path.join(args.work_path, "data")
    ligand_pdb = os.path.join(data_dir, f"{args.pdb}_ligand.pdb")
    meeko_pdb  = os.path.join(data_dir, f"{args.pdb}_meeko.pdb")

    # Подготовка Meeko, если нужно
    if not (os.path.exists(ligand_pdb) and os.path.exists(meeko_pdb)):
        logging.info(f"Reference files not found for {args.pdb}. Running Meeko preparation...")
        prep_cmd = [
            "python", "prepare_protein_meeko.py",
            "-p", args.pdb,
            "-w", args.work_path,
            "-c", args.chain
        ]
        if args.ref_ligand:
            prep_cmd.extend(["-r", args.ref_ligand])
        
        result = subprocess.run(prep_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.critical(f"Meeko preparation failed:\n{result.stderr}")
            sys.exit(1)
        logging.info("Meeko preparation completed.")

    # Копируем файлы
    shutil.copy(ligand_pdb, os.path.join(raw_dir, f"{args.pdb}_ligand.pdb"))
    shutil.copy(os.path.join(data_dir, f"{args.df_lig}.csv"), original_csv_path)
    shutil.copy(meeko_pdb, os.path.join(raw_dir, "pocket", f"{args.pdb}_meeko.pdb"))

    # Копируем UFF SDF в raw_{pdb} — это критично!
    uff_src = os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf")
    uff_dst = os.path.join(raw_dir, f"{args.pdb}_uff.sdf")
    if os.path.exists(uff_src):
        shutil.copy(uff_src, uff_dst)
        logging.info(f"Copied UFF SDF: {uff_src} → {uff_dst}")
    else:
        raise FileNotFoundError(f"UFF SDF not found: {uff_src}. Run prepare_ligands_new.py first.")

    # Подготовка ресурсов
    try:
        assets_dir = prepare_global_resources(args.work_path, args.pdb)
    except Exception as e:
        logging.critical(f"FATAL RESOURCE PREP: {e}")
        return
   
    # Подготовка лигандов
    logging.info("Запуск prepare_ligands_new.py...")
    successful_names, failed_names = prepare_ligands_for_interformer(
        original_csv_path,
        os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf")
    )
   
    # Загрузка молекул с подробной отладкой
    logging.info("Loading molecules from UFF SDF...")
    suppl = Chem.SDMolSupplier(uff_dst)  # используем копию в raw_{pdb}
    mol_dict = {}
    for i, mol in enumerate(suppl):
        if mol is None:
            logging.warning(f"Molecule {i} is None in UFF SDF")
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"NO_NAME_{i}"
        atoms = mol.GetNumAtoms()
        logging.info(f"Mol {i}: name='{name}', atoms={atoms}")
        if mol.HasProp("_Name"):
            mol_dict[name] = mol
    
    logging.info(f"Loaded {len(mol_dict)} valid molecules from UFF SDF")
    if len(mol_dict) == 0:
        raise ValueError("No valid molecules loaded from UFF SDF. Check names in SDF and prepare_ligands_new.py")

    # ------------------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ ОТЧЕТА
    # ------------------------------------------------------------------
    all_report_data = []
    report_file_path = os.path.join(FINAL_RESULTS_DIR, f"pipeline_summary_report_{args.pdb}.csv")
    logging.info(f"Обработка ошибок подготовки ({len(failed_names)} лигандов)...")
    for item in failed_names:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            name, reason = item[0], item[1]
        else:
            name, reason = str(item), "Preparation Failed (Unknown reason)"
        all_report_data.append({
            "ligand_name": name,
            "status": "failed",
            "failed_stage": "Ligand Preparation",
            "error_details": str(reason),
            "duration_sec": 0.0
        })
    pd.DataFrame(all_report_data).to_csv(report_file_path, index=False)
   
    # ДОКИНГ ЦИКЛ
    logging.info(f"Начало докинга для {len(successful_names)} успешных лигандов...")
   
    for idx, lig_name in enumerate(successful_names):
        mol = mol_dict.get(lig_name)
        if not mol:
            res = {
                "ligand_name": lig_name,
                "status": "failed",
                "failed_stage": "Loading (Mol missing)",
                "error_details": f"Mol '{lig_name}' not found in UFF SDF (loaded {len(mol_dict)} mols)",
                "duration_sec": 0
            }
        else:
            res = run_isolated_docking(idx+1, lig_name, mol, args.pdb, assets_dir, args.docking_path)
       
        all_report_data.append(res)
        pd.DataFrame(all_report_data).to_csv(report_file_path, index=False)
   
    # Финализация
    logging.info("Сборка финальных файлов результатов...")
    collect_final_csv(args.pdb)
    collect_final_sdf(args.pdb, args.docking_path)
   
    if os.path.exists("analyze.py") and len(successful_names) > 0:
        run_command([
            "python", "analyze.py",
            "--results-csv", os.path.join(FINAL_RESULTS_DIR, f"{args.pdb}_docked_infer_ensemble.csv"),
            "--original-csv", original_csv_path,
            "--experimental-col", "pValue", "--output-folder", FINAL_RESULTS_DIR
        ])
       
    logging.info(f"ПАЙПЛАЙН ЗАВЕРШЁН. Полный отчет: {report_file_path}")

if __name__ == "__main__":
    main()
