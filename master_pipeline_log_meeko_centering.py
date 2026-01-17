import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D
from prepare_ligands_new import prepare_ligands_for_interformer

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
FINAL_RESULTS_DIR = ""            # Устанавливается в main динамически
TEMP_ROOT = ""                    # Устанавливается в main динамически
MAX_LIGAND_DURATION = 1200        # 20 минут на лиганд (Hard Limit)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# УТИЛИТЫ И ИСКЛЮЧЕНИЯ
# ==============================================================================
class PipelineStepFailed(Exception):
    pass

class LigandTimeout(Exception):
    pass

def run_command(command, custom_env=None, timeout=None):
    """Запуск внешней команды с перехватом ошибок."""
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
        # Логируем только хвост ошибки, чтобы не засорять лог
        error_snippet = e.stderr.strip()[-1000:] if e.stderr else "No stderr"
        logging.error(f"CMD ERROR: {cmd_str}\nSTDERR TAIL: ...{error_snippet}")
        raise PipelineStepFailed(f"Exit code {e.returncode} | {error_snippet}")

def check_timeout(start_time, stage_name):
    elapsed = time.time() - start_time
    if elapsed > MAX_LIGAND_DURATION:
        raise LigandTimeout(f"Лимит времени ({MAX_LIGAND_DURATION}с) истек на этапе: {stage_name}")

# ==============================================================================
# ГЕОМЕТРИЯ: ЦЕНТРИРОВАНИЕ ЛИГАНДА
# ==============================================================================
def get_mol_center(mol):
    """Считает геометрический центр молекулы (центроид)."""
    conf = mol.GetConformer()
    if not conf:
        return Point3D(0, 0, 0)
    pos = conf.GetPositions()
    centroid = np.mean(pos, axis=0)
    return Point3D(float(centroid[0]), float(centroid[1]), float(centroid[2]))

def align_mol_to_ref(target_mol, ref_mol_path):
    """
    Сдвигает target_mol так, чтобы его центр совпадал с центром ref_mol.
    Необходимо, т.к. RDKit генерирует 3D в (0,0,0), а белок находится в координатах кристалла.
    """
    try:
        ref_suppl = Chem.SDMolSupplier(ref_mol_path, sanitize=False)
        ref_mol = next(ref_suppl)
        if not ref_mol:
            raise ValueError("Reference SDF is empty/invalid")
        
        ref_center = get_mol_center(ref_mol)
        target_center = get_mol_center(target_mol)
        
        # Вектор смещения
        shift = ref_center - target_center
        
        conf = target_mol.GetConformer()
        n_atoms = target_mol.GetNumAtoms()
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, pos + shift)
            
        return shift
    except Exception as e:
        logging.warning(f"Alignment failed: {e}. Using raw coordinates (High risk of failure).")
        return Point3D(0,0,0)

# ==============================================================================
# ПОДГОТОВКА РЕСУРСОВ
# ==============================================================================
def prepare_global_resources(work_path, pdb, raw_dir):
    """
    Готовит общие файлы (Pocket и Reference Ligand) один раз для всех лигандов.
    """
    logging.info(">>> STEP: Подготовка глобальных ресурсов...")
    assets_dir = os.path.join(work_path, f"global_assets_{pdb}")
    if os.path.exists(assets_dir):
        shutil.rmtree(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Конвертация Reference Ligand PDB -> SDF
    ref_ligand_sdf = os.path.join(assets_dir, "reference_ligand.sdf")
    
    # Ищем файл лиганда (приоритет: точный файл, сгенерированный prepare_protein)
    ligand_src = os.path.join(raw_dir, f"{pdb}_ligand.pdb")
    if not os.path.exists(ligand_src):
         # Фолбэк на старое имя
         ligand_src = os.path.join(raw_dir, f"{pdb}_ligand_LIG.pdb")

    if not os.path.exists(ligand_src):
        raise FileNotFoundError(f"Ligand PDB not found in {raw_dir}")
        
    logging.info(f"Converting Reference Ligand: {ligand_src} -> SDF")
    run_command(["obabel", ligand_src, "-p", "7.4", "-O", ref_ligand_sdf])

    # 2. Подготовка Кармана (Pocket)
    # ВАЖНО: Мы пропускаем extract_pocket_by_ligand.py, так как он ненадежен.
    # Мы используем полный очищенный белок как pocket.pdb.
    
    prot_candidates = [
        os.path.join(raw_dir, "pocket", f"{pdb}_meeko.pdb"), 
        os.path.join(work_path, "data", f"{pdb}_meeko.pdb"),
        os.path.join(work_path, "data", f"{pdb}.pdb")
    ]
    prot_src = next((f for f in prot_candidates if os.path.exists(f)), None)
    
    if not prot_src:
        raise FileNotFoundError(f"Protein PDB not found for {pdb}")
    
    logging.info(f"BYPASSING EXTRACTION: Using full protein as pocket: {prot_src}")
    
    # Просто копируем полный белок как pocket.pdb
    # Interformer сам выберет нужные атомы в радиусе 10A во время inference
    shutil.copy(prot_src, os.path.join(assets_dir, "pocket.pdb"))

    return assets_dir

# ==============================================================================
# ИЗОЛИРОВАННЫЙ ДОКИНГ (ЯДРО)
# ==============================================================================
def run_isolated_docking(idx, ligand_name, mol, original_pdb, assets_dir, docking_path):
    """
    Запускает полный цикл докинга для одного лиганда в изолированной папке.
    """
    start_time = time.time()
    fake_pdb = f"L{idx:03d}"
    iso_dir = os.path.join(TEMP_ROOT, fake_pdb)
    current_stage = "Init"

    try:
        logging.info(f"[{ligand_name}] Processing in {iso_dir}...")

        # --- STAGE 1: SETUP ---
        current_stage = "Setup"
        check_timeout(start_time, current_stage)
        if os.path.exists(iso_dir): shutil.rmtree(iso_dir)
        for d in ["ligand", "pocket", "uff", "infer", "complex", "result"]:
            os.makedirs(os.path.join(iso_dir, d), exist_ok=True)

        # 1. Pocket (Полный белок)
        shutil.copy(os.path.join(assets_dir, "pocket.pdb"), os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb"))
        
        # 2. Reference Ligand (нужен для определения центра коробки в inference.py)
        ref_ligand_path = os.path.join(assets_dir, "reference_ligand.sdf")
        shutil.copy(ref_ligand_path, os.path.join(iso_dir, "ligand", f"{fake_pdb}_docked.sdf"))

        # 3. UFF Ligand (Target) - ЦЕНТРИРОВАНИЕ
        # Сдвигаем наш UFF лиганд в центр референсного, иначе граф будет пустой
        shift = align_mol_to_ref(mol, ref_ligand_path)
        logging.info(f"[{ligand_name}] Centered UFF to Ref. Shift: {shift.x:.1f}, {shift.y:.1f}, {shift.z:.1f}")

        uff_path = os.path.join(iso_dir, "uff", f"{fake_pdb}_uff.sdf")
        with Chem.SDWriter(uff_path) as w:
            w.write(mol)

        # 4. Input CSV
        input_csv = os.path.join(iso_dir, f"{fake_pdb}_input.csv")
        with open(input_csv, 'w') as f:
            f.write(f"Target,Molecule ID,pose_rank\n{fake_pdb},{ligand_name},0\n")

        env = {"PYTHONPATH": "interformer/"}

        # --- STAGE 2: ENERGY PREDICTION ---
        current_stage = "Energy Prediction"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        check_timeout(start_time, current_stage)
        
        # Добавляем --uff_as_ligand, так как мы хотим докать наш UFF, а не референс
        run_command([
            "python", "inference.py", "-test_csv", input_csv, "-work_path", iso_dir,
            "-ensemble", "checkpoints/v0.2_energy_model", "-gpus", "1", "-batch_size", "1",
            "-posfix", "*val_loss*", "-energy_output_folder", iso_dir, 
            "-uff_as_ligand", # ВАЖНО: использовать UFF как стартовую конформацию
            "-debug", "-reload"
        ], custom_env=env, timeout=900)

        # --- STAGE 3: POCKET RESTORE ---
        current_stage = "Pocket Restore"
        # Восстанавливаем pocket, если inference его удалил или изменил
        pocket_root = os.path.join(iso_dir, f"{fake_pdb}_pocket.pdb")
        if not os.path.exists(pocket_root):
            shutil.copy(os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb"), pocket_root)

        # --- STAGE 4: RECONSTRUCTION (DOCKING) ---
        current_stage = "Reconstruction"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        check_timeout(start_time, current_stage)
        
        # Запуск Monte Carlo сэмплинга
        run_command([
            "python", "docking/reconstruct_ligands.py", "-y", "--cwd", iso_dir,
            "--find_all", "--uff_folder", "uff", "find"
        ], timeout=900)
        
        recon_sdf = os.path.join(iso_dir, "ligand_reconstructing", f"{fake_pdb}_docked.sdf")
        if not os.path.exists(recon_sdf):
            raise PipelineStepFailed("Reconstruction failed (no SDF output)")

        # --- STAGE 5: SCORING (AFFINITY) ---
        current_stage = "Scoring"
        logging.info(f"[{ligand_name}] >>> Start Stage: {current_stage}")
        check_timeout(start_time, current_stage)
        score_sdf = os.path.join(iso_dir, "infer", f"{fake_pdb}_docked.sdf")
        shutil.copy(recon_sdf, score_sdf)
        
        # Создаем CSV для скоринга
        run_command(["python", "tools/inference/inter_sdf2csv.py", score_sdf, "0"])
        
        # Запуск предсказания аффинности
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
        
        # Fallback check для путей
        if not os.path.exists(gen_res) and os.path.exists(f"result/{fake_pdb}_docked_infer_ensemble.csv"):
            shutil.move(f"result/{fake_pdb}_docked_infer_ensemble.csv", gen_res)
            
        if not os.path.exists(gen_res):
            raise PipelineStepFailed("Result CSV missing")

        df = pd.read_csv(gen_res)
        if df.empty: raise PipelineStepFailed("Result CSV empty")
        
        # Обогащаем данными
        df['Target'] = original_pdb
        df['Molecule ID'] = ligand_name
        df['ligand_name'] = ligand_name
        
        # Save Final CSV per ligand
        final_dir = os.path.join(FINAL_RESULTS_DIR, ligand_name)
        os.makedirs(final_dir, exist_ok=True)
        df.to_csv(os.path.join(final_dir, f"{ligand_name}_ensemble.csv"), index=False)
        
        # Save Final SDF
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
        # Очистка временной папки для экономии места
        if os.path.exists(iso_dir): shutil.rmtree(iso_dir)

# ==============================================================================
# ФИНАЛИЗАЦИЯ И ОТЧЕТЫ
# ==============================================================================
def collect_final_csv(pdb):
    """Собирает все CSV отдельных лигандов в один большой файл."""
    logging.info("Сборка финального CSV...")
    all_dfs = []
    for root, _, files in os.walk(FINAL_RESULTS_DIR):
        for f in files:
            # Игнорируем сам отчет и уже собранные файлы
            if f.endswith("_ensemble.csv") and not f.startswith(pdb) and "pipeline_summary" not in f:
                try:
                    all_dfs.append(pd.read_csv(os.path.join(root, f)))
                except: pass
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_infer_ensemble.csv"), index=False)

def collect_final_sdf(pdb, docking_path):
    """Собирает все SDF в один файл."""
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
    parser = argparse.ArgumentParser(description="Interformer Pipeline (Centering + Full Protein Fix)")
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-w", "--work_path", required=True)
    parser.add_argument("-d", "--docking_path", required=True)
    parser.add_argument("-l", "--df_lig", required=True)
    parser.add_argument("-o", "--omp_num_threads", default="32,32")
    parser.add_argument("-c", "--chain", default="A")
    parser.add_argument("-r", "--ref_ligand", default=None)
    args = parser.parse_args()

    # Установка глобальных переменных путей
    global FINAL_RESULTS_DIR, TEMP_ROOT
    FINAL_RESULTS_DIR = f"results_{args.pdb}"
    TEMP_ROOT = f"temp_isolation_work_{args.pdb}"

    # Очистка старых результатов (защита от смешивания данных)
    if os.path.exists(FINAL_RESULTS_DIR): shutil.rmtree(FINAL_RESULTS_DIR)
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    if os.path.exists(TEMP_ROOT): shutil.rmtree(TEMP_ROOT)
    os.makedirs(TEMP_ROOT, exist_ok=True)
    
    # ------------------------------------------------------------------
    # 0. ИНИЦИАЛИЗАЦИЯ И ПОДГОТОВКА ФАЙЛОВ
    # ------------------------------------------------------------------
    raw_dir = os.path.join(args.work_path, f"raw_{args.pdb}")
    if os.path.exists(raw_dir): shutil.rmtree(raw_dir) 
    os.makedirs(os.path.join(raw_dir, "pocket"), exist_ok=True)
    
    data_dir = os.path.join(args.work_path, "data")
    
    # Проверка исходного PDB
    raw_pdb_source = os.path.join(data_dir, f"{args.pdb}.pdb")
    if not os.path.exists(raw_pdb_source):
        logging.critical(f"FATAL: Source PDB not found: {raw_pdb_source}")
        logging.critical("Please run 'bash download_pdbs.sh' to fetch missing structures.")
        sys.exit(1)

    # Проверка и запуск предобработки (prepare_protein_meeko.py)
    ligand_pdb = os.path.join(data_dir, f"{args.pdb}_ligand.pdb")
    meeko_pdb = os.path.join(data_dir, f"{args.pdb}_meeko.pdb")
    
    if not (os.path.exists(ligand_pdb) and os.path.exists(meeko_pdb)):
        logging.info(f"Pre-processed files not found. Running prepare_protein_meeko.py...")
        prep_cmd = ["python", "prepare_protein_meeko.py", "-p", args.pdb, "-w", args.work_path, "-c", args.chain]
        if args.ref_ligand: prep_cmd.extend(["-r", args.ref_ligand])
        
        res = subprocess.run(prep_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logging.critical(f"Preparation Failed:\n{res.stderr}")
            sys.exit(1)
        else:
            logging.info("Preparation done.")

    # Копирование в рабочую директорию
    shutil.copy(ligand_pdb, os.path.join(raw_dir, f"{args.pdb}_ligand.pdb"))
    shutil.copy(meeko_pdb, os.path.join(raw_dir, "pocket", f"{args.pdb}_meeko.pdb"))
    shutil.copy(os.path.join(data_dir, f"{args.df_lig}.csv"), os.path.join(raw_dir, f"{args.df_lig}.csv"))

    # Подготовка глобальных ресурсов (Pocket и Ref Ligand)
    try:
        assets_dir = prepare_global_resources(args.work_path, args.pdb, raw_dir)
    except Exception as e:
        logging.critical(f"FATAL RESOURCE PREP: {e}")
        return

    # ------------------------------------------------------------------
    # 1. ГЕНЕРАЦИЯ UFF (3D Generation)
    # ------------------------------------------------------------------
    logging.info("Generating UFF conformations...")
    original_csv_path = os.path.join(raw_dir, f"{args.df_lig}.csv")
    uff_sdf_path = os.path.join(raw_dir, f"{args.pdb}_uff.sdf")
    
    successful_names, failed_names = prepare_ligands_for_interformer(
        original_csv_path,
        uff_sdf_path
    )
    
    # Загрузка молекул в память
    suppl = Chem.SDMolSupplier(uff_sdf_path)
    mol_dict = {m.GetProp("_Name"): m for m in suppl if m and m.HasProp("_Name")}
    
    # ------------------------------------------------------------------
    # 2. ДОКИНГ ЦИКЛ
    # ------------------------------------------------------------------
    all_report_data = []
    report_file_path = os.path.join(FINAL_RESULTS_DIR, f"pipeline_summary_report_{args.pdb}.csv")

    # Добавляем отчет по провалившимся на этапе подготовки
    for item in failed_names:
        name = item[0] if isinstance(item, (tuple, list)) else str(item)
        reason = item[1] if isinstance(item, (tuple, list)) else "Prep Failed"
        all_report_data.append({
            "ligand_name": name, "status": "failed", 
            "failed_stage": "Ligand Preparation", "error_details": str(reason), "duration_sec": 0
        })

    logging.info(f"Starting Docking for {len(successful_names)} ligands...")
    
    for idx, lig_name in enumerate(successful_names):
        mol = mol_dict.get(lig_name)
        if not mol:
             res = {"ligand_name": lig_name, "status": "failed", "failed_stage": "Loading", "error_details": "Mol missing in SDF", "duration_sec": 0}
        else:
            # ЗАПУСК ИЗОЛИРОВАННОГО ДОКИНГА
            res = run_isolated_docking(idx+1, lig_name, mol, args.pdb, assets_dir, args.docking_path)
        
        all_report_data.append(res)
        # Сохраняем отчет после каждого шага
        pd.DataFrame(all_report_data).to_csv(report_file_path, index=False)

    # ------------------------------------------------------------------
    # 3. ФИНАЛИЗАЦИЯ
    # ------------------------------------------------------------------
    collect_final_csv(args.pdb)
    collect_final_sdf(args.pdb, args.docking_path)
    
    # Опциональный анализ
    if os.path.exists("analyze.py") and len(successful_names) > 0:
        try:
            run_command([
                "python", "analyze.py",
                "--results-csv", os.path.join(FINAL_RESULTS_DIR, f"{args.pdb}_docked_infer_ensemble.csv"),
                "--original-csv", original_csv_path,
                "--experimental-col", "pValue", "--output-folder", FINAL_RESULTS_DIR
            ])
        except Exception: 
            logging.warning("Analysis script failed or missing, but docking completed successfully.")

    logging.info(f"PIPELINE COMPLETE. Report: {report_file_path}")

if __name__ == "__main__":
    main()
