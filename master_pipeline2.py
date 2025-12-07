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
FINAL_RESULTS_DIR = "results"
TEMP_ROOT = "temp_isolation_work" # Здесь будут создаваться папки L001, L002...

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("master_pipeline.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class PipelineStepFailed(Exception):
    pass

class LigandTimeout(Exception):
    pass

# ==============================================================================
# ФУНКЦИИ
# ==============================================================================
def run_command(command, custom_env=None, timeout=None):
    cmd_str = ' '.join(command)
    # logging.info(f"→ {cmd_str}") # Слишком много спама, оставим только важные

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
    except subprocess.TimeoutExpired as e:
        raise PipelineStepFailed(f"TIMEOUT {timeout}s: {cmd_str}")
    except subprocess.CalledProcessError as e:
        # Логируем stderr только при ошибке
        logging.error(f"CMD ERROR: {cmd_str}\nSTDERR: {e.stderr}")
        raise PipelineStepFailed(f"Exit code {e.returncode}")

def generate_fake_pdb_id(index):
    """Генерирует ID вида L001, L002... L999, LA01..."""
    # Для простоты используем L + 3 цифры. Если больше 999, можно расширить логику.
    # Но L{index:03d} хватит до 1000. Для 4000 можно использовать hex или просто цифры.
    # Используем A001, A002... чтобы точно 4 символа.
    # Простой вариант: L + index (до 9999)
    return f"L{index:03d}"[-4:] if index > 999 else f"L{index:03d}"

def prepare_global_resources(work_path, pdb, df_lig):
    """
    Подготавливает исходные данные один раз в папке global_assets.
    Оттуда мы будем копировать их в изолированные папки.
    """
    assets_dir = os.path.join(work_path, "global_assets")
    if os.path.exists(assets_dir):
        shutil.rmtree(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Исходные файлы
    logging.info("Подготовка глобальных ресурсов (белок, покет)...")
    
    # Копируем исходники в raw для обработки
    raw_dir = os.path.join(work_path, "raw")
    raw_pocket = os.path.join(raw_dir, "pocket")
    os.makedirs(raw_pocket, exist_ok=True)
    
    shutil.copy(os.path.join(work_path, "data", f"{pdb}_ligand.pdb"), os.path.join(raw_dir, f"{pdb}_ligand.pdb"))
    shutil.copy(os.path.join(work_path, "data", f"{pdb}_meeko.pdb"), os.path.join(raw_pocket, f"{pdb}_meeko.pdb"))
    
    # 2. Обработка obabel (получаем reference ligand sdf)
    ref_ligand_sdf = os.path.join(assets_dir, "reference_ligand.sdf")
    run_command([
        "obabel", os.path.join(raw_dir, f"{pdb}_ligand.pdb"),
        "-p", "7.4", "-O", ref_ligand_sdf
    ])
    
    # 3. Extract Pocket (получаем pocket pdb)
    # Скрипт требует специфической структуры папок, эмулируем её временно
    temp_extract_lig = os.path.join(work_path, "temp_extract_ligand")
    os.makedirs(temp_extract_lig, exist_ok=True)
    shutil.copy(ref_ligand_sdf, os.path.join(temp_extract_lig, f"{pdb}_docked.sdf")) # Имя важно для скрипта
    
    run_command([
        "python", "tools/extract_pocket_by_ligand.py",
        raw_pocket + "/",
        temp_extract_lig + "/",
        "0"
    ])
    
    generated_pocket = os.path.join(raw_pocket, "output", f"{pdb}_pocket.pdb")
    if not os.path.exists(generated_pocket):
        raise Exception("Не удалось создать pocket.pdb")
        
    shutil.copy(generated_pocket, os.path.join(assets_dir, "pocket.pdb"))
    shutil.rmtree(temp_extract_lig)
    
    logging.info("Глобальные ресурсы готовы.")
    return assets_dir

def run_isolated_docking(idx, ligand_name, mol, original_pdb, assets_dir, docking_path):
    """
    Запускает полный цикл для одного лиганда в полной изоляции с фейковым PDB ID.
    """
    fake_pdb = f"L{idx:03d}" # L001, L002...
    iso_dir = os.path.join(TEMP_ROOT, fake_pdb) # temp_isolation_work/L001
    
    # Создаем структуру папок Interformer внутри изоляции
    for subdir in ["ligand", "pocket", "uff", "infer", "complex", "result"]:
        os.makedirs(os.path.join(iso_dir, subdir), exist_ok=True)
        
    try:
        # 1. КОПИРОВАНИЕ И ПЕРЕИМЕНОВАНИЕ РЕСУРСОВ
        # Теперь для нейросети этот белок называется L001, а не 1tqn
        shutil.copy(os.path.join(assets_dir, "pocket.pdb"), os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb"))
        shutil.copy(os.path.join(assets_dir, "reference_ligand.sdf"), os.path.join(iso_dir, "ligand", f"{fake_pdb}_docked.sdf"))
        
        # 2. СОЗДАНИЕ UFF ЛИГАНДА
        # Сохраняем текущий лиганд как L001_uff.sdf
        uff_path = os.path.join(iso_dir, "uff", f"{fake_pdb}_uff.sdf")
        with Chem.SDWriter(uff_path) as w:
            w.write(mol)
            
        # 3. СОЗДАНИЕ CSV ЗАДАНИЯ
        # В колонке Target пишем L001. Это заставит Interformer искать файлы L001_*
        input_csv = os.path.join(iso_dir, f"{fake_pdb}_input.csv")
        with open(input_csv, 'w') as f:
            f.write("Target,Molecule ID,pose_rank\n")
            f.write(f"{fake_pdb},{ligand_name},0\n")
            
        interformer_env = {"PYTHONPATH": "interformer/"}
        start_time = time.time()

        # 4. ENERGY PREDICTION
        # -work_path указывает на iso_dir. Кэш создастся в iso_dir/tmp_beta
        run_command([
            "python", "inference.py",
            "-test_csv", input_csv,
            "-work_path", iso_dir,
            "-ensemble", "checkpoints/v0.2_energy_model",
            "-gpus", "1", "-batch_size", "1",
            "-posfix", "*val_loss*",
            "-energy_output_folder", iso_dir, # Результат кладём в корень изоляции
            "-uff_as_ligand", "-debug", "-reload"
        ], custom_env=interformer_env, timeout=600)

        # 5. RESTORE POCKET (на всякий случай)
        # Если inference удалил его из корня iso_dir, копируем из папки pocket
        pocket_in_root = os.path.join(iso_dir, f"{fake_pdb}_pocket.pdb")
        if not os.path.exists(pocket_in_root):
            shutil.copy(os.path.join(iso_dir, "pocket", f"{fake_pdb}_pocket.pdb"), pocket_in_root)

        # 6. RECONSTRUCTION
        run_command([
            "python", "docking/reconstruct_ligands.py",
            "-y", "--cwd", iso_dir,
            "--find_all", "--uff_folder", "uff", "find"
        ], timeout=600)
        
        recon_sdf = os.path.join(iso_dir, "ligand_reconstructing", f"{fake_pdb}_docked.sdf")
        if not os.path.exists(recon_sdf):
            raise PipelineStepFailed("Reconstruction failed (no SDF)")

        # 7. SCORING PREPARATION
        # Копируем результат в infer/, чтобы подготовить CSV для скоринга
        # Называем его L001_docked.sdf
        score_sdf = os.path.join(iso_dir, "infer", f"{fake_pdb}_docked.sdf")
        shutil.copy(recon_sdf, score_sdf)
        
        # Генерируем CSV для скоринга (L001_docked_infer.csv)
        run_command(["python", "tools/inference/inter_sdf2csv.py", score_sdf, "0"])
        score_csv = score_sdf.replace(".sdf", "_infer.csv")
        
        # 8. SCORING (AFFINITY)
        # Результат будет в iso_dir/result/L001_docked_infer_ensemble.csv
        run_command([
            "python", "inference.py",
            "-test_csv", score_csv,
            "-work_path", iso_dir,
            "-ligand_folder", "infer",
            "-ensemble", "checkpoints/v0.2_affinity_model/model*",
            "-use_ff_ligands", "''", "-vs",
            "-gpus", "1", "-batch_size", "20",
            "-posfix", "*val_loss*", "--pose_sel", "True"
        ], custom_env=interformer_env, timeout=600)

        # 9. EXTRACT & FIX RESULT
        generated_result = os.path.join(iso_dir, "result", f"{fake_pdb}_docked_infer_ensemble.csv")
        
        if not os.path.exists(generated_result):
            # Иногда падает в корень ./result, проверяем
            fallback = os.path.join("result", f"{fake_pdb}_docked_infer_ensemble.csv")
            if os.path.exists(fallback):
                shutil.move(fallback, generated_result)
        
        if not os.path.exists(generated_result):
            raise PipelineStepFailed("Result file not created")

        # Читаем результат
        df = pd.read_csv(generated_result)
        if df.empty: raise PipelineStepFailed("Empty result")
        
        # Исправляем данные: меняем L001 обратно на реальный 1tqn и ligand_name
        df['Target'] = original_pdb
        df['Molecule ID'] = ligand_name
        df['ligand_name'] = ligand_name
        
        # Сохраняем в финальную папку
        final_ligand_dir = os.path.join(FINAL_RESULTS_DIR, ligand_name)
        os.makedirs(final_ligand_dir, exist_ok=True)
        final_csv = os.path.join(final_ligand_dir, f"{ligand_name}_ensemble.csv")
        df.to_csv(final_csv, index=False)
        
        # Копируем SDF с позами
        final_sdf = os.path.join(docking_path, ligand_name, f"{original_pdb}_docked.sdf")
        os.makedirs(os.path.dirname(final_sdf), exist_ok=True)
        # Нужно поменять имя внутри SDF с L001 на реальное? Желательно, но не критично для PyMOL
        shutil.copy(recon_sdf, final_sdf)

        return round(time.time() - start_time, 2)

    finally:
        # 10. УНИЧТОЖЕНИЕ ИЗОЛЯТОРА
        # Удаляем всю папку temp_isolation_work/L001
        if os.path.exists(iso_dir):
            shutil.rmtree(iso_dir)

# ==============================================================================
# MAIN LOGIC
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
        merged = pd.concat(all_dfs, ignore_index=True)
        merged.to_csv(os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_infer_ensemble.csv"), index=False)

def collect_final_sdf(pdb):
    logging.info("Сборка финального SDF...")
    outfile = os.path.join(FINAL_RESULTS_DIR, f"{pdb}_docked_ALL.sdf")
    # Простая конкатенация всех SDF из docking_path
    # Реализуем через RDKit для надежности, или просто cat
    # Здесь лучше просто собрать пути, так как файлы разбросаны
    pass # Реализуйте, если критично, но CSV важнее

def main():
    parser = argparse.ArgumentParser(description="Interformer Pipeline — Total Isolation")
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-w", "--work_path", required=True)
    parser.add_argument("-d", "--docking_path", required=True)
    parser.add_argument("-l", "--df_lig", required=True)
    parser.add_argument("-o", "--omp_num_threads", default="32,32")
    args = parser.parse_args()

    # Инициализация
    if os.path.exists(FINAL_RESULTS_DIR): shutil.rmtree(FINAL_RESULTS_DIR)
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    if os.path.exists(TEMP_ROOT): shutil.rmtree(TEMP_ROOT)
    os.makedirs(TEMP_ROOT, exist_ok=True)
    
    # 1. Подготовка глобальных ресурсов (настоящий 1tqn)
    assets_dir = prepare_global_resources(args.work_path, args.pdb, args.df_lig)
    
    # 2. Чтение лигандов
    logging.info("Чтение и подготовка списка лигандов...")
    # Здесь мы используем вашу функцию подготовки, чтобы получить список RDKit Mol
    successful_mols = [] # Список кортежей (name, mol)
    
    # Загружаем CSV с лигандами
    df = pd.read_csv(os.path.join(args.work_path, "data", f"{args.df_lig}.csv"))
    # Предполагаем, что prepare_ligands_new.py делает сложную фильтрацию, 
    # но для скорости используем упрощенную загрузку или вызываем его
    # Для интеграции с вашим кодом:
    successful_names, failed_prep = prepare_ligands_for_interformer(
        os.path.join(args.work_path, "raw", f"{args.df_lig}.csv"),
        os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf")
    )
    
    # Читаем созданный uff файл, чтобы получить объекты Mol
    suppl = Chem.SDMolSupplier(os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf"))
    mol_dict = {m.GetProp("_Name"): m for m in suppl if m and m.HasProp("_Name")}
    
    pipeline_report = []
    
    # 3. ГЛАВНЫЙ ЦИКЛ
    logging.info(f"Запуск докинга для {len(successful_names)} лигандов...")
    
    for idx, lig_name in enumerate(successful_names):
        logging.info(f"\n>>> PROCESSING [{idx+1}/{len(successful_names)}]: {lig_name}")
        
        mol = mol_dict.get(lig_name)
        if not mol:
            pipeline_report.append({"ligand_name": lig_name, "status": "failed_mol_load"})
            continue
            
        try:
            # ЗАПУСК В ИЗОЛЯЦИИ С ФЕЙКОВЫМ ID (например, L001)
            # Мы передаем idx+1, чтобы ID начинались с L001
            elapsed = run_isolated_docking(
                idx + 1, 
                lig_name, 
                mol, 
                args.pdb, 
                assets_dir, 
                args.docking_path
            )
            pipeline_report.append({"ligand_name": lig_name, "status": "success", "time": elapsed})
            logging.info(f"<<< SUCCESS: {lig_name} in {elapsed}s")
            
        except Exception as e:
            logging.error(f"<<< FAILED: {lig_name} - {str(e)}")
            pipeline_report.append({"ligand_name": lig_name, "status": "failed", "error": str(e)})

    # 4. Финализация
    collect_final_csv(args.pdb)
    pd.DataFrame(pipeline_report).to_csv(os.path.join(FINAL_RESULTS_DIR, "report.csv"), index=False)
    
    # Анализ
    if os.path.exists("analyze.py"):
        run_command([
            "python", "analyze.py",
            "--results-csv", os.path.join(FINAL_RESULTS_DIR, f"{args.pdb}_docked_infer_ensemble.csv"),
            "--original-csv", os.path.join(args.work_path, "raw", f"{args.df_lig}.csv"),
            "--experimental-col", "pValue",
            "--output-folder", FINAL_RESULTS_DIR
        ])
        
    logging.info("DONE.")

if __name__ == "__main__":
    main()
