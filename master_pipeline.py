import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import pandas as pd
from rdkit import Chem
# Предполагается, что файл prepare_ligands_new.py лежит рядом
from prepare_ligands_new import prepare_ligands_for_interformer 

# ==============================================================================
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

def run_command(command, custom_env=None, timeout=None):
    cmd_str = ' '.join(command)
    logging.info(f"→ {cmd_str}" + (f" | таймаут = {timeout}s" if timeout else ""))

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
        if result.stdout.strip():
            logging.info(result.stdout.strip())
        return result.stdout
    except subprocess.TimeoutExpired as e:
        error_msg = f"ТАЙМАУТ {timeout}s: {cmd_str}"
        logging.error(error_msg)
        if e.stdout: logging.error(f"STDOUT:\n{e.stdout}")
        if e.stderr: logging.error(f"STDERR:\n{e.stderr}")
        raise PipelineStepFailed(error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"ОШИБКА (код {e.returncode}): {cmd_str}\n{e.stderr.strip()}"
        logging.error(error_msg)
        if e.stdout: logging.error(f"STDOUT:\n{e.stdout}") 
        raise PipelineStepFailed(error_msg)

def prepare_directories(work_path):
    for subdir in ["data", "raw", "raw/pocket", "ligand", "pocket", "uff", "infer", "result"]:
        os.makedirs(os.path.join(work_path, subdir), exist_ok=True)

def copy_input_files(work_path, pdb, df_lig):
    logging.info("Копирование входных файлов + полная очистка raw/pocket...")
    raw_pocket_dir = os.path.join(work_path, "raw", "pocket")
    if os.path.exists(raw_pocket_dir):
        shutil.rmtree(raw_pocket_dir)
    os.makedirs(raw_pocket_dir, exist_ok=True)

    shutil.copy(os.path.join(work_path, "data", f"{pdb}_ligand.pdb"), os.path.join(work_path, "raw", f"{pdb}_ligand.pdb"))
    shutil.copy(os.path.join(work_path, "data", f"{df_lig}.csv"), os.path.join(work_path, "raw", f"{df_lig}.csv"))
    shutil.copy(os.path.join(work_path, "data", f"{pdb}_meeko.pdb"), os.path.join(work_path, "raw", "pocket", f"{pdb}_meeko.pdb"))

def prepare_protein_and_ligand(work_path, pdb):
    logging.info("Подготовка белка и референсного лиганда...")
    for folder in ["ligand", "pocket", "uff", "infer"]:
        path = os.path.join(work_path, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            logging.info(f"ОЧИЩЕНО: {path}")
        os.makedirs(path, exist_ok=True)

    run_command([
        "obabel",
        os.path.join(work_path, "raw", f"{pdb}_ligand.pdb"),
        "-p", "7.4",
        "-O", os.path.join(work_path, "ligand", f"{pdb}_docked.sdf")
    ])

    run_command([
        "python", "tools/extract_pocket_by_ligand.py",
        os.path.join(work_path, "raw", "pocket/"),
        os.path.join(work_path, "ligand/"),
        "0"
    ])

    source = os.path.join(work_path, "raw", "pocket", "output", f"{pdb}_pocket.pdb")
    dest = os.path.join(work_path, "pocket", f"{pdb}_pocket.pdb")
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(source, dest)
    logging.info(f"Белок и референсный лиганд готовы. Pocket PDB в {dest}")


def run_docking_per_ligand(work_path, docking_path, pdb, omp_num_threads, successful_preparation_ligands):
    logging.info("Запуск покалигандного докинга (лимит 20 минут на лиганд)...")
    os.environ["OMP_NUM_THREADS"] = omp_num_threads

    docking_success = []
    docking_failed = {}
    ligand_processing_times = {}

    all_ligands_sdf = os.path.join(work_path, "uff", f"{pdb}_uff.sdf")
    tmp_dir = os.path.join(work_path, "tmp_individual_ligands")
    os.makedirs(tmp_dir, exist_ok=True)

    interformer_env = {"PYTHONPATH": "interformer/"}
    supplier = Chem.SDMolSupplier(all_ligands_sdf)

    for mol in supplier:
        if not mol or not mol.HasProp("_Name"):
            continue
        ligand_name = mol.GetProp("_Name")
        if ligand_name not in successful_preparation_ligands:
            continue

        start_time = time.time()
        logging.info(f"\n{'='*80}\nЛИГАНД: {ligand_name} | {time.strftime('%H:%M:%S')}\n{'='*80}")

        ligand_dir = os.path.join(docking_path, ligand_name)
        single_sdf = os.path.join(tmp_dir, f"{pdb}_{ligand_name}.sdf")

        try:
            # 1. Запись лиганда в отдельный SDF
            with Chem.SDWriter(single_sdf) as w:
                w.write(mol)

            # 2. Подготовка директории и копирование общих файлов
            os.makedirs(ligand_dir, exist_ok=True)
            os.makedirs(os.path.join(ligand_dir, "uff"), exist_ok=True)
            shutil.copy(single_sdf, os.path.join(ligand_dir, "uff", f"{pdb}_uff.sdf"))

            # Копируем ligand, pocket, complex
            for folder in ["ligand", "pocket", "complex"]:
                src = os.path.join(work_path, folder)
                dst = os.path.join(ligand_dir, folder)
                if os.path.exists(src):
                    # Используем rsync-подобное копирование
                    if os.path.exists(dst):
                         shutil.rmtree(dst)
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    os.makedirs(dst, exist_ok=True)

            csv_path = single_sdf.replace(".sdf", "_infer.csv")
            run_command(["python", "tools/inference/inter_sdf2csv.py", single_sdf, "1"])

            if time.time() - start_time > 1200:
                raise LigandTimeout("Общий лимит 20 минут")

            # 3. Energy prediction
            run_command([
                "python", "inference.py",
                "-test_csv", csv_path,
                "-work_path", work_path,
                "-ensemble", "checkpoints/v0.2_energy_model",
                "-gpus", "1", "-batch_size", "1",
                "-posfix", "*val_loss*", "-energy_output_folder", ligand_dir,
                "-uff_as_ligand", "-debug", "-reload"
            ], custom_env=interformer_env, timeout=900)

            if time.time() - start_time > 1200:
                raise LigandTimeout("20 минут")

            # 4. ВАЖНОЕ ИСПРАВЛЕНИЕ: Гарантируем наличие pocket.pdb в корне ligand_dir перед реконструкцией
            pocket_src = os.path.join(work_path, "pocket", f"{pdb}_pocket.pdb")
            pocket_in_root = os.path.join(ligand_dir, f"{pdb}_pocket.pdb")
            
            if not os.path.exists(pocket_src):
                 raise PipelineStepFailed(f"Исходный {pdb}_pocket.pdb не найден в {work_path}/pocket. Не могу продолжить.")

            # Копируем файл повторно, так как inference.py мог его удалить
            shutil.copy(pocket_src, pocket_in_root)
            logging.info(f"Успешно восстановлен {pdb}_pocket.pdb в корне ligand_dir. Запускаем reconstruction.")
            
            # 5. Docking/Reconstruction
            run_command([
                "python", "docking/reconstruct_ligands.py",
                "-y", "--cwd", ligand_dir,
                "--find_all", "--uff_folder", "uff", "find"
            ], timeout=900)

            recon_sdf = os.path.join(ligand_dir, "ligand_reconstructing", f"{pdb}_docked.sdf")
            if not os.path.exists(recon_sdf):
                raise PipelineStepFailed("reconstruct не создал SDF")

            if time.time() - start_time > 1200:
                raise LigandTimeout("20 минут")

            # 6. Scoring
            infer_dir = os.path.join(tmp_dir, "infer", ligand_name)
            os.makedirs(infer_dir, exist_ok=True)
            shutil.copy(recon_sdf, os.path.join(infer_dir, f"{pdb}_docked.sdf"))

            run_command(["python", "tools/inference/inter_sdf2csv.py", os.path.join(infer_dir, f"{pdb}_docked.sdf"), "0"])

            docked_csv = os.path.join(infer_dir, f"{pdb}_docked_infer.csv")
            run_command([
                "python", "inference.py",
                "-test_csv", docked_csv,
                "-work_path", work_path,
                "-ligand_folder", os.path.relpath(infer_dir, work_path),
                "-ensemble", "checkpoints/v0.2_affinity_model/model*",
                "-use_ff_ligands", "''", "-vs",
                "-gpus", "1", "-batch_size", "20",
                "-posfix", "*val_loss*", "--pose_sel", "True"
            ], custom_env=interformer_env, timeout=600)

            elapsed = time.time() - start_time
            logging.info(f"УСПЕХ {ligand_name} | время: {elapsed:.1f} сек ({elapsed/60:.1f} мин)\n{'='*80}")
            docking_success.append(ligand_name)
            ligand_processing_times[ligand_name] = round(elapsed, 2)

        except Exception as e:
            elapsed = time.time() - start_time
            err = str(e)
            if isinstance(e, LigandTimeout):
                 err = "ТАЙМАУТ 20 МИНУТ → " + err
            logging.error(f"ПРОВАЛ {ligand_name} | время: {elapsed:.1f} сек | {err}\n{'='*80}")
            docking_failed[ligand_name] = f"{err} ({elapsed:.1f} сек)"
            ligand_processing_times[ligand_name] = round(elapsed, 2)
        
        finally:
            if os.path.exists(single_sdf):
                os.remove(single_sdf)

    # Сборка результатов
    all_dfs = []
    for ligand_name in docking_success:
        tmp_csv = os.path.join(tmp_dir, "infer", ligand_name, f"{pdb}_docked_infer_ensemble.csv")
        if os.path.exists(tmp_csv):
            df = pd.read_csv(tmp_csv)
            df_ligand = df[df['Molecule ID'] == ligand_name]
            df_ligand['ligand_name'] = ligand_name
            all_dfs.append(df_ligand)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(os.path.join("result", f"{pdb}_docked_infer_ensemble.csv"), index=False)

    final_sdf = os.path.join(work_path, "infer", f"{pdb}_docked_ALL.sdf")
    with Chem.SDWriter(final_sdf) as w:
        for ligand_name in docking_success:
            path = os.path.join(docking_path, ligand_name, "ligand_reconstructing", f"{pdb}_docked.sdf")
            if os.path.exists(path):
                for m in Chem.SDMolSupplier(path):
                    if m:
                        m.SetProp("_Name", ligand_name)
                        w.write(m)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info(f"Финальный объединенный SDF сохранен в {final_sdf}")
    return docking_success, docking_failed, ligand_processing_times

def analyze_results(work_path, pdb, df_lig):
    results_csv = f"result/{pdb}_docked_infer_ensemble.csv"
    if not os.path.exists(results_csv):
        logging.warning("Нет результатов для анализа (final_ensemble.csv не найден)")
        return
    run_command([
        "python", "analyze.py",
        "--results-csv", results_csv,
        "--original-csv", os.path.join(work_path, "raw", f"{df_lig}.csv"),
        "--experimental-col", "pValue",
        "--output-folder", "result"
    ])

def main():
    parser = argparse.ArgumentParser(description="Interformer master pipeline — 100% рабочий")
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-w", "--work_path", required=True)
    parser.add_argument("-d", "--docking_path", required=True)
    parser.add_argument("-l", "--df_lig", required=True)
    parser.add_argument("-o", "--omp_num_threads", default="32,32")
    args = parser.parse_args()

    if os.path.exists("result"):
        shutil.rmtree("result")
    os.makedirs("result", exist_ok=True)

    logging.info(f"Старт пайплайна: {args}")

    pipeline_report = []
    
    try:
        prepare_directories(args.work_path)
        copy_input_files(args.work_path, args.pdb, args.df_lig)
        prepare_protein_and_ligand(args.work_path, args.pdb)

        # Шаг подготовки лигандов
        logging.info("Подготовка лигандов (prepare_ligands_new.py)...")
        successful_prep, failed_prep = prepare_ligands_for_interformer(
            os.path.join(args.work_path, "raw", f"{args.df_lig}.csv"),
            os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf")
        )
        logging.info(f"Подготовлено успешно: {len(successful_prep)}, провалено: {len(failed_prep)}")

        # Инициализация отчета
        for lig in successful_prep:
            pipeline_report.append({"ligand_name": lig, "preparation": "success", "docking": "pending", "analysis": "pending", "error": "", "docking_time_sec": 0})
        
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: failed_prep — это список, а не словарь ---
        for lig in failed_prep:
            pipeline_report.append({"ligand_name": lig, "preparation": "failed", "docking": "skipped", "analysis": "skipped", "error": "preparation failed", "docking_time_sec": 0})
        # ----------------------------------------------------------------

        if successful_prep:
            success, failed, times = run_docking_per_ligand(
                args.work_path, args.docking_path, args.pdb, args.omp_num_threads, successful_prep
            )
            for item in pipeline_report:
                name = item["ligand_name"]
                if name in success:
                    item["docking"] = "success"
                    item["docking_time_sec"] = times[name]
                    try:
                        analyze_results(args.work_path, args.pdb, args.df_lig)
                        item["analysis"] = "success"
                    except Exception as e:
                        logging.error(f"Анализ упал для {name}: {e}")
                        item["analysis"] = "failed"
                        item["error"] += f" | Analysis failed: {str(e)}"

                elif name in failed:
                    item["docking"] = "failed"
                    item["error"] = failed[name]
                    item["docking_time_sec"] = times[name]

    except Exception as full_pipeline_e:
        logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА ПАЙПЛАЙНА (до цикла лигандов): {full_pipeline_e}")
        if not pipeline_report:
             pipeline_report.append({"ligand_name": "N/A", "preparation": "failed", "docking": "skipped", "analysis": "skipped", "error": f"Critical pipeline failure: {full_pipeline_e}", "docking_time_sec": 0})

    pd.DataFrame(pipeline_report).to_csv(f"result/pipeline_summary_report_{args.pdb}.csv", index=False)
    logging.info("ПАЙПЛАЙН ЗАВЕРШЁН (с ошибками или без)! Отчет сохранен.")

if __name__ == "__main__":
    main()
