import argparse
import logging
import os
import shutil
import subprocess
import sys
import pandas as pd
from rdkit import Chem

from prepare_ligands import prepare_ligands_for_interformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("master_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class PipelineStepFailed(Exception):
    """Custom exception for failed pipeline steps."""
    pass

def run_command(command):
    """Executes a command and raises an exception on failure."""
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Command successful:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}:\n{e.stderr}")
        raise PipelineStepFailed(f"Command failed: {' '.join(command)}")

def prepare_directories(work_path):
    """Creates the necessary directories for the pipeline."""
    for subdir in ["data", "raw", "raw/pocket", "ligand", "pocket", "uff", "infer"]:
        os.makedirs(os.path.join(work_path, subdir), exist_ok=True)

def copy_input_files(work_path, pdb, df_lig):
    """Copies input files to the appropriate directories."""
    logging.info("Copying input files...")
    shutil.copy(os.path.join(work_path, "data", f"{pdb}_ligand.pdb"), os.path.join(work_path, "raw", f"{pdb}_ligand.pdb"))
    shutil.copy(os.path.join(work_path, "data", f"{df_lig}.csv"), os.path.join(work_path, "raw", f"{df_lig}.csv"))
    shutil.copy(os.path.join(work_path, "data", f"{pdb}_meeko.pdb"), os.path.join(work_path, "raw", "pocket", f"{pdb}_meeko.pdb"))

def prepare_protein_and_ligand(work_path, pdb):
    """Prepares the protein and reference ligand."""
    logging.info("Preparing protein and reference ligand...")
    run_command(["obabel", os.path.join(work_path, "raw", f"{pdb}_ligand.pdb"), "-p", "7.4", "-O", os.path.join(work_path, "ligand", f"{pdb}_docked.sdf")])
    run_command(["python", "tools/extract_pocket_by_ligand.py", os.path.join(work_path, "raw", "pocket/"), os.path.join(work_path, "ligand/"), "0"])
    shutil.move(os.path.join(work_path, "raw", "pocket", "output", f"{pdb}_pocket.pdb"), os.path.join(work_path, "pocket"))

def run_docking_per_ligand(work_path, docking_path, pdb, omp_num_threads, successful_preparation_ligands):
    """Runs the docking process for each ligand individually."""
    logging.info("Starting per-ligand docking process...")
    os.environ["OMP_NUM_THREADS"] = omp_num_threads

    docking_success = []
    docking_failed = {}

    all_ligands_sdf_path = os.path.join(work_path, "uff", f"{pdb}_uff.sdf")
    tmp_ligand_dir = os.path.join(work_path, "tmp_individual_ligands")
    os.makedirs(tmp_ligand_dir, exist_ok=True)

    supplier = Chem.SDMolSupplier(all_ligands_sdf_path)
    for mol in supplier:
        if not mol or not mol.HasProp("_Name"):
            continue

        ligand_name = mol.GetProp("_Name")
        if ligand_name not in successful_preparation_ligands:
            continue

        logging.info(f"--- Processing ligand: {ligand_name} ---")
        ligand_docking_dir = os.path.join(docking_path, ligand_name)
        single_sdf_path = os.path.join(tmp_ligand_dir, f"{ligand_name}.sdf")

        with Chem.SDWriter(single_sdf_path) as writer:
            writer.write(mol)

        try:
            # Create a self-contained environment for the docking script
            os.makedirs(ligand_docking_dir, exist_ok=True)
            ligand_docking_uff_dir = os.path.join(ligand_docking_dir, "uff")
            os.makedirs(ligand_docking_uff_dir, exist_ok=True)
            shutil.copy(single_sdf_path, os.path.join(ligand_docking_uff_dir, f"{pdb}_uff.sdf"))
            for required_dir in ["ligand", "pocket", "complex"]:
                if os.path.exists(os.path.join(work_path, required_dir)):
                    shutil.copytree(os.path.join(work_path, required_dir), os.path.join(ligand_docking_dir, required_dir), dirs_exist_ok=True)

            # 1. Create query CSV for the single ligand
            single_uff_infer_csv = os.path.join(tmp_ligand_dir, f"{ligand_name}_infer.csv")
            run_command(["python", "tools/inference/inter_sdf2csv.py", single_sdf_path, "1"])

            # 2. Predict energy
            run_command(["python", "inference.py", "-test_csv", single_uff_infer_csv, "-work_path", work_path, "-ensemble", "checkpoints/v0.2_energy_model", "-gpus", "1", "-batch_size", "1", "-posfix", "*val_loss*", "-energy_output_folder", ligand_docking_dir, "-uff_as_ligand", "-debug", "-reload"])

            # 3. Reconstruct (Docking)
            run_command(["python", "docking/reconstruct_ligands.py", "-y", "--cwd", ligand_docking_dir, "--find_all", "--uff_folder", "uff", "find"])

            # 4. Scoring
            reconstructed_sdf = os.path.join(ligand_docking_dir, "ligand_reconstructing", f"{pdb}_docked.sdf")
            if not os.path.exists(reconstructed_sdf):
                raise PipelineStepFailed(f"Reconstructed SDF not found for {ligand_name}")

            ligand_infer_dir = os.path.join(tmp_ligand_dir, "infer", ligand_name)
            os.makedirs(ligand_infer_dir, exist_ok=True)
            shutil.copy(reconstructed_sdf, os.path.join(ligand_infer_dir, f"{pdb}_docked.sdf"))

            run_command(["python", "tools/inference/inter_sdf2csv.py", os.path.join(ligand_infer_dir, f"{pdb}_docked.sdf"), "0"])
            docked_infer_csv = os.path.join(ligand_infer_dir, f"{pdb}_docked_infer.csv")

            run_command(["python", "inference.py", "-test_csv", docked_infer_csv, "-work_path", work_path, "-ligand_folder", os.path.relpath(ligand_infer_dir, work_path), "-ensemble", "checkpoints/v0.2_affinity_model/model*", "-use_ff_ligands", "''", "-vs", "-gpus", "1", "-batch_size", "20", "-posfix", "*val_loss*", "--pose_sel", "True"])

            docking_success.append(ligand_name)

        except PipelineStepFailed as e:
            logging.error(f"Docking failed for ligand {ligand_name}: {e}")
            docking_failed[ligand_name] = str(e)

    logging.info("--- Merging individual docking results ---")
    # Merge CSV results
    all_dfs = []
    for ligand_name in docking_success:
        ensemble_csv = os.path.join(tmp_ligand_dir, "infer", ligand_name, f"{pdb}_docked_infer_ensemble.csv")
        if os.path.exists(ensemble_csv):
            all_dfs.append(pd.read_csv(ensemble_csv))

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_csv_path = os.path.join("result", f"{pdb}_docked_infer_ensemble.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        logging.info(f"Merged ensemble CSV saved to {merged_csv_path}")

    # Merge SDF results
    final_docked_sdf_path = os.path.join(work_path, "infer", f"{pdb}_docked.sdf")
    with Chem.SDWriter(final_docked_sdf_path) as writer:
        for ligand_name in docking_success:
            reconstructed_sdf = os.path.join(docking_path, ligand_name, "ligand_reconstructing", f"{pdb}_docked.sdf")
            if os.path.exists(reconstructed_sdf):
                for mol in Chem.SDMolSupplier(reconstructed_sdf):
                    if mol: writer.write(mol)
    logging.info(f"Merged docked SDF saved to {final_docked_sdf_path}")

    shutil.rmtree(tmp_ligand_dir)
    return docking_success, docking_failed

def analyze_results(work_path, pdb, df_lig):
    """Analyzes the docking results."""
    logging.info("Analyzing results...")
    results_csv = f"result/{pdb}_docked_infer_ensemble.csv"
    if not os.path.exists(results_csv):
        logging.warning(f"Results CSV not found: {results_csv}. Skipping analysis.")
        return
    run_command(["python", "analyze.py", "--results-csv", results_csv, "--original-csv", os.path.join(work_path, "raw", f"{df_lig}.csv"), "--experimental-col", "pValue", "--output-folder", "result"])

def main():
    parser = argparse.ArgumentParser(description="Master pipeline for Interformer docking.")
    parser.add_argument("-p", "--pdb", required=True, help="PDB code (e.g., 1ere, 1tqn).")
    parser.add_argument("-w", "--work_path", required=True, help="Working directory.")
    parser.add_argument("-d", "--docking_path", required=True, help="Directory for docking results.")
    parser.add_argument("-l", "--df_lig", required=True, help="Name of the CSV file with ligands.")
    parser.add_argument("-o", "--omp_num_threads", default="32,32", help="Number of threads for OpenMP.")
    args = parser.parse_args()

    logging.info(f"Starting master pipeline with parameters: {args}")

    pipeline_report = []

    try:
        prepare_directories(args.work_path)
        copy_input_files(args.work_path, args.pdb, args.df_lig)
        prepare_protein_and_ligand(args.work_path, args.pdb)
    except (PipelineStepFailed, IOError, OSError) as e:
        logging.critical(f"Setup phase failed: {e}. Aborting.")
        sys.exit(1)

    successful_prep, failed_prep = prepare_ligands_for_interformer(os.path.join(args.work_path, "raw", f"{args.df_lig}.csv"), os.path.join(args.work_path, "uff", f"{args.pdb}_uff.sdf"))

    for ligand in successful_prep:
        pipeline_report.append({"ligand_name": ligand, "preparation": "success", "docking": "pending", "analysis": "pending", "error": ""})
    for ligand in failed_prep:
        pipeline_report.append({"ligand_name": ligand, "preparation": "failed", "docking": "skipped", "analysis": "skipped", "error": "Failed during ligand preparation"})

    if not successful_prep:
        logging.error("No ligands were successfully prepared. Aborting.")
    else:
        docking_success, docking_failed = run_docking_per_ligand(args.work_path, args.docking_path, args.pdb, args.omp_num_threads, successful_prep)

        for item in pipeline_report:
            if item["ligand_name"] in docking_success:
                item["docking"] = "success"
            elif item["ligand_name"] in docking_failed:
                item["docking"] = "failed"
                item["analysis"] = "skipped"
                item["error"] = docking_failed[item["ligand_name"]]

        try:
            analyze_results(args.work_path, args.pdb, args.df_lig)
            for item in pipeline_report:
                if item["docking"] == "success":
                    item["analysis"] = "success"
        except PipelineStepFailed as e:
            logging.error(f"Analysis phase failed: {e}")
            for item in pipeline_report:
                if item["docking"] == "success":
                    item["analysis"] = "failed"
                    item["error"] = str(e)

    report_df = pd.DataFrame(pipeline_report)
    report_df.to_csv(os.path.join("result", f"pipeline_summary_report_{args.pdb}.csv"), index=False)
    logging.info(f"Pipeline summary report saved to result/pipeline_summary_report_{args.pdb}.csv")
    logging.info("Master pipeline finished.")

if __name__ == "__main__":
    main()
