Here is the English translation for your GitHub repository. I have maintained the technical terminology and formatting used in the bioinformatics and molecular modeling community.

---

# Interformer Implementation Details

The main script driving the pipeline is **```pipeline.sh```**. We have included detailed comments for each stage within that script.

### 1. Protein and Ligand Preparation
The protein was pre-processed using `meeko` (no further modifications were made).
The reference ligand was protonated as described in the original README:

```bash
obabel ${WORK_PATH}/raw/${PDB}_ligand.pdb -p 7.4 -O ${WORK_PATH}/ligand/${PDB}_docked.sdf
```

For the preparation of the target ligands, we developed the **```prepare_ligands.py```** script. 
**The primary objective of the script** is to read molecular structures from a CSV file, transform them, filter them, and save them in SDF format.

#### Key Functions of **```prepare_ligands.py```**:

* **Molecular Weight Filtering:** The molecular weight of each molecule is checked. Molecules exceeding a set threshold (default is 500 Da) are filtered out and excluded from further processing.
* **Hydrogen Addition:** Hydrogen atoms are added to the molecule (`Chem.AddHs`).
* **3D Coordinate Generation:** 3D coordinates are generated using the ETKDGv3 algorithm (`AllChem.EmbedMolecule`).
* **Geometry Optimization:** Geometry optimization is performed using the UFF force field (`AllChem.UFFOptimizeMolecule`) to minimize the molecule's energy.
* **SDF Export:** Prepared 3D structures are saved as SDF (Structure-Data File).
* **Naming:** Each saved molecule is assigned a name corresponding to its original row number in the CSV file.


### 2. Internal Interformer Protein and Ligand Processing

#### General Preparation Workflow

#### Step 1: Protein Preparation

The main task here is to isolate the part of the protein that interacts with the ligand (the binding pocket) from the overall structure and prepare it for analysis.

#### 1.1. Hydrogen Addition and Protonation State Assignment

*   **Process:** Hydrogen atoms are added to the protein structure, as they are absent in most PDB files. Additionally, the script determines which amino acid residues (e.g., Histidine, Aspartate, Glutamate) should be protonated (carry a charge) at physiological pH.
*   **Tool:** The README suggests using **Reduce**, a standard bioinformatics tool for this task.
    ```bash
    # Command from README
    reduce examples/raw/2qbr.pdb > examples/raw/pocket/2qbr_reduce.pdb
    ```
*   **Result:** A new PDB file (`2qbr_reduce.pdb`) containing the complete protein structure with correctly assigned hydrogen atoms.

*Note: For some proteins, Reduce failed to process the structure. In those cases, we replaced the command with `obabel` (following the same logic used for ligand preparation in section 1).*

#### 1.2. Pocket Extraction

This is a critical stage utilizing the **`extract_pocket_by_ligand.py`** script.

*   **Process:** The script "clips" only the atoms and amino acid residues that are in close proximity to the reference ligand from the full protein structure.
*   **Mechanism of `extract_pocket_by_ligand.py`:**
    1.  **Input:** The script accepts:
        *   The path to the processed protein PDB file from step 1.1 (`2qbr_reduce.pdb`).
        *   The path to the reference ligand SDF file (`2qbr_docked.sdf`).
    2.  **Ligand Identification:** The script loads both structures using the **RDKit** library.
    3.  **Environment Extraction:** Using the reference ligand's atomic coordinates, the script identifies all protein atoms within a **10 Å** radius of the ligand. This operation is performed using the `ExtractPocketAndLigand` function from the `oddt` library (a toolkit built on RDKit).
    4.  **Reference Ligand Removal (Optional):** Importantly, the script can remove the original reference ligand from the protein structure (`rm_ccd=True`) to ensure no "extra" molecules interfere with the docking of the new ligand.
    5.  **Cofactor Preservation:** The script is designed to preserve critical cofactors (e.g., Zn, Mg metal ions) if they fall within the 10 Å radius.
*   **Result:** A compact PDB file (`2qbr_pocket.pdb`) containing only the binding pocket atoms. This significantly accelerates subsequent computations as the neural network does not need to process the entire protein.

This script was executed and performed without issues.

#### Step 2: Ligand Preparation

In this stage, we prepare the molecule that we intend to "dock" into the protein pocket.

##### 2.1. Ligand Protonation

*   **Process:** Similar to the protein, the most probable protonation state of the ligand at a given pH (usually 7.4) is determined. This is critical for correctly modeling interactions, specifically hydrogen bonds.
*   **Tool:** The README utilizes **OpenBabel (`obabel`)**.
    ```bash
    # Command from README
    obabel examples/raw/2qbr_ligand.sdf -p 7.4 -O examples/ligand/2qbr_docked.sdf
    ```
*   **Result:** A new SDF file (`2qbr_docked.sdf`) with added hydrogens and correct charges.

#### 2.2. Initial 3D Conformation Generation

The second key stage involves the **`rdkit_ETKDG_3d_gen.py`** script.

*   **Mechanism:**
    1.  **Input:** The script takes the SDF file from step 2.1.
    2.  **Conformational Sampling:** Using RDKit's **ETKDGv3** algorithm, several possible 3D structures (defined as `n_confs=30` in the code) are generated. This algorithm is known for reconstructing accurate molecular geometries.
    3.  **Energy Minimization:** Each generated conformation is optimized using the **UFF (Universal Force Field)**. This process "shakes" the molecule so that its atoms settle into the most stable, low-energy positions.
    4.  **Selection:** The conformation with the lowest energy is selected from the optimized pool.
*   **Result:** A new SDF file (`2qbr_uff.sdf`) containing a single, energetically favorable 3D structure, serving as the starting pose for docking.

*Our **```prepare_ligands.py```** script is based on this logic but includes enhanced error handling, logging, and molecular weight filtering.*

#### Step 3: Final Directory Structure and Docking Execution

Upon completion, the files are organized into the structure expected by Interformer:

*   `examples/pocket/2qbr_pocket.pdb`: Extracted and processed **protein pocket**.
*   `examples/ligand/2qbr_docked.sdf`: Protonated **reference ligand** (used to define the docking center).
*   `examples/uff/2qbr_uff.sdf`: Low-energy **3D structure of the target ligand** for docking.

### 3. Installation
The script for a fresh, successful installation is located at `installation.sh`.

### 4. Error Debugging
For most systems, few errors occurred, and we manually restarted simulations a couple of times after removing the specific ligand that caused the crash. The most significant issues were encountered with the `1tqn_ic40` system. For this system, the final version includes a dedicated bash script for simulation execution, **```pipeline_mmff_uff.sh```**, and a dedicated preparation script, **```prepare_ligands_mmff_uff.py```**. 

Initially, we attempted to fix errors without modifying the original Interformer scripts. Therefore, for certain ligands in this system, if UFF optimization failed, the more universal **MMFF** force field was applied. Additionally, all sulfur-containing ligands were removed (there were few), as error logs specifically pointed to them. These "trouble molecules" are saved in `result/trouble_ligands_{pdb_id}.csv`. Otherwise, the scripts are identical to those used for other systems.

We also added minor debugging logic to one of the original scripts. In ```docking/pdbqt_ligand/wrappers_for_third_party_tools/wrapper_rdkitmol/wrapper_rdkitmol.py```, we modified the `save_sdf_given_wrappers` class method.

The primary modification involves adding logic that **redirects file saving from the temporary `/tmp/` directory to a permanent directory.**

**Original Version:**
```python
# ...
else:
    # rewrite the sdf
    io_writer = open(abspath_sdf_to_save, 'w')
# ...
```

**Modified Version:**
```python
# ...
else:
    # rewrite the sdf
    if abspath_sdf_to_save.startswith('/tmp/'):
        import os
        filename = os.path.basename(abspath_sdf_to_save)
        new_path_dir = '/mnt/tank/scratch/ikarpushkina/Interformer/tmp_output'
        abspath_sdf_to_save = os.path.join(new_path_dir, filename)
    io_writer = open(abspath_sdf_to_save, 'w')
# ...
```

The script now performs a check before creating the file:
1.  **Condition:** `if abspath_sdf_to_save.startswith('/tmp/')` checks if the path starts with the Linux temporary directory.
2.  **Action:** If true, it extracts the filename and redirects it to the permanent directory: `/mnt/tank/scratch/ikarpushkina/Interformer/tmp_output`.
3.  **Result:** Files originally destined for `/tmp` (which are often deleted on reboot) are saved permanently for debugging purposes.

---

# Pipeline Modifications: Original Intent vs. Implemented Robust Pipeline

We implemented an approach based on multiple restarts with "context spoofing," which is necessary for stable performance. This differs significantly from the original Interformer design.

### 1. Canonical Approach (Original)
*As described in the README and used in `inference.py` and `pipeline.sh`.*

**Concept:** **Batch Processing.**
The authors intend for the user to prepare one large CSV file containing all ligands and feed it to the neural network at once.

*   **Identification:** The system relies on the **PDB ID** (the first 4 characters of the filename or "Target" column).
*   **Caching:** To speed up performance, Interformer aggressively caches pre-processed data (graphs, features) in the `tmp_beta` folder. The cache is bound to the filename and PDB ID.
*   **Data Flow:**
    1.  `inference.py` (Energy) reads the entire CSV, calculates energies for all rows, and saves results.
    2.  `reconstruct_ligands.py` (C++) takes the energy folder and reconstructs poses for the entire batch.
    3.  `inference.py` (Affinity) evaluates the entire batch of poses simultaneously.
*   **Cons:**
    *   **Fragility:** If the C++ docking code crashes on one complex ligand (e.g., segfault), the entire batch process stops.
    *   **File Conflicts:** Scripts frequently overwrite files with identical names (e.g., `pocket.pdb`) unless the directory structure is managed perfectly.
    *   **"Data Leakage" in loops:** If one attempts to run this process in a `for` loop for a single protein, the `tmp_beta` cache does not refresh, causing the network to return results from the previous ligand for the new one (a behavior we observed).

---

### 2. Implemented Robust Pipeline (Our Approach)
*Implemented in the final `master_pipeline_log.py`.*

**Concept:** **Iterative Isolation.**
We bypass the caching issue by making Interformer "think" that every ligand is a **completely new, unique project with a new protein** it has never seen before.

#### Comparison Table:

| Feature | Canonical Interformer | Robust Master Pipeline (Ours) |
| :--- | :--- | :--- |
| **Execution Mode** | One run for ~1000 ligands (Batch). | ~1000 runs of 1 ligand each (Loop). |
| **Identification** | Uses the real PDB ID (e.g., `1tqn`). | **Generates a "Fake" ID** (`L001`, `L002`...) for each ligand. |
| **Caching** | Relies on `tmp_beta` for speed. | **Forced deletion** of `tmp_beta` before each step + Fake ID ensures cache uniqueness. |
| **File System** | All results saved in one folder (`energy_output`). | Each ligand works in an **isolated sandbox** (`temp_isolation_work/L001`), deleted after success. |
| **Robustness** | Single ligand error crashes the entire process. | Errors are caught via `try-except`, logged, and the script moves to the next ligand. |
| **Runtime** | Faster (lower Python startup overhead). | Slightly slower (restarting environment per ligand) but **guarantees results**. |

#### Step-by-Step Logic:

1.  **Preparation:** We take the real protein (e.g., `1tqn`) and save a copy.
2.  **Ligand Loop:** Iterate through ligands one by one.
3.  **ID Spoofing:**
    *   Assign an ID like `L001`.
    *   Create a temporary folder `temp/L001`.
    *   Copy the protein but rename it to `L001_pocket.pdb`.
    *   Copy the ligand as `L001_uff.sdf`.
4.  **Execution:** Run Interformer. It sees "Protein L001," fails to find it in the cache, and performs all calculations from scratch.
5.  **Harvesting:** Collect results from the `L001` folder, map the ID back to the real name (e.g., `1tqn`), and save to the final output.
6.  **Cleanup:** Delete the `L001` folder and the cache so the next ligand (`L002`) starts with a clean slate.

### Conclusion

We adopted this strategy to overcome an architectural limitation of Interformer: its **heavy reliance on caching and specific filenames**. By using **Fake IDs (`L001`)**, we guaranteed an bypass of the internal caching mechanisms without needing to rewrite the core source code. This makes the process more resource-intensive per run but ensures it is highly reliable and reproducible.
