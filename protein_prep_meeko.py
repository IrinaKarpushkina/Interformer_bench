import os
import logging
import sys

# ==============================================================================
# БЛОК ДИАГНОСТИКИ ОКРУЖЕНИЯ
# ==============================================================================
print(f"DEBUG: [protein_prep_meeko] Initializing...")
# (Оставляем диагностику импортов как была, она работает)
try:
    import meeko
    from meeko import MoleculePreparation, PDBQTMolecule, Polymer, ResidueChemTemplates
except ImportError as e:
    print(f"DEBUG: REAL IMPORT ERROR: {e}")
    sys.exit(1)
# ==============================================================================

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MEEKO_PREP] - %(message)s')

def prepare_protein_and_ref(raw_pdb_path, output_dir, pdb_code, chain_id, ref_ligand_code):
    setup_logger()
    
    logging.info(f"--- ЗАПУСК РАЗДЕЛЕНИЯ БЕЛКА И ЛИГАНДА ---")
    logging.info(f"Входной файл: {raw_pdb_path}")
    
    if not os.path.exists(raw_pdb_path):
        logging.error(f"Файл {raw_pdb_path} НЕ СУЩЕСТВУЕТ!")
        sys.exit(1)

    out_protein_path = os.path.join(output_dir, f"{pdb_code}_meeko.pdb") 
    out_ref_path = os.path.join(output_dir, f"{pdb_code}_ligand.pdb")
    
    logging.info(f"Filter: Chain='{chain_id}', RefLigand='{ref_ligand_code}'")
    
    kept_lines_protein = [] 
    kept_lines_ref = []     
    
    valid_res = ref_ligand_code.split("+") if "+" in ref_ligand_code else [ref_ligand_code]

    try:
        with open(raw_pdb_path, 'r') as f:
            pdb_content = f.readlines()
    except Exception as e:
        logging.error(f"Ошибка чтения файла: {e}")
        sys.exit(1)

    # --- ДИАГНОСТИКА СОДЕРЖИМОГО ---
    # Мы соберем все найденные HETATM, чтобы показать пользователю, если что-то пойдет не так
    found_ligands = set()

    for line in pdb_content:
        # Белок
        if line.startswith("ATOM"):
            if line[21] == chain_id:
                kept_lines_protein.append(line)
        
        # Лиганд
        elif line.startswith("HETATM"):
            res_name = line[17:20].strip()
            curr_chain = line[21]
            
            if res_name == "HOH": continue 
            
            # Сохраняем для отчета, что мы вообще нашли
            found_ligands.add(f"'{res_name}' in Chain '{curr_chain}'")
            
            # Проверка условий
            if curr_chain == chain_id and res_name in valid_res:
                kept_lines_ref.append(line)

    # --- ПРОВЕРКИ ---
    logging.info(f"Найдено строк белка: {len(kept_lines_protein)}")
    logging.info(f"Найдено строк лиганда: {len(kept_lines_ref)}")

    if not kept_lines_protein:
        raise ValueError(f"CRITICAL: Не найдены атомы белка (ATOM) в цепи {chain_id}!")
    
    if not kept_lines_ref:
        # ВЫВОДИМ ПОЛЕЗНУЮ ИНФОРМАЦИЮ ПЕРЕД ОШИБКОЙ
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.error(f"ОШИБКА: Лиганд {ref_ligand_code} не найден в цепи {chain_id}!")
        logging.error("ВОТ СПИСОК ВСЕХ ЛИГАНДОВ, НАЙДЕННЫХ В ФАЙЛЕ:")
        for l in sorted(found_ligands):
            logging.error(f"  -> Найдено: {l}")
        logging.error("ПОЖАЛУЙСТА, ОБНОВИТЕ pipeline_config.py В СООТВЕТСТВИИ С ЭТИМ СПИСКОМ.")
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise ValueError(f"CRITICAL: Референсный лиганд {ref_ligand_code} не найден в цепи {chain_id}!")

    # 1. Сохранение ЛИГАНДА
    with open(out_ref_path, 'w') as f:
        f.writelines(kept_lines_ref)
    logging.info(f"Файл лиганда записан: {out_ref_path}")

    # 2. Сохранение БЕЛКА (с обработкой Meeko)
    raw_protein_str = "".join(kept_lines_protein)
    try:
        logging.info("Запуск Meeko Polymer preparation...")
        templates = ResidueChemTemplates.create_from_defaults()
        mk_prep = MoleculePreparation.from_config({})
        
        polymer = Polymer.from_pdb_string(
            raw_protein_str, 
            templates, 
            mk_prep, 
            allow_bad_res=True, 
            default_altloc='A'
        )
        cleaned_pdb_str = polymer.to_pdb()
        with open(out_protein_path, 'w') as f:
            f.write(cleaned_pdb_str)
        logging.info(f"Файл белка (Meeko) записан: {out_protein_path}")
        
    except Exception as e:
        logging.error(f"MEEKO PROCESSING FAILED: {e}")
        logging.warning("Сохраняем белок БЕЗ обработки Meeko (Fallback mode).")
        with open(out_protein_path, 'w') as f:
            f.writelines(kept_lines_protein)

    return out_protein_path, out_ref_path
