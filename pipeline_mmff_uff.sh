#!/bin/bash

# Функция для вывода справки
usage() {
    echo "════════════════════════════════════════════════════"
    echo " Справка по использованию скрипта                   "
    echo "════════════════════════════════════════════════════"
    echo " Использование: $0 -p PDB -w WORK_PATH -d DOCKING_PATH -l DF_LIG -o OMP_NUM_THREADS      "
    echo "                                                                                         "
    echo " Обязательные параметры:                                                                 "
    echo "   -p PDB              Код PDB (1ere/1tqn/4k71/4wnv)                                     "
    echo "   -w WORK_PATH        Рабочая директория (например, vs)                                 "
    echo "   -d DOCKING_PATH     Директория для докинга (например, energy_VS)                      "
    echo "   -l DF_LIG           Имя CSV-файла с лигандами (ERalpha_ki_df/ERalpha_ic50_df/CYP3a4_ki_df/CYP3a4_ic50_df/AR_ki_df/AR_ic50_df/CYP2d6_ki_df/CYP2d6_ic50_df)         "
    echo "   -o OMP_NUM_THREADS  Количество потоков для OpenMP (например, 32,32)                   "
    echo "   -h                  Показать эту справку                                              "
    echo "                                                                                         "
    echo " Все входные файлы должны быть в папке vs/data:                                          "
    echo "   - ${PDB}_ligand.pdb                                                                 "
    echo "   - ${DF_LIG}.csv                                                                     "
    echo "   - ${PDB}_meeko.pdb                                                                  "
    echo " Примеры использования:                                                                "
    echo "bash pipeline.sh -p 1tqn -w vs -d energy_VS -l CYP2d6_ki_df -o 32,32 "
    echo "sbatch   --cpus-per-task=10   -p aichem  --gres=gpu:1   --mem=50G   --time=16:00:00   --error=dock.err pipeline.sh -p 1tqn -w vs -d energy_VS -l CYP2d6_ki_df -o 32,32 "


    echo "════════════════════════════════════════════════════"
    exit 1
}

# Обработка аргументов командной строки
while getopts "p:w:d:l:o:h" opt; do
    case $opt in
        p) PDB="$OPTARG" ;;
        w) WORK_PATH="$OPTARG" ;;
        d) DOCKING_PATH="$OPTARG" ;;
        l) DF_LIG="$OPTARG" ;;
        o) OMP_NUM_THREADS="$OPTARG" ;;
        h) usage ;;
        \?) echo "Ошибка: Неверный флаг -$OPTARG"; usage ;;
    esac
done

# Проверка, что все обязательные параметры заданы
if [ -z "$PDB" ] || [ -z "$WORK_PATH" ] || [ -z "$DOCKING_PATH" ] || [ -z "$DF_LIG" ] || [ -z "$OMP_NUM_THREADS" ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не указаны все обязательные параметры      "
    echo "════════════════════════════════════════════════════"
    usage
fi

# Создание необходимых директорий
#mkdir -p ${WORK_PATH}/data ${WORK_PATH}/raw ${WORK_PATH}/raw/pocket ${WORK_PATH}/ligand ${WORK_PATH}/pocket ${WORK_PATH}/uff ${WORK_PATH}/infer

echo "════════════════════════════════════════════════════"
echo " Параметры пайплайна                               "
echo "════════════════════════════════════════════════════"
echo " PDB: $PDB                                                                               "
echo " WORK_PATH: $WORK_PATH                                                                   "
echo " DOCKING_PATH: $DOCKING_PATH                                                             "
echo " DF_LIG: $DF_LIG                                                                         "
echo " OMP_NUM_THREADS: $OMP_NUM_THREADS                                                       "
echo "════════════════════════════════════════════════════"

echo "════════════════════════════════════════════════════"
echo " Шаг 1: Копирование входных данных                 "
echo "════════════════════════════════════════════════════"
echo " Копируем файлы из ${WORK_PATH}/data в рабочие директории                                "
echo "════════════════════════════════════════════════════"

# Проверка и копирование входных файлов
for file in "${WORK_PATH}/data/${PDB}_ligand.pdb" "${WORK_PATH}/data/${DF_LIG}.csv" "${WORK_PATH}/data/${PDB}_meeko.pdb"; do
    if [ ! -f "$file" ]; then
        echo "════════════════════════════════════════════════════"
        echo " Ошибка: Файл $file не найден                     "
        echo "════════════════════════════════════════════════════"
        exit 1
    fi
done

cp ${WORK_PATH}/data/${PDB}_ligand.pdb ${WORK_PATH}/raw/${PDB}_ligand.pdb
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось скопировать ${PDB}_ligand.pdb "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

cp ${WORK_PATH}/data/${DF_LIG}.csv ${WORK_PATH}/raw/${DF_LIG}.csv
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось скопировать ${DF_LIG}.csv     "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

cp ${WORK_PATH}/data/${PDB}_meeko.pdb ${WORK_PATH}/raw/pocket/${PDB}_meeko.pdb
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось скопировать ${PDB}_meeko.pdb  "
    echo "════════════════════════════════════════════════════"
    exit 1
fi
echo "════════════════════════════════════════════════════"
echo "════════════════════════════════════════════════════"
echo " Шаг 2: Проверка структуры папок и файлов          "
echo "════════════════════════════════════════════════════"
echo " Убедитесь, что у вас правильная структура папок и файлов в них.                           "
echo " Названия должны быть строго как в примерах далее.                                        "
echo "════════════════════════════════════════════════════"

echo "════════════════════════════════════════════════════"
echo " Шаг 3: Подготовка лиганда в PyMOL                 "
echo "════════════════════════════════════════════════════"
echo " Откройте оригинальный файл белка в PyMOL.                                                "
echo " Выделите нужный лиганд и сохраните его в формате pdb в папку ${WORK_PATH}/data/${PDB}_ligand.pdb "
echo "════════════════════════════════════════════════════"

# Проверка входного файла лиганда
if [ ! -f "${WORK_PATH}/raw/${PDB}_ligand.pdb" ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Файл ${WORK_PATH}/raw/${PDB}_ligand.pdb не найден "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo " Шаг 4: Подготовка белка в PyMOL                   "
echo "════════════════════════════════════════════════════"
echo " Откройте в этом же рабочем окне PyMOL файл белка meeko и проверьте, что всё нормально.   "
echo " Сохраните этот файл без лиганда в папку ${WORK_PATH}/data/${PDB}_meeko.pdb              "
echo "════════════════════════════════════════════════════"

# Проверка входного файла белка
if [ ! -f "${WORK_PATH}/raw/pocket/${PDB}_meeko.pdb" ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Файл ${WORK_PATH}/raw/pocket/${PDB}_meeko.pdb не найден "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo " Шаг 5: Протонирование лиганда                     "
echo "════════════════════════════════════════════════════"
echo " Теперь протонируем и сохраняем в нужном формате лиганд                                    "
echo "════════════════════════════════════════════════════"

obabel ${WORK_PATH}/raw/${PDB}_ligand.pdb -p 7.4 -O ${WORK_PATH}/ligand/${PDB}_docked.sdf
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить obabel               "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

python tools/extract_pocket_by_ligand.py ${WORK_PATH}/raw/pocket/ ${WORK_PATH}/ligand/ 0
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить extract_pocket_by_ligand.py "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

mv ${WORK_PATH}/raw/pocket/output/${PDB}_pocket.pdb ${WORK_PATH}/pocket
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось переместить ${PDB}_pocket.pdb "
    echo "════════════════════════════════════════════════════"
    exit 1
fi
echo "════════════════════════════════════════════════════"

echo "════════════════════════════════════════════════════"
echo " Шаг 6: Проверка структуры рабочей папки           "
echo "════════════════════════════════════════════════════"
echo " Сейчас структура рабочей папки должна быть такая                                         "
echo " (+ папка raw, которая далее используется только для CSV):                                "
echo "                                                                                         "
echo " ${WORK_PATH}/                                                                           "
echo " ├── ligand                                                                              "
echo " │   └── ${PDB}_docked.sdf (Bound ligand conformation inside the binding pocket, from RCSB) "
echo " ├── pocket                                                                              "
echo " │   └── ${PDB}_pocket.pdb (Prepared protein structure, entire protein or pocket)         "
echo " └── uff                                                                                 "
echo "     └── ${PDB}_uff.sdf (Force field minimized SDF file with all ligands to dock)        "
echo "════════════════════════════════════════════════════"

echo "════════════════════════════════════════════════════"
echo " Шаг 7: Подготовка лигандов                        "
echo "════════════════════════════════════════════════════"
echo " Запускаем скрипт для подготовки лигандов                                                 "
echo "════════════════════════════════════════════════════"

# Проверка CSV-файла с лигандами
if [ ! -f "${WORK_PATH}/raw/${DF_LIG}.csv" ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Файл ${WORK_PATH}/raw/${DF_LIG}.csv не найден "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

python  prepare_ligands_mmff_uff.py ${WORK_PATH}/raw/${DF_LIG}.csv ${WORK_PATH}/uff/${PDB}_uff.sdf
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить prepare_ligands_mmff.py   "
    echo "════════════════════════════════════════════════════"
    exit 1
fi
echo "════════════════════════════════════════════════════"

echo "════════════════════════════════════════════════════"
echo " Шаг 8: Удаление файлов WSL                        "
echo "════════════════════════════════════════════════════"
echo " Удаляем файлы WSL, которые почему-то мешаются                                            "
echo "════════════════════════════════════════════════════"

cd checkpoints
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Директория checkpoints не найдена         "
    echo "════════════════════════════════════════════════════"
    exit 1
fi
find . -type f -name "*:Zone.Identifier" -delete
cd ..

echo "════════════════════════════════════════════════════"
echo " Шаг 9: Запуск молекулярного докинга               "
echo "════════════════════════════════════════════════════"
echo " Запускаем докинг                                                                         "
echo "════════════════════════════════════════════════════"

# 1. Create query csv
python tools/inference/inter_sdf2csv.py ${WORK_PATH}/uff/${PDB}_uff.sdf 1
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить inter_sdf2csv.py (UFF) "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

# 2. Predict energy files
PYTHONPATH=interformer/ python inference.py -test_csv ${WORK_PATH}/uff/${PDB}_uff_infer.csv \
-work_path ${WORK_PATH}/ \
-ensemble checkpoints/v0.2_energy_model \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder ${DOCKING_PATH} \
-uff_as_ligand \
-debug \
-reload
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить inference.py (energy) "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

# 3. Docking
# [VS mode] need to refresh the uff ligand file
cp ${WORK_PATH}/uff/${PDB}_uff.sdf ${DOCKING_PATH}/uff/
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось скопировать ${PDB}_uff.sdf     "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

OMP_NUM_THREADS="${OMP_NUM_THREADS}" python docking/reconstruct_ligands.py -y --cwd ${DOCKING_PATH} --find_all --uff_folder uff find
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить reconstruct_ligands.py "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

# 4. Scoring
mkdir -p ${WORK_PATH}/infer && cp -r ${DOCKING_PATH}/ligand_reconstructing/* ${WORK_PATH}/infer/
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось скопировать файлы докинга      "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

python tools/inference/inter_sdf2csv.py ${WORK_PATH}/infer/${PDB}_docked.sdf 0
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить inter_sdf2csv.py (docked) "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

PYTHONPATH=interformer/ python inference.py -test_csv ${WORK_PATH}/infer/${PDB}_docked_infer.csv  \
-work_path ${WORK_PATH}/ \
-ligand_folder infer/ \
-ensemble checkpoints/v0.2_affinity_model/model* \
-use_ff_ligands '' \
-vs \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* \
--pose_sel True
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить inference.py (affinity) "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

# 5. Review result
cat result/${PDB}_docked_infer_ensemble.csv
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Файл result/${PDB}_docked_infer_ensemble.csv не найден "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo " Шаг 10: Анализ результатов докинга                "
echo "════════════════════════════════════════════════════"
echo " Докинг прошёл успешно! Наверное. Сейчас узнаем. Запускаем анализ...                      "
echo "════════════════════════════════════════════════════"

python analyze.py \
    --results-csv result/${PDB}_docked_infer_ensemble.csv \
    --original-csv ${WORK_PATH}/raw/${DF_LIG}.csv \
    --experimental-col pValue \
    --output-folder result
if [ $? -ne 0 ]; then
    echo "════════════════════════════════════════════════════"
    echo " Ошибка: Не удалось выполнить analyze.py           "
    echo "════════════════════════════════════════════════════"
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo " Завершение пайплайна                              "
echo "════════════════════════════════════════════════════"
echo " Пайплайн успешно выполнен! Результаты анализа сохранены в папке result                   "
echo "════════════════════════════════════════════════════"
echo "Очищаем папку vs/raw..."
rm vs/raw/${PDB}_ligand.pdb
rm vs/raw/${DF_LIG}.csv
rm vs/raw/pocket/${PDB}_meeko.pdb
