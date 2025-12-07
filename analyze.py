import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import argparse

def generate_analysis_for_dataset(df, df_name, output_dir, experimental_col_name):
    """
    Выполняет полный анализ для набора данных: рассчитывает статистики, 
    строит графики и возвращает результаты и пути к файлам.
    """
    print(f"\n--- Анализ для набора: '{df_name}' ({len(df)} точек) ---")

    if df.empty or len(df) < 2:
        print("Недостаточно данных для анализа.")
        return None, None, None

    # 1. Расчет статистик
    exp_vals = df['experimental_value']
    pred_vals = df['pred_pIC50']
    
    stats = {
        'count': len(df),
        'pearson_r': pearsonr(exp_vals, pred_vals)[0],
        'spearman_rho': spearmanr(exp_vals, pred_vals)[0],
        'rmse': np.sqrt(mean_squared_error(exp_vals, pred_vals)),
        'mae': mean_absolute_error(exp_vals, pred_vals)
    }

    # Сохранение статистики в текстовый файл
    stats_path = os.path.join(output_dir, f"{df_name}_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Статистика для '{df_name}':\n" + "="*40 + "\n" +
                f"Кол-во точек: {stats['count']}\n" +
                f"Pearson r: {stats['pearson_r']:.4f}\n" +
                f"Spearman ρ: {stats['spearman_rho']:.4f}\n" +
                f"RMSE: {stats['rmse']:.4f}\n" +
                f"MAE: {stats['mae']:.4f}\n")
    print(f"Статистики сохранены: {stats_path}")

    # 2. Построение графика корреляции
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.regplot(x='experimental_value', y='pred_pIC50', data=df, ax=ax_corr,
                scatter_kws={'alpha': 0.7, 'edgecolor': 'w'}, line_kws={'color': 'red'})
    ax_corr.set_title(f'Корреляция ({df_name})', fontsize=16)
    ax_corr.set_xlabel(f'Эксперимент ({experimental_col_name})', fontsize=12)
    ax_corr.set_ylabel('Предсказание (pIC50)', fontsize=12)
    min_val, max_val = min(exp_vals.min(), pred_vals.min()) - 0.5, max(exp_vals.max(), pred_vals.max()) + 0.5
    ax_corr.plot([min_val, max_val], [min_val, max_val], color='navy', linestyle='--', label='y=x')
    ax_corr.legend(loc='upper left')
    fig_corr.tight_layout(pad=1.5)
    corr_plot_path = os.path.join(output_dir, f"{df_name}_correlation.png")
    fig_corr.savefig(corr_plot_path, dpi=150)
    plt.close(fig_corr)
    print(f"График корреляции сохранен: {corr_plot_path}")

    # 3. Построение графика остатков
    residuals = exp_vals - pred_vals
    fig_res, ax_res = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=pred_vals, y=residuals, ax=ax_res, alpha=0.7, edgecolor='w')
    ax_res.axhline(y=0, color='red', linestyle='--')
    ax_res.set_title(f'График остатков ({df_name})', fontsize=16)
    ax_res.set_xlabel('Предсказанное значение', fontsize=12)
    ax_res.set_ylabel('Остатки (Эксперимент - Предсказание)', fontsize=12)
    fig_res.tight_layout(pad=1.5)
    res_plot_path = os.path.join(output_dir, f"{df_name}_residuals.png")
    fig_res.savefig(res_plot_path, dpi=150)
    plt.close(fig_res)
    print(f"График остатков сохранен: {res_plot_path}")

    return stats, corr_plot_path, res_plot_path

def create_summary_dashboard(stats_all, paths_all, stats_filtered, paths_filtered, output_dir):
    """
    Создает единую картинку-дэшборд со всеми графиками и статистиками.
    """
    fig, axes = plt.subplots(2, 3, figsize=(28, 16), 
                             gridspec_kw={'width_ratios': [3.5, 3.5, 2.5]})
    analysis_title = os.path.basename(output_dir)
    fig.suptitle(f'Сводный отчет по анализу: {analysis_title}', fontsize=24, y=0.98)

    # Функция для отрисовки одной строки дэшборда
    def draw_row(row_idx, title_prefix, stats, paths):
        corr_path, res_path = paths[0], paths[1]
        axes[row_idx, 0].imshow(mpimg.imread(corr_path))
        axes[row_idx, 0].set_title(f'{title_prefix}: Корреляция', fontsize=18)
        axes[row_idx, 1].imshow(mpimg.imread(res_path))
        axes[row_idx, 1].set_title(f'{title_prefix}: Остатки', fontsize=18)
        stats_text = (f"Количество точек: {stats['count']}\n\n"
                      f"Pearson r: {stats['pearson_r']:.3f}\n"
                      f"Spearman ρ: {stats['spearman_rho']:.3f}\n\n"
                      f"RMSE: {stats['rmse']:.3f}\n"
                      f"MAE: {stats['mae']:.3f}")
        axes[row_idx, 2].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=18, 
                              bbox=dict(boxstyle='round,pad=1', fc='aliceblue', ec='grey'))
        axes[row_idx, 2].set_title(f'{title_prefix}: Статистики', fontsize=18)
    
    # --- Ряд 1: Все данные ---
    draw_row(0, 'Все данные', stats_all, paths_all)

    # --- Ряд 2: Отфильтрованные данные ---
    if stats_filtered:
        draw_row(1, 'Без нулей', stats_filtered, paths_filtered)
    else:
        for i in range(3):
            axes[1, i].text(0.5, 0.5, 'Нет данных для анализа', ha='center', va='center', fontsize=14)
            axes[1, i].set_title('Без нулей', fontsize=18)
    
    for ax in axes.flatten(): ax.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    summary_path = os.path.join(output_dir, "summary_dashboard.png")
    fig.savefig(summary_path, dpi=200, bbox_inches='tight')
    print(f"\nСводный дэшборд сохранен: {summary_path}")
    plt.close(fig)

def run_full_analysis_workflow(selection_col, selection_name, df_results, df_original_exp, experimental_col_name, output_folder):
    """
    Запускает полный цикл анализа для заданного метода отбора поз.
    """
    print(f"\n{'='*20} НАЧАЛО АНАЛИЗА (отбор по: {selection_col}) {'='*20}")

    # Создание папки для результатов
    base_name = os.path.splitext(os.path.basename(df_original_exp_path))[0]
    output_dir = os.path.join(output_folder, f'analysis_{base_name}_{selection_name}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Все результаты для этого анализа будут сохранены в: {output_dir}")

    # Выбор лучшей позы и соединение данных
    best_poses_idx = df_results.groupby('Molecule ID')[selection_col].idxmax()
    df_best_poses = df_results.loc[best_poses_idx]
    df_merged = pd.merge(df_best_poses, df_original_exp, on='Molecule ID', how='inner')

    # Анализ №1: все данные
    stats_all, corr_path_all, res_path_all = generate_analysis_for_dataset(
        df_merged, 'all_data', output_dir, experimental_col_name
    )

    # Анализ №2: данные без нулей
    df_filtered = df_merged[df_merged['experimental_value'] != 0].copy()
    stats_filtered, corr_path_filtered, res_path_filtered = generate_analysis_for_dataset(
        df_filtered, 'filtered_data_no_zeros', output_dir, experimental_col_name
    )

    # Создание итогового дэшборда
    if stats_all:
        create_summary_dashboard(
            stats_all, (corr_path_all, res_path_all),
            stats_filtered, (corr_path_filtered, res_path_filtered),
            output_dir
        )
    print(f"\n{'='*20} АНАЛИЗ ЗАВЕРШЕН (отбор по: {selection_col}) {'='*20}")

def main():
    """Главная функция для запуска всех процессов анализа."""
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description="Анализ результатов докинга с генерацией статистики и графиков.")
    parser.add_argument('--results-csv', required=True, help="Путь к CSV-файлу с результатами докинга (например, result/4wnv_docked_infer_ensemble.csv)")
    parser.add_argument('--original-csv', required=True, help="Путь к CSV-файлу с экспериментальными данными (например, vs/raw/CYP2d6_ki_df_short.csv)")
    parser.add_argument('--experimental-col', required=True, help="Имя колонки с экспериментальными значениями (например, pValue)")
    parser.add_argument('--output-folder', required=True, help="Папка для сохранения результатов анализа (например, result)")
    
    args = parser.parse_args()

    print("--- Запуск комплексного анализа ---")

    # Проверка наличия файлов
    for f in [args.results_csv, args.original_csv]:
        if not os.path.exists(f): 
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл не найден: {f}")
            return
    
    # Загрузка и подготовка данных (выполняется один раз)
    df_results = pd.read_csv(args.results_csv)
    df_original = pd.read_csv(args.original_csv)

    if args.experimental_col not in df_original.columns:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: Колонка '{args.experimental_col}' не найдена.")
        return

    global df_original_exp_path
    df_original_exp_path = args.original_csv  # Для использования в run_full_analysis_workflow
    df_original_exp = df_original.copy()
    df_original_exp['Molecule ID'] = [f"ligand_source_row_{i+2}" for i in df_original_exp.index]
    df_original_exp = df_original_exp.rename(columns={args.experimental_col: 'experimental_value'})
    
    # ЗАПУСК ДВУХ АНАЛИЗОВ
    # Анализ 1: Отбор по лучшей предсказанной аффинности
    run_full_analysis_workflow('pred_pIC50', 'by_affinity', df_results, df_original_exp, args.experimental_col, args.output_folder)
    
    # Анализ 2: Отбор по лучшей уверенности в позе
    run_full_analysis_workflow('pred_pose', 'by_pose_score', df_results, df_original_exp, args.experimental_col, args.output_folder)
    
    print("\n--- Все анализы завершены. ---")

if __name__ == '__main__':
    main()