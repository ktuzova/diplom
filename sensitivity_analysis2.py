"""
Анализ чувствительности композитного показателя к выбору весов.

Четыре набора весов × 5 методов × 3 размера выборки (n = 150, 300, 700).
Строит:
  - тепловую карту рангов (строки = наборы весов, столбцы = методы)
  - тепловую карту composite scores
  - таблицу средних рангов (CSV)

ВКР — Тузова К. К.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ
# ════════════════════════════════════════════════════════════════

TARGET_NS = [150, 300, 700]

METRIC_KEYS = [
    'rel_error_mean_score',
    'rel_error_mean_mark',
    'ks_stat',
    'mmd',
    'cramers_v',
    'max_mark_dev',
]

# Четыре набора весов (сумма = 1.0)
WEIGHT_SETS = {
    'W1: Равные (1/6)': {
        'rel_error_mean_score': 1/6,
        'rel_error_mean_mark':  1/6,
        'ks_stat':              1/6,
        'mmd':                  1/6,
        'cramers_v':            1/6,
        'max_mark_dev':         1/6,
    },
    'W2: Акцент на распределении': {
        'rel_error_mean_score': 0.10,
        'rel_error_mean_mark':  0.10,
        'ks_stat':              0.30,
        'mmd':                  0.30,
        'cramers_v':            0.10,
        'max_mark_dev':         0.10,
    },
    'W3: Акцент на средних': {
        'rel_error_mean_score': 0.35,
        'rel_error_mean_mark':  0.35,
        'ks_stat':              0.10,
        'mmd':                  0.10,
        'cramers_v':            0.05,
        'max_mark_dev':         0.05,
    },
    'W4: Текущие веса': {
        'rel_error_mean_score': 0.25,
        'rel_error_mean_mark':  0.20,
        'ks_stat':              0.20,
        'mmd':                  0.15,
        'cramers_v':            0.10,
        'max_mark_dev':         0.10,
    },
}

# Порядок методов для отображения (лучший → худший по текущим весам)
METHOD_ORDER = [
    '4. Facility location',
    '5. Kernel herding',
    '2. Стратифицированная',
    '1. SRS',
    '3. k-center greedy',
]

METHOD_SHORT = {
    '1. SRS':               'SRS',
    '2. Стратифицированная': 'Stratified',
    '3. k-center greedy':   'k-center',
    '4. Facility location': 'Facility\nLocation',
    '5. Kernel herding':    'Kernel\nHerding',
}

COLORS = {
    '4. Facility location': '#2e8b57',
    '5. Kernel herding':    '#d4802a',
    '2. Стратифицированная': '#3266ad',
    '1. SRS':               '#73726c',
    '3. k-center greedy':   '#9370DB',
}



# 1. ЗАГРУЗКА И ФИЛЬТРАЦИЯ


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df_filtered = df[df['n'].isin(TARGET_NS)].copy()
    if df_filtered.empty:
        raise ValueError(
            f"В файле нет строк с n ∈ {TARGET_NS}. "
            f"Доступные n: {sorted(df['n'].unique().tolist())}"
        )
    found_ns = sorted(df_filtered['n'].unique().tolist())
    print(f"  Загружено: {len(df_filtered)} строк, n = {found_ns}")
    return df_filtered



# 2. ПЕРЕСЧЁТ COMPOSITE SCORE С НОВЫМИ ВЕСАМИ


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой пары (n, метод) вычисляет composite score
    по каждому из 4 наборов весов.

    Нормализация: метрика / srs_norm, где srs_norm — среднее SRS при данном n.
    Это точно воспроизводит логику compute_composite_score() из sampling_system.py.
    """
    rows = []

    for n_val in TARGET_NS:
        sub = df[df['n'] == n_val]
        if sub.empty:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: нет данных для n={n_val}, пропускаю.")
            continue

        # Нормализовочный базис — строка SRS
        srs_row = sub[sub['Метод'] == '1. SRS']
        if srs_row.empty:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: нет строки SRS для n={n_val}, пропускаю.")
            continue

        srs_norm = {
            mk: float(srs_row[f'{mk}_mean'].values[0])
            for mk in METRIC_KEYS
        }

        for _, row in sub.iterrows():
            method = row['Метод']
            # Нормализованные сырые метрики
            norm_vals = {
                mk: row[f'{mk}_mean'] / srs_norm[mk]
                if srs_norm[mk] > 0 else 0.0
                for mk in METRIC_KEYS
            }

            result = {'n': n_val, 'Метод': method}
            for ws_name, weights in WEIGHT_SETS.items():
                score = sum(weights[mk] * norm_vals[mk] for mk in METRIC_KEYS)
                result[ws_name] = round(score, 6)

            rows.append(result)

    return pd.DataFrame(rows)



# 3. РАНЖИРОВАНИЕ


def compute_ranks(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой пары (n, набор_весов) ранжирует методы по composite score
    (1 = лучший = наименьший score).
    Возвращает датафрейм тех же размеров, но со значениями-рангами.
    """
    rank_rows = []
    ws_names = list(WEIGHT_SETS.keys())

    for n_val in TARGET_NS:
        sub = scores_df[scores_df['n'] == n_val]
        if sub.empty:
            continue
        row_base = {'n': n_val}
        for ws in ws_names:
            # Сортируем методы по score, присваиваем ранги
            sorted_methods = sub.sort_values(ws)['Метод'].tolist()
            for rank, method in enumerate(sorted_methods, start=1):
                row_base_copy = dict(row_base)
                row_base_copy['Метод'] = method
                row_base_copy[ws] = rank
                # Объединяем позже
        # Делаем аккуратнее: строим по методам
        for _, row in sub.iterrows():
            method = row['Метод']
            rank_row = {'n': n_val, 'Метод': method}
            for ws in ws_names:
                sorted_methods = sub.sort_values(ws)['Метод'].tolist()
                rank_row[ws] = sorted_methods.index(method) + 1
            rank_rows.append(rank_row)

    return pd.DataFrame(rank_rows)


def compute_avg_ranks(ranks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Усредняет ранги по всем n для каждой пары (метод, набор_весов).
    Возвращает DataFrame: строки = наборы весов, столбцы = методы.
    """
    ws_names = list(WEIGHT_SETS.keys())
    result = {}
    for ws in ws_names:
        result[ws] = {}
        for method in METHOD_ORDER:
            sub = ranks_df[ranks_df['Метод'] == method][ws]
            result[ws][method] = round(float(sub.mean()), 3)
    return pd.DataFrame(result).T  # строки = наборы весов, столбцы = методы



# 4. ВИЗУАЛИЗАЦИЯ


def plot_score_heatmap(scores_df: pd.DataFrame, ranks_df: pd.DataFrame,
                       out_path: str):
    """
    Тепловые карты composite scores: вертикальная компоновка.
    Строки подграфиков = размеры выборки (n), столбцы = 1 (одна таблица на n).
    Ось X — методы, ось Y — наборы весов (с подписями слева).
    В каждой ячейке — крупный ранг (1–5) и мелко значение score.

    """
    ws_names = list(WEIGHT_SETS.keys())
    ws_short = [
        'W1: Равные (1/6)',
        'W2: Акцент на\nраспределении',
        'W3: Акцент на\nсредних',
        'W4: Текущие веса',
    ]
    n_vals = [n for n in TARGET_NS if n in scores_df['n'].values]

    methods_display = [METHOD_SHORT[m] for m in METHOD_ORDER]

    #  Размеры фигуры

    cell_w   = 2.0   # ширина
    cell_h   = 1.8   # высота
    n_rows   = len(n_vals)      # 3
    n_cols   = len(METHOD_ORDER)  # 5

    left_margin  = 3.2          # место слева под подписи схем весов
    right_margin = 1.2          # место справа под colorbar
    top_margin   = 2.0          # место сверху под общий заголовок

    fig_w = left_margin + n_cols * cell_w + right_margin
    fig_h = top_margin + n_rows * (len(ws_names) * cell_h + 0.3)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(fig_w, fig_h),
    )
    if n_rows == 1:
        axes = [axes]

    # Общий глобальный vmax по всем данным для единой шкалы
    all_scores = []
    for n_val in n_vals:
        sub_s = scores_df[scores_df['n'] == n_val]
        for method in METHOD_ORDER:
            s_row = sub_s[sub_s['Метод'] == method]
            for ws in ws_names:
                all_scores.append(float(s_row[ws].values[0]))
    global_vmax = min(float(np.percentile(all_scores, 85)), 1.5)
    global_vmin = 0.0

    im_last = None
    for row_idx, (ax, n_val) in enumerate(zip(axes, n_vals)):
        sub_s = scores_df[scores_df['n'] == n_val]
        sub_r = ranks_df[ranks_df['n'] == n_val]

        scores = np.zeros((len(ws_names), len(METHOD_ORDER)))
        ranks  = np.zeros((len(ws_names), len(METHOD_ORDER)), dtype=int)

        for j, method in enumerate(METHOD_ORDER):
            s_row = sub_s[sub_s['Метод'] == method]
            r_row = sub_r[sub_r['Метод'] == method]
            for i, ws in enumerate(ws_names):
                scores[i, j] = float(s_row[ws].values[0])
                ranks[i, j]  = int(r_row[ws].values[0])

        im = ax.imshow(scores, cmap='Blues',
                       vmin=global_vmin, vmax=global_vmax,
                       aspect='auto')
        im_last = im

        # Текст в ячейках
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                score_val = scores[i, j]
                rank_val  = ranks[i, j]
                norm_val  = min(score_val / global_vmax, 1.0) \
                            if global_vmax > 0 else 0.0
                text_color = 'white' if norm_val > 0.58 else '#0d2a4a'

                # Крупный ранг
                ax.text(j, i - 0.12, str(rank_val),
                        ha='center', va='center',
                        fontsize=27, fontweight='bold', color=text_color)
                # Мелкое значение score
                ax.text(j, i + 0.28, f'{score_val:.3f}',
                        ha='center', va='center',
                        fontsize=22, color=text_color, alpha=0.82)

        #  Подписи осей
        # X: названия методов — только на последнем подграфике
        if row_idx == n_rows - 1:
            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels(methods_display, fontsize=21, linespacing=1.3)
        else:
            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels([], fontsize=21)

        # Y: подписи схем весов
        ax.set_yticks(range(len(ws_names)))
        ax.set_yticklabels(ws_short, fontsize=21)
        ax.tick_params(axis='y', length=0, pad=10)

        # Заголовок подграфика (n = ...)
        ax.set_title(f'n = {n_val}', fontsize=21, fontweight='bold', pad=10)

        # Сетка между ячейками
        ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ws_names), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', bottom=False, left=False)

    #  Единый colorbar справа
    title_frac = top_margin / fig_h   # доля фигуры, занятая отступом сверху
    fig.subplots_adjust(
        left=left_margin / fig_w,
        right=1.0 - right_margin / fig_w,
        top=1.0 - title_frac,
        hspace=0.15,                   # минимальный зазор между таблицами
    )
    cbar_ax = fig.add_axes([
        1.0 - (right_margin - 0.3) / fig_w,  # x
        0.08,                                   # y
        0.018,                                  # width
        0.82 * (1.0 - title_frac),              # height — согласован с top
    ])
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label('Score (1.0 = SRS)', fontsize=16, labelpad=12)
    cbar.ax.tick_params(labelsize=14)

    # Общий заголовок — размещаем посередине зоны top_margin
    suptitle_y = 1.0 - title_frac * 0.42   # середина верхнего поля
    fig.suptitle(
        'Composite score по наборам весов\n'
        'Цифра = ранг метода (1 — лучший)     мелко = значение score'
        '     (1.0 = уровень SRS, ниже = лучше)',
        fontsize=21, y=suptitle_y, va='center',
    )

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path}')



# 5. КОНСОЛЬНЫЙ ВЫВОД И CSV


def print_summary(avg_ranks: pd.DataFrame, scores_df: pd.DataFrame):
    ws_names = list(WEIGHT_SETS.keys())
    print()
    print('=' * 75)
    print('СРЕДНИЕ РАНГИ (по n = 150, 300, 700)')
    print('=' * 75)
    header = f"  {'Набор весов':<35}" + ''.join(
        f"{METHOD_SHORT[m].replace(chr(10), ' '):>14}" for m in METHOD_ORDER)
    print(header)
    print('  ' + '─' * 73)
    for ws in ws_names:
        row_str = f"  {ws:<35}"
        for method in METHOD_ORDER:
            row_str += f"{avg_ranks.loc[ws, method]:>14.2f}"
        print(row_str)

    print()
    print('=' * 75)
    print('COMPOSITE SCORES ПО КАЖДОМУ n')
    print('=' * 75)
    for n_val in TARGET_NS:
        sub = scores_df[scores_df['n'] == n_val]
        if sub.empty:
            continue
        print(f"\n  n = {n_val}")
        header2 = f"  {'Набор весов':<35}" + ''.join(
            f"{METHOD_SHORT[m].replace(chr(10), ' '):>14}" for m in METHOD_ORDER)
        print(header2)
        print('  ' + '─' * 73)
        for ws in ws_names:
            row_str = f"  {ws:<35}"
            for method in METHOD_ORDER:
                val = float(sub[sub['Метод'] == method][ws].values[0])
                row_str += f"{val:>14.4f}"
            print(row_str)

    print()
    print('=' * 75)
    print('ВЫВОД: УСТОЙЧИВОСТЬ ИЕРАРХИИ')
    print('=' * 75)
    for method in METHOD_ORDER:
        ranks = [avg_ranks.loc[ws, method] for ws in ws_names]
        r_min, r_max = min(ranks), max(ranks)
        spread = r_max - r_min
        stable = 'стабильно' if spread <= 1.0 else '△ варьируется'
        name_short = METHOD_SHORT[method].replace('\n', ' ')
        print(f"  {name_short:<20}  ранги {r_min:.1f}–{r_max:.1f}  "
              f"(разброс {spread:.1f})  {stable}")


def save_csv(avg_ranks: pd.DataFrame, scores_df: pd.DataFrame,
             ranks_df: pd.DataFrame):
    out = avg_ranks[METHOD_ORDER].copy()
    out.index.name = 'Набор весов'
    out.to_csv('sensitivity_avg_ranks.csv', encoding='utf-8-sig')
    print('  -> sensitivity_avg_ranks.csv')

    scores_df.to_csv('sensitivity_scores.csv', index=False, encoding='utf-8-sig')
    print('  -> sensitivity_scores.csv')

    ranks_df.to_csv('sensitivity_ranks_full.csv', index=False,
                    encoding='utf-8-sig')
    print('  -> sensitivity_ranks_full.csv')



# ТОЧКА ВХОДА


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'batch_summary.csv'

    print('=' * 75)
    print('АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К ВЫБОРУ ВЕСОВ')
    print(f'  Файл: {csv_path}')
    print(f'  Размеры выборки: n = {TARGET_NS}')
    print('=' * 75)

    for ws_name, weights in WEIGHT_SETS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f'{ws_name}: сумма весов = {total}'
    print('  Веса: суммы = 1.0')

    df = load_data(csv_path)
    scores_df  = compute_scores(df)
    ranks_df   = compute_ranks(scores_df)
    avg_ranks  = compute_avg_ranks(ranks_df)

    print_summary(avg_ranks, scores_df)

    print()
    print('Сохранение графиков и CSV...')
    save_csv(avg_ranks, scores_df, ranks_df)
    plot_score_heatmap(scores_df, ranks_df, 'sensitivity_score_heatmap.png')

    print()
    print('Готово! Файлы:')
    print('  sensitivity_score_heatmap.png — тепловая карта scores + ранги')
    print('  sensitivity_avg_ranks.csv')
    print('  sensitivity_scores.csv')
    print('  sensitivity_ranks_full.csv')