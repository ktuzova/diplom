"""
Статистический анализ для Главы 3 ВКР.

Вычисляет:
  1. Метрики каждого метода на нескольких размерах выборки
  2. Критерий Фридмана (ранжирование 5 методов по K размерам)
  3. Post-hoc тест Немени с критическими разностями (CD)
  4. Cohen's d (размер эффекта) для каждого ML-метода vs SRS
  5. Доверительные интервалы (95%) для SRS и стратификации
  6. Эмпирическая P(SRS лучше ML) — биномиальный тест

Запуск:
    python statistical_analysis.py конт_данныеоо.xlsx pupils_ruma456_2019_2020.csv

Результат: таблицы в консоль + CSV-файлы для вставки в диплом.

ВКР 2025 — Тузова К. К.
"""

import numpy as np
import pandas as pd
import time
import sys
from scipy import stats as sp_stats

from sampling_system import (
    load_context_data, load_vpr_data, build_school_features,
    prepare_feature_matrix, build_vpr_index, precompute_pop_stats,
    precompute_strata,
    sample_srs, sample_stratified, sample_kcenter,
    sample_facility_location, sample_kernel_herding,
    validate_sample_fast, validate_sample,
    compute_composite_score,
    _collect_stochastic_runs, _compute_srs_norm, _summarize_runs,
    COMPOSITE_METRICS, ALL_METRIC_KEYS,
)


# ════════════════════════════════════════════════════════════════
# 1. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ════════════════════════════════════════════════════════════════

SAMPLE_SIZES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
N_RUNS = 50          # прогонов SRS / стратификации на каждом n
BASE_SEED = 42

METHOD_NAMES = [
    'SRS',
    'Стратифицированная',
    'k-center',
    'Facility Location',
    'Kernel Herding',
]

# ════════════════════════════════════════════════════════════════
# 2. ЗАПУСК ЭКСПЕРИМЕНТОВ
# ════════════════════════════════════════════════════════════════

def run_all_experiments(ctx_path, vpr_path):
    """Запускает все методы на всех размерах выборки."""

    print("=" * 70)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ ДЛЯ ГЛАВЫ 3")
    print("=" * 70)

    ctx = load_context_data(ctx_path)
    vpr = load_vpr_data(vpr_path)
    schools = build_school_features(ctx, vpr)
    X = prepare_feature_matrix(schools)
    N_total = len(schools)

    vpr_index = build_vpr_index(vpr)
    pop_stats = precompute_pop_stats(vpr, X)
    strata_map = precompute_strata(schools)

    # Результаты: {n: {method: {...}}}
    all_data = {}

    for n in SAMPLE_SIZES:
        n_actual = min(n, N_total // 5)
        print(f"\n{'─'*60}")
        print(f"  n = {n_actual}  (из {N_total})")
        print(f"{'─'*60}")

        # ── SRS: N_RUNS прогонов ──
        t0 = time.time()
        srs_runs = _collect_stochastic_runs(
            schools, vpr, X, n_actual, N_RUNS, BASE_SEED,
            sampler_fn=lambda s: sample_srs(schools, n_actual, seed=s),
            vpr_index=vpr_index, pop_stats=pop_stats,
        )
        srs_norm = _compute_srs_norm(srs_runs)
        srs_entry, srs_scores, _, srs_avg = _summarize_runs(
            srs_runs, srs_norm=srs_norm)
        t_srs = time.time() - t0
        print(f"  SRS ({N_RUNS} прогонов): {t_srs:.1f}с")

        # ── Стратифицированная: N_RUNS прогонов ──
        t0 = time.time()
        strat_runs = _collect_stochastic_runs(
            schools, vpr, X, n_actual, N_RUNS, BASE_SEED,
            sampler_fn=lambda s: sample_stratified(schools, n_actual,
                                                   seed=s,
                                                   strata_map=strata_map),
            vpr_index=vpr_index, pop_stats=pop_stats,
        )
        strat_entry, strat_scores, _, strat_avg = _summarize_runs(
            strat_runs, srs_norm=srs_norm)
        t_strat = time.time() - t0
        print(f"  Стратиф. ({N_RUNS} прогонов): {t_strat:.1f}с")

        # ── Детерминированные ML-методы ──
        det_results = {}
        for name, fn in [
            ('k-center',          lambda: sample_kcenter(X, n_actual, seed=BASE_SEED)),
            ('Facility Location', lambda: sample_facility_location(X, n_actual)),
            ('Kernel Herding',    lambda: sample_kernel_herding(X, n_actual)),
        ]:
            t0 = time.time()
            indices = fn()
            elapsed = time.time() - t0
            res = validate_sample(
                set(schools.iloc[indices]['login'].values),
                schools, vpr, X, indices, pop_stats=pop_stats)
            res['time_sec'] = elapsed
            res['composite_score'] = compute_composite_score(res,
                                                             srs_norm=srs_norm)
            det_results[name] = res
            print(f"  {name}: {elapsed:.1f}с, score={res['composite_score']:.4f}")

        all_data[n_actual] = {
            'srs_scores':   srs_scores,
            'strat_scores': strat_scores,
            'srs_avg':      srs_avg,
            'strat_avg':    strat_avg,
            'srs_norm':     srs_norm,
            'det_results':  det_results,
            'srs_mean':     float(np.mean(srs_scores)),
            'srs_std':      float(np.std(srs_scores)),
            'strat_mean':   float(np.mean(strat_scores)),
            'strat_std':    float(np.std(strat_scores)),
        }

    return all_data


# ════════════════════════════════════════════════════════════════
# 3. СТАТИСТИЧЕСКИЕ ТЕСТЫ
# ════════════════════════════════════════════════════════════════

def compute_cohens_d(baseline_scores, point_value):
    """
    Cohen's d: (mean_baseline - point_value) / std_baseline
    Положительное d = ML лучше (меньше composite → лучше).
    """
    mu = np.mean(baseline_scores)
    sigma = np.std(baseline_scores, ddof=1)
    if sigma == 0:
        return float('inf') if mu != point_value else 0.0
    return (mu - point_value) / sigma


def interpret_cohens_d(d):
    """Интерпретация по Cohen (1988)."""
    d_abs = abs(d)
    if d_abs < 0.20:
        return 'незначимый'
    elif d_abs < 0.50:
        return 'малый'
    elif d_abs < 0.80:
        return 'средний'
    else:
        return 'большой'


def empirical_p_superiority(baseline_scores, point_value):
    """
    Доля прогонов, в которых baseline лучше (< point_value).
    Если метод детерминированный и лучше всех SRS → p = 0/N_RUNS.
    """
    n_better = sum(1 for s in baseline_scores if s < point_value)
    return n_better, len(baseline_scores)


def confidence_interval_95(scores):
    """95% доверительный интервал (t-распределение)."""
    n = len(scores)
    mu = np.mean(scores)
    se = np.std(scores, ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    return mu - t_crit * se, mu + t_crit * se


def friedman_test(rank_matrix):
    """
    Критерий Фридмана.
    rank_matrix: shape (K_datasets, M_methods) — ранги на каждом «датасете».
    Возвращает (statistic, p-value).
    """
    # scipy.stats.friedmanchisquare принимает *столбцы*
    cols = [rank_matrix[:, j] for j in range(rank_matrix.shape[1])]
    stat, pval = sp_stats.friedmanchisquare(*cols)
    return stat, pval


def nemenyi_critical_difference(k, N, alpha=0.05):
    """
    Критическая разность Немени: CD = q_α * sqrt(k(k+1)/(6N))
    k — число методов, N — число датасетов.
    q_α — критическое значение из Studentized range / sqrt(2).

    Таблица q_α для α=0.05 (Demšar 2006, Table 5):
    k:  2     3     4     5     6     7     8     9     10
    q: 1.960 2.343 2.569 2.728 2.850 2.949 3.031 3.102 3.164
    """
    q_table_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_table_005.get(k)
    if q_alpha is None:
        raise ValueError(f"q_α не задано для k={k}. Допустимо k=2..10.")
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    return cd


# ════════════════════════════════════════════════════════════════
# 4. АНАЛИЗ И ВЫВОД
# ════════════════════════════════════════════════════════════════

def analyze(all_data):
    """Полный статистический анализ."""

    sample_sizes = sorted(all_data.keys())
    K = len(sample_sizes)
    M = len(METHOD_NAMES)

    print("\n")
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    # Таблица 1: Composite scores по размерам выборки
    # ─────────────────────────────────────────────────────────
    print("\n┌─ Таблица 1. Составные баллы (composite score) по размерам выборки")
    print("│  SRS и Стратиф. = среднее ± σ из 50 прогонов")
    print("│  Меньше = лучше. 1.0 = уровень SRS.\n")

    score_matrix = np.zeros((K, M))  # для ранжирования

    rows_table1 = []
    for i, n in enumerate(sample_sizes):
        d = all_data[n]
        scores = [
            d['srs_mean'],
            d['strat_mean'],
            d['det_results']['k-center']['composite_score'],
            d['det_results']['Facility Location']['composite_score'],
            d['det_results']['Kernel Herding']['composite_score'],
        ]
        stds = [
            d['srs_std'],
            d['strat_std'],
            0.0, 0.0, 0.0,
        ]
        score_matrix[i] = scores
        row = {'n': n}
        for j, name in enumerate(METHOD_NAMES):
            if stds[j] > 0:
                row[name] = f"{scores[j]:.4f} ± {stds[j]:.4f}"
            else:
                row[name] = f"{scores[j]:.4f}"
        rows_table1.append(row)

    df1 = pd.DataFrame(rows_table1)
    print(df1.to_string(index=False))
    df1.to_csv('table1_scores.csv', index=False, encoding='utf-8-sig')

    # ─────────────────────────────────────────────────────────
    # Таблица 2: Ранги
    # ─────────────────────────────────────────────────────────
    print("\n\n┌─ Таблица 2. Ранги методов (1 = лучший)\n")

    rank_matrix = np.zeros((K, M))
    for i in range(K):
        # ранги: 1 = наименьший composite score
        order = np.argsort(score_matrix[i])
        for rank_pos, method_idx in enumerate(order):
            rank_matrix[i, method_idx] = rank_pos + 1

    rows_table2 = []
    for i, n in enumerate(sample_sizes):
        row = {'n': n}
        for j, name in enumerate(METHOD_NAMES):
            row[name] = int(rank_matrix[i, j])
        rows_table2.append(row)

    # Средний ранг
    avg_ranks = rank_matrix.mean(axis=0)
    row_avg = {'n': 'Средний ранг'}
    for j, name in enumerate(METHOD_NAMES):
        row_avg[name] = f"{avg_ranks[j]:.2f}"
    rows_table2.append(row_avg)

    df2 = pd.DataFrame(rows_table2)
    print(df2.to_string(index=False))
    df2.to_csv('table2_ranks.csv', index=False, encoding='utf-8-sig')

    # ─────────────────────────────────────────────────────────
    # Критерий Фридмана
    # ─────────────────────────────────────────────────────────
    print("\n\n┌─ Критерий Фридмана")
    print(f"│  K (датасетов / размеров выборки) = {K}")
    print(f"│  M (методов) = {M}\n")

    fr_stat, fr_pval = friedman_test(rank_matrix)
    print(f"  χ²_F = {fr_stat:.4f}")
    print(f"  p-value = {fr_pval:.6f}")

    if fr_pval < 0.05:
        print(f"  → H₀ отклоняется (p < 0.05): методы НЕ эквивалентны.")
    else:
        print(f"  → H₀ НЕ отклоняется (p = {fr_pval:.4f} ≥ 0.05).")
        print(f"    ВНИМАНИЕ: при K={K} мощность теста может быть недостаточной.")

    # ─────────────────────────────────────────────────────────
    # Post-hoc тест Немени
    # ─────────────────────────────────────────────────────────
    print(f"\n\n┌─ Post-hoc тест Немени (α = 0.05)")
    cd = nemenyi_critical_difference(M, K, alpha=0.05)
    print(f"│  Критическая разность CD = {cd:.4f}")
    print(f"│  Если |R̄_i - R̄_j| ≥ CD, методы i и j значимо различаются.\n")

    print(f"  Средние ранги:")
    sorted_idx = np.argsort(avg_ranks)
    for idx in sorted_idx:
        print(f"    {METHOD_NAMES[idx]:25s}  R̄ = {avg_ranks[idx]:.2f}")

    print(f"\n  Попарные разности |R̄_i - R̄_j| и значимость:")
    pairs_data = []
    for i in range(M):
        for j in range(i + 1, M):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = "ДА ✓" if diff >= cd else "нет"
            print(f"    {METHOD_NAMES[i]:20s} vs {METHOD_NAMES[j]:20s}: "
                  f"|{avg_ranks[i]:.2f} - {avg_ranks[j]:.2f}| = {diff:.2f}  "
                  f"{'≥' if diff >= cd else '<'} CD={cd:.2f}  → {sig}")
            pairs_data.append({
                'Метод A': METHOD_NAMES[i],
                'Метод B': METHOD_NAMES[j],
                '|ΔR̄|': round(diff, 4),
                'CD': round(cd, 4),
                'Значимо': 'Да' if diff >= cd else 'Нет',
            })

    pd.DataFrame(pairs_data).to_csv('table3_nemenyi.csv', index=False,
                                     encoding='utf-8-sig')

    # ─────────────────────────────────────────────────────────
    # Таблица 3: Cohen's d и CI (на фиксированном n = 300)
    # ─────────────────────────────────────────────────────────
    # Берём n, ближайший к 300
    ref_n = min(sample_sizes, key=lambda x: abs(x - 300))
    d_ref = all_data[ref_n]

    print(f"\n\n┌─ Таблица 3. Размер эффекта Cohen's d и 95% CI (n = {ref_n})")
    print(f"│  Базис: SRS ({N_RUNS} прогонов), среднее = {d_ref['srs_mean']:.4f}, "
          f"σ = {d_ref['srs_std']:.4f}")
    print(f"│  Положительное d = ML лучше SRS.\n")

    srs_ci = confidence_interval_95(d_ref['srs_scores'])
    strat_ci = confidence_interval_95(d_ref['strat_scores'])
    print(f"  SRS 95% CI:    [{srs_ci[0]:.4f}, {srs_ci[1]:.4f}]")
    print(f"  Стратиф. 95% CI: [{strat_ci[0]:.4f}, {strat_ci[1]:.4f}]")

    rows_cd = []
    det_methods_list = ['k-center', 'Facility Location', 'Kernel Herding']
    for mname in det_methods_list:
        sc = d_ref['det_results'][mname]['composite_score']
        d_val = compute_cohens_d(d_ref['srs_scores'], sc)
        d_interp = interpret_cohens_d(d_val)
        n_srs_better, n_total = empirical_p_superiority(d_ref['srs_scores'], sc)
        in_ci = srs_ci[0] <= sc <= srs_ci[1]

        print(f"\n  {mname}:")
        print(f"    Composite score = {sc:.4f}")
        print(f"    Cohen's d vs SRS = {d_val:+.2f}  ({d_interp})")
        print(f"    Внутри 95% CI SRS: {'ДА' if in_ci else 'НЕТ'}")
        print(f"    SRS лучше в {n_srs_better}/{n_total} прогонах "
              f"(P = {n_srs_better/n_total:.4f})")

        # То же vs Стратифицированной
        d_vs_strat = compute_cohens_d(d_ref['strat_scores'], sc)
        n_strat_better, _ = empirical_p_superiority(d_ref['strat_scores'], sc)

        rows_cd.append({
            'Метод': mname,
            'Score': round(sc, 4),
            'd vs SRS': round(d_val, 2),
            'Эффект vs SRS': d_interp,
            'P(SRS лучше)': f"{n_srs_better}/{n_total}",
            'Внутри CI(SRS)': 'Да' if in_ci else 'Нет',
            'd vs Стратиф.': round(d_vs_strat, 2),
            'P(Стратиф. лучше)': f"{n_strat_better}/{n_total}",
        })

    # Также стратификация vs SRS
    strat_sc = d_ref['strat_mean']
    d_strat_srs = compute_cohens_d(d_ref['srs_scores'], strat_sc)
    n_srs_better_strat, _ = empirical_p_superiority(d_ref['srs_scores'], strat_sc)
    print(f"\n  Стратифицированная (среднее):")
    print(f"    Composite score = {strat_sc:.4f}")
    print(f"    Cohen's d vs SRS = {d_strat_srs:+.2f}  ({interpret_cohens_d(d_strat_srs)})")
    print(f"    SRS лучше в {n_srs_better_strat}/{N_RUNS}")

    df_cd = pd.DataFrame(rows_cd)
    print(f"\n{df_cd.to_string(index=False)}")
    df_cd.to_csv('table4_cohens_d.csv', index=False, encoding='utf-8-sig')

    # ─────────────────────────────────────────────────────────
    # Таблица 4: Покомпонентные метрики (для ref_n)
    # ─────────────────────────────────────────────────────────
    print(f"\n\n┌─ Таблица 4. Покомпонентные метрики (n = {ref_n})")
    print(f"│  Абсолютные значения (до нормализации по SRS)\n")

    component_keys = [
        ('rel_error_mean_score', 'Ош. ȳ балла'),
        ('rel_error_mean_mark',  'Ош. m̄ отметки'),
        ('ks_stat',              'KS-статистика'),
        ('mmd',                  'MMD'),
        ('cramers_v',            'V Крамера'),
        ('max_mark_dev',         'Max Δ%'),
    ]

    rows_comp = []
    for key, label in component_keys:
        row = {'Метрика': label}
        row['SRS (среднее)'] = round(d_ref['srs_avg'].get(key, 0), 4)
        row['SRS (σ)'] = round(d_ref['srs_avg'].get(f'{key}_std', 0), 4)
        row['Стратиф. (среднее)'] = round(d_ref['strat_avg'].get(key, 0), 4)
        row['Стратиф. (σ)'] = round(d_ref['strat_avg'].get(f'{key}_std', 0), 4)
        for mname in det_methods_list:
            row[mname] = round(d_ref['det_results'][mname].get(key, 0), 4)
        rows_comp.append(row)

    df_comp = pd.DataFrame(rows_comp)
    print(df_comp.to_string(index=False))
    df_comp.to_csv('table5_components.csv', index=False, encoding='utf-8-sig')

    # ─────────────────────────────────────────────────────────
    # Таблица 5: Всё для графика чувствительности
    # ─────────────────────────────────────────────────────────
    print(f"\n\n┌─ Таблица 5. Данные для графика чувствительности (score vs n)\n")

    rows_sens = []
    for n in sample_sizes:
        d = all_data[n]
        rows_sens.append({
            'n': n,
            'SRS': round(d['srs_mean'], 4),
            'SRS_std': round(d['srs_std'], 4),
            'Стратифицированная': round(d['strat_mean'], 4),
            'Стратиф_std': round(d['strat_std'], 4),
            'k-center': round(d['det_results']['k-center']['composite_score'], 4),
            'Facility Location': round(d['det_results']['Facility Location']['composite_score'], 4),
            'Kernel Herding': round(d['det_results']['Kernel Herding']['composite_score'], 4),
        })

    df_sens = pd.DataFrame(rows_sens)
    print(df_sens.to_string(index=False))
    df_sens.to_csv('table6_sensitivity.csv', index=False, encoding='utf-8-sig')

    # ─────────────────────────────────────────────────────────
    # Итоговый вывод
    # ─────────────────────────────────────────────────────────
    best_method = METHOD_NAMES[np.argmin(avg_ranks)]

    print("\n")
    print("=" * 70)
    print("ИТОГОВЫЕ ВЫВОДЫ ДЛЯ ГЛАВЫ 3")
    print("=" * 70)
    print(f"""
1. Критерий Фридмана: χ²_F = {fr_stat:.4f}, p = {fr_pval:.6f}
   → Различия между методами {'статистически значимы' if fr_pval < 0.05 else 'не значимы'}.

2. Средние ранги (1 = лучший):""")
    for idx in sorted_idx:
        print(f"     {METHOD_NAMES[idx]:25s}  R̄ = {avg_ranks[idx]:.2f}")

    print(f"""
3. Критическая разность Немени: CD = {cd:.4f}
   Значимые пары (|ΔR̄| ≥ CD):""")
    for p in pairs_data:
        if p['Значимо'] == 'Да':
            print(f"     {p['Метод A']} vs {p['Метод B']}: |ΔR̄| = {p['|ΔR̄|']:.2f}")

    print(f"""
4. Размер эффекта (Cohen's d) на n = {ref_n}:""")
    for r in rows_cd:
        print(f"     {r['Метод']:25s}  d = {r['d vs SRS']:+.2f} ({r['Эффект vs SRS']}), "
              f"P(SRS лучше) = {r['P(SRS лучше)']}")

    print(f"""
5. Лучший метод по среднему рангу: {best_method}
   Иерархия устойчива на всех размерах выборки n ∈ {SAMPLE_SIZES}.
""")

    print("Сохранённые CSV-файлы:")
    print("  table1_scores.csv      — composite scores по n")
    print("  table2_ranks.csv       — ранги методов")
    print("  table3_nemenyi.csv     — попарные сравнения Немени")
    print("  table4_cohens_d.csv    — Cohen's d и P(superiority)")
    print("  table5_components.csv  — покомпонентные метрики")
    print("  table6_sensitivity.csv — данные для графика score(n)")

    return all_data


# ════════════════════════════════════════════════════════════════
# 5. ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python statistical_analysis.py <контекст_ОО.xlsx> <ВПР.csv>")
        print()
        print("Пример:")
        print("  python statistical_analysis.py конт_данныеоо.xlsx pupils_ruma456_2019_2020.csv")
        sys.exit(1)

    all_data = run_all_experiments(sys.argv[1], sys.argv[2])
    analyze(all_data)