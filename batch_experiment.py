"""
Пакетный эксперимент: сравнение методов сэмплинга на разных размерах выборки.

Загружает данные один раз, прогоняет все методы для каждого n,
усредняет по нескольким seed'ам, выводит сводку и графики.

Запуск:
    python batch_experiment.py

Файлы данных (в той же папке):
    конт_данныеоо.xlsx, pupils_ruma456_2019_2020.csv

Результат:
    batch_summary.csv  — сводная таблица
    batch_composite.png — графики composite score vs n
    batch_metrics.png   — графики 6 метрик vs n
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

from sampling_system import (
    load_context_data, load_vpr_data, build_school_features,
    prepare_feature_matrix, sample_srs, sample_stratified,
    sample_kcenter, sample_facility_location, sample_kernel_herding,
    validate_sample, validate_sample_fast, compute_composite_score,
    build_vpr_index, precompute_pop_stats, precompute_strata,
    _collect_stochastic_runs, _compute_srs_norm,
    COMPOSITE_METRICS, ALL_METRIC_KEYS,
)

# ════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ (менять здесь)
# ════════════════════════════════════════════════════════════════
CTX_PATH = 'конт_данныеоо.xlsx'
VPR_PATH = 'pupils_ruma456_2019_2020.csv'

SAMPLE_SIZES = [100, 150, 200, 300, 500, 700, 1000]
SEEDS = [42, 123, 456, 789, 1024]
N_SRS_RUNS = 50  # прогонов SRS/стратиф. на каждый seed

# ════════════════════════════════════════════════════════════════

METHOD_NAMES = {
    'srs': '1. SRS',
    'strat': '2. Стратифицированная',
    'kcenter': '3. k-center greedy',
    'fl': '4. Facility location',
    'kh': '5. Kernel herding',
}

METRICS_FOR_TABLE = [
    'rel_error_mean_score', 'rel_error_mean_mark', 'ks_stat',
    'mmd', 'cramers_v', 'max_mark_dev', 'composite_score',
]

METRIC_LABELS = {
    'rel_error_mean_score': 'Ош. ȳ балла',
    'rel_error_mean_mark': 'Ош. m̄ отметки',
    'ks_stat': 'KS',
    'mmd': 'MMD',
    'cramers_v': 'Cramér V',
    'max_mark_dev': 'Max Δ%',
    'composite_score': 'Балл (норм.)',
}


def run_batch():
    # ── Загрузка (один раз) ──────────────────────────────────────────
    print("=" * 70)
    print("ПАКЕТНЫЙ ЭКСПЕРИМЕНТ")
    print("=" * 70)

    ctx = load_context_data(CTX_PATH)
    vpr = load_vpr_data(VPR_PATH)
    schools = build_school_features(ctx, vpr)
    X = prepare_feature_matrix(schools)
    N_total = len(schools)

    t0 = time.time()
    vpr_index = build_vpr_index(vpr)
    pop_stats = precompute_pop_stats(vpr, X)
    strata_map = precompute_strata(schools)
    print(f"  Предвычисления: {time.time() - t0:.1f}с")
    print(f"  Размеры выборок: {SAMPLE_SIZES}")
    print(f"  Seed'ы: {SEEDS}")
    print(f"  Прогонов SRS/стратиф.: {N_SRS_RUNS}")
    print()

    rows = []  # (n, method, seed, metric_key: value, ...)

    for n_target in SAMPLE_SIZES:
        n = min(n_target, N_total // 5)
        print(f"{'─' * 60}")
        print(f"  n = {n} (из {N_total})")
        print(f"{'─' * 60}")

        # ── Детерминированные: один раз на n ─────────────────────────
        det_raw = {}  # method_key → raw results dict
        det_indices = {}

        for key, fn in [
            ('kcenter', lambda: sample_kcenter(X, n, seed=0)),
            ('fl', lambda: sample_facility_location(X, n)),
            ('kh', lambda: sample_kernel_herding(X, n)),
        ]:
            t1 = time.time()
            indices = fn()
            elapsed = time.time() - t1
            res = validate_sample(
                set(schools.iloc[indices]['login'].values),
                schools, vpr, X, indices, pop_stats=pop_stats)
            res['time_sec'] = elapsed
            det_raw[key] = res
            det_indices[key] = indices
            print(f"    {METHOD_NAMES[key]}: {elapsed:.1f}с")

        # ── По каждому seed: SRS/стратиф. + нормализация ────────────
        for seed in SEEDS:
            # SRS прогоны → нормализация
            srs_runs = _collect_stochastic_runs(
                schools, vpr, X, n, N_SRS_RUNS, seed,
                sampler_fn=lambda s: sample_srs(schools, n, seed=s),
                vpr_index=vpr_index, pop_stats=pop_stats,
            )
            srs_norm = _compute_srs_norm(srs_runs)

            # SRS composite scores
            srs_scores = [compute_composite_score(r, srs_norm=srs_norm)
                          for r in srs_runs]
            srs_raw_avg = {}
            for mk in METRICS_FOR_TABLE[:-1]:  # всё кроме composite
                srs_raw_avg[mk] = float(np.mean([r.get(mk, 0)
                                                  for r in srs_runs]))
            srs_raw_avg['composite_score'] = float(np.mean(srs_scores))

            rows.append({
                'n': n, 'method': 'srs', 'seed': seed,
                **srs_raw_avg,
            })

            # Стратифицированная прогоны
            strat_runs = _collect_stochastic_runs(
                schools, vpr, X, n, N_SRS_RUNS, seed,
                sampler_fn=lambda s: sample_stratified(
                    schools, n, seed=s, strata_map=strata_map),
                vpr_index=vpr_index, pop_stats=pop_stats,
            )
            strat_scores = [compute_composite_score(r, srs_norm=srs_norm)
                            for r in strat_runs]
            strat_raw_avg = {}
            for mk in METRICS_FOR_TABLE[:-1]:
                strat_raw_avg[mk] = float(np.mean([r.get(mk, 0)
                                                    for r in strat_runs]))
            strat_raw_avg['composite_score'] = float(np.mean(strat_scores))

            rows.append({
                'n': n, 'method': 'strat', 'seed': seed,
                **strat_raw_avg,
            })

            # Детерминированные: нормализуем по этому seed'у
            for key in ['kcenter', 'fl', 'kh']:
                res = det_raw[key]
                cs = compute_composite_score(res, srs_norm=srs_norm)
                row = {'n': n, 'method': key, 'seed': seed}
                for mk in METRICS_FOR_TABLE[:-1]:
                    row[mk] = res.get(mk, 0)
                row['composite_score'] = cs
                rows.append(row)

        print(f"    ✓ {len(SEEDS)} seed'ов обработано")

    # ── Сводка ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    summary = df.groupby(['n', 'method']).agg(
        **{f'{mk}_mean': (mk, 'mean') for mk in METRICS_FOR_TABLE},
        **{f'{mk}_std': (mk, 'std') for mk in METRICS_FOR_TABLE},
    ).reset_index()

    # Красивые имена
    summary['Метод'] = summary['method'].map(METHOD_NAMES)
    summary = summary.sort_values(['n', 'composite_score_mean'])

    # ── Вывод в консоль ──────────────────────────────────────────────
    print()
    print("=" * 80)
    print("СВОДКА")
    print("=" * 80)
    for n_val in sorted(summary['n'].unique()):
        sub = summary[summary['n'] == n_val]
        print(f"\n  n = {n_val}")
        print(f"  {'Метод':<30} {'Ош.ȳ':>8} {'Ош.m̄':>8} {'KS':>8} "
              f"{'MMD':>8} {'CramV':>8} {'MaxΔ%':>8} {'Балл':>10}")
        print(f"  {'─' * 98}")
        for _, row in sub.iterrows():
            print(f"  {row['Метод']:<30} "
                  f"{row['rel_error_mean_score_mean']:8.4f} "
                  f"{row['rel_error_mean_mark_mean']:8.4f} "
                  f"{row['ks_stat_mean']:8.4f} "
                  f"{row['mmd_mean']:8.4f} "
                  f"{row['cramers_v_mean']:8.4f} "
                  f"{row['max_mark_dev_mean']:8.4f} "
                  f"{row['composite_score_mean']:8.4f} "
                  f"±{row['composite_score_std']:6.4f}")

    # ── CSV ──────────────────────────────────────────────────────────
    out_cols = ['n', 'Метод']
    for mk in METRICS_FOR_TABLE:
        out_cols += [f'{mk}_mean', f'{mk}_std']
    summary[out_cols].to_csv('batch_summary.csv', index=False)
    print(f"\n  Сохранено: batch_summary.csv")

    # ── Графики ──────────────────────────────────────────────────────
    _plot_composite(summary)
    _plot_metrics(summary)

    print("\n  Сохранено: batch_composite.png, batch_metrics.png")
    print("  Готово!")


def _plot_composite(summary):
    """Composite score vs n для каждого метода."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'srs': '#8b949e', 'strat': '#2ea4ff', 'kcenter': '#d2a8ff',
        'fl': '#7ee787', 'kh': '#ffa657',
    }
    markers = {
        'srs': 's', 'strat': 'D', 'kcenter': 'v', 'fl': '^', 'kh': 'o',
    }

    for method_key in ['kh', 'fl', 'strat', 'srs', 'kcenter']:
        sub = summary[summary['method'] == method_key].sort_values('n')
        ns = sub['n'].values
        means = sub['composite_score_mean'].values
        stds = sub['composite_score_std'].values

        ax.plot(ns, means, marker=markers[method_key], color=colors[method_key],
                label=METHOD_NAMES[method_key], linewidth=2, markersize=7)
        ax.fill_between(ns, means - stds, means + stds,
                        color=colors[method_key], alpha=0.15)

    ax.axhline(y=1.0, color='#8b949e', linestyle='--', alpha=0.5,
               label='SRS baseline (1.0)')
    ax.set_xlabel('Размер выборки (n)', fontsize=12)
    ax.set_ylabel('Составной балл (1.0 = SRS)', fontsize=12)
    ax.set_title('Сравнение методов сэмплинга по размеру выборки', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig('batch_composite.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_metrics(summary):
    """6 метрик vs n, каждая на отдельном подграфике."""
    metrics = METRICS_FOR_TABLE[:-1]  # без composite
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    colors = {
        'srs': '#8b949e', 'strat': '#2ea4ff', 'kcenter': '#d2a8ff',
        'fl': '#7ee787', 'kh': '#ffa657',
    }

    for idx, mk in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]

        for method_key in ['kh', 'fl', 'strat', 'srs', 'kcenter']:
            sub = summary[summary['method'] == method_key].sort_values('n')
            ns = sub['n'].values
            means = sub[f'{mk}_mean'].values

            ax.plot(ns, means, marker='o', color=colors[method_key],
                    label=METHOD_NAMES[method_key], linewidth=1.5,
                    markersize=5)

        ax.set_title(METRIC_LABELS[mk], fontsize=11)
        ax.set_xlabel('n', fontsize=9)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Метрики валидации по размеру выборки', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig('batch_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    t_start = time.time()
    run_batch()
    print(f"\n  Общее время: {(time.time() - t_start) / 60:.1f} мин")