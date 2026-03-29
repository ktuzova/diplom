"""
Система сэмплинга данных для формирования репрезентативной выборки
образовательных организаций (ОО) на основе данных ВПР.

Реализует 5 методов отбора и валидирует выборки по совпадению
распределений результатов ВПР с генеральной совокупностью.

ВКР, 2025 — Тузова Ксения Кирилловна
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, chisquare
from sklearn.preprocessing import StandardScaler
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================
# КОНСТАНТЫ
# ============================================================
KIM_MAX_SCORES = {
    (4, 1): 38, (5, 1): 45, (6, 1): 45,
    (4, 2): 20, (5, 2): 20, (6, 2): 20,
}
SUBJECT_NAMES = {1: 'РУ', 2: 'МА'}
VALID_MARKS = [2, 3, 4, 5]

# ============================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================

def load_context_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.iloc[0, 0] == df.columns[0] or df.iloc[0, 0] == 'Логин ОО':
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = ['login', 'location_type', 'locality_name',
                  'locality_size', 'is_correctional', 'n_students']
    df['region'] = df['login'].str[3:5]
    df['n_students'] = pd.to_numeric(df['n_students'], errors='coerce')
    df['is_correctional'] = df['is_correctional'].str.lower().map(
        {'да': 1, 'нет': 0}).fillna(0).astype(int)
    df['location_code'] = df['location_type'].str[0].astype(int)
    df['size_code'] = df['locality_size'].str[0].astype(int)
    df = df.dropna(subset=['n_students']).reset_index(drop=True)
    print(f"  Загружено ОО: {len(df)}, регионов: {df['region'].nunique()}")
    return df


def load_vpr_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = ['grade', 'subject_code', 'login', 'student_id',
                  'score', 'mark', 'year']
    for c in ['score', 'mark', 'grade', 'subject_code']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['score', 'mark']).reset_index(drop=True)
    n_before = len(df)
    df = df[df['mark'].isin(VALID_MARKS)].reset_index(drop=True)
    print(f"  ВПР: {n_before} → {len(df)} (удалено {n_before - len(df)} записей с mark=0)")
    print(f"  Классы: {sorted(df['grade'].unique().tolist())}, "
          f"Предметы: {sorted(df['subject_code'].unique().tolist())}")
    return df


def build_school_features(ctx, vpr):
    school_stats = vpr.groupby('login').agg(
        mean_score=('score', 'mean'),
        std_score=('score', 'std'),
        mean_mark=('mark', 'mean'),
        n_students_vpr=('student_id', 'nunique'),
        pct_mark_5=('mark', lambda x: (x == 5).mean()),
        pct_mark_4=('mark', lambda x: (x == 4).mean()),
        pct_mark_3=('mark', lambda x: (x == 3).mean()),
        pct_mark_2=('mark', lambda x: (x == 2).mean()),
    ).reset_index()
    school_stats['std_score'] = school_stats['std_score'].fillna(0)
    merged = ctx.merge(school_stats, on='login', how='inner')
    print(f"  ОО с контекстом и ВПР: {len(merged)}")
    return merged


def prepare_feature_matrix(schools):
    parts = []
    parts.append(pd.get_dummies(schools['region'], prefix='reg'))
    parts.append(pd.get_dummies(schools['location_code'], prefix='loc'))
    parts.append(pd.get_dummies(schools['size_code'], prefix='sz'))
    parts.append(pd.DataFrame({'is_correctional': schools['is_correctional'].values}))
    parts.append(pd.DataFrame({
        'log_n_students': np.log1p(schools['n_students'].values),
        'mean_score': schools['mean_score'].values,
        'std_score': schools['std_score'].values,
        'pct_mark_2': schools['pct_mark_2'].values,
        'pct_mark_3': schools['pct_mark_3'].values,
        'pct_mark_4': schools['pct_mark_4'].values,
        'pct_mark_5': schools['pct_mark_5'].values,
    }))
    features = pd.concat(parts, axis=1)
    X = StandardScaler().fit_transform(features.values.astype(np.float32))
    print(f"  Признаков: {X.shape[1]} (one-hot, float32)")
    return X


# ============================================================
# 1b. ПРЕДВЫЧИСЛЕНИЯ ДЛЯ БЫСТРОЙ ВАЛИДАЦИИ
# ============================================================

def build_vpr_index(vpr):
    """login → массив индексов строк ВПР. O(1) вместо isin() на 11М строк."""
    idx = {}
    for login, group in vpr.groupby('login'):
        idx[login] = group.index.values
    return idx


def precompute_pop_stats(vpr, X_all):
    """Статистики генеральной совокупности — вычисляются один раз."""
    rng = np.random.RandomState(0)

    pop_mark_counts = vpr['mark'].value_counts().sort_index()
    marks = sorted(pop_mark_counts.index)
    pop_mark_probs = np.array([pop_mark_counts.get(m, 0) for m in marks], dtype=np.float64)
    pop_mark_probs = pop_mark_probs / pop_mark_probs.sum()

    pop_mean_score = float(vpr['score'].mean())
    pop_mean_mark = float(vpr['mark'].mean())

    Xp = X_all[rng.choice(len(X_all), min(3000, len(X_all)), replace=False)]
    d_sub = cdist(Xp[:400], Xp[:400], 'sqeuclidean')
    sigma2 = float(np.median(d_sub[d_sub > 0]))
    kpp = float(np.exp(-cdist(Xp, Xp, 'sqeuclidean') / (2 * sigma2)).mean())

    slice_groups = {}
    for (g, s), grp in vpr.groupby(['grade', 'subject_code']):
        slice_groups[(g, s)] = grp

    return {
        'marks': marks,
        'pop_mark_probs': pop_mark_probs,
        'pop_mean_score': pop_mean_score,
        'pop_mean_mark': pop_mean_mark,
        'n_pop': len(vpr),
        'Xp_sub': Xp,
        'sigma2': sigma2,
        'kpp': kpp,
        'slice_groups': slice_groups,
    }


def precompute_strata(schools):
    """Предвычисление страта→индексы (один раз, не на каждый прогон)."""
    strata = schools['region'] + '_' + schools['location_code'].astype(str)
    strata_map = {}
    for s_name, group in strata.groupby(strata):
        strata_map[s_name] = group.index.values
    return strata_map


# ============================================================
# 2. МЕТОДЫ СЭМПЛИНГА
# ============================================================

def sample_srs(schools, n, seed=42):
    return np.random.RandomState(seed).choice(len(schools), size=n, replace=False)


def sample_stratified(schools, n, seed=42, strata_map=None):
    """
    Стратифицированная выборка: пропорциональное размещение
    по методу Хэра (наибольших остатков).
    strata_map — предвычисленный словарь {имя_страты: np.array индексов}.
    """
    rng = np.random.RandomState(seed)
    N_total = len(schools)

    if strata_map is None:
        strata_map = precompute_strata(schools)

    # Пропорциональное размещение (Hare largest-remainder)
    exact = {s: n * len(idx) / N_total for s, idx in strata_map.items()}
    alloc = {s: int(v) for s, v in exact.items()}
    rems = {s: exact[s] - alloc[s] for s in exact}
    deficit = n - sum(alloc.values())
    for s in sorted(rems, key=rems.get, reverse=True)[:deficit]:
        alloc[s] += 1

    selected = []
    for stratum, n_s in alloc.items():
        if n_s == 0:
            continue
        idx = strata_map[stratum]
        if len(idx) <= n_s:
            selected.extend(idx)
        else:
            selected.extend(rng.choice(idx, size=n_s, replace=False))

    return np.array(selected)


def sample_kcenter(X, n, seed=42):
    """
    Жадный k-center: последовательно добавляет наиболее удалённую
    от уже выбранных точку. 2-аппроксимация задачи k-center.
    """
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    first = rng.randint(N)
    selected = [first]
    min_dist = np.sum((X - X[first]) ** 2, axis=1)
    for step in range(1, n):
        farthest = int(np.argmax(min_dist))
        selected.append(farthest)
        dist_new = np.sum((X - X[farthest]) ** 2, axis=1)
        np.minimum(min_dist, dist_new, out=min_dist)
        if (step + 1) % 500 == 0:
            print(f"    k-center: {step + 1}/{n}")
    return np.array(selected)


def sample_facility_location(X, n):
    """Субмодулярная оптимизация facility location (жадный алгоритм)."""
    N = X.shape[0]
    if N <= 8000:
        return _fl_exact(X, n)
    n_cand = min(max(10 * n, 5000), N)
    cand_idx = np.random.RandomState(0).choice(N, size=n_cand, replace=False)
    print(f"    FL: {n_cand} кандидатов → отбор {n}...")
    return _fl_on_candidates(X, cand_idx, n)


def _fl_exact(X, n):
    N = X.shape[0]
    dists = cdist(X, X, metric='sqeuclidean')
    sigma2 = float(np.median(dists[dists > 0]))
    sim = np.exp(-dists / (2 * sigma2)).astype(np.float32)
    selected, mask, cov = [], np.zeros(N, dtype=bool), np.zeros(N, dtype=np.float32)
    for _ in range(n):
        gains = np.maximum(0, sim - cov[:, None]).sum(axis=0)
        gains[mask] = -1.0
        best = int(gains.argmax())
        selected.append(best)
        mask[best] = True
        cov = np.maximum(cov, sim[:, best])
    return np.array(selected)


def _fl_on_candidates(X_all, cand_idx, n):
    N = len(X_all)
    n_cand = len(cand_idx)
    X_cand = X_all[cand_idx]
    rng = np.random.RandomState(1)

    eval_size = min(10000, N)
    X_eval = X_all[rng.choice(N, size=eval_size, replace=False)]

    sub_n = min(3000, n_cand)
    sub_idx = rng.choice(n_cand, size=sub_n, replace=False)
    sub_d = cdist(X_cand[sub_idx], X_cand[sub_idx], metric='sqeuclidean')
    sigma2 = float(np.median(sub_d[sub_d > 0]))

    sim = np.empty((eval_size, n_cand), dtype=np.float32)
    block = 2000
    for s in range(0, eval_size, block):
        e = min(s + block, eval_size)
        sim[s:e] = np.exp(-cdist(X_eval[s:e], X_cand, metric='sqeuclidean')
                          / (2 * sigma2)).astype(np.float32)

    sel, mask, cov = [], np.zeros(n_cand, dtype=bool), np.zeros(eval_size, dtype=np.float32)
    for step in range(n):
        gains = np.maximum(0, sim - cov[:, None]).sum(axis=0)
        gains[mask] = -1.0
        best = int(gains.argmax())
        sel.append(best)
        mask[best] = True
        cov = np.maximum(cov, sim[:, best])
        if (step + 1) % 200 == 0:
            print(f"    FL: {step + 1}/{n}")
    return cand_idx[np.array(sel)]


def sample_kernel_herding(X, n):
    """
    Kernel herding (Chen, Welling, Smola 2010):
    детерминистический выбор подмножества, минимизирующего MMD.
    μ_P вычисляется точно по всей совокупности (блоками).
    """
    N = X.shape[0]
    rng = np.random.RandomState(0)

    # σ² по подвыборке
    sub_idx = rng.choice(N, size=min(2000, N), replace=False)
    sub_d = cdist(X[sub_idx], X[sub_idx], metric='sqeuclidean')
    sigma2 = float(np.median(sub_d[sub_d > 0]))
    inv_2s = 1.0 / (2 * sigma2)

    # μ_P — точное среднее ядра по ВСЕЙ совокупности (блоками)
    mu = np.zeros(N, dtype=np.float64)
    block = 2000
    for s in range(0, N, block):
        e = min(s + block, N)
        mu += np.exp(-cdist(X[s:e], X, metric='sqeuclidean') * inv_2s).sum(axis=0)
    mu /= N
    print(f"    KH: μ_P вычислено точно (N={N})")

    # Жадный отбор
    selected, mask = [], np.zeros(N, dtype=bool)
    w = mu.copy()
    for step in range(n):
        wm = w.copy()
        wm[mask] = -np.inf
        best = int(np.argmax(wm))
        selected.append(best)
        mask[best] = True
        k_best = np.exp(-np.sum((X - X[best]) ** 2, axis=1).astype(np.float64) * inv_2s)
        w = w + mu - k_best
        if (step + 1) % 100 == 0:
            print(f"    KH: {step + 1}/{n}")
    return np.array(selected)


# ============================================================
# 3. МЕТРИКИ ВАЛИДАЦИИ
# ============================================================

def _compute_chi2_from_counts(sample_marks_series, marks, pop_probs):
    sc = sample_marks_series.value_counts().sort_index()
    obs = np.array([sc.get(m, 0) for m in marks])
    exp_arr = pop_probs * obs.sum()
    mask = exp_arr > 0
    stat, pval = chisquare(obs[mask], exp_arr[mask])
    n_obs, k = obs[mask].sum(), mask.sum()
    cv = (stat / (n_obs * max(k - 1, 1))) ** 0.5 if n_obs > 0 else 0.0
    sp = obs / max(obs.sum(), 1)
    return {'chi2_stat': stat, 'chi2_pvalue': pval, 'cramers_v': cv,
            'max_mark_dev': float(np.max(np.abs(sp - pop_probs)))}


def compute_chi2_marks(sample_marks, pop_marks):
    pc = pop_marks.value_counts().sort_index()
    marks = sorted(pc.index)
    ep = np.array([pc.get(m, 0) for m in marks], dtype=np.float64)
    ep = ep / ep.sum()
    return _compute_chi2_from_counts(sample_marks, marks, ep)


def compute_mmd(X_sample, X_pop, sigma=None):
    rng = np.random.RandomState(0)
    Xp = X_pop[rng.choice(len(X_pop), min(3000, len(X_pop)), replace=False)]
    Xs = X_sample[rng.choice(len(X_sample), min(3000, len(X_sample)), replace=False)]
    if sigma is None:
        c = np.vstack([Xs[:200], Xp[:200]])
        d = cdist(c, c, 'sqeuclidean')
        sigma2 = float(np.median(d[d > 0]))
    else:
        sigma2 = sigma ** 2
    kss = np.exp(-cdist(Xs, Xs, 'sqeuclidean') / (2 * sigma2)).mean()
    kpp = np.exp(-cdist(Xp, Xp, 'sqeuclidean') / (2 * sigma2)).mean()
    ksp = np.exp(-cdist(Xs, Xp, 'sqeuclidean') / (2 * sigma2)).mean()
    return max(0, kss + kpp - 2 * ksp) ** 0.5


def _compute_mmd_fast(X_sample, pop_stats):
    rng = np.random.RandomState(0)
    Xs = X_sample
    if len(Xs) > 3000:
        Xs = Xs[rng.choice(len(Xs), 3000, replace=False)]
    sigma2 = pop_stats['sigma2']
    kpp = pop_stats['kpp']
    Xp = pop_stats['Xp_sub']
    kss = float(np.exp(-cdist(Xs, Xs, 'sqeuclidean') / (2 * sigma2)).mean())
    ksp = float(np.exp(-cdist(Xs, Xp, 'sqeuclidean') / (2 * sigma2)).mean())
    return max(0, kss + kpp - 2 * ksp) ** 0.5


def validate_slice(samp_sl, pop_sl, grade, subj):
    res = {
        'grade': int(grade), 'subject_code': int(subj),
        'subject_name': SUBJECT_NAMES.get(int(subj), f'?{subj}'),
        'kim_max': KIM_MAX_SCORES.get((int(grade), int(subj))),
        'n_pop': len(pop_sl), 'n_sample': len(samp_sl),
    }
    if len(samp_sl) < 10 or len(pop_sl) < 10:
        res['error'] = 'Мало данных'
        return res
    km = res['kim_max']
    if km:
        pop_sl = pop_sl[pop_sl['score'] <= km]
        samp_sl = samp_sl[samp_sl['score'] <= km]
    if len(samp_sl) < 5:
        res['error'] = 'Мало данных после КИМ-фильтра'
        return res
    res.update(compute_chi2_marks(samp_sl['mark'], pop_sl['mark']))
    ks_s, ks_p = ks_2samp(samp_sl['score'], pop_sl['score'])
    res['ks_stat'] = ks_s
    res['ks_pvalue'] = ks_p
    res['mean_score_pop'] = pop_sl['score'].mean()
    res['mean_score_sample'] = samp_sl['score'].mean()
    res['mean_mark_pop'] = pop_sl['mark'].mean()
    res['mean_mark_sample'] = samp_sl['mark'].mean()
    for m in VALID_MARKS:
        res[f'pop_pct_{m}'] = (pop_sl['mark'] == m).mean()
        res[f'sample_pct_{m}'] = (samp_sl['mark'] == m).mean()
    return res


# ============================================================
# 3b. БЫСТРАЯ ВАЛИДАЦИЯ (для стохастических прогонов)
# ============================================================

def validate_sample_fast(sample_indices, schools, vpr, X_all,
                         vpr_index, pop_stats, compute_slices=False):
    logins = schools.iloc[sample_indices]['login'].values
    idx_arrays = [vpr_index[l] for l in logins if l in vpr_index]
    if not idx_arrays:
        return {'error': 'Нет данных ВПР для выбранных ОО'}
    sample_vpr = vpr.iloc[np.concatenate(idx_arrays)]

    marks = pop_stats['marks']
    pop_probs = pop_stats['pop_mark_probs']
    pm = pop_stats['pop_mean_score']
    pmk = pop_stats['pop_mean_mark']

    r = {'n_schools_selected': len(logins),
         'n_students_sample': len(sample_vpr),
         'n_students_pop': pop_stats['n_pop']}
    r.update(_compute_chi2_from_counts(sample_vpr['mark'], marks, pop_probs))

    ks_s, ks_p = ks_2samp(sample_vpr['score'].values, vpr['score'].values)
    r['ks_stat'] = ks_s
    r['ks_pvalue'] = ks_p

    sm = float(sample_vpr['score'].mean())
    smk = float(sample_vpr['mark'].mean())
    r['mean_score_pop'] = pm
    r['mean_score_sample'] = sm
    r['rel_error_mean_score'] = abs(sm - pm) / pm
    r['mean_mark_pop'] = pmk
    r['mean_mark_sample'] = smk
    r['rel_error_mean_mark'] = abs(smk - pmk) / pmk
    r['mmd'] = _compute_mmd_fast(X_all[sample_indices], pop_stats)

    if compute_slices:
        slices = []
        for (g, s), pop_sl in pop_stats['slice_groups'].items():
            samp_sl = sample_vpr[
                (sample_vpr['grade'] == g) & (sample_vpr['subject_code'] == s)]
            slices.append(validate_slice(samp_sl, pop_sl, g, s))
        r['slices'] = slices
    else:
        r['slices'] = []
    return r


def validate_sample(sample_logins, schools, vpr, X_all, sample_indices,
                    pop_stats=None):
    """Полная валидация (для детерминированных методов).
    Если передан pop_stats — MMD считается с той же σ², что и для стохастических."""
    logins = set(schools.iloc[sample_indices]['login'].values)
    sample_vpr = vpr[vpr['login'].isin(logins)]
    if len(sample_vpr) == 0:
        return {'error': 'Нет данных ВПР для выбранных ОО'}

    r = {'n_schools_selected': len(logins),
         'n_students_sample': len(sample_vpr),
         'n_students_pop': len(vpr)}
    r.update(compute_chi2_marks(sample_vpr['mark'], vpr['mark']))
    ks_s, ks_p = ks_2samp(sample_vpr['score'], vpr['score'])
    r['ks_stat'] = ks_s
    r['ks_pvalue'] = ks_p

    pm, sm = vpr['score'].mean(), sample_vpr['score'].mean()
    r['mean_score_pop'] = pm
    r['mean_score_sample'] = sm
    r['rel_error_mean_score'] = abs(sm - pm) / pm
    pmk, smk = vpr['mark'].mean(), sample_vpr['mark'].mean()
    r['mean_mark_pop'] = pmk
    r['mean_mark_sample'] = smk
    r['rel_error_mean_mark'] = abs(smk - pmk) / pmk

    if pop_stats is not None:
        r['mmd'] = _compute_mmd_fast(X_all[sample_indices], pop_stats)
    else:
        r['mmd'] = compute_mmd(X_all[sample_indices], X_all)

    slices = []
    if pop_stats is not None:
        for (g, s), pop_sl in pop_stats['slice_groups'].items():
            samp_sl = sample_vpr[
                (sample_vpr['grade'] == g) & (sample_vpr['subject_code'] == s)]
            slices.append(validate_slice(samp_sl, pop_sl, g, s))
    else:
        for (g, s), pop_sl in vpr.groupby(['grade', 'subject_code']):
            samp_sl = sample_vpr[
                (sample_vpr['grade'] == g) & (sample_vpr['subject_code'] == s)]
            slices.append(validate_slice(samp_sl, pop_sl, g, s))
    r['slices'] = slices
    return r


# Метрики, входящие в composite score (и используемые для нормализации)
COMPOSITE_WEIGHTS = [
    ('rel_error_mean_score', 0.25),
    ('rel_error_mean_mark', 0.20),
    ('ks_stat', 0.20),
    ('mmd', 0.15),
    ('cramers_v', 0.10),
    ('max_mark_dev', 0.10),
]
COMPOSITE_METRICS = [k for k, _ in COMPOSITE_WEIGHTS]

ALL_METRIC_KEYS = [
    'rel_error_mean_score', 'rel_error_mean_mark', 'ks_stat',
    'mmd', 'cramers_v', 'max_mark_dev', 'composite_score',
    'mean_score_pop', 'mean_score_sample',
    'mean_mark_pop', 'mean_mark_sample', 'ks_pvalue',
]


def compute_composite_score(results, srs_norm=None):
    """
    Составной балл качества выборки.
    Если передан srs_norm — каждая метрика нормализуется по среднему SRS:
      1.0 = уровень SRS, <1.0 = лучше SRS, >1.0 = хуже SRS.
    """
    if 'error' in results:
        return float('inf')
    s = 0.0
    for key, weight in COMPOSITE_WEIGHTS:
        val = results.get(key, 1.0)
        if srs_norm is not None and srs_norm.get(key, 0) > 0:
            val = val / srs_norm[key]
        s += weight * val
    return s


# ============================================================
# 4. УСРЕДНЕНИЕ СТОХАСТИЧЕСКИХ ПРОГОНОВ
# ============================================================


def _collect_stochastic_runs(schools, vpr, X, n, n_runs, seed,
                             sampler_fn, vpr_index, pop_stats):
    """Прогон стохастического метода N раз, сбор сырых метрик."""
    all_runs = []
    for run in range(n_runs):
        idx = sampler_fn(seed + run)
        res = validate_sample_fast(
            idx, schools, vpr, X, vpr_index, pop_stats,
            compute_slices=False)
        all_runs.append(res)
    return all_runs


def _compute_srs_norm(srs_runs):
    """Среднее каждой метрики по SRS-прогонам (базис нормализации)."""
    norm = {}
    valid = [r for r in srs_runs if 'error' not in r]
    for key in COMPOSITE_METRICS:
        vals = [r.get(key, 0) for r in valid]
        norm[key] = float(np.mean(vals)) if vals else 1.0
    return norm


def _summarize_runs(all_runs, srs_norm=None):
    """Вычисление нормализованных composite scores и средних."""
    for res in all_runs:
        res['composite_score'] = compute_composite_score(res, srs_norm=srs_norm)
    scores = [r['composite_score'] for r in all_runs]

    avg = {}
    for key in ALL_METRIC_KEYS:
        vals = [r.get(key, 0) for r in all_runs]
        avg[key] = float(np.mean(vals))
        avg[f'{key}_std'] = float(np.std(vals))

    avg_entry = {k: avg[k] for k in ALL_METRIC_KEYS}
    avg_entry['time_sec'] = 0.0
    avg_entry['composite_score'] = avg['composite_score']
    avg_entry['slices'] = []
    return avg_entry, scores, all_runs, avg


# ============================================================
# 5. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def run_sampling_experiment(ctx_path, vpr_path, sample_size=300,
                            seed=42, n_srs_runs=50):
    print("=" * 70)
    print("СИСТЕМА СЭМПЛИНГА ОО")
    print("=" * 70)

    ctx = load_context_data(ctx_path)
    vpr = load_vpr_data(vpr_path)
    schools = build_school_features(ctx, vpr)
    X = prepare_feature_matrix(schools)
    n = min(sample_size, len(schools) // 5)
    print(f"  Выборка: {n} из {len(schools)}")

    t0 = time.time()
    vpr_index = build_vpr_index(vpr)
    pop_stats = precompute_pop_stats(vpr, X)
    strata_map = precompute_strata(schools)
    print(f"  Предвычисления: {time.time() - t0:.1f}с")

    # ── Фаза 1: SRS прогоны → базис нормализации ──
    srs_runs = _collect_stochastic_runs(
        schools, vpr, X, n, n_srs_runs, seed,
        sampler_fn=lambda s: sample_srs(schools, n, seed=s),
        vpr_index=vpr_index, pop_stats=pop_stats,
    )
    srs_norm = _compute_srs_norm(srs_runs)
    print(f"  SRS-нормализация: { {k: f'{v:.4f}' for k, v in srs_norm.items()} }")

    # ── Фаза 2: все composite scores нормализуются по SRS ──
    srs_entry, srs_scores, _, srs_avg = _summarize_runs(srs_runs, srs_norm=srs_norm)

    strat_runs = _collect_stochastic_runs(
        schools, vpr, X, n, n_srs_runs, seed,
        sampler_fn=lambda s: sample_stratified(schools, n, seed=s,
                                               strata_map=strata_map),
        vpr_index=vpr_index, pop_stats=pop_stats,
    )
    strat_entry, strat_scores, _, strat_avg = _summarize_runs(
        strat_runs, srs_norm=srs_norm)

    det_methods = {
        '3. k-center greedy':    lambda: sample_kcenter(X, n, seed=seed),
        '4. Facility location':  lambda: sample_facility_location(X, n),
        '5. Kernel herding':     lambda: sample_kernel_herding(X, n),
    }

    all_results = {}
    for name, fn in det_methods.items():
        t0 = time.time()
        indices = fn()
        elapsed = time.time() - t0
        res = validate_sample(set(schools.iloc[indices]['login'].values),
                              schools, vpr, X, indices, pop_stats=pop_stats)
        res['time_sec'] = elapsed
        res['composite_score'] = compute_composite_score(res, srs_norm=srs_norm)
        all_results[name] = res

    return all_results, srs_avg, srs_scores, schools, vpr, X, n


if __name__ == '__main__':
    results, srs_avg, srs_scores, _, _, _, _ = run_sampling_experiment(
        'конт_данныеоо.xlsx', 'pupils_ruma456_2019_2020.csv')