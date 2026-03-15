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
# subject_code: 1 = Русский язык (РУ), 2 = Математика (МА)
KIM_MAX_SCORES = {
    (4, 1): 38,   # РУ 4 класс
    (5, 1): 45,   # РУ 5 класс
    (6, 1): 45,   # РУ 6 класс
    (4, 2): 20,   # МА 4 класс
    (5, 2): 20,   # МА 5 класс
    (6, 2): 20,   # МА 6 класс
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
    X = StandardScaler().fit_transform(features.values.astype(float))
    print(f"  Признаков: {X.shape[1]} (one-hot)")
    return X


# ============================================================
# 2. МЕТОДЫ СЭМПЛИНГА
# ============================================================

def sample_srs(schools, n, seed=42):
    return np.random.RandomState(seed).choice(len(schools), size=n, replace=False)


def sample_stratified(schools, n, seed=42):
    rng = np.random.RandomState(seed)
    schools = schools.copy()
    schools['stratum'] = schools['region'] + '_' + schools['location_code'].astype(str)
    strata_counts = schools['stratum'].value_counts()
    N_total = len(schools)

    exact = {s: n * cnt / N_total for s, cnt in strata_counts.items()}
    floor = {s: int(v) for s, v in exact.items()}
    rems = {s: exact[s] - floor[s] for s in exact}
    deficit = n - sum(floor.values())
    for s in sorted(rems, key=rems.get, reverse=True)[:deficit]:
        floor[s] += 1

    selected = []
    for stratum, n_s in floor.items():
        if n_s == 0:
            continue
        idx = schools.index[schools['stratum'] == stratum].values
        if len(idx) <= n_s:
            selected.extend(idx)
        else:
            selected.extend(rng.choice(idx, size=n_s, replace=False))

    selected = list(set(selected))
    if len(selected) > n:
        selected = list(rng.choice(selected, size=n, replace=False))
    elif len(selected) < n:
        remaining = list(set(range(len(schools))) - set(selected))
        selected.extend(rng.choice(remaining, size=n - len(selected), replace=False))
    return np.array(selected[:n])


def sample_kmedoids(X, n, seed=42):
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    first = rng.randint(N)
    selected = [first]
    min_dist = np.sum((X - X[first]) ** 2, axis=1)
    for step in range(1, n):
        farthest = np.argmax(min_dist)
        selected.append(farthest)
        min_dist = np.minimum(min_dist, np.sum((X - X[farthest]) ** 2, axis=1))
        if (step + 1) % 500 == 0:
            print(f"    k-center: {step + 1}/{n}")
    return np.array(selected)


def sample_facility_location(X, n):
    N = X.shape[0]
    if N <= 5000:
        return _fl_exact(X, n)
    n_cand = min(5 * n, N)
    cand_idx = np.random.RandomState(0).choice(N, size=n_cand, replace=False)
    print(f"    FL: {n_cand} кандидатов → отбор {n}...")
    return _fl_on_candidates(X, cand_idx, n)


def _fl_exact(X, n):
    N = X.shape[0]
    dists = cdist(X, X, metric='sqeuclidean')
    sigma2 = float(np.median(dists[dists > 0]))
    sim = np.exp(-dists / (2 * sigma2))
    selected, mask, cov = [], np.zeros(N, dtype=bool), np.zeros(N)
    for _ in range(n):
        gains = np.maximum(0, sim - cov[:, None]).sum(axis=0)
        gains[mask] = -1.0
        best = gains.argmax()
        selected.append(best)
        mask[best] = True
        cov = np.maximum(cov, sim[:, best])
    return np.array(selected)


def _fl_on_candidates(X_all, cand_idx, n):
    n_cand = len(cand_idx)
    X_cand = X_all[cand_idx]
    rng = np.random.RandomState(1)
    eval_size = min(5000, len(X_all))
    X_eval = X_all[rng.choice(len(X_all), size=eval_size, replace=False)]

    dists = cdist(X_eval, X_cand, metric='sqeuclidean')
    sub_n = min(2000, n_cand)
    sub_d = cdist(X_cand[:sub_n], X_cand[:sub_n], metric='sqeuclidean')
    sigma2 = float(np.median(sub_d[sub_d > 0]))
    sim = np.exp(-dists / (2 * sigma2))

    sel, mask, cov = [], np.zeros(n_cand, dtype=bool), np.zeros(eval_size)
    for step in range(n):
        gains = np.maximum(0, sim - cov[:, None]).sum(axis=0)
        gains[mask] = -1.0
        best = gains.argmax()
        sel.append(best)
        mask[best] = True
        cov = np.maximum(cov, sim[:, best])
        if (step + 1) % 100 == 0:
            print(f"    FL: {step + 1}/{n}")
    return cand_idx[np.array(sel)]


def sample_kernel_herding(X, n):
    N = X.shape[0]
    rng = np.random.RandomState(0)
    sub_idx = rng.choice(N, size=min(2000, N), replace=False)
    sub_d = cdist(X[sub_idx], X[sub_idx], metric='sqeuclidean')
    sigma2 = float(np.median(sub_d[sub_d > 0]))

    eval_size = min(5000, N)
    X_eval = X[rng.choice(N, size=eval_size, replace=False)]
    mu = np.zeros(N)
    for s in range(0, eval_size, 2000):
        e = min(s + 2000, eval_size)
        mu += np.exp(-cdist(X_eval[s:e], X, metric='sqeuclidean') / (2 * sigma2)).sum(axis=0)
    mu /= eval_size

    selected, mask, w = [], np.zeros(N, dtype=bool), mu.copy()
    for step in range(n):
        wm = w.copy()
        wm[mask] = -np.inf
        best = np.argmax(wm)
        selected.append(best)
        mask[best] = True
        w = w + mu - np.exp(-np.sum((X - X[best]) ** 2, axis=1) / (2 * sigma2))
        if (step + 1) % 100 == 0:
            print(f"    KH: {step + 1}/{n}")
    return np.array(selected)


# ============================================================
# 3. МЕТРИКИ ВАЛИДАЦИИ
# ============================================================

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


def compute_chi2_marks(sample_marks, pop_marks):
    pc = pop_marks.value_counts().sort_index()
    sc = sample_marks.value_counts().sort_index()
    marks = sorted(set(pc.index) | set(sc.index))
    obs = np.array([sc.get(m, 0) for m in marks])
    ep = np.array([pc.get(m, 0) for m in marks])
    ep = ep / ep.sum()
    exp = ep * obs.sum()
    mask = exp > 0
    stat, pval = chisquare(obs[mask], exp[mask])
    n_obs, k = obs[mask].sum(), mask.sum()
    cv = (stat / (n_obs * max(k - 1, 1))) ** 0.5 if n_obs > 0 else 0.0
    sp = obs / max(obs.sum(), 1)
    return {'chi2_stat': stat, 'chi2_pvalue': pval, 'cramers_v': cv,
            'max_mark_dev': float(np.max(np.abs(sp - ep)))}


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


def validate_sample(sample_logins, schools, vpr, X_all, sample_indices):
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
    r['mmd'] = compute_mmd(X_all[sample_indices], X_all)

    slices = []
    for (g, s), pop_sl in vpr.groupby(['grade', 'subject_code']):
        samp_sl = sample_vpr[(sample_vpr['grade'] == g) & (sample_vpr['subject_code'] == s)]
        slices.append(validate_slice(samp_sl, pop_sl, g, s))
    r['slices'] = slices
    return r


def compute_composite_score(results):
    if 'error' in results:
        return float('inf')
    s = 0.0
    s += 0.25 * results.get('rel_error_mean_score', 1.0)
    s += 0.20 * results.get('rel_error_mean_mark', 1.0)
    s += 0.20 * results.get('ks_stat', 1.0)
    s += 0.15 * min(results.get('mmd', 1.0), 1.0)
    s += 0.10 * min(results.get('cramers_v', 1.0), 1.0)
    s += 0.10 * min(results.get('max_mark_dev', 1.0), 1.0)
    return s


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

    srs_scores, srs_all = [], []
    for run in range(n_srs_runs):
        idx = sample_srs(schools, n, seed=seed + run)
        res = validate_sample(set(schools.iloc[idx]['login'].values), schools, vpr, X, idx)
        res['composite_score'] = compute_composite_score(res)
        srs_all.append(res)
        srs_scores.append(res['composite_score'])

    srs_avg = {}
    for key in ['rel_error_mean_score', 'rel_error_mean_mark', 'ks_stat',
                'mmd', 'cramers_v', 'max_mark_dev', 'composite_score']:
        vals = [r.get(key, 0) for r in srs_all]
        srs_avg[key] = float(np.mean(vals))
        srs_avg[f'{key}_std'] = float(np.std(vals))

    methods = {
        '2. Стратифицированная': lambda: sample_stratified(schools, n, seed),
        '3. k-center greedy': lambda: sample_kmedoids(X, n, seed=seed),
        '4. Facility location': lambda: sample_facility_location(X, n),
        '5. Kernel herding': lambda: sample_kernel_herding(X, n),
    }

    all_results = {}
    for name, fn in methods.items():
        t0 = time.time()
        indices = fn()
        elapsed = time.time() - t0
        res = validate_sample(set(schools.iloc[indices]['login'].values), schools, vpr, X, indices)
        res['time_sec'] = elapsed
        res['composite_score'] = compute_composite_score(res)
        all_results[name] = res

    return all_results, srs_avg, srs_scores, schools, vpr, X, n


if __name__ == '__main__':
    results, srs_avg, srs_scores, _, _, _, _ = run_sampling_experiment(
        'Конт_данныеОО.xlsx', 'pupils_ruma456_2019_2020.csv')