"""
Система сэмплинга данных для формирования репрезентативной выборки
образовательных организаций (ОО) на основе данных ВПР.

Реализует 5 методов отбора и валидирует выборки по совпадению
распределений результатов ВПР с генеральной совокупностью.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
import time
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================

def load_context_data(path: str) -> pd.DataFrame:
    """
    Загрузка контекстных данных ОО.
    Ожидаемые столбцы: Логин ОО, Расположение, Наименование НП,
    Размер НП, Коррекционная, Количество обучающихся.
    """
    df = pd.read_excel(path)
    # Пропускаем строку-заголовок, если она дублируется
    if df.iloc[0, 0] == df.columns[0] or df.iloc[0, 0] == 'Логин ОО':
        df = df.iloc[1:].reset_index(drop=True)

    # Унифицируем имена столбцов
    df.columns = ['login', 'location_type', 'locality_name',
                  'locality_size', 'is_correctional', 'n_students']

    # Извлекаем код региона из логина (sch[XX]...)
    df['region'] = df['login'].str[3:5]

    # Чистка данных
    df['n_students'] = pd.to_numeric(df['n_students'], errors='coerce')
    df['is_correctional'] = df['is_correctional'].str.lower().map(
        {'да': 1, 'нет': 0}).fillna(0).astype(int)

    # Извлекаем числовой код расположения (1-4)
    df['location_code'] = df['location_type'].str[0].astype(int)

    # Извлекаем числовой код размера НП (1-8)
    df['size_code'] = df['locality_size'].str[0].astype(int)

    df = df.dropna(subset=['n_students']).reset_index(drop=True)
    print(f"  Загружено ОО: {len(df)}")
    print(f"  Регионов: {df['region'].nunique()}")
    return df


def load_vpr_data(path: str) -> pd.DataFrame:
    """
    Загрузка результатов ВПР.
    Ожидаемые столбцы: Класс, Код_предмета, ЛогинОО,
    Код_ученика, Балл, Отметка, Year
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = ['grade', 'subject_code', 'login', 'student_id',
                  'score', 'mark', 'year']
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['mark'] = pd.to_numeric(df['mark'], errors='coerce')
    df['grade'] = pd.to_numeric(df['grade'], errors='coerce')
    df['subject_code'] = pd.to_numeric(df['subject_code'], errors='coerce')
    df = df.dropna(subset=['score', 'mark']).reset_index(drop=True)
    print(f"  Загружено записей ВПР: {len(df)}")
    print(f"  Уникальных ОО: {df['login'].nunique()}")
    print(f"  Классы: {sorted(df['grade'].dropna().unique().tolist())}")
    print(f"  Предметы: {sorted(df['subject_code'].dropna().unique().tolist())}")
    return df


def build_school_features(ctx: pd.DataFrame, vpr: pd.DataFrame) -> pd.DataFrame:
    """
    Строит вектор признаков для каждой ОО на основе контекстных данных
    и агрегированных результатов ВПР.
    """
    # Агрегируем результаты ВПР по ОО
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

    # Объединяем с контекстными данными
    merged = ctx.merge(school_stats, on='login', how='inner')
    print(f"  ОО с контекстом и результатами ВПР: {len(merged)}")

    return merged


def prepare_feature_matrix(schools: pd.DataFrame) -> np.ndarray:
    """
    Создаёт числовую матрицу признаков для алгоритмов отбора.
    Признаки: region (one-hot), location_code, size_code,
    is_correctional, log(n_students), mean_score, std_score.
    """
    features = pd.DataFrame()

    # Категориальные → числовые
    features['location_code'] = schools['location_code']
    features['size_code'] = schools['size_code']
    features['is_correctional'] = schools['is_correctional']

    # Непрерывные
    features['log_n_students'] = np.log1p(schools['n_students'])
    features['mean_score'] = schools['mean_score']
    features['std_score'] = schools['std_score']

    # Регион — LabelEncoder (для расстояний в k-medoids используем
    # one-hot, но для экономии памяти — только код)
    le = LabelEncoder()
    features['region_code'] = le.fit_transform(schools['region'])

    X = features.values.astype(float)

    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# ============================================================
# 2. МЕТОДЫ СЭМПЛИНГА
# ============================================================

def sample_srs(schools: pd.DataFrame, n: int, seed: int = 42) -> np.ndarray:
    """Простая случайная выборка (SRS)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(schools), size=n, replace=False)
    return idx


def sample_stratified(schools: pd.DataFrame, n: int, seed: int = 42) -> np.ndarray:
    """
    Стратифицированная выборка: страты = регион × тип расположения.
    Пропорциональное размещение (proportional allocation).
    """
    rng = np.random.RandomState(seed)
    schools = schools.copy()
    schools['stratum'] = schools['region'] + '_' + schools['location_code'].astype(str)

    strata_counts = schools['stratum'].value_counts()
    N_total = len(schools)

    selected = []
    # Пропорциональное размещение
    for stratum, count in strata_counts.items():
        n_stratum = max(1, int(round(n * count / N_total)))
        stratum_idx = schools.index[schools['stratum'] == stratum].values
        if len(stratum_idx) <= n_stratum:
            selected.extend(stratum_idx)
        else:
            chosen = rng.choice(stratum_idx, size=n_stratum, replace=False)
            selected.extend(chosen)

    # Корректируем до нужного размера
    selected = list(set(selected))
    if len(selected) > n:
        selected = list(rng.choice(selected, size=n, replace=False))
    elif len(selected) < n:
        remaining = list(set(range(len(schools))) - set(selected))
        extra = rng.choice(remaining, size=n - len(selected), replace=False)
        selected.extend(extra)

    return np.array(selected[:n])


def sample_kmedoids(X: np.ndarray, n: int, max_iter: int = 100,
                    seed: int = 42) -> np.ndarray:
    """
    k-center greedy (farthest-first traversal).
    Выбирает n точек, последовательно добавляя самую далёкую
    от уже выбранных. Гарантия: 2-приближение к оптимуму k-center.
    Сложность: O(N * n) — линейная по числу объектов.
    """
    rng = np.random.RandomState(seed)
    N = X.shape[0]

    # Начинаем со случайной точки
    first = rng.randint(N)
    selected = [first]

    # min_dist[i] = расстояние от i до ближайшего выбранного
    min_dist = np.sum((X - X[first]) ** 2, axis=1)

    for step in range(1, n):
        # Выбираем точку с максимальным min_dist
        farthest = np.argmax(min_dist)
        selected.append(farthest)

        # Обновляем min_dist
        new_dist = np.sum((X - X[farthest]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, new_dist)

        if (step + 1) % 500 == 0:
            print(f"    k-center greedy: {step + 1}/{n}")

    return np.array(selected)


def sample_facility_location(X: np.ndarray, n: int) -> np.ndarray:
    """
    Жадный алгоритм facility location (субмодулярная оптимизация).
    f(S) = sum_i max_{j in S} s(i, j)
    Гарантия: (1 - 1/e) ≈ 0.632 от оптимума.

    Для больших N (>5000) используется двухэтапная схема:
    1) k-center на полных данных выбирает 5×n кандидатов
    2) Facility location на кандидатах выбирает финальные n
    Память: O(N) на этапе 1, O(candidates²) на этапе 2.
    """
    N = X.shape[0]

    if N <= 5000:
        return _facility_location_exact(X, n)

    # Этап 1: предварительный отбор кандидатов через k-center
    n_candidates = min(3 * n, N)
    print(f"    FL: предотбор {n_candidates} кандидатов через k-center...")
    candidate_idx = sample_kmedoids(X, n_candidates, seed=0)

    # Этап 2: точный facility location на кандидатах,
    # но оценивая покрытие по ВСЕЙ совокупности
    print(f"    FL: точный отбор {n} из {n_candidates} кандидатов...")
    return _facility_location_on_candidates(X, candidate_idx, n)


def _facility_location_exact(X: np.ndarray, n: int) -> np.ndarray:
    """Точный facility location для малых N (≤5000)."""
    N = X.shape[0]
    dists = cdist(X, X, metric='sqeuclidean')
    sigma2 = float(np.median(dists[dists > 0]))
    similarity = np.exp(-dists / (2 * sigma2))

    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    current_coverage = np.zeros(N)

    for step in range(n):
        gains = np.maximum(0, similarity - current_coverage[:, None]).sum(axis=0)
        gains[selected_mask] = -1.0
        best = gains.argmax()
        selected.append(best)
        selected_mask[best] = True
        current_coverage = np.maximum(current_coverage, similarity[:, best])

    return np.array(selected)


def _facility_location_on_candidates(X_all: np.ndarray,
                                      candidate_idx: np.ndarray,
                                      n: int) -> np.ndarray:
    """
    Facility location: выбирает n из candidate_idx,
    оценивая покрытие по всей совокупности X_all (через подвыборку).
    """
    N_all = len(X_all)
    n_cand = len(candidate_idx)
    X_cand = X_all[candidate_idx]

    # Подвыборка совокупности для оценки покрытия (экономия памяти)
    rng = np.random.RandomState(0)
    eval_size = min(3000, N_all)
    eval_idx = rng.choice(N_all, size=eval_size, replace=False)
    X_eval = X_all[eval_idx]

    # Матрица сходства: eval_size × n_cand
    dists = cdist(X_eval, X_cand, metric='sqeuclidean')
    sub_dists = cdist(X_cand[:min(2000, n_cand)],
                      X_cand[:min(2000, n_cand)], metric='sqeuclidean')
    sigma2 = float(np.median(sub_dists[sub_dists > 0]))
    similarity = np.exp(-dists / (2 * sigma2))  # eval_size × n_cand

    selected_local = []  # indices in candidate_idx
    selected_mask = np.zeros(n_cand, dtype=bool)
    current_coverage = np.zeros(eval_size)

    for step in range(n):
        gains = np.maximum(0, similarity - current_coverage[:, None]).sum(axis=0)
        gains[selected_mask] = -1.0
        best_local = gains.argmax()
        selected_local.append(best_local)
        selected_mask[best_local] = True
        current_coverage = np.maximum(current_coverage, similarity[:, best_local])

        if (step + 1) % 100 == 0:
            print(f"    Facility location: {step + 1}/{n}")

    return candidate_idx[np.array(selected_local)]


def sample_kernel_herding(X: np.ndarray, n: int) -> np.ndarray:
    """
    Kernel herding (Chen, Welling, Smola 2010).
    Детерминистический выбор, минимизирующий MMD.
    Сходимость O(1/T) vs O(1/√T) для случайной выборки.
    Для больших N mean embedding оценивается по подвыборке.
    """
    N = X.shape[0]
    rng = np.random.RandomState(0)

    # Оценка σ по подвыборке
    sub_idx = rng.choice(N, size=min(2000, N), replace=False)
    sub_dists = cdist(X[sub_idx], X[sub_idx], metric='sqeuclidean')
    sigma2 = float(np.median(sub_dists[sub_dists > 0]))

    # mean embedding по подвыборке (для больших N — не по всей совокупности)
    eval_size = min(5000, N)
    eval_idx = rng.choice(N, size=eval_size, replace=False)
    X_eval = X[eval_idx]

    # μ_P(j) ≈ (1/eval_size) Σ_{i in eval} k(x_i, x_j)
    mean_embedding = np.zeros(N)
    batch = 2000
    for start in range(0, eval_size, batch):
        end = min(start + batch, eval_size)
        dists_batch = cdist(X_eval[start:end], X, metric='sqeuclidean')
        mean_embedding += np.exp(-dists_batch / (2 * sigma2)).sum(axis=0)
    mean_embedding /= eval_size

    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    w = mean_embedding.copy()

    for step in range(n):
        w_masked = w.copy()
        w_masked[selected_mask] = -np.inf
        best = np.argmax(w_masked)
        selected.append(best)
        selected_mask[best] = True

        k_best = np.exp(-np.sum((X - X[best]) ** 2, axis=1) / (2 * sigma2))
        w = w + mean_embedding - k_best

        if (step + 1) % 100 == 0:
            print(f"    Kernel herding: {step + 1}/{n}")

    return np.array(selected)


# ============================================================
# 3. МЕТРИКИ ВАЛИДАЦИИ
# ============================================================

def compute_mmd(X_sample: np.ndarray, X_population: np.ndarray,
                sigma: float = None) -> float:
    """
    Maximum Mean Discrepancy (Gretton et al., 2012).
    MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    Для больших совокупностей используется подвыборка.
    """
    rng = np.random.RandomState(0)
    max_pop = min(len(X_population), 3000)
    pop_idx = rng.choice(len(X_population), size=max_pop, replace=False)
    X_pop = X_population[pop_idx]

    max_samp = min(len(X_sample), 3000)
    samp_idx = rng.choice(len(X_sample), size=max_samp, replace=False)
    X_samp = X_sample[samp_idx]

    if sigma is None:
        combined = np.vstack([X_samp[:200], X_pop[:200]])
        dists = cdist(combined, combined, metric='sqeuclidean')
        sigma2 = float(np.median(dists[dists > 0]))
    else:
        sigma2 = sigma ** 2

    k_ss = np.exp(-cdist(X_samp, X_samp, 'sqeuclidean') / (2 * sigma2))
    k_pp = np.exp(-cdist(X_pop, X_pop, 'sqeuclidean') / (2 * sigma2))
    k_sp = np.exp(-cdist(X_samp, X_pop, 'sqeuclidean') / (2 * sigma2))

    mmd2 = k_ss.mean() + k_pp.mean() - 2 * k_sp.mean()
    return max(0, mmd2) ** 0.5


def compute_chi2_marks(sample_marks: pd.Series,
                       pop_marks: pd.Series) -> dict:
    """
    Метрики распределения отметок (2, 3, 4, 5):
    - χ²-статистика и p-value
    - V Крамера (Cramér's V) — мера размера эффекта, не зависит от n
    - Максимальное абсолютное отклонение долей (max |p_sample - p_pop|)
    """
    pop_counts = pop_marks.value_counts().sort_index()
    sample_counts = sample_marks.value_counts().sort_index()

    # Выравниваем категории
    all_marks = sorted(set(pop_counts.index) | set(sample_counts.index))
    observed = np.array([sample_counts.get(m, 0) for m in all_marks])
    expected_prop = np.array([pop_counts.get(m, 0) for m in all_marks])
    expected_prop = expected_prop / expected_prop.sum()
    expected = expected_prop * observed.sum()

    # Убираем нулевые категории
    mask = expected > 0
    observed_m = observed[mask]
    expected_m = expected[mask]

    from scipy.stats import chisquare
    stat, pvalue = chisquare(observed_m, expected_m)

    # V Крамера: V = sqrt(χ²/(n*(k-1))), k = число категорий
    n_obs = observed_m.sum()
    k = len(observed_m)
    cramers_v = (stat / (n_obs * max(k - 1, 1))) ** 0.5 if n_obs > 0 else 0.0

    # Максимальное абсолютное отклонение долей
    sample_prop = observed / max(observed.sum(), 1)
    max_abs_dev = float(np.max(np.abs(sample_prop - expected_prop)))

    return {
        'chi2_stat': stat,
        'chi2_pvalue': pvalue,
        'cramers_v': cramers_v,
        'max_mark_dev': max_abs_dev,
    }


def compute_ks_score(sample_scores: pd.Series,
                     pop_scores: pd.Series) -> dict:
    """
    Критерий Колмогорова–Смирнова для распределения баллов.
    """
    stat, pvalue = ks_2samp(sample_scores, pop_scores)
    return {'ks_stat': stat, 'ks_pvalue': pvalue}


def compute_relative_errors(sample_df: pd.DataFrame,
                            pop_df: pd.DataFrame) -> dict:
    """
    Относительные ошибки оценки ключевых статистик.
    """
    metrics = {}

    # Средний балл
    pop_mean = pop_df['score'].mean()
    sample_mean = sample_df['score'].mean()
    metrics['mean_score_pop'] = pop_mean
    metrics['mean_score_sample'] = sample_mean
    metrics['rel_error_mean_score'] = abs(sample_mean - pop_mean) / pop_mean

    # Средняя отметка
    pop_mark = pop_df['mark'].mean()
    sample_mark = sample_df['mark'].mean()
    metrics['mean_mark_pop'] = pop_mark
    metrics['mean_mark_sample'] = sample_mark
    metrics['rel_error_mean_mark'] = abs(sample_mark - pop_mark) / pop_mark

    # Доля двоек
    pop_pct2 = (pop_df['mark'] == 2).mean()
    sample_pct2 = (sample_df['mark'] == 2).mean()
    metrics['pct_mark2_pop'] = pop_pct2
    metrics['pct_mark2_sample'] = sample_pct2
    if pop_pct2 > 0:
        metrics['rel_error_pct_mark2'] = abs(sample_pct2 - pop_pct2) / pop_pct2
    else:
        metrics['rel_error_pct_mark2'] = 0.0

    # Дисперсия баллов
    pop_var = pop_df['score'].var()
    sample_var = sample_df['score'].var()
    metrics['var_score_pop'] = pop_var
    metrics['var_score_sample'] = sample_var
    if pop_var > 0:
        metrics['rel_error_var_score'] = abs(sample_var - pop_var) / pop_var
    else:
        metrics['rel_error_var_score'] = 0.0

    return metrics


def validate_sample(sample_logins: set, schools: pd.DataFrame,
                    vpr: pd.DataFrame, X_all: np.ndarray,
                    sample_indices: np.ndarray) -> dict:
    """
    Комплексная валидация выборки по всем метрикам.
    """
    # Получаем логины выбранных ОО
    logins = set(schools.iloc[sample_indices]['login'].values)

    # Фильтруем данные ВПР
    pop_vpr = vpr
    sample_vpr = vpr[vpr['login'].isin(logins)]

    if len(sample_vpr) == 0:
        return {'error': 'Нет данных ВПР для выбранных ОО'}

    results = {}
    results['n_schools_selected'] = len(logins)
    results['n_students_sample'] = len(sample_vpr)
    results['n_students_pop'] = len(pop_vpr)

    # 1. Хи-квадрат для отметок
    chi2_res = compute_chi2_marks(sample_vpr['mark'], pop_vpr['mark'])
    results.update(chi2_res)

    # 2. KS-тест для баллов
    ks_res = compute_ks_score(sample_vpr['score'], pop_vpr['score'])
    results.update(ks_res)

    # 3. Относительные ошибки
    rel_res = compute_relative_errors(sample_vpr, pop_vpr)
    results.update(rel_res)

    # 4. MMD на признаках ОО
    X_sample = X_all[sample_indices]
    mmd = compute_mmd(X_sample, X_all)
    results['mmd'] = mmd

    # 5. Хи-квадрат для категориальных контекстных признаков
    for col in ['location_code', 'size_code', 'region']:
        pop_dist = schools[col].value_counts(normalize=True)
        sample_dist = schools.iloc[sample_indices][col].value_counts(normalize=True)
        # Выравниваем
        all_cats = sorted(set(pop_dist.index) | set(sample_dist.index))
        obs = np.array([sample_dist.get(c, 0) for c in all_cats])
        exp = np.array([pop_dist.get(c, 0) for c in all_cats])
        if obs.sum() > 0:
            obs_counts = obs * len(sample_indices)
            exp_counts = exp * len(sample_indices)
            mask = exp_counts > 0
            if mask.sum() > 1:
                from scipy.stats import chisquare
                stat, pval = chisquare(obs_counts[mask], exp_counts[mask])
                results[f'chi2_{col}_pvalue'] = pval

    return results


# ============================================================
# 4. СВОДНЫЙ РЕЙТИНГ МЕТОДОВ
# ============================================================

def compute_composite_score(results: dict) -> float:
    """
    Составной балл качества выборки (чем МЕНЬШЕ, тем лучше).
    Компоненты:
    - Относительная ошибка среднего балла (вес 0.25)
    - Относительная ошибка средней отметки (вес 0.20)
    - KS-статистика (вес 0.20)
    - MMD (вес 0.15)
    - V Крамера для отметок (вес 0.10)
    - Max отклонение долей отметок (вес 0.10)
    """
    if 'error' in results:
        return float('inf')

    score = 0.0
    score += 0.25 * results.get('rel_error_mean_score', 1.0)
    score += 0.20 * results.get('rel_error_mean_mark', 1.0)
    score += 0.20 * results.get('ks_stat', 1.0)
    score += 0.15 * min(results.get('mmd', 1.0), 1.0)
    score += 0.10 * min(results.get('cramers_v', 1.0), 1.0)
    score += 0.10 * min(results.get('max_mark_dev', 1.0), 1.0)

    return score


# ============================================================
# 5. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def run_sampling_experiment(ctx_path: str, vpr_path: str,
                            sample_size: int = 300,
                            seed: int = 42,
                            n_srs_runs: int = 50):
    """
    Запускает эксперимент: все 5 методов + валидация + рейтинг.
    SRS запускается n_srs_runs раз для оценки стабильности.

    Параметры:
    ----------
    ctx_path : str
        Путь к Excel-файлу с контекстными данными ОО
    vpr_path : str
        Путь к CSV-файлу с результатами ВПР
    sample_size : int
        Размер выборки (количество ОО)
    seed : int
        Seed для воспроизводимости
    n_srs_runs : int
        Количество прогонов SRS для оценки дисперсии
    """
    print("=" * 70)
    print("СИСТЕМА СЭМПЛИНГА ОБРАЗОВАТЕЛЬНЫХ ОРГАНИЗАЦИЙ")
    print("=" * 70)

    # Загрузка данных
    print("\n[1/6] Загрузка данных...")
    ctx = load_context_data(ctx_path)
    vpr = load_vpr_data(vpr_path)

    # Построение признаков
    print("\n[2/6] Построение признаков ОО...")
    schools = build_school_features(ctx, vpr)
    X = prepare_feature_matrix(schools)
    print(f"  Матрица признаков: {X.shape}")

    n = min(sample_size, len(schools) // 5)
    print(f"  Размер выборки: {n} из {len(schools)} ОО")

    # ---- Многократный SRS (бейзлайн) ----
    print(f"\n[3/6] Многократный SRS ({n_srs_runs} прогонов)...")
    srs_scores = []
    srs_all_results = []
    for run in range(n_srs_runs):
        idx = sample_srs(schools, n, seed=seed + run)
        res = validate_sample(
            set(schools.iloc[idx]['login'].values),
            schools, vpr, X, idx
        )
        res['composite_score'] = compute_composite_score(res)
        srs_all_results.append(res)
        srs_scores.append(res['composite_score'])

    srs_mean = np.mean(srs_scores)
    srs_std = np.std(srs_scores)
    srs_best_run = np.argmin(srs_scores)
    srs_best_res = srs_all_results[srs_best_run]
    srs_best_res['time_sec'] = 0.0

    # Средние метрики по SRS
    srs_avg = {}
    metric_keys = ['rel_error_mean_score', 'rel_error_mean_mark', 'ks_stat',
                   'mmd', 'cramers_v', 'max_mark_dev', 'composite_score']
    for key in metric_keys:
        vals = [r.get(key, 0) for r in srs_all_results]
        srs_avg[key] = np.mean(vals)
        srs_avg[f'{key}_std'] = np.std(vals)

    print(f"  SRS средний балл: {srs_mean:.4f} ± {srs_std:.4f}")
    print(f"  SRS лучший прогон: {srs_scores[srs_best_run]:.4f} (run {srs_best_run})")

    # ---- Детерминированные методы ----
    methods = {
        '2. Стратифицированная': lambda: sample_stratified(schools, n, seed),
        '3. k-center greedy': lambda: sample_kmedoids(X, n, seed=seed),
        '4. Facility location': lambda: sample_facility_location(X, n),
        '5. Kernel herding': lambda: sample_kernel_herding(X, n),
    }

    print(f"\n[4/6] Формирование выборок (детерминированные методы)...")
    all_results = {}

    # Добавляем SRS результаты
    all_results[f'1. SRS (лучший из {n_srs_runs})'] = srs_best_res

    for name, method_fn in methods.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        indices = method_fn()
        elapsed = time.time() - t0
        print(f"  Время: {elapsed:.1f} с")

        print(f"  Валидация...")
        res = validate_sample(
            set(schools.iloc[indices]['login'].values),
            schools, vpr, X, indices
        )
        res['time_sec'] = elapsed
        res['composite_score'] = compute_composite_score(res)
        all_results[name] = res

    # ---- Вывод результатов ----
    print("\n" + "=" * 70)
    print("[5/6] РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ")
    print("=" * 70)

    header = (f"{'Метод':<35} {'Ош. ȳ':>8} {'Ош. m̄':>8} "
              f"{'KS':>7} {'MMD':>7} {'CramérV':>8} {'MaxΔ%':>7} {'Балл':>7} {'t,с':>6}")
    print(header)
    print("-" * len(header))

    for name, res in sorted(all_results.items(),
                            key=lambda x: x[1].get('composite_score', 99)):
        if 'error' in res:
            print(f"{name:<35} ОШИБКА: {res['error']}")
            continue
        print(f"{name:<35} "
              f"{res['rel_error_mean_score']:>8.4f} "
              f"{res['rel_error_mean_mark']:>8.4f} "
              f"{res['ks_stat']:>7.4f} "
              f"{res['mmd']:>7.4f} "
              f"{res.get('cramers_v', 0):>8.4f} "
              f"{res.get('max_mark_dev', 0)*100:>6.2f}% "
              f"{res['composite_score']:>7.4f} "
              f"{res['time_sec']:>5.1f}")

    # Бейзлайн SRS
    print()
    print(f"{'SRS среднее ('+str(n_srs_runs)+' прогонов)':<35} "
          f"{srs_avg['rel_error_mean_score']:>8.4f} "
          f"{srs_avg['rel_error_mean_mark']:>8.4f} "
          f"{srs_avg['ks_stat']:>7.4f} "
          f"{srs_avg['mmd']:>7.4f} "
          f"{srs_avg['cramers_v']:>8.4f} "
          f"{srs_avg['max_mark_dev']*100:>6.2f}% "
          f"{srs_avg['composite_score']:>7.4f} "
          f"{'—':>5}")
    print(f"{'  ± std':<35} "
          f"{srs_avg['rel_error_mean_score_std']:>8.4f} "
          f"{srs_avg['rel_error_mean_mark_std']:>8.4f} "
          f"{srs_avg['ks_stat_std']:>7.4f} "
          f"{srs_avg['mmd_std']:>7.4f} "
          f"{srs_avg['cramers_v_std']:>8.4f} "
          f"{srs_avg['max_mark_dev_std']*100:>6.2f}% "
          f"{srs_avg['composite_score_std']:>7.4f} ")

    # Лучший метод
    best_name = min(all_results.keys(),
                    key=lambda k: all_results[k].get('composite_score', 99))
    best_res = all_results[best_name]

    # Сколько прогонов SRS бьёт лучший ML-метод
    ml_methods = {k: v for k, v in all_results.items() if 'SRS' not in k}
    if ml_methods:
        best_ml_name = min(ml_methods.keys(),
                           key=lambda k: ml_methods[k].get('composite_score', 99))
        best_ml_score = ml_methods[best_ml_name]['composite_score']
        srs_wins = sum(1 for s in srs_scores if s < best_ml_score)
    else:
        best_ml_name = best_name
        best_ml_score = best_res['composite_score']
        srs_wins = 0

    print(f"\n{'=' * 70}")
    print(f"[6/6] РЕКОМЕНДАЦИЯ: {best_name}")
    print(f"{'=' * 70}")
    print(f"  Составной балл: {best_res['composite_score']:.4f}")
    print(f"  Относительная ошибка среднего балла: "
          f"{best_res['rel_error_mean_score']:.4f} "
          f"({best_res['mean_score_pop']:.2f} vs {best_res['mean_score_sample']:.2f})")
    print(f"  Относительная ошибка средней отметки: "
          f"{best_res['rel_error_mean_mark']:.4f} "
          f"({best_res['mean_mark_pop']:.2f} vs {best_res['mean_mark_sample']:.2f})")
    print(f"  KS-статистика (баллы): {best_res['ks_stat']:.4f} "
          f"(p={best_res['ks_pvalue']:.4f})")
    print(f"  V Крамера (отметки): {best_res.get('cramers_v', 0):.4f}")
    print(f"  Max отклонение доли отметок: "
          f"{best_res.get('max_mark_dev', 0)*100:.2f}%")
    print(f"  MMD (признаки ОО): {best_res['mmd']:.4f}")
    print()
    print(f"  --- Сравнение с SRS ---")
    print(f"  SRS среднее ({n_srs_runs} прогонов): {srs_mean:.4f} ± {srs_std:.4f}")
    print(f"  Лучший ML-метод ({best_ml_name}): {best_ml_score:.4f}")
    improvement = (srs_mean - best_ml_score) / srs_mean * 100
    print(f"  Улучшение: {improvement:+.1f}% vs среднее SRS")
    print(f"  SRS побеждает лучший ML в {srs_wins}/{n_srs_runs} "
          f"прогонов ({srs_wins/n_srs_runs*100:.0f}%)")

    # Сохраняем выбранные ОО лучшего метода
    best_indices = None
    if 'SRS' in best_name:
        best_indices = sample_srs(schools, n, seed=seed + srs_best_run)
    else:
        for mname, mfn in methods.items():
            if mname == best_name:
                best_indices = mfn()
                break
    if best_indices is None:
        best_indices = sample_srs(schools, n, seed=seed)

    best_schools = schools.iloc[best_indices][
        ['login', 'region', 'location_type', 'locality_size',
         'n_students', 'mean_score', 'mean_mark']
    ].copy()

    return all_results, best_schools, srs_avg


# ============================================================
# 6. ТОЧКА ВХОДА
# ============================================================


if __name__ == '__main__':


    ctx_path = 'Конт_данныеОО.xlsx'
    vpr_path = 'pupils_ruma456_2019_2020.csv'
    print(ctx_path)

    results, best_sample, srs_avg = run_sampling_experiment(ctx_path, vpr_path)

    # Сохраняем лучшую выборку
    output_path = 'best_sample.csv'
    best_sample.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nЛучшая выборка сохранена в: {output_path}")

    # Сохраняем метрики
    metrics_df = pd.DataFrame(results).T
    cols = ['rel_error_mean_score', 'rel_error_mean_mark', 'ks_stat',
            'mmd', 'cramers_v', 'max_mark_dev', 'composite_score', 'time_sec']
    cols = [c for c in cols if c in metrics_df.columns]
    metrics_df[cols].to_csv('sampling_metrics.csv', encoding='utf-8-sig')
    print(f"Метрики сохранены в: sampling_metrics.csv")
