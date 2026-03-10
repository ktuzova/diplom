"""
Streamlit-приложение: Система сэмплинга образовательных организаций
ВКР, 2025 — Тузова Ксения Кирилловна
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
import time
import io
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Конфигурация страницы ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Сэмплинг ОО | ВПР",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Стили ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geologica:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --accent: #2ea4ff;
    --accent2: #7ee787;
    --accent3: #f78166;
    --accent4: #d2a8ff;
    --text: #e6edf3;
    --muted: #8b949e;
}

html, body, [class*="css"] {
    font-family: 'Geologica', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

/* Шапка */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 2rem;
    margin: -1.5rem -2rem 2rem -2rem;
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(46,164,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(126,231,135,0.05) 0%, transparent 60%);
}

.header-icon {
    font-size: 2.5rem;
    position: relative;
    z-index: 1;
}

.header-text { position: relative; z-index: 1; }
.header-text h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.header-text p {
    margin: 0.2rem 0 0 0;
    font-size: 0.8rem;
    color: var(--muted);
    font-weight: 300;
}

/* Карточки метрик */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text);
}
.metric-value.good  { color: var(--accent2); }
.metric-value.warn  { color: #e3b341; }
.metric-value.bad   { color: var(--accent3); }
.metric-sub {
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* Бейдж метода */
.method-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-srs      { background: rgba(139,148,158,0.2); color: #8b949e; }
.badge-strat    { background: rgba(46,164,255,0.15); color: #2ea4ff; }
.badge-kcenter  { background: rgba(210,168,255,0.15); color: #d2a8ff; }
.badge-facility { background: rgba(126,231,135,0.15); color: #7ee787; }
.badge-herding  { background: rgba(255,166,87,0.15); color: #ffa657; }

/* Лучший результат */
.winner-badge {
    background: linear-gradient(135deg, rgba(126,231,135,0.2), rgba(46,164,255,0.15));
    border: 1px solid rgba(126,231,135,0.4);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}
.winner-badge h3 { margin: 0 0 0.3rem 0; color: var(--accent2); font-size: 1rem; }
.winner-badge p  { margin: 0; color: var(--muted); font-size: 0.85rem; }

/* Секции */
.section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Прогресс-лог */
.log-box {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.6;
}

/* Сайдбар */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Кнопка */
.stButton > button {
    background: linear-gradient(135deg, #2ea4ff, #1a7fd4) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Geologica', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(46,164,255,0.3) !important;
}

/* Таблица */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Upload zone */
[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 0.5rem;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Geologica', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Info box */
.info-box {
    background: rgba(46,164,255,0.08);
    border: 1px solid rgba(46,164,255,0.25);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.6;
}
.info-box strong { color: var(--accent); }

div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Шапка ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-icon">🎓</div>
    <div class="header-text">
        <h1>Система сэмплинга образовательных организаций</h1>
        <p>Формирование репрезентативной выборки ОО для оценки качества образования (ВПР/НИКО) · ВКР 2025</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Сайдбар ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Данные")

    ctx_file = st.file_uploader(
        "Контекстные данные ОО (.xlsx)",
        type=["xlsx"],
        help="Столбцы: Логин ОО, Расположение, Наименование НП, Размер НП, Коррекционная, Кол-во обучающихся"
    )
    vpr_file = st.file_uploader(
        "Результаты ВПР (.csv)",
        type=["csv"],
        help="Столбцы: Класс, Код_предмета, ЛогинОО, Код_ученика, Балл, Отметка, Year"
    )

    st.markdown("### ⚙️ Параметры")

    sample_size = st.slider(
        "Размер выборки (ОО)",
        min_value=50, max_value=1000, value=300, step=10,
        help="Количество образовательных организаций в выборке"
    )

    n_srs_runs = st.slider(
        "Прогонов SRS (бейзлайн)",
        min_value=10, max_value=100, value=50, step=5,
        help="Больше прогонов = надёжнее оценка стабильности случайной выборки"
    )

    seed = st.number_input(
        "Seed воспроизводимости",
        min_value=0, max_value=9999, value=42,
        help="Фиксирует случайность для воспроизводимости результатов"
    )

    st.markdown("### 🔬 Методы")
    st.markdown("""
<div style="font-size:0.75rem; color:#8b949e; line-height:1.8">
<span class="method-badge badge-srs">SRS</span> Простая случайная<br>
<span class="method-badge badge-strat">STRAT</span> Стратифицированная<br>
<span class="method-badge badge-kcenter">K-CTR</span> k-center greedy<br>
<span class="method-badge badge-facility">FAC-LOC</span> Facility Location<br>
<span class="method-badge badge-herding">KH</span> Kernel Herding
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶ Запустить эксперимент", type="primary")

# ── Вспомогательные функции ──────────────────────────────────────────────────

def color_score(val):
    """Цвет метрики по качеству (меньше = лучше)."""
    if val < 0.02:
        return "good"
    elif val < 0.05:
        return "warn"
    return "bad"


def method_badge(name):
    mapping = {
        'SRS': ('badge-srs', 'SRS'),
        'Стратифицированная': ('badge-strat', 'STRAT'),
        'k-center': ('badge-kcenter', 'K-CTR'),
        'Facility': ('badge-facility', 'FAC-LOC'),
        'Kernel': ('badge-herding', 'KH'),
    }
    for key, (cls, label) in mapping.items():
        if key in name:
            return f'<span class="method-badge {cls}">{label}</span>'
    return f'<span class="method-badge badge-srs">{name[:6]}</span>'


@st.cache_data(show_spinner=False)
def run_cached(ctx_bytes, vpr_bytes, sample_size, seed, n_srs_runs):
    """Кэшируем результаты, чтобы не пересчитывать при смене вкладки."""
    # Импортируем здесь, чтобы кэш работал без глобального импорта
    from sampling_system import (
        load_context_data, load_vpr_data, build_school_features,
        prepare_feature_matrix, sample_srs, sample_stratified,
        sample_kmedoids, sample_facility_location, sample_kernel_herding,
        validate_sample, compute_composite_score
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx_path = os.path.join(tmpdir, "ctx.xlsx")
        vpr_path = os.path.join(tmpdir, "vpr.csv")
        with open(ctx_path, "wb") as f:
            f.write(ctx_bytes)
        with open(vpr_path, "wb") as f:
            f.write(vpr_bytes)

        ctx = load_context_data(ctx_path)
        vpr = load_vpr_data(vpr_path)
        schools = build_school_features(ctx, vpr)
        X = prepare_feature_matrix(schools)
        n = min(sample_size, len(schools) // 5)

        # SRS многократный
        srs_scores = []
        srs_all_results = []
        for run in range(n_srs_runs):
            idx = sample_srs(schools, n, seed=seed + run)
            res = validate_sample(set(schools.iloc[idx]['login'].values),
                                  schools, vpr, X, idx)
            res['composite_score'] = compute_composite_score(res)
            srs_all_results.append(res)
            srs_scores.append(res['composite_score'])

        srs_mean = float(np.mean(srs_scores))
        srs_std = float(np.std(srs_scores))
        srs_best_run = int(np.argmin(srs_scores))
        srs_best_res = dict(srs_all_results[srs_best_run])
        srs_best_res['time_sec'] = 0.0

        metric_keys = ['rel_error_mean_score', 'rel_error_mean_mark', 'ks_stat',
                       'mmd', 'cramers_v', 'max_mark_dev', 'composite_score']
        srs_avg = {}
        for key in metric_keys:
            vals = [r.get(key, 0) for r in srs_all_results]
            srs_avg[key] = float(np.mean(vals))
            srs_avg[f'{key}_std'] = float(np.std(vals))

        # Детерминированные методы
        det_methods = {
            '2. Стратифицированная': lambda: sample_stratified(schools, n, seed),
            '3. k-center greedy':    lambda: sample_kmedoids(X, n, seed=seed),
            '4. Facility location':  lambda: sample_facility_location(X, n),
            '5. Kernel herding':     lambda: sample_kernel_herding(X, n),
        }

        all_results = {f'1. SRS (лучший из {n_srs_runs})': srs_best_res}

        for name, fn in det_methods.items():
            t0 = time.time()
            indices = fn()
            elapsed = time.time() - t0
            res = validate_sample(set(schools.iloc[indices]['login'].values),
                                  schools, vpr, X, indices)
            res['time_sec'] = elapsed
            res['composite_score'] = compute_composite_score(res)
            all_results[name] = res

        # Лучший метод — получаем ОО
        best_name = min(all_results, key=lambda k: all_results[k].get('composite_score', 99))

        if 'SRS' in best_name:
            best_idx = sample_srs(schools, n, seed=seed + srs_best_run)
        else:
            fn_map = {
                '2. Стратифицированная': lambda: sample_stratified(schools, n, seed),
                '3. k-center greedy':    lambda: sample_kmedoids(X, n, seed=seed),
                '4. Facility location':  lambda: sample_facility_location(X, n),
                '5. Kernel herding':     lambda: sample_kernel_herding(X, n),
            }
            best_idx = fn_map[best_name]()

        best_schools = schools.iloc[best_idx][
            ['login', 'region', 'location_type', 'locality_size',
             'n_students', 'mean_score', 'mean_mark']
        ].copy().reset_index(drop=True)

        # Данные ВПР для лучшего метода (для графиков)
        best_logins = set(schools.iloc[best_idx]['login'].values)
        best_vpr = vpr[vpr['login'].isin(best_logins)].copy()

        return {
            'all_results': all_results,
            'best_name': best_name,
            'best_schools': best_schools,
            'best_vpr': best_vpr,
            'vpr': vpr,
            'schools': schools,
            'srs_avg': srs_avg,
            'srs_mean': srs_mean,
            'srs_std': srs_std,
            'srs_scores': srs_scores,
            'n_actual': n,
            'N_total': len(schools),
        }


# ── Главное содержимое ────────────────────────────────────────────────────────

if not run_btn:
    # Приветственный экран
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class="metric-card">
    <div class="metric-label">Методы отбора</div>
    <div class="metric-value" style="font-size:2rem">5</div>
    <div class="metric-sub">SRS · Стратификация · k-center · Facility Location · Kernel Herding</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class="metric-card">
    <div class="metric-label">Метрик валидации</div>
    <div class="metric-value" style="font-size:2rem">6</div>
    <div class="metric-sub">χ² · KS · MMD · Cramér V · Отн. ошибка · MaxΔ</div>
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div class="metric-card">
    <div class="metric-label">Гарантия алгоритма</div>
    <div class="metric-value" style="font-size:2rem">63%</div>
    <div class="metric-sub">Facility Location: (1−1/e) от оптимума</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
<strong>Как использовать:</strong><br>
1. Загрузите файл контекстных данных ОО (.xlsx) и файл с результатами ВПР (.csv) в боковой панели.<br>
2. Настройте параметры: размер выборки, количество прогонов SRS и seed.<br>
3. Нажмите <strong>▶ Запустить эксперимент</strong> — система сравнит все методы и выберет лучший.<br>
4. Скачайте итоговую выборку и метрики в виде CSV.
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Описание методов
    st.markdown('<div class="section-title">Описание методов</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    methods_info = [
        ("SRS", "badge-srs", "Простая случайная выборка", "Базовый стохастический метод. Запускается 50 раз для оценки стабильности."),
        ("STRAT", "badge-strat", "Стратифицированная", "Пропорциональное размещение по стратам регион × тип расположения."),
        ("K-CTR", "badge-kcenter", "k-center greedy", "Farthest-first traversal. Гарантия 2-приближения к оптимуму."),
        ("FAC-LOC", "badge-facility", "Facility Location", "Субмодулярная оптимизация. Гарантия ≥63.2% от оптимума."),
        ("KH", "badge-herding", "Kernel Herding", "Минимизация MMD. Сходимость O(1/T) vs O(1/√T) для SRS."),
    ]
    for col, (label, cls, title, desc) in zip([m1, m2, m3, m4, m5], methods_info):
        with col:
            st.markdown(f"""
<div class="metric-card" style="text-align:left; height:160px">
    <span class="method-badge {cls}">{label}</span>
    <div style="font-size:0.8rem; font-weight:600; margin: 0.5rem 0 0.3rem 0; color:#e6edf3">{title}</div>
    <div style="font-size:0.72rem; color:#8b949e; line-height:1.5">{desc}</div>
</div>""", unsafe_allow_html=True)

elif not ctx_file or not vpr_file:
    st.warning("⚠️ Загрузите оба файла данных в боковой панели перед запуском.")

else:
    # ── ЗАПУСК ────────────────────────────────────────────────────────────────
    with st.spinner("Выполняется эксперимент... Это может занять несколько минут для больших данных."):
        ctx_bytes = ctx_file.read()
        vpr_bytes = vpr_file.read()

        try:
            data = run_cached(ctx_bytes, vpr_bytes, sample_size, seed, n_srs_runs)
        except Exception as e:
            st.error(f"❌ Ошибка при выполнении: {e}")
            st.stop()

    all_results = data['all_results']
    best_name   = data['best_name']
    best_schools = data['best_schools']
    best_vpr    = data['best_vpr']
    vpr         = data['vpr']
    schools     = data['schools']
    srs_avg     = data['srs_avg']
    srs_mean    = data['srs_mean']
    srs_std     = data['srs_std']
    srs_scores  = data['srs_scores']
    n_actual    = data['n_actual']
    N_total     = data['N_total']

    best_res = all_results[best_name]

    # ── Победитель ────────────────────────────────────────────────────────────
    ml_methods = {k: v for k, v in all_results.items() if 'SRS' not in k}
    best_ml_name  = min(ml_methods, key=lambda k: ml_methods[k].get('composite_score', 99))
    best_ml_score = ml_methods[best_ml_name]['composite_score']
    improvement   = (srs_mean - best_ml_score) / srs_mean * 100
    srs_wins      = sum(1 for s in srs_scores if s < best_ml_score)

    st.markdown(f"""
<div class="winner-badge">
    <h3>🏆 Рекомендованный метод: {best_name}</h3>
    <p>Составной балл: <strong>{best_res['composite_score']:.4f}</strong> &nbsp;·&nbsp;
       Улучшение vs среднее SRS: <strong>{improvement:+.1f}%</strong> &nbsp;·&nbsp;
       SRS побеждает в <strong>{srs_wins}/{n_srs_runs}</strong> прогонах ({srs_wins/n_srs_runs*100:.0f}%)
    </p>
</div>
""", unsafe_allow_html=True)

    # ── Ключевые метрики (карточки) ───────────────────────────────────────────
    st.markdown('<div class="section-title">Ключевые показатели лучшего метода</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "Ош. среднего балла", best_res['rel_error_mean_score'],
         f"{best_res['mean_score_pop']:.2f} → {best_res['mean_score_sample']:.2f}"),
        (c2, "Ош. средней отметки", best_res['rel_error_mean_mark'],
         f"{best_res['mean_mark_pop']:.2f} → {best_res['mean_mark_sample']:.2f}"),
        (c3, "KS-статистика", best_res['ks_stat'],
         f"p = {best_res['ks_pvalue']:.3f}"),
        (c4, "MMD", best_res['mmd'], "ядерное расстояние"),
        (c5, "Cramér V", best_res.get('cramers_v', 0), "отметки"),
        (c6, "Max Δ долей", best_res.get('max_mark_dev', 0),
         f"{best_res.get('max_mark_dev', 0)*100:.2f}%"),
    ]
    for col, label, val, sub in cards:
        with col:
            cls = color_score(val)
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value {cls}">{val:.4f}</div>
    <div class="metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Вкладки ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Сравнение методов",
        "📈 Графики распределений",
        "🏫 Выбранные ОО",
        "🗺 Анализ по регионам",
        "⬇️ Скачать"
    ])

    # ── TAB 1: Сравнительная таблица ─────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Сводная таблица метрик</div>', unsafe_allow_html=True)

        rows = []
        for name, res in sorted(all_results.items(),
                                 key=lambda x: x[1].get('composite_score', 99)):
            if 'error' in res:
                continue
            rows.append({
                'Метод': name,
                'Ош. ȳ балла': res['rel_error_mean_score'],
                'Ош. m̄ отметки': res['rel_error_mean_mark'],
                'KS': res['ks_stat'],
                'MMD': res['mmd'],
                'Cramér V': res.get('cramers_v', 0),
                'Max Δ%': res.get('max_mark_dev', 0) * 100,
                'Балл ↓': res['composite_score'],
                'Время, с': res['time_sec'],
            })

        df_table = pd.DataFrame(rows)

        # Подсветка лучшей строки
        def highlight_best(row):
            if row['Метод'] == best_name:
                return ['background-color: rgba(126,231,135,0.08); '
                        'border-left: 3px solid #7ee787'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_table.style
                .apply(highlight_best, axis=1)
                .format({
                    'Ош. ȳ балла': '{:.4f}',
                    'Ош. m̄ отметки': '{:.4f}',
                    'KS': '{:.4f}',
                    'MMD': '{:.4f}',
                    'Cramér V': '{:.4f}',
                    'Max Δ%': '{:.2f}%',
                    'Балл ↓': '{:.4f}',
                    'Время, с': '{:.1f}',
                })
                .background_gradient(subset=['Балл ↓'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=280
        )

        # Бейзлайн SRS
        st.markdown("**Бейзлайн SRS (среднее по прогонам):**")
        bline_cols = st.columns(7)
        bline_items = [
            ('Ош. ȳ', srs_avg['rel_error_mean_score'], srs_avg['rel_error_mean_score_std']),
            ('Ош. m̄', srs_avg['rel_error_mean_mark'], srs_avg['rel_error_mean_mark_std']),
            ('KS', srs_avg['ks_stat'], srs_avg['ks_stat_std']),
            ('MMD', srs_avg['mmd'], srs_avg['mmd_std']),
            ('Cramér V', srs_avg['cramers_v'], srs_avg['cramers_v_std']),
            ('Max Δ', srs_avg['max_mark_dev'], srs_avg['max_mark_dev_std']),
            ('Балл', srs_avg['composite_score'], srs_avg['composite_score_std']),
        ]
        for col, (label, mean, std) in zip(bline_cols, bline_items):
            col.metric(label, f"{mean:.4f}", f"±{std:.4f}")

        # Scatter plot: композитный балл vs время
        st.markdown('<div class="section-title" style="margin-top:1.5rem">Балл vs Время вычисления</div>',
                    unsafe_allow_html=True)
        colors_map = {
            '1. SRS': '#8b949e',
            '2. Стратифицированная': '#2ea4ff',
            '3. k-center': '#d2a8ff',
            '4. Facility': '#7ee787',
            '5. Kernel': '#ffa657',
        }

        fig_scatter = go.Figure()
        for name, res in all_results.items():
            if 'error' in res:
                continue
            color = next((v for k, v in colors_map.items() if k[:3] in name[:3]), '#8b949e')
            is_best = name == best_name
            fig_scatter.add_trace(go.Scatter(
                x=[res['time_sec']],
                y=[res['composite_score']],
                mode='markers+text',
                name=name,
                text=[name.split('.')[-1].strip()],
                textposition='top center',
                marker=dict(
                    size=20 if is_best else 12,
                    color=color,
                    symbol='star' if is_best else 'circle',
                    line=dict(width=2 if is_best else 0, color='white')
                )
            ))

        fig_scatter.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,1)',
            height=320,
            xaxis_title='Время, с',
            yaxis_title='Составной балл (↓ лучше)',
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d'),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── TAB 2: Графики ────────────────────────────────────────────────────────
    with tab2:
        col_l, col_r = st.columns(2)

        # Распределение отметок: популяция vs лучшая выборка
        with col_l:
            st.markdown('<div class="section-title">Распределение отметок</div>',
                        unsafe_allow_html=True)
            pop_marks  = vpr['mark'].value_counts(normalize=True).sort_index()
            samp_marks = best_vpr['mark'].value_counts(normalize=True).sort_index()
            all_marks  = sorted(set(pop_marks.index) | set(samp_marks.index))

            fig_marks = go.Figure()
            fig_marks.add_trace(go.Bar(
                x=[f'Отметка {m}' for m in all_marks],
                y=[pop_marks.get(m, 0) for m in all_marks],
                name='Ген. совокупность',
                marker_color='#2ea4ff',
                opacity=0.8
            ))
            fig_marks.add_trace(go.Bar(
                x=[f'Отметка {m}' for m in all_marks],
                y=[samp_marks.get(m, 0) for m in all_marks],
                name=f'Выборка ({best_name.split(".")[-1].strip()})',
                marker_color='#7ee787',
                opacity=0.8
            ))
            fig_marks.update_layout(
                barmode='group',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,1)',
                height=320,
                yaxis_title='Доля',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                margin=dict(l=40, r=10, t=30, b=40),
                xaxis=dict(gridcolor='#21262d'),
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_marks, use_container_width=True)

        # ЭФР баллов
        with col_r:
            st.markdown('<div class="section-title">ЭФР распределения баллов (KS-тест)</div>',
                        unsafe_allow_html=True)
            pop_scores_sorted  = np.sort(vpr['score'].dropna().values)
            samp_scores_sorted = np.sort(best_vpr['score'].dropna().values)
            pop_ecdf  = np.arange(1, len(pop_scores_sorted)+1) / len(pop_scores_sorted)
            samp_ecdf = np.arange(1, len(samp_scores_sorted)+1) / len(samp_scores_sorted)

            # Децимация для скорости
            step_p = max(1, len(pop_scores_sorted) // 3000)
            step_s = max(1, len(samp_scores_sorted) // 3000)

            fig_ecdf = go.Figure()
            fig_ecdf.add_trace(go.Scatter(
                x=pop_scores_sorted[::step_p], y=pop_ecdf[::step_p],
                name='Ген. совокупность',
                line=dict(color='#2ea4ff', width=2)
            ))
            fig_ecdf.add_trace(go.Scatter(
                x=samp_scores_sorted[::step_s], y=samp_ecdf[::step_s],
                name=f'Выборка ({best_name.split(".")[-1].strip()})',
                line=dict(color='#7ee787', width=2, dash='dash')
            ))
            ks_stat = best_res['ks_stat']
            fig_ecdf.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,1)',
                height=320,
                xaxis_title='Балл',
                yaxis_title='ЭФР',
                title=dict(text=f'KS = {ks_stat:.4f}', font=dict(size=12)),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                margin=dict(l=40, r=10, t=40, b=40),
                xaxis=dict(gridcolor='#21262d'),
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_ecdf, use_container_width=True)

        # Сравнение составных баллов всех методов
        st.markdown('<div class="section-title">Составные баллы всех методов</div>',
                    unsafe_allow_html=True)
        names_sorted  = sorted(all_results, key=lambda k: all_results[k].get('composite_score', 99))
        scores_sorted = [all_results[k]['composite_score'] for k in names_sorted]
        short_names   = [n.split('.')[-1].strip()[:25] for n in names_sorted]

        bar_colors = []
        for n in names_sorted:
            if n == best_name:
                bar_colors.append('#7ee787')
            elif 'SRS' in n:
                bar_colors.append('#8b949e')
            elif 'Страт' in n:
                bar_colors.append('#2ea4ff')
            elif 'k-center' in n:
                bar_colors.append('#d2a8ff')
            elif 'Facility' in n:
                bar_colors.append('#ffa657')
            else:
                bar_colors.append('#58a6ff')

        fig_bar = go.Figure(go.Bar(
            x=short_names,
            y=scores_sorted,
            marker_color=bar_colors,
            text=[f'{s:.4f}' for s in scores_sorted],
            textposition='outside',
            textfont=dict(size=10, family='JetBrains Mono'),
        ))
        # SRS бейзлайн
        fig_bar.add_hline(y=srs_mean, line_dash='dash', line_color='#8b949e',
                          annotation_text=f'SRS среднее ({srs_mean:.4f})',
                          annotation_position='top right')

        fig_bar.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,1)',
            height=280,
            yaxis_title='Составной балл (↓ лучше)',
            showlegend=False,
            margin=dict(l=40, r=10, t=30, b=40),
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d'),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Распределение SRS баллов (violin)
        st.markdown('<div class="section-title">Стабильность SRS (распределение составных баллов)</div>',
                    unsafe_allow_html=True)
        fig_violin = go.Figure()
        fig_violin.add_trace(go.Violin(
            y=srs_scores,
            box_visible=True,
            meanline_visible=True,
            fillcolor='rgba(139,148,158,0.2)',
            line_color='#8b949e',
            name='SRS',
        ))
        fig_violin.add_hline(y=best_ml_score, line_dash='dash', line_color='#7ee787',
                             annotation_text=f'Лучший ML ({best_ml_name.split(".")[-1].strip()})',
                             annotation_position='top right')
        fig_violin.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,1)',
            height=250,
            yaxis_title='Составной балл',
            showlegend=False,
            margin=dict(l=40, r=10, t=20, b=40),
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d'),
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── TAB 3: Выбранные ОО ───────────────────────────────────────────────────
    with tab3:
        st.markdown(f'<div class="section-title">Выбранные образовательные организации ({len(best_schools)} ОО)</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ОО в выборке", n_actual)
        c2.metric("Всего ОО", N_total)
        c3.metric("Доля отбора", f"{n_actual/N_total*100:.1f}%")
        c4.metric("Регионов охвачено", best_schools['region'].nunique())

        st.dataframe(
            best_schools.style.format({
                'n_students': '{:.0f}',
                'mean_score': '{:.2f}',
                'mean_mark': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )

        # Гистограмма среднего балла по выборке
        st.markdown('<div class="section-title">Распределение среднего балла по ОО в выборке vs совокупность</div>',
                    unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=schools['mean_score'],
            nbinsx=50,
            name='Ген. совокупность',
            marker_color='rgba(46,164,255,0.4)',
            opacity=0.8,
            histnorm='probability density'
        ))
        fig_hist.add_trace(go.Histogram(
            x=best_schools['mean_score'],
            nbinsx=30,
            name='Выборка',
            marker_color='rgba(126,231,135,0.6)',
            opacity=0.8,
            histnorm='probability density'
        ))
        fig_hist.update_layout(
            barmode='overlay',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,1)',
            height=270,
            xaxis_title='Средний балл ОО',
            yaxis_title='Плотность',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            margin=dict(l=40, r=10, t=30, b=40),
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d'),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── TAB 4: Анализ по регионам ─────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Сравнение долей регионов: генеральная совокупность vs выборка</div>',
                    unsafe_allow_html=True)

        pop_regions  = schools['region'].value_counts(normalize=True).sort_index()
        samp_regions = best_schools['region'].value_counts(normalize=True).sort_index()
        all_regions  = sorted(set(pop_regions.index) | set(samp_regions.index))

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(
            x=all_regions,
            y=[pop_regions.get(r, 0)*100 for r in all_regions],
            name='Ген. совокупность',
            marker_color='rgba(46,164,255,0.6)',
        ))
        fig_reg.add_trace(go.Bar(
            x=all_regions,
            y=[samp_regions.get(r, 0)*100 for r in all_regions],
            name='Выборка',
            marker_color='rgba(126,231,135,0.7)',
        ))
        fig_reg.update_layout(
            barmode='group',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,1)',
            height=350,
            xaxis_title='Регион',
            yaxis_title='Доля, %',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            margin=dict(l=40, r=10, t=30, b=60),
            xaxis=dict(gridcolor='#21262d', tickangle=-45),
            yaxis=dict(gridcolor='#21262d'),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # Тип расположения
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-title">Тип расположения</div>', unsafe_allow_html=True)
            pop_loc  = schools['location_code'].value_counts(normalize=True).sort_index()
            samp_loc = best_schools['location_type'].map(
                lambda x: int(str(x)[0]) if pd.notna(x) else np.nan
            ).value_counts(normalize=True).sort_index()

            fig_loc = go.Figure()
            loc_labels = {1: 'Город', 2: 'Пгт', 3: 'Райцентр', 4: 'Село'}
            cats = sorted(set(pop_loc.index) | set(samp_loc.index))
            fig_loc.add_trace(go.Bar(
                x=[loc_labels.get(c, str(c)) for c in cats],
                y=[pop_loc.get(c, 0) for c in cats],
                name='Ген. совокупность', marker_color='rgba(46,164,255,0.6)'))
            fig_loc.add_trace(go.Bar(
                x=[loc_labels.get(c, str(c)) for c in cats],
                y=[samp_loc.get(c, 0) for c in cats],
                name='Выборка', marker_color='rgba(126,231,135,0.7)'))
            fig_loc.update_layout(
                barmode='group', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)',
                height=270, showlegend=False,
                margin=dict(l=40, r=10, t=10, b=40),
                xaxis=dict(gridcolor='#21262d'),
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_loc, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-title">Размер населённого пункта</div>',
                        unsafe_allow_html=True)
            pop_sz  = schools['size_code'].value_counts(normalize=True).sort_index()
            samp_sz = best_schools['locality_size'].map(
                lambda x: int(str(x)[0]) if pd.notna(x) else np.nan
            ).value_counts(normalize=True).sort_index()

            cats_sz = sorted(set(pop_sz.index) | set(samp_sz.index))
            fig_sz = go.Figure()
            fig_sz.add_trace(go.Bar(
                x=[str(c) for c in cats_sz],
                y=[pop_sz.get(c, 0) for c in cats_sz],
                name='Ген. совокупность', marker_color='rgba(46,164,255,0.6)'))
            fig_sz.add_trace(go.Bar(
                x=[str(c) for c in cats_sz],
                y=[samp_sz.get(c, 0) for c in cats_sz],
                name='Выборка', marker_color='rgba(126,231,135,0.7)'))
            fig_sz.update_layout(
                barmode='group', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)',
                height=270, showlegend=False,
                margin=dict(l=40, r=10, t=10, b=40),
                xaxis=dict(gridcolor='#21262d', title='Код размера НП'),
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_sz, use_container_width=True)

    # ── TAB 5: Скачать ────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-title">Экспорт результатов</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # Лучшая выборка
        with col1:
            st.markdown("**📋 Лучшая выборка ОО**")
            st.markdown(f"<small style='color:#8b949e'>Метод: {best_name} · {len(best_schools)} ОО</small>",
                        unsafe_allow_html=True)
            csv_sample = best_schools.to_csv(index=False, encoding='utf-8').encode('utf-8')
            st.download_button(
                label="⬇️ Скачать best_sample.csv",
                data=csv_sample,
                file_name="best_sample.csv",
                mime="text/csv",
            )

        # Метрики всех методов
        with col2:
            st.markdown("**📊 Метрики всех методов**")
            st.markdown("<small style='color:#8b949e'>CSV с полным сравнением методов</small>",
                        unsafe_allow_html=True)
            metric_rows = []
            for name, res in all_results.items():
                if 'error' in res:
                    continue
                metric_rows.append({
                    'Метод': name,
                    'rel_error_mean_score': res.get('rel_error_mean_score'),
                    'rel_error_mean_mark': res.get('rel_error_mean_mark'),
                    'ks_stat': res.get('ks_stat'),
                    'ks_pvalue': res.get('ks_pvalue'),
                    'mmd': res.get('mmd'),
                    'cramers_v': res.get('cramers_v'),
                    'max_mark_dev': res.get('max_mark_dev'),
                    'composite_score': res.get('composite_score'),
                    'time_sec': res.get('time_sec'),
                })
            # Добавляем SRS среднее
            metric_rows.append({
                'Метод': f'SRS среднее ({n_srs_runs} прогонов)',
                'rel_error_mean_score': srs_avg['rel_error_mean_score'],
                'rel_error_mean_mark': srs_avg['rel_error_mean_mark'],
                'ks_stat': srs_avg['ks_stat'],
                'mmd': srs_avg['mmd'],
                'cramers_v': srs_avg['cramers_v'],
                'max_mark_dev': srs_avg['max_mark_dev'],
                'composite_score': srs_avg['composite_score'],
            })
            csv_metrics = pd.DataFrame(metric_rows).to_csv(index=False, encoding='utf-8').encode('utf-8')
            st.download_button(
                label="⬇️ Скачать sampling_metrics.csv",
                data=csv_metrics,
                file_name="sampling_metrics.csv",
                mime="text/csv",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Параметры запуска</div>', unsafe_allow_html=True)
        st.json({
            'sample_size_requested': sample_size,
            'sample_size_actual': n_actual,
            'N_total': N_total,
            'fraction': round(n_actual / N_total, 4),
            'seed': seed,
            'n_srs_runs': n_srs_runs,
            'best_method': best_name,
            'improvement_vs_srs_mean_pct': round(improvement, 2),
            'srs_wins': f'{srs_wins}/{n_srs_runs}',
        })
