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
import tempfile, os, time, warnings

warnings.filterwarnings("ignore")

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
    --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
    --border: #30363d; --accent: #2ea4ff; --accent2: #7ee787;
    --accent3: #f78166; --accent4: #d2a8ff;
    --text: #e6edf3; --muted: #8b949e;
}

html, body, [class*="css"] {
    font-family: 'Geologica', sans-serif;
    background-color: var(--bg); color: var(--text);
}
.main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

.app-header {
    display: flex; align-items: center; gap: 1rem;
    padding: 1.5rem 2rem; margin: -1.5rem -2rem 2rem -2rem;
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border); position: relative; overflow: hidden;
}
.app-header::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(46,164,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(126,231,135,0.05) 0%, transparent 60%);
}
.header-icon { font-size: 2.5rem; position: relative; z-index: 1; }
.header-text { position: relative; z-index: 1; }
.header-text h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
.header-text p { margin: 0.2rem 0 0 0; font-size: 0.8rem; color: var(--muted); }

.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 600; }
.metric-value.good { color: var(--accent2); }
.metric-value.warn { color: #e3b341; }
.metric-value.bad { color: var(--accent3); }
.metric-sub { font-size: 0.68rem; color: var(--muted); margin-top: 0.2rem; }

.method-badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.badge-srs { background: rgba(139,148,158,0.2); color: #8b949e; }
.badge-strat { background: rgba(46,164,255,0.15); color: #2ea4ff; }
.badge-kcenter { background: rgba(210,168,255,0.15); color: #d2a8ff; }
.badge-facility { background: rgba(126,231,135,0.15); color: #7ee787; }
.badge-herding { background: rgba(255,166,87,0.15); color: #ffa657; }

.winner-badge {
    background: linear-gradient(135deg, rgba(126,231,135,0.2), rgba(46,164,255,0.15));
    border: 1px solid rgba(126,231,135,0.4); border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0;
}
.winner-badge h3 { margin: 0 0 0.3rem 0; color: var(--accent2); font-size: 1rem; }
.winner-badge p { margin: 0; color: var(--muted); font-size: 0.85rem; }

.section-title {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}

.info-box {
    background: rgba(46,164,255,0.08); border: 1px solid rgba(46,164,255,0.25);
    border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.82rem; color: var(--muted); line-height: 1.6;
}
.info-box strong { color: var(--accent); }

[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown h3 { color: var(--accent); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }

.stButton > button {
    background: linear-gradient(135deg, #2ea4ff, #1a7fd4) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Geologica', sans-serif !important; font-weight: 600 !important;
    font-size: 0.9rem !important; padding: 0.6rem 1.5rem !important; width: 100% !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(46,164,255,0.3) !important; }

div[data-testid="metric-container"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem; }
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
    ctx_file = st.file_uploader("Контекстные данные ОО (.xlsx)", type=["xlsx"])
    vpr_file = st.file_uploader("Результаты ВПР (.csv)", type=["csv"])

    st.markdown("### ⚙️ Параметры")
    sample_size = st.number_input("Размер выборки (ОО)", min_value=50, value=300, step=10)
    n_srs_runs = st.number_input("Прогонов SRS / Стратиф.", min_value=5, value=50, step=5)
    seed = st.number_input("Seed", 0, 9999, 42)

    st.markdown("### 🔬 Методы")
    st.markdown("""
<div style="font-size:0.75rem; color:#8b949e; line-height:1.8">
<span class="method-badge badge-srs">SRS</span> Простая случайная<br>
<span class="method-badge badge-strat">STRAT</span> Стратифицированная<br>
<span class="method-badge badge-kcenter">K-CTR</span> k-center greedy<br>
<span class="method-badge badge-facility">FAC-LOC</span> Facility Location<br>
<span class="method-badge badge-herding">KH</span> Kernel Herding
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    run_btn = st.button("▶ Запустить эксперимент", type="primary")


def color_score(val):
    if val < 0.02: return "good"
    elif val < 0.05: return "warn"
    return "bad"


# ── Кэшированный запуск ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_cached(ctx_bytes, vpr_bytes, sample_size, seed, n_srs_runs):
    from sampling_system import (
        load_context_data, load_vpr_data, build_school_features,
        prepare_feature_matrix, sample_srs, sample_stratified,
        sample_kmedoids, sample_facility_location, sample_kernel_herding,
        validate_sample, validate_sample_fast, compute_composite_score,
        build_vpr_index, precompute_pop_stats,
        VALID_MARKS, KIM_MAX_SCORES, SUBJECT_NAMES, ALL_METRIC_KEYS,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx_path = os.path.join(tmpdir, "ctx.xlsx")
        vpr_path = os.path.join(tmpdir, "vpr.csv")
        with open(ctx_path, "wb") as f: f.write(ctx_bytes)
        with open(vpr_path, "wb") as f: f.write(vpr_bytes)

        ctx = load_context_data(ctx_path)
        vpr = load_vpr_data(vpr_path)
        schools = build_school_features(ctx, vpr)
        X = prepare_feature_matrix(schools)
        n = min(sample_size, len(schools) // 5)

        # ─── Предвычисления (один раз) ──────────────────────────────────
        vpr_index = build_vpr_index(vpr)
        pop_stats = precompute_pop_stats(vpr, X)

        # ─── Вспомогательная: прогон стохастического метода N раз ────────
        def _run_stochastic(sampler_fn, n_runs):
            scores_list, all_runs = [], []
            for run in range(n_runs):
                idx = sampler_fn(seed + run)
                res = validate_sample_fast(
                    idx, schools, vpr, X, vpr_index, pop_stats,
                    compute_slices=False)
                res['composite_score'] = compute_composite_score(res)
                all_runs.append(res)
                scores_list.append(res['composite_score'])

            avg_raw = {}
            for key in ALL_METRIC_KEYS:
                vals = [r.get(key, 0) for r in all_runs]
                avg_raw[key] = float(np.mean(vals))
                avg_raw[f'{key}_std'] = float(np.std(vals))

            avg_entry = {k: avg_raw[k] for k in ALL_METRIC_KEYS}
            avg_entry['time_sec'] = 0.0
            avg_entry['composite_score'] = avg_raw['composite_score']
            avg_entry['slices'] = []
            return avg_entry, scores_list, all_runs, avg_raw

        # ─── SRS: N прогонов ────────────────────────────────────────────
        srs_avg_entry, srs_scores, srs_all, srs_avg = _run_stochastic(
            sampler_fn=lambda s: sample_srs(schools, n, seed=s),
            n_runs=n_srs_runs,
        )
        srs_mean = float(np.mean(srs_scores))
        srs_std = float(np.std(srs_scores))

        # ─── Стратифицированная: N прогонов ─────────────────────────────
        strat_avg_entry, strat_scores, strat_all, strat_avg = _run_stochastic(
            sampler_fn=lambda s: sample_stratified(schools, n, seed=s),
            n_runs=n_srs_runs,
        )
        strat_mean = float(np.mean(strat_scores))
        strat_std = float(np.std(strat_scores))

        # ─── Детерминированные ML-методы ────────────────────────────────
        det_methods = {
            '3. k-center greedy':    lambda: sample_kmedoids(X, n, seed=seed),
            '4. Facility location':  lambda: sample_facility_location(X, n),
            '5. Kernel herding':     lambda: sample_kernel_herding(X, n),
        }

        all_results = {
            f'1. SRS (среднее из {n_srs_runs})': srs_avg_entry,
            f'2. Стратифицированная (среднее из {n_srs_runs})': strat_avg_entry,
        }
        all_indices = {}

        for name, fn in det_methods.items():
            t0 = time.time()
            indices = fn()
            elapsed = time.time() - t0
            res = validate_sample(
                set(schools.iloc[indices]['login'].values),
                schools, vpr, X, indices)
            res['time_sec'] = elapsed
            res['composite_score'] = compute_composite_score(res)
            all_results[name] = res
            all_indices[name] = indices

        # ─── Лучший ML-метод (детерм.) ─────────────────────────────────
        ml_only = {k: v for k, v in all_results.items()
                   if 'SRS' not in k and 'Страт' not in k}
        best_ml_name = min(ml_only, key=lambda k: ml_only[k].get('composite_score', 99))
        best_idx = all_indices[best_ml_name]

        best_schools = schools.iloc[best_idx][
            ['login', 'region', 'location_type', 'locality_size',
             'n_students', 'mean_score', 'mean_mark']
        ].copy().reset_index(drop=True)

        best_logins = set(schools.iloc[best_idx]['login'].values)
        best_vpr = vpr[vpr['login'].isin(best_logins)].copy()

        return {
            'all_results': all_results,
            'best_ml_name': best_ml_name,
            'best_schools': best_schools,
            'best_vpr': best_vpr,
            'vpr': vpr,
            'schools': schools,
            'srs_avg': srs_avg,
            'srs_mean': srs_mean,
            'srs_std': srs_std,
            'srs_scores': srs_scores,
            'strat_avg': strat_avg,
            'strat_mean': strat_mean,
            'strat_std': strat_std,
            'strat_scores': strat_scores,
            'n_actual': n,
            'N_total': len(schools),
        }


# ── Главное содержимое ────────────────────────────────────────────────────────

if not run_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card"><div class="metric-label">Методы отбора</div>
<div class="metric-value" style="font-size:2rem">5</div>
<div class="metric-sub">SRS · Стратификация · k-center · FL · KH</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-label">Метрик валидации</div>
<div class="metric-value" style="font-size:2rem">6</div>
<div class="metric-sub">χ² · KS · MMD · Cramér V · Отн. ошибка · MaxΔ</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="metric-label">Per-slice валидация</div>
<div class="metric-value" style="font-size:2rem">6</div>
<div class="metric-sub">РУ/МА × 4/5/6 класс с КИМ-ограничениями</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
<strong>Как использовать:</strong><br>
1. Загрузите контекстные данные ОО (.xlsx) и результаты ВПР (.csv).<br>
2. Настройте параметры: размер выборки, прогоны SRS/Стратиф., seed.<br>
3. Нажмите <strong>▶ Запустить эксперимент</strong>.<br>
4. Анализируйте результаты по вкладкам, включая валидацию по срезам (класс × предмет).<br>
<em>SRS и Стратифицированная усредняются по N прогонов. Отметка 0 (непройденные темы) исключена из анализа.</em>
</div>""", unsafe_allow_html=True)

elif not ctx_file or not vpr_file:
    st.warning("⚠️ Загрузите оба файла данных в боковой панели.")

else:
    with st.spinner("Выполняется эксперимент..."):
        try:
            data = run_cached(ctx_file.read(), vpr_file.read(), sample_size, seed, n_srs_runs)
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            st.stop()

    R            = data['all_results']
    best_ml      = data['best_ml_name']
    best_schools = data['best_schools']
    best_vpr     = data['best_vpr']
    vpr          = data['vpr']
    schools      = data['schools']
    srs_avg      = data['srs_avg']
    srs_mean     = data['srs_mean']
    srs_std      = data['srs_std']
    srs_scores   = data['srs_scores']
    strat_avg    = data['strat_avg']
    strat_mean   = data['strat_mean']
    strat_std    = data['strat_std']
    strat_scores = data['strat_scores']
    n_actual     = data['n_actual']
    N_total      = data['N_total']

    ml_res = R[best_ml]
    ml_score = ml_res['composite_score']
    improvement_srs = (srs_mean - ml_score) / srs_mean * 100
    improvement_strat = (strat_mean - ml_score) / strat_mean * 100
    srs_wins = sum(1 for s in srs_scores if s < ml_score)
    strat_wins = sum(1 for s in strat_scores if s < ml_score)

    # ── Победитель ────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="winner-badge">
    <h3>🏆 Рекомендованный метод: {best_ml}</h3>
    <p>Составной балл: <strong>{ml_score:.4f}</strong> &nbsp;·&nbsp;
       SRS среднее: <strong>{srs_mean:.4f} ± {srs_std:.4f}</strong> &nbsp;·&nbsp;
       Стратиф. среднее: <strong>{strat_mean:.4f} ± {strat_std:.4f}</strong><br>
       Улучшение vs SRS: <strong>{improvement_srs:+.1f}%</strong> &nbsp;·&nbsp;
       Улучшение vs Стратиф.: <strong>{improvement_strat:+.1f}%</strong> &nbsp;·&nbsp;
       SRS лучше в <strong>{srs_wins}/{n_srs_runs}</strong> &nbsp;·&nbsp;
       Стратиф. лучше в <strong>{strat_wins}/{n_srs_runs}</strong>
    </p>
</div>""", unsafe_allow_html=True)

    # ── Карточки ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Ключевые показатели лучшего метода</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "Ош. ȳ балла", ml_res['rel_error_mean_score'],
         f"{ml_res['mean_score_pop']:.2f} → {ml_res['mean_score_sample']:.2f}"),
        (c2, "Ош. m̄ отметки", ml_res['rel_error_mean_mark'],
         f"{ml_res['mean_mark_pop']:.2f} → {ml_res['mean_mark_sample']:.2f}"),
        (c3, "KS", ml_res['ks_stat'], f"p = {ml_res['ks_pvalue']:.3f}"),
        (c4, "MMD", ml_res['mmd'], "ядерное расстояние"),
        (c5, "Cramér V", ml_res.get('cramers_v', 0), "отметки 2–5"),
        (c6, "Max Δ%", ml_res.get('max_mark_dev', 0),
         f"{ml_res.get('max_mark_dev', 0)*100:.2f}%"),
    ]
    for col, label, val, sub in cards:
        with col:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
<div class="metric-value {color_score(val)}">{val:.4f}</div>
<div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Вкладки ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Сравнение методов",
        "📈 Графики распределений",
        "📐 Валидация по срезам",
        "🏫 Выбранные ОО",
        "🗺 Регионы",
        "⬇️ Скачать"
    ])

    # ── TAB 1: Таблица ───────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Сводная таблица метрик</div>', unsafe_allow_html=True)
        rows = []
        for name, res in sorted(R.items(), key=lambda x: x[1].get('composite_score', 99)):
            if 'error' in res: continue
            rows.append({
                'Метод': name,
                'Ош. ȳ балла': res.get('rel_error_mean_score', 0),
                'Ош. m̄ отметки': res.get('rel_error_mean_mark', 0),
                'KS': res.get('ks_stat', 0),
                'MMD': res.get('mmd', 0),
                'Cramér V': res.get('cramers_v', 0),
                'Max Δ%': res.get('max_mark_dev', 0) * 100,
                'Балл ↓': res.get('composite_score', 0),
                'Время, с': res.get('time_sec', 0),
            })
        df_table = pd.DataFrame(rows)

        def hl(row):
            if row['Метод'] == best_ml:
                return ['background-color: rgba(126,231,135,0.08); border-left: 3px solid #7ee787'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_table.style.apply(hl, axis=1)
                .format({'Ош. ȳ балла': '{:.4f}', 'Ош. m̄ отметки': '{:.4f}',
                         'KS': '{:.4f}', 'MMD': '{:.4f}', 'Cramér V': '{:.4f}',
                         'Max Δ%': '{:.2f}%', 'Балл ↓': '{:.4f}', 'Время, с': '{:.1f}'})
                .background_gradient(subset=['Балл ↓'], cmap='RdYlGn_r'),
            use_container_width=True, height=300
        )

        # ± SRS и ± Стратиф.
        st.markdown("**Разброс SRS (±σ):**")
        bc = st.columns(7)
        for col, (lbl, k) in zip(bc, [
            ('Ош.ȳ', 'rel_error_mean_score'), ('Ош.m̄', 'rel_error_mean_mark'),
            ('KS', 'ks_stat'), ('MMD', 'mmd'), ('CramérV', 'cramers_v'),
            ('MaxΔ', 'max_mark_dev'), ('Балл', 'composite_score')]):
            col.metric(lbl, f"±{srs_avg.get(f'{k}_std', 0):.4f}")

        st.markdown("**Разброс Стратифицированной (±σ):**")
        bc2 = st.columns(7)
        for col, (lbl, k) in zip(bc2, [
            ('Ош.ȳ', 'rel_error_mean_score'), ('Ош.m̄', 'rel_error_mean_mark'),
            ('KS', 'ks_stat'), ('MMD', 'mmd'), ('CramérV', 'cramers_v'),
            ('MaxΔ', 'max_mark_dev'), ('Балл', 'composite_score')]):
            col.metric(lbl, f"±{strat_avg.get(f'{k}_std', 0):.4f}")

        # Bar chart
        st.markdown('<div class="section-title" style="margin-top:1.5rem">Составные баллы</div>', unsafe_allow_html=True)
        ns = sorted(R, key=lambda k: R[k].get('composite_score', 99))
        fig_bar = go.Figure(go.Bar(
            x=[n.split('.')[-1].strip()[:30] for n in ns],
            y=[R[n]['composite_score'] for n in ns],
            marker_color=['#7ee787' if n == best_ml else '#8b949e' if 'SRS' in n
                          else '#2ea4ff' if 'Страт' in n else '#d2a8ff' if 'k-center' in n
                          else '#ffa657' for n in ns],
            text=[f"{R[n]['composite_score']:.4f}" for n in ns],
            textposition='outside', textfont=dict(size=10, family='JetBrains Mono'),
        ))
        fig_bar.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,1)', height=280, yaxis_title='Балл (↓ лучше)',
            showlegend=False, margin=dict(l=40, r=10, t=30, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB 2: Графики ────────────────────────────────────────────────────────
    with tab2:
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="section-title">Распределение отметок (2–5)</div>', unsafe_allow_html=True)
            pm = vpr['mark'].value_counts(normalize=True).sort_index()
            sm = best_vpr['mark'].value_counts(normalize=True).sort_index()
            marks = sorted(set(pm.index) | set(sm.index))
            marks = [m for m in marks if m in [2, 3, 4, 5]]

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(x=[f'Отметка {m}' for m in marks],
                y=[pm.get(m, 0) for m in marks], name='Ген. совокупность', marker_color='#2ea4ff', opacity=0.8))
            fig_m.add_trace(go.Bar(x=[f'Отметка {m}' for m in marks],
                y=[sm.get(m, 0) for m in marks],
                name=f'Выборка ({best_ml.split(".")[-1].strip()})', marker_color='#7ee787', opacity=0.8))
            fig_m.update_layout(barmode='group', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)',
                height=320, yaxis_title='Доля',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                margin=dict(l=40, r=10, t=30, b=40))
            st.plotly_chart(fig_m, use_container_width=True)

        with cr:
            st.markdown('<div class="section-title">ECDF баллов (KS-тест)</div>', unsafe_allow_html=True)
            ps = np.sort(vpr['score'].dropna().values)
            ss = np.sort(best_vpr['score'].dropna().values)
            pe = np.arange(1, len(ps)+1) / len(ps)
            se = np.arange(1, len(ss)+1) / len(ss)
            sp, ssp = max(1, len(ps)//3000), max(1, len(ss)//3000)
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=ps[::sp], y=pe[::sp], name='Ген. совокупность', line=dict(color='#2ea4ff', width=2)))
            fig_e.add_trace(go.Scatter(x=ss[::ssp], y=se[::ssp],
                name=f'Выборка ({best_ml.split(".")[-1].strip()})', line=dict(color='#7ee787', width=2, dash='dash')))
            fig_e.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,1)', height=320, xaxis_title='Балл', yaxis_title='ECDF',
                title=dict(text=f'KS = {ml_res["ks_stat"]:.4f}', font=dict(size=12)),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                margin=dict(l=40, r=10, t=40, b=40))
            st.plotly_chart(fig_e, use_container_width=True)

        # Violin: SRS + Стратиф.
        st.markdown('<div class="section-title">Стабильность стохастических методов</div>', unsafe_allow_html=True)
        fig_v = go.Figure()
        fig_v.add_trace(go.Violin(y=srs_scores, box_visible=True, meanline_visible=True,
            fillcolor='rgba(139,148,158,0.2)', line_color='#8b949e', name='SRS', x0='SRS'))
        fig_v.add_trace(go.Violin(y=strat_scores, box_visible=True, meanline_visible=True,
            fillcolor='rgba(46,164,255,0.2)', line_color='#2ea4ff', name='Стратиф.', x0='Стратиф.'))
        fig_v.add_hline(y=ml_score, line_dash='dash', line_color='#7ee787',
            annotation_text=f'{best_ml.split(".")[-1].strip()}', annotation_position='top right')
        fig_v.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,1)', height=280, yaxis_title='Составной балл',
            showlegend=True, margin=dict(l=40, r=10, t=20, b=40))
        st.plotly_chart(fig_v, use_container_width=True)

    # ── TAB 3: Валидация по срезам ────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Валидация по срезам (класс × предмет) — лучший метод</div>',
                    unsafe_allow_html=True)

        slices = ml_res.get('slices', [])
        if not slices:
            st.info("Нет данных по срезам.")
        else:
            slice_rows = []
            for sl in slices:
                row = {
                    'Срез': f"{sl['subject_name']} {sl['grade']} кл.",
                    'КИМ макс.': sl.get('kim_max', '—'),
                    'N (поп.)': sl.get('n_pop', 0),
                    'N (выб.)': sl.get('n_sample', 0),
                }
                if 'error' in sl:
                    row['Cramér V'] = '—'
                    row['MaxΔ%'] = '—'
                    row['KS'] = '—'
                    row['Ош. ȳ'] = sl['error']
                else:
                    row['Cramér V'] = f"{sl['cramers_v']:.4f}"
                    row['MaxΔ%'] = f"{sl['max_mark_dev']*100:.2f}%"
                    row['KS'] = f"{sl['ks_stat']:.4f}"
                    pop_m = sl.get('mean_score_pop', 0)
                    samp_m = sl.get('mean_score_sample', 0)
                    row['Ош. ȳ'] = f"{abs(samp_m - pop_m) / pop_m:.4f}" if pop_m > 0 else '—'
                slice_rows.append(row)

            st.dataframe(pd.DataFrame(slice_rows), use_container_width=True, height=280)

            valid_slices = [sl for sl in slices if 'error' not in sl]

            if valid_slices:
                st.markdown('<div class="section-title" style="margin-top:1rem">Распределение отметок по срезам</div>',
                            unsafe_allow_html=True)

                n_slices = len(valid_slices)
                cols_per_row = min(3, n_slices)
                for i in range(0, n_slices, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx >= n_slices:
                            break
                        sl = valid_slices[idx]
                        label = f"{sl['subject_name']} {sl['grade']} кл. (КИМ≤{sl['kim_max']})"

                        with col:
                            fig_sl = go.Figure()
                            m_labels = [f'{m}' for m in [2, 3, 4, 5]]
                            pop_pcts = [sl.get(f'pop_pct_{m}', 0) * 100 for m in [2, 3, 4, 5]]
                            samp_pcts = [sl.get(f'sample_pct_{m}', 0) * 100 for m in [2, 3, 4, 5]]

                            fig_sl.add_trace(go.Bar(x=m_labels, y=pop_pcts,
                                name='Поп.', marker_color='#2ea4ff', opacity=0.8))
                            fig_sl.add_trace(go.Bar(x=m_labels, y=samp_pcts,
                                name='Выб.', marker_color='#7ee787', opacity=0.8))
                            fig_sl.update_layout(
                                barmode='group', template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)',
                                height=220, title=dict(text=label, font=dict(size=11)),
                                yaxis_title='%', showlegend=(idx == 0),
                                legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0),
                                margin=dict(l=30, r=5, t=35, b=30),
                                xaxis=dict(title='Отметка'),
                            )
                            st.plotly_chart(fig_sl, use_container_width=True)

                st.markdown('<div class="section-title" style="margin-top:1rem">ECDF баллов по срезам</div>',
                            unsafe_allow_html=True)

                for i in range(0, n_slices, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx >= n_slices:
                            break
                        sl = valid_slices[idx]
                        g, s = sl['grade'], sl['subject_code']
                        km = sl.get('kim_max')
                        label = f"{sl['subject_name']} {sl['grade']} кл."

                        pop_sl = vpr[(vpr['grade'] == g) & (vpr['subject_code'] == s)]
                        samp_sl = best_vpr[(best_vpr['grade'] == g) & (best_vpr['subject_code'] == s)]
                        if km:
                            pop_sl = pop_sl[pop_sl['score'] <= km]
                            samp_sl = samp_sl[samp_sl['score'] <= km]

                        with col:
                            if len(pop_sl) > 0 and len(samp_sl) > 0:
                                ps2 = np.sort(pop_sl['score'].values)
                                ss2 = np.sort(samp_sl['score'].values)
                                pe2 = np.arange(1, len(ps2)+1)/len(ps2)
                                se2 = np.arange(1, len(ss2)+1)/len(ss2)
                                stp = max(1, len(ps2)//1000)
                                sts = max(1, len(ss2)//1000)

                                fig_ec = go.Figure()
                                fig_ec.add_trace(go.Scatter(x=ps2[::stp], y=pe2[::stp],
                                    name='Поп.', line=dict(color='#2ea4ff', width=2)))
                                fig_ec.add_trace(go.Scatter(x=ss2[::sts], y=se2[::sts],
                                    name='Выб.', line=dict(color='#7ee787', width=2, dash='dash')))
                                fig_ec.update_layout(
                                    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(13,17,23,1)', height=220,
                                    title=dict(text=f'{label} (KS={sl["ks_stat"]:.4f})', font=dict(size=11)),
                                    xaxis_title='Балл', yaxis_title='ECDF',
                                    showlegend=(idx == 0),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0),
                                    margin=dict(l=30, r=5, t=35, b=30),
                                )
                                st.plotly_chart(fig_ec, use_container_width=True)

    # ── TAB 4: Выбранные ОО ──────────────────────────────────────────────────
    with tab4:
        st.markdown(f'<div class="section-title">Выбранные ОО ({len(best_schools)})</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ОО", n_actual)
        c2.metric("Всего", N_total)
        c3.metric("Доля", f"{n_actual/N_total*100:.1f}%")
        c4.metric("Регионов", best_schools['region'].nunique())

        st.dataframe(best_schools.style.format({
            'n_students': '{:.0f}', 'mean_score': '{:.2f}', 'mean_mark': '{:.2f}'
        }), use_container_width=True, height=400)

        st.markdown('<div class="section-title">Средний балл ОО: выборка vs совокупность</div>', unsafe_allow_html=True)
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(x=schools['mean_score'], nbinsx=50, name='Совокупность',
            marker_color='rgba(46,164,255,0.4)', histnorm='probability density'))
        fig_h.add_trace(go.Histogram(x=best_schools['mean_score'], nbinsx=30, name='Выборка',
            marker_color='rgba(126,231,135,0.6)', histnorm='probability density'))
        fig_h.update_layout(barmode='overlay', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)', height=270,
            xaxis_title='Средний балл', yaxis_title='Плотность',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            margin=dict(l=40, r=10, t=30, b=40))
        st.plotly_chart(fig_h, use_container_width=True)

    # ── TAB 5: Регионы ───────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-title">Доли регионов</div>', unsafe_allow_html=True)
        pr = schools['region'].value_counts(normalize=True).sort_index()
        sr = best_schools['region'].value_counts(normalize=True).sort_index()
        regs = sorted(set(pr.index) | set(sr.index))

        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=regs, y=[pr.get(r, 0)*100 for r in regs],
            name='Совокупность', marker_color='rgba(46,164,255,0.6)'))
        fig_r.add_trace(go.Bar(x=regs, y=[sr.get(r, 0)*100 for r in regs],
            name='Выборка', marker_color='rgba(126,231,135,0.7)'))
        fig_r.update_layout(barmode='group', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)',
            height=350, xaxis_title='Регион', yaxis_title='Доля, %',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            margin=dict(l=40, r=10, t=30, b=60), xaxis=dict(tickangle=-45))
        st.plotly_chart(fig_r, use_container_width=True)

        cl2, cr2 = st.columns(2)
        with cl2:
            st.markdown('<div class="section-title">Тип расположения</div>', unsafe_allow_html=True)
            ploc = schools['location_code'].value_counts(normalize=True).sort_index()
            sloc = best_schools['location_type'].map(
                lambda x: int(str(x)[0]) if pd.notna(x) else np.nan
            ).value_counts(normalize=True).sort_index()
            loc_labels = {1: 'Город', 2: 'Пгт', 3: 'Райцентр', 4: 'Село'}
            cats = sorted(set(ploc.index) | set(sloc.index))
            fig_l = go.Figure()
            fig_l.add_trace(go.Bar(x=[loc_labels.get(c, str(c)) for c in cats],
                y=[ploc.get(c, 0) for c in cats], name='Совокупность', marker_color='rgba(46,164,255,0.6)'))
            fig_l.add_trace(go.Bar(x=[loc_labels.get(c, str(c)) for c in cats],
                y=[sloc.get(c, 0) for c in cats], name='Выборка', marker_color='rgba(126,231,135,0.7)'))
            fig_l.update_layout(barmode='group', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)',
                height=270, showlegend=False, margin=dict(l=40, r=10, t=10, b=40))
            st.plotly_chart(fig_l, use_container_width=True)

        with cr2:
            st.markdown('<div class="section-title">Размер НП</div>', unsafe_allow_html=True)
            psz = schools['size_code'].value_counts(normalize=True).sort_index()
            ssz = best_schools['locality_size'].map(
                lambda x: int(str(x)[0]) if pd.notna(x) else np.nan
            ).value_counts(normalize=True).sort_index()
            cats_s = sorted(set(psz.index) | set(ssz.index))
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(x=[str(c) for c in cats_s],
                y=[psz.get(c, 0) for c in cats_s], name='Совокупность', marker_color='rgba(46,164,255,0.6)'))
            fig_s.add_trace(go.Bar(x=[str(c) for c in cats_s],
                y=[ssz.get(c, 0) for c in cats_s], name='Выборка', marker_color='rgba(126,231,135,0.7)'))
            fig_s.update_layout(barmode='group', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,1)',
                height=270, showlegend=False, margin=dict(l=40, r=10, t=10, b=40),
                xaxis=dict(title='Код размера НП'))
            st.plotly_chart(fig_s, use_container_width=True)

    # ── TAB 6: Скачать ────────────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-title">Экспорт</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📋 Лучшая выборка ОО** ({best_ml})")
            st.download_button("⬇️ best_sample.csv",
                best_schools.to_csv(index=False).encode('utf-8'),
                "best_sample.csv", "text/csv")
        with col2:
            st.markdown("**📊 Метрики всех методов**")
            mr = []
            for name, res in R.items():
                if 'error' in res: continue
                mr.append({k: res.get(k) for k in
                    ['rel_error_mean_score', 'rel_error_mean_mark', 'ks_stat',
                     'mmd', 'cramers_v', 'max_mark_dev', 'composite_score', 'time_sec']}
                    | {'Метод': name})
            st.download_button("⬇️ sampling_metrics.csv",
                pd.DataFrame(mr).to_csv(index=False).encode('utf-8'),
                "sampling_metrics.csv", "text/csv")

        st.markdown("<br>", unsafe_allow_html=True)
        st.json({
            'sample_size': n_actual, 'N_total': N_total,
            'fraction': round(n_actual / N_total, 4),
            'seed': seed, 'n_runs': n_srs_runs,
            'best_method': best_ml, 'score': round(ml_score, 4),
            'srs_mean': round(srs_mean, 4), 'srs_std': round(srs_std, 4),
            'strat_mean': round(strat_mean, 4), 'strat_std': round(strat_std, 4),
            'improvement_vs_srs': f'{improvement_srs:+.1f}%',
            'improvement_vs_strat': f'{improvement_strat:+.1f}%',
            'mark_0_excluded': True,
            'kim_constraints': {
                'РУ 4кл': '≤38', 'РУ 5кл': '≤45', 'РУ 6кл': '≤45',
                'МА 4кл': '≤20', 'МА 5кл': '≤20', 'МА 6кл': '≤20'},
        })