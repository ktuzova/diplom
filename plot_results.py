"""
Построение графиков для Главы 3 ВКР.
Данные из прогона statistical_analysis.py (K=20 размеров выборки).

Запуск:
    python plot_results.py

ВКР 2025 — Тузова К. К.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════════════
# ДАННЫЕ (K=20, N_RUNS=50)
# ════════════════════════════════════════════════════════════════

NS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
      550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

SRS =       [1.0]*20
SRS_STD =   [0.5090, 0.4432, 0.4959, 0.4448, 0.4110, 0.4172, 0.5003,
             0.5636, 0.5673, 0.5739, 0.5032, 0.5233, 0.5048, 0.4896,
             0.4983, 0.5474, 0.5264, 0.5188, 0.5230, 0.5121]

STRAT =     [0.9017, 0.9106, 0.8936, 0.8002, 0.7716, 0.7812, 0.9652,
             0.9629, 0.8573, 0.9123, 0.8245, 0.7397, 0.7376, 0.7155,
             0.8856, 0.8386, 0.8738, 0.9141, 0.8425, 0.7306]
STRAT_STD = [0.4248, 0.4102, 0.4409, 0.3624, 0.3516, 0.4780, 0.5342,
             0.4994, 0.5000, 0.4430, 0.3854, 0.3551, 0.4058, 0.3731,
             0.4040, 0.4006, 0.4034, 0.4408, 0.4228, 0.3358]

KC =        [2.7705, 2.8948, 2.5029, 3.8971, 4.7254, 5.1527, 5.4624,
             5.3151, 5.5517, 5.1253, 4.9276, 4.7788, 4.7125, 4.2146,
             4.5229, 4.4450, 4.5172, 4.7947, 4.7220, 4.6237]

FL =        [0.5186, 0.4819, 0.4754, 0.4110, 0.4673, 0.4722, 0.8180,
             0.8615, 0.8865, 1.1054, 0.9172, 0.9490, 1.0136, 0.7480,
             0.6488, 0.8142, 0.8251, 0.8389, 1.0592, 0.8249]

KH =        [1.0869, 0.6424, 0.6180, 0.5513, 0.6688, 0.7734, 0.7246,
             0.5659, 0.3151, 0.2116, 0.2575, 0.4746, 0.3979, 0.2815,
             0.3074, 0.3704, 0.3651, 0.2602, 0.3371, 0.4938]


# ════════════════════════════════════════════════════════════════
# 1. Чувствительность: composite score vs n
# ════════════════════════════════════════════════════════════════

def plot_sensitivity():
    srs_hi = [s + d for s, d in zip(SRS, SRS_STD)]
    srs_lo = [s - d for s, d in zip(SRS, SRS_STD)]

    fig, ax1 = plt.subplots(figsize=(12, 5.5))

    ax1.fill_between(NS, srs_lo, srs_hi, alpha=0.12, color='#73726c')
    ax1.plot(NS, SRS, '--', color='#73726c', linewidth=2, marker='o',
             markersize=4, label='SRS (mean +/- sigma)')

    strat_hi = [s + d for s, d in zip(STRAT, STRAT_STD)]
    strat_lo = [s - d for s, d in zip(STRAT, STRAT_STD)]
    ax1.fill_between(NS, strat_lo, strat_hi, alpha=0.08, color='#3266ad')
    ax1.plot(NS, STRAT, '-', color='#3266ad', linewidth=1.5, marker='s',
             markersize=4, label='Stratified (+/- sigma)')

    ax1.plot(NS, FL, '-', color='#2e8b57', linewidth=2.5, marker='D',
             markersize=5, label='Facility Location')
    ax1.plot(NS, KH, '-', color='#d4802a', linewidth=2.5, marker='^',
             markersize=5, label='Kernel Herding')

    ax1.set_ylim(0, 1.8)
    ax1.set_xlabel('Sample size n', fontsize=12)
    ax1.set_ylabel('Composite score (1.0 = SRS level, lower = better)',
                   fontsize=12)
    ax1.set_xticks(NS[::2])

    ax2 = ax1.twinx()
    ax2.plot(NS, KC, '--', color='#9370DB', linewidth=1.5, marker='v',
             markersize=4, label='k-center (right axis)')
    ax2.set_ylabel('k-center (right axis)', fontsize=12, color='#9370DB')
    ax2.set_ylim(0, 6.5)
    ax2.tick_params(axis='y', labelcolor='#9370DB')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=9, framealpha=0.9)

    ax1.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig('fig1_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  -> fig1_sensitivity.png')


# ════════════════════════════════════════════════════════════════
# 2. CD-диаграмма Немени (оригинальный дизайн, мелкий шрифт)
# ════════════════════════════════════════════════════════════════

def plot_cd_diagram():
    methods = ['Kernel\nHerding', 'Facility\nLocation', 'Stratified',
               'SRS', 'k-center']
    ranks =   [1.40, 2.25, 2.55, 3.80, 5.00]
    colors =  ['#d4802a', '#2e8b57', '#3266ad', '#73726c', '#9370DB']
    cd = 1.364

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-1.8, 4.5)

    # Ось рангов
    ax.plot([1, 5], [3, 3], '-', color='#444', linewidth=1)
    for r in range(1, 6):
        ax.plot([r, r], [2.85, 3.15], '-', color='#444', linewidth=1)
        ax.text(r, 3.35, str(r), ha='center', va='bottom', fontsize=10)

    # CD-отрезок сверху
    cd_start = 1.0
    cd_end = cd_start + cd
    ax.annotate('', xy=(cd_end, 3.7), xytext=(cd_start, 3.7),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
    ax.text((cd_start + cd_end) / 2, 3.85, f'CD = {cd:.2f}',
            ha='center', va='bottom', fontsize=9, color='#333')

    # Методы: точки + подписи (увеличенный шаг 0.80 вместо 0.65)
    for i, (name, rank, color) in enumerate(zip(methods, ranks, colors)):
        y_pos = 2.2 - i * 0.80
        ax.plot(rank, y_pos, 'o', color=color, markersize=9, zorder=5)
        ax.plot([rank, rank], [y_pos, 3], '--', color=color,
                linewidth=1, alpha=0.5)
        if rank < 3:
            ax.text(rank - 0.12, y_pos, name, ha='right', va='center',
                    fontsize=8.5, color=color, fontweight='bold')
            ax.text(rank - 0.12, y_pos - 0.28, f'R = {rank:.2f}',
                    ha='right', va='center', fontsize=8, color=color)
        else:
            ax.text(rank + 0.12, y_pos, name, ha='left', va='center',
                    fontsize=8.5, color=color, fontweight='bold')
            ax.text(rank + 0.12, y_pos - 0.28, f'R = {rank:.2f}',
                    ha='left', va='center', fontsize=8, color=color)

    # Скобки: незначимые группы
    ax.plot([1.40, 2.55], [0.55, 0.55], '-', color='#888', linewidth=2.5,
            solid_capstyle='round')
    ax.text((1.40 + 2.55) / 2, 0.35, 'not significant', ha='center',
            fontsize=8, color='#888')

    ax.plot([2.55, 5.00], [-0.25, -0.25], '-', color='#888', linewidth=2.5,
            solid_capstyle='round')
    ax.text((2.55 + 5.00) / 2, -0.45, 'not significant', ha='center',
            fontsize=8, color='#888')

    # Значимые пары
    sig_pairs = [
        (2.25, 3.80, 'SRS vs FL: 1.55'),
        (1.40, 3.80, 'SRS vs KH: 2.40'),
        (2.55, 5.00, 'k-ctr vs Strat: 2.45'),
        (2.25, 5.00, 'k-ctr vs FL: 2.75'),
        (1.40, 5.00, 'k-ctr vs KH: 3.60'),
    ]
    y_sig = -0.9
    for r1, r2, label in sig_pairs:
        ax.annotate('', xy=(r1, y_sig), xytext=(r2, y_sig),
                    arrowprops=dict(arrowstyle='-', color='#cc3333',
                                    lw=0.8, linestyle='--'))
        ax.text((r1 + r2) / 2, y_sig - 0.15, label, ha='center',
                fontsize=7, color='#cc3333')
        y_sig -= 0.30

    ax.set_xlabel('Average rank (1 = best)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('fig2_cd_nemenyi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  -> fig2_cd_nemenyi.png')


# ════════════════════════════════════════════════════════════════
# 3. Покомпонентные метрики (n=300)
# ════════════════════════════════════════════════════════════════

def plot_components():
    labels = ['Rel. error\nmean score', 'Rel. error\nmean mark',
              'KS\nstatistic', 'MMD', "Cramer's\nV", 'Max mark\ndeviation']
    srs_v =   [0.0084, 0.0052, 0.0083, 0.0383, 0.0157, 0.0089]
    strat_v = [0.0067, 0.0041, 0.0071, 0.0246, 0.0125, 0.0072]
    fl_v =    [0.0018, 0.0001, 0.0052, 0.0492, 0.0069, 0.0048]
    kh_v =    [0.0067, 0.0044, 0.0086, 0.0144, 0.0109, 0.0064]

    norm = lambda v: [vi / si if si > 0 else 0 for vi, si in zip(v, srs_v)]

    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, norm(srs_v), w, label='SRS', color='#73726c',
           alpha=0.7, edgecolor='#73726c')
    ax.bar(x - 0.5*w, norm(strat_v), w, label='Stratified',
           color='#3266ad', alpha=0.7, edgecolor='#3266ad')
    ax.bar(x + 0.5*w, norm(fl_v), w, label='Facility Location',
           color='#2e8b57', alpha=0.7, edgecolor='#2e8b57')
    ax.bar(x + 1.5*w, norm(kh_v), w, label='Kernel Herding',
           color='#d4802a', alpha=0.7, edgecolor='#d4802a')

    ax.axhline(y=1.0, color='#73726c', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(5.6, 1.03, 'SRS = 1.0', fontsize=9, color='#73726c', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Ratio to SRS (1.0 = SRS, lower = better)', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, axis='y', alpha=0.15)
    ax.set_ylim(0, 1.5)
    plt.tight_layout()
    plt.savefig('fig3_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  -> fig3_components.png')


# ════════════════════════════════════════════════════════════════
# 4. Cohen's d
# ════════════════════════════════════════════════════════════════

def plot_cohens_d():
    methods = ['Facility Location', 'Kernel Herding', 'Stratified']
    d_values = [1.25, 0.54, 0.52]
    colors = ['#2e8b57', '#d4802a', '#3266ad']
    p_labels = ['P(SRS better) = 4/50',
                'P(SRS better) = 17/50',
                'P(SRS better) = 17/50']

    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = np.arange(len(methods))

    bars = ax.barh(y_pos, d_values, height=0.5, color=colors, alpha=0.7,
                   edgecolor=colors)

    for i, (bar, d, p) in enumerate(zip(bars, d_values, p_labels)):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f'd = +{d:.2f}   ({p})',
                va='center', fontsize=10)

    thresholds = [(0.20, 'small', '#aaa'), (0.50, 'medium', '#888'),
                  (0.80, 'large', '#555')]
    for val, label, color in thresholds:
        ax.axvline(x=val, color=color, linestyle='--', linewidth=1, alpha=0.5)
        ax.text(val, len(methods) - 0.1, label, ha='center', va='bottom',
                fontsize=9, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel("Cohen's d vs SRS (positive = better than SRS)", fontsize=12)
    ax.set_xlim(0, 1.7)
    ax.grid(True, axis='x', alpha=0.15)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('fig4_cohens_d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  -> fig4_cohens_d.png')


# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Building charts for Chapter 3...\n')
    plot_sensitivity()
    plot_cd_diagram()
    plot_components()
    plot_cohens_d()
    print(f'\nDone! Saved:')
    print('  fig1_sensitivity.png  — composite score vs n (20 points)')
    print('  fig2_cd_nemenyi.png   — Nemenyi CD diagram (CD=1.36)')
    print('  fig3_components.png   — per-component metrics (n=300)')
    print('  fig4_cohens_d.png     — Cohen\'s d (n=300)')