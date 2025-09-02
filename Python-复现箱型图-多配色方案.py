# -*- coding: utf-8 -*-
"""
循环绘制多种学术配色风格的温度敏感性箱线图
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.patches import Patch


ACADEMIC_COLOR_SCHEMES = {
    'nature': {
       
        'ts_fill': '#0072B2', 'ts_edge': '#004D7A',
        'ta_fill': '#D55E00', 'ta_edge': '#A04000',
        'ts_line': '#0072B2', 'ta_line': '#D55E00',
        'bg_alt': '#F8F8F8', 'grid': '#E0E0E0'
    },
    'science': {
       
        'ts_fill': '#1B4F72', 'ts_edge': '#0D2A3F',
        'ta_fill': '#C0392B', 'ta_edge': '#922B21',
        'ts_line': '#2874A6', 'ta_line': '#E74C3C',
        'bg_alt': '#FAFAFA', 'grid': '#DDDDDD'
    },
    'cell': {
      
        'ts_fill': '#2E86AB', 'ts_edge': '#1A5B7A',
        'ta_fill': '#A23B72', 'ta_edge': '#7A2B56',
        'ts_line': '#3498DB', 'ta_line': '#B7472A',
        'bg_alt': '#F7F9FC', 'grid': '#E8EAED'
    },
    'nejm': {
      
        'ts_fill': '#005580', 'ts_edge': '#003D5C',
        'ta_fill': '#8B0000', 'ta_edge': '#5C0000',
        'ts_line': '#0077AA', 'ta_line': '#AA0000',
        'bg_alt': '#F5F7FA', 'grid': '#DDE4ED'
    },
    'pnas': {
       
        'ts_fill': '#4472C4', 'ts_edge': '#2F4F8F',
        'ta_fill': '#E15759', 'ta_edge': '#C13133',
        'ts_line': '#5B9BD5', 'ta_line': '#FF6B6B',
        'bg_alt': '#F9F9FB', 'grid': '#E6E6E6'
    },
    'elegant_mono': {
      
        'ts_fill': '#34495E', 'ts_edge': '#2C3E50',
        'ta_fill': '#7F8C8D', 'ta_edge': '#65727A',
        'ts_line': '#34495E', 'ta_line': '#95A5A6',
        'bg_alt': '#FCFCFC', 'grid': '#F0F0F0'
    },
    'colorblind_friendly': {
      
        'ts_fill': '#0173B2', 'ts_edge': '#013A5F',
        'ta_fill': '#DE8F05', 'ta_edge': '#A06D04',
        'ts_line': '#029E73', 'ta_line': '#CC78BC',
        'bg_alt': '#F8F8F8', 'grid': '#E0E0E0'
    },
    'high_contrast': {
     
        'ts_fill': '#000080', 'ts_edge': '#000055',
        'ta_fill': '#8B0000', 'ta_edge': '#5C0000',
        'ts_line': '#0000CD', 'ta_line': '#DC143C',
        'bg_alt': '#FAFAFA', 'grid': '#CCCCCC'
    }
}


def generate_synthetic_data(bins, n_samples=140, seed=36):
    np.random.seed(seed)
    ts_groups, ta_groups = [], []
    for i in range(len(bins)):
        mean_ts, mean_ta = 1.2 - 0.55*i, 0.3 - 0.12*i
        std_ts, std_ta = 0.6 + 0.12*i, 0.9 + 0.18*i
        ts = np.random.normal(mean_ts, std_ts, n_samples)
        ta = np.random.normal(mean_ta, std_ta, n_samples)
        ts_groups.append(ts)
        ta_groups.append(ta)
    return ts_groups, ta_groups


def plot_temperature_boxplot(ts_groups, ta_groups, bins, scheme, 
                           title='', 
                           xlabel='Temperature Sensitivity (K)',
                           ylabel=r'$\delta T$ (K)',
                           figsize=(14, 8),
                           save_path=None,
                           dpi=200):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=figsize)

    x_centers = np.arange(len(bins))

    # 背景交替带
    for i in range(len(x_centers)):
        if i % 2 == 1:
            ax.axvspan(i-0.5, i+0.5, color=scheme['bg_alt'], zorder=0)

    # 箱线图
    pos_left, pos_right = x_centers - 0.18, x_centers + 0.18
    width = 0.30

    boxprops_ts = dict(facecolor=scheme['ts_fill'], edgecolor=scheme['ts_edge'],
                       linewidth=1.5, alpha=0.9)
    boxprops_ta = dict(facecolor=scheme['ta_fill'], edgecolor=scheme['ta_edge'],
                       linewidth=1.5, alpha=0.9)
    medianprops = dict(color='black', linewidth=1.6)
    whiskerprops_ts = dict(color=scheme['ts_edge'], linewidth=1.1)
    whiskerprops_ta = dict(color=scheme['ta_edge'], linewidth=1.1)
    capprops = dict(color='black', linewidth=1.1)

    ax.boxplot(ts_groups, positions=pos_left, widths=width, patch_artist=True,
               boxprops=boxprops_ts, medianprops=medianprops,
               whiskerprops=whiskerprops_ts, capprops=capprops,
               showcaps=True, zorder=3)

    ax.boxplot(ta_groups, positions=pos_right, widths=width, patch_artist=True,
               boxprops=boxprops_ta, medianprops=medianprops,
               whiskerprops=whiskerprops_ta, capprops=capprops,
               showcaps=True, zorder=3)

    # 回归线
    medians_ts = np.array([np.median(group) for group in ts_groups])
    medians_ta = np.array([np.median(group) for group in ta_groups])

    slope_ts, intercept_ts, r_ts, _, _ = linregress(x_centers, medians_ts)
    slope_ta, intercept_ta, r_ta, _, _ = linregress(x_centers, medians_ta)

    x_line = np.linspace(x_centers[0]-0.3, x_centers[-1]+0.3, 200)
    ax.plot(x_line, intercept_ts + slope_ts*x_line, linestyle='--', linewidth=2,
            color=scheme['ts_line'], dashes=(8,6), zorder=3)
    ax.plot(x_line, intercept_ta + slope_ta*x_line, linestyle='--', linewidth=2,
            color=scheme['ta_line'], dashes=(8,6), zorder=3)

    # 坐标轴
    ax.set_xticks(x_centers)
    ax.set_xticklabels(bins, fontweight='bold', fontsize=13)
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_ylim(-6, 6)
    ax.set_yticks(np.arange(-6, 6.1, 2))

    ax.set_title(f"{scheme_name}", fontsize=18, fontweight='bold', pad=15)
    ax.grid(color=scheme['grid'], linestyle='-', linewidth=0.8, alpha=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # 图例
    box_legend = [
        Patch(facecolor=scheme['ts_fill'], edgecolor=scheme['ts_edge'], label=r'$\delta T_s$ (Box Plot)'),
        Patch(facecolor=scheme['ta_fill'], edgecolor=scheme['ta_edge'], label=r'$\delta T_a$ (Box Plot)')
    ]
    reg_legend = [
        plt.Line2D([0], [0], linestyle='--', color=scheme['ts_line'], label=r'$\delta T_s$ Regression', linewidth=2.2),
        plt.Line2D([0], [0], linestyle='--', color=scheme['ta_line'], label=r'$\delta T_a$ Regression', linewidth=2.2)
    ]
    ax.legend(handles=box_legend+reg_legend, loc='best', frameon=False, fontsize=12)

    # 回归文字
    ax.text(0.98, 0.98, f"δTs: k={slope_ts:.2f}, R={r_ts:.2f}",
            transform=ax.transAxes, fontsize=15, color=scheme['ts_line'],
            ha='right', va='top', weight='bold')
    ax.text(0.98, 0.95, f"δTa: k={slope_ta:.2f}, R={r_ta:.2f}",
            transform=ax.transAxes, fontsize=15, color=scheme['ta_line'],
            ha='right', va='top', weight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    return fig, ax


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    bins = ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8']
    ts_data, ta_data = generate_synthetic_data(bins, n_samples=200, seed=42)

    for scheme_name, scheme in ACADEMIC_COLOR_SCHEMES.items():
        fig, ax = plot_temperature_boxplot(
            ts_groups=ts_data,
            ta_groups=ta_data,
            bins=bins,
            scheme=scheme,
            save_path=f'./boxplot_{scheme_name}.png',
            dpi=300
        )
        