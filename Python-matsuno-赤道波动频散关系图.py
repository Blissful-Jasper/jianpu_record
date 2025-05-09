# %%
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

@email : xianpuji@hhu.edu.cn
"""


import matplotlib.pyplot as plt
import numpy as np

# ================================================================================================
# Author: %(Jianpu)s | Affiliation: Hohai
# email : xianpuji@hhu.edu.cn
# =================================================================================================
# 定义各类波的频散关系
def omega_gravity(k, n):
    return np.sqrt(k**2 + 2 * n + 1)

def omega_rossby(k, n):
    return -k / (k**2 + 2 * n + 1)

def omega_mixed_rossby_gravity(k):
    return 0.5 * (k + np.sqrt(k**2 + 4))


def plot_kelvin(ax, k, color, lw):
    k_pos = k[k > 0]
    ax.plot(k_pos, k_pos, label='Kelvin波', color=color, linewidth=lw)

def plot_mrg(ax, k, omega_func, color, lw):
    k_pos = k[k > 0]
    k_neg = k[k <= 0]
    ax.plot(k_neg, omega_func(k_neg), color=color, label='MRG波', linewidth=lw)
    ax.plot(k_pos, omega_func(k_pos), color=color, linewidth=lw)

def plot_rossby(ax, k, omega_func, color, lw, n_max=4):
    k_neg = k[k < 0]
    for n in range(1, n_max+1):
        omega_r = omega_func(k_neg, n)
        ax.plot(k_neg, omega_r, color=color, linewidth=lw)

def plot_gravity(ax, k, omega_func, color, lw, n_max=4):
    for n in range(1, n_max+1):
        omega_g = omega_func(k, n)
        ax.plot(k, omega_g, color=color, linewidth=lw)
        if n == 1:
            ax.plot([], [], color=color, label='惯性重力波')

def style_axes(ax, color, lw):
    ax.axhline(0, color=color, linewidth=lw)
    ax.axvline(0, color=color, linewidth=lw)
    ax.set_xlabel(r'$k^*$', fontsize=14, color=color)
    ax.set_ylabel(r'$\omega^*$', fontsize=14, color=color)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(0, 4.5)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['bottom'].set_color(color)

    ax.tick_params(axis='both', colors=color, width=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color(color)

    ax.plot(1, 0, ">", transform=ax.get_yaxis_transform(), clip_on=False, color=color)
    ax.plot(0, 1, "^", transform=ax.get_xaxis_transform(), clip_on=False, color=color)

def annotate(ax, color):
    ax.text(1.2, 0.7, 'Kelvin波 \n (n=-1)', fontsize=10, rotation=55, color=color)
    ax.text(-3.8, 0.35, 'Rossby波 \n (n=1,2,3,4)', fontsize=10, color=color)
    ax.text(-1, 0.9, 'MRG波 \n (n=0)', fontsize=10, rotation=25, color=color)
    ax.set_title('赤道浅水波频散关系图（Matsuno 1966）', fontsize=15, color=color)
    ax.set_title('东传惯性重力波', fontsize=10, loc='right', color=color)
    ax.set_title('西传惯性重力波', fontsize=10, loc='left', color=color)

def plot_equatorial_waves(k, omega_mrg, omega_rossby, omega_gravity, main_color='k', linewidth=1.8):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # 绘制各类波动
    plot_kelvin(ax, k, main_color, linewidth)
    plot_mrg(ax, k, omega_mrg, main_color, linewidth)
    plot_rossby(ax, k, omega_rossby, main_color, linewidth)
    plot_gravity(ax, k, omega_gravity, main_color, linewidth)

    # 样式设置
    style_axes(ax, main_color, linewidth)
    annotate(ax, main_color)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    fig.savefig('频散曲线理论关系图.png',dpi=600)
    
plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #正常显示负号
# 示例 k 轴（你应该根据实际数据定义）
k = np.linspace(-4.2, 4.2, 500)
# 通用配色（深蓝紫色）
main_color = '#4B0082'
linewidth = 1.8

plot_equatorial_waves(
    k=k,
    omega_mrg=omega_mixed_rossby_gravity,
    omega_rossby=omega_rossby,
    omega_gravity=omega_gravity,
    main_color=main_color,
    linewidth=1.8
)



# %%