# -*- coding: utf-8 -*-
"""
简洁高效的箱线图绘制函数
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.patches import Patch


def generate_synthetic_data(bins, n_samples=140, seed=36):
    """
    生成合成温度数据
    
    参数:
        bins: 温度区间标签列表，如 ['0.1-0.2', '0.2-0.3', ...]
        n_samples: 每个区间的样本数量
        seed: 随机种子，确保结果可重现
    
    返回:
        ts_groups: δTs数据组列表
        ta_groups: δTa数据组列表
    """
    np.random.seed(seed)
    
    ts_groups = []  # δTs数据组
    ta_groups = []  # δTa数据组
    
    for i in range(len(bins)):
        # 随区间递减的均值趋势
        mean_ts = 1.2 - 0.55 * i  # δTs衰减更明显
        mean_ta = 0.3 - 0.12 * i  # δTa衰减较缓
        
        # 随区间增加的方差
        std_ts = 0.6 + 0.12 * i
        std_ta = 0.9 + 0.18 * i
        
        # 生成正态分布数据
        ts = np.random.normal(loc=mean_ts, scale=std_ts, size=n_samples)
        ta = np.random.normal(loc=mean_ta, scale=std_ta, size=n_samples)
        
        ts_groups.append(ts)
        ta_groups.append(ta)
    
    return ts_groups, ta_groups


def plot_temperature_boxplot(ts_groups, ta_groups, bins, 
                           title='温度敏感性回归分析', 
                           xlabel='Temperature Sensitivity (K)',
                           ylabel=r'$\delta T$ (K)',
                           figsize=(14, 6),
                           save_path='./temperature_boxplot.png',
                           dpi=200):
    """
    绘制温度敏感性箱线图及回归分析
    
    参数:
        ts_groups: δTs数据组列表
        ta_groups: δTa数据组列表  
        bins: 区间标签列表
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        figsize: 图形尺寸 (宽, 高)
        save_path: 保存路径
        dpi: 图像分辨率
    
    返回:
        fig, ax: matplotlib图形和轴对象
    """
    
    # 设置字体和图形参数
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=figsize)
    
    # 基本设置
    x_centers = np.arange(len(bins))  # X轴中心位置
    
    # 1. 绘制交替背景带
    for i in range(len(x_centers)):
        if i % 2 == 1:  # 奇数索引添加背景色
            ax.axvspan(i-0.5, i+0.5, color='0.93', zorder=0)
    
    # 2. 箱线图位置和样式设置
    pos_left = x_centers - 0.18   # δTs位置（左侧）
    pos_right = x_centers + 0.18  # δTa位置（右侧）
    width = 0.30                  # 箱体宽度
    
    # 箱线图样式定义
    boxprops_ts = dict(facecolor='#8ecae6', edgecolor='navy', linewidth=1.5,alpha=0.9)
    boxprops_ta = dict(facecolor='#f4a6a6', edgecolor='darkred', linewidth=1.5,alpha=0.9)
    medianprops = dict(color='black', linewidth=1.6)
    whiskerprops_ts = dict(color='navy', linewidth=1.1)
    whiskerprops_ta = dict(color='darkred', linewidth=1.1)
    capprops = dict(color='black', linewidth=1.1)
    
    # 3. 绘制箱线图
    # bp_ts = ax.boxplot(ts_groups, positions=pos_left, widths=width, patch_artist=True,
    #                    boxprops=boxprops_ts, medianprops=medianprops,
    #                    whiskerprops=whiskerprops_ts, capprops=capprops, 
    #                    showcaps=True, zorder=3)
    
    # bp_ta = ax.boxplot(ta_groups, positions=pos_right, widths=width, patch_artist=True,
    #                    boxprops=boxprops_ta, medianprops=medianprops,
    #                    whiskerprops=whiskerprops_ta, capprops=capprops, 
    #                    showcaps=True, zorder=3)
    bp_ts = ax.boxplot(ts_groups,0,'', positions=pos_left, widths=width, patch_artist=True,
                       boxprops=boxprops_ts, medianprops=medianprops,
                       whiskerprops=whiskerprops_ts, capprops=capprops, 
                       showcaps=True, zorder=3)
    
    bp_ta = ax.boxplot(ta_groups,0,'', positions=pos_right, widths=width, patch_artist=True,
                       boxprops=boxprops_ta, medianprops=medianprops,
                       whiskerprops=whiskerprops_ta, capprops=capprops, 
                       showcaps=True, zorder=3)
    
    # 4. 回归分析和趋势线
    # 计算中位数用于回归
    medians_ts = np.array([np.median(group) for group in ts_groups])
    medians_ta = np.array([np.median(group) for group in ta_groups])
    
    # 线性回归
    slope_ts, intercept_ts, r_ts, p_ts, se_ts = linregress(x_centers, medians_ts)
    slope_ta, intercept_ta, r_ta, p_ta, se_ta = linregress(x_centers, medians_ta)
    
    # 绘制回归线
    x_line    = np.linspace(x_centers[0]-0.3, x_centers[-1]+0.3, 200)
    y_ts_line = intercept_ts + slope_ts * x_line
    y_ta_line = intercept_ta + slope_ta * x_line
    
    ax.plot(x_line, y_ts_line, linestyle='--', linewidth=2, color='royalblue', 
            dashes=(8,6), zorder=3)
    ax.plot(x_line, y_ta_line, linestyle='--', linewidth=2, color='crimson', 
            dashes=(8,6), zorder=3)
    
    # 5. 坐标轴设置
    ax.set_xticks(x_centers)
    ax.set_xticklabels(bins, fontweight='bold', fontsize=13)
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_ylim(-6, 6)
    ax.set_yticks(np.arange(-6, 6.1, 2))
    
    # 6. 标题和网格
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # 7. 图例设置 - 分层布局
    
    box_legend_elems = [
        Patch(facecolor='#8ecae6', edgecolor='navy', label=r'$\delta T_s$ (Box Plot)'),
        Patch(facecolor='#f4a6a6', edgecolor='darkred', label=r'$\delta T_a$ (Box Plot)'),
    ]
    
    ax.legend(handles=box_legend_elems, loc='lower left', frameon=True, 
                          fontsize=12, edgecolor='gray', fancybox=True, shadow=True)
    
    # 7.2 回归线图例 
    
    regression_legend_elems = [
        plt.Line2D([0], [0], linestyle='--', color='royalblue', 
                   label=r'$\delta T_s$ Regression', linewidth=2.2),
        plt.Line2D([0], [0], linestyle='--', color='crimson', 
                   label=r'$\delta T_a$ Regression', linewidth=2.2)
    ]
    
    # 添加回归线到箱线图图例中，创建组合图例
    combined_legend_elems = box_legend_elems + regression_legend_elems
    ax.legend(handles=combined_legend_elems, loc='best', frameon=False, 
              fontsize=12, edgecolor='gray', fancybox=True, shadow=True)
    
   
    ax.text(0.98, 0.98, f"δTs: k={slope_ts:.2f}, R={r_ts:.2f}", 
            transform=ax.transAxes, fontsize=12, color='royalblue', 
            ha='right', va='top', weight='bold')
    ax.text(0.98, 0.92, f"δTa: k={slope_ta:.2f}, R={r_ta:.2f}", 
            transform=ax.transAxes, fontsize=12, color='crimson', 
            ha='right', va='top', weight='bold')
    
    # 10. 最终调整和保存
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


# 使用示例 
if __name__ == "__main__":
    
    # 定义温度敏感性区间
    bins = ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
            '0.5-0.6', '0.6-0.7', '0.7-0.8']
    
    # 生成合成数据
    ts_data, ta_data = generate_synthetic_data(bins, n_samples=200, seed=42)
    print(np.array(ts_data).shape,np.array(ta_data).shape)
    # 绘制图表 
    fig, ax = plot_temperature_boxplot(
        ts_groups=ts_data,
        ta_groups=ta_data,
        bins=bins,
        title='Regression Analysis of δT across Temperature Sensitivity Intervals',
        xlabel='Temperature Sensitivity (K)',
        ylabel='δT (K)',
        figsize=(14, 8),  # 稍微调大以匹配图片比例
        save_path='./replicated_kelvin_boxplot.png',
        dpi=300  # 更高分辨率
    )
    
    plt.show()
    
    
