# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

@email : xianpuji@hhu.edu.cn
"""

import matplotlib.ticker as ticker
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import ListedColormap 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import glob
from datetime import datetime, timedelta
import cmaps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cftime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches


# ================================================================================================
# Author: %(Jianpu)s | Affiliation: Hohai
# email : xianpuji@hhu.edu.cn
# Last modified:  %(date)s
# Filename: 
# =================================================================================================

    
plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #正常显示负号
# 网格设置


def plot_kelvin_wave(k=1.0, omega=1.0, t=0.0, 
                     xlim=(0, 4*np.pi), ylim=(-3, 3), 
                     nx=180, ny=91):
    """
    绘制 Kelvin 波的水平风场（u, v=0）、位势场（Φ）和散度（∇·u）
    
    参数：
        k, omega : 波数与频率
        t        : 时间截面
        xlim     : x 方向范围 (起点, 终点)
        ylim     : y 方向范围 (起点, 终点)
        nx, ny   : x/y 网格点数量
    """
    # 创建网格
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y)

    # Kelvin 波理论解
    u = np.exp(-0.5 * Y**2) * np.cos(k * X - omega * t)
    v = np.zeros_like(u)
    Phi = (omega / k) * u
    
    # 计算散度 ∂u/∂x
    du_dx = np.gradient(u, x, axis=1)
    divergence = du_dx
    cmap =  cmaps.BlueWhiteOrangeRed
    # 绘图
    plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #正常显示负号
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    plt.title(r"$Kelvin (K^*=1$)", fontsize=10,loc='left')
    # 散度填色背景
    cf = plt.contourf(X, Y, divergence, levels=21, cmap=cmap, extend='both')
    # cbar = plt.colorbar(cf, label='散度')
    # cbar.ax.tick_params(axis='both', direction='in',  )
    # 风矢量
    step = 3
    plt.quiver(X[::step,::step], Y[::step,::step], u[::step,::step], v[::step,::step], scale=40, width=0.002, color='black')
    ax.text(4.4, 0, '赤道', va='center', ha='right', fontsize=10)
    # 位势场等值线
    cs = plt.contour(X, Y, Phi,  levels=[-0.75,-0.5,-0.25,0.25,0.5,0.75], colors='k', linewidths=1)
    plt.clabel(cs, inline=True, fontsize=8)
    ax.set_ylim(-2,2)
    ax.set_xlim(4.5,11)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', direction='in',  )
    # 轴标签
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.3)
    # plt.tight_layout()
    plt.show()
    fig.savefig('kelvin波水平分布图.png',dpi=600, bbox_inches='tight')
plot_kelvin_wave(k=1, omega=1.0, t=0.0)
