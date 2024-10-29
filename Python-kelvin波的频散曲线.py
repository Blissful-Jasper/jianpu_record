# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: jianpu
@email : xianpuji@hhu.edu.cn

# ================================================================================================
# Author: %(Jianpu)s | Affiliation: Hohai
# Email : %(email)s
# Last modified:  'date': time.strftime("%Y-%m-%d %H:%M"),
# Filename: 
# =================================================================================================

"""

import matplotlib.ticker as ticker
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import pandas as pd
from matplotlib import gridspec
import cartopy.crs as ccrs





# Constants
g = 9.8  # Gravity acceleration (m/s^2)
s2d = 86400  # Seconds in a day
re = 6371 * 1000  # Earth radius (m)

# Titles for subplots
title = ['(a)', '(b)', '(c)','(d)','(e)','(f)']

# Initialize figure and subplots
plt.rcParams.update({'font.size': 6.5})  # Set font size for all
fig, ax = plt.subplots(2, 3, figsize=(5.8, 3.9), dpi=600)
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.15, hspace=0.22)

# Calculate dispersion curves and CCKW band only once
he_all = np.array([8,  12, 25, 50])

fmax = np.array([1/3, 1/2.25, 0.5])

cp = (g * he_all) ** 0.5            # Phase speed (m/s)
zwnum_goal =  np.pi * re / cp /s2d 

# Prepare for CCKW band
kw_x = []
kw_y = []
for v in range(3):  # For each depth category
    he = [8, 25, 50] if v == 0 else [12, 25, 90] if v == 1 else [25, 90, 150]
    s_min = (g * he[0]) ** 0.5 / (2 * np.pi * re) * s2d  # Min slope  1/day
    s_max = (g * he[2]) ** 0.5 / (2 * np.pi * re) * s2d  # Max slope
    kw_tmax = 20

    kw_x.append(np.array([1 / kw_tmax / s_max, 1 / kw_tmax / s_min, 14, 14, fmax[v] / s_max, 1 / kw_tmax / s_max]))
    kw_y.append(np.array([1 / kw_tmax, 1 / kw_tmax, 14 * s_min, fmax[v], fmax[v], 1 / kw_tmax]))

# Draw subplots
for i in range(2):
    for v in range(3):
        print(i, v)
        ax[i, v] = plt.subplot(2, 3, v + 1 + 3 * i)

        # Mark 3, 6, 20 day period:
        for dd, d in enumerate([3, 6, 20]):
            plt.plot([-20, 20], [1 / d, 1 / d], 'k', linewidth=0.5, linestyle=':')
            plt.text(-14.8, 1 / d + 0.01, ['3d', '6d', '20d'][dd], fontsize=6)

        # Mark CCKW dispersion relationship:
        for hh in range(len(he_all)):
            plt.plot([0, zwnum_goal[hh]], [0, 0.5], 'grey', linewidth=0.5, linestyle='dashed')

        # Mark zwnum == 0:
        plt.plot([0, 0], [0, 0.5], 'k', linewidth=0.5, linestyle=':')

        # Mark CCKW band:
        plt.plot(kw_x[0], kw_y[0], 'purple', linewidth=1.2, linestyle='solid')

        if i == 0:
            plt.title(title[v], pad=3, loc='center', fontsize=9)
        if v == 0:
            plt.ylabel('Frequency (1/day)', fontsize=7.5)
        if i == 1:
            plt.xlabel('Zonal wavenumber', fontsize=7.5)
            plt.title(title[v+3], pad=3, loc='center', fontsize=9)
        plt.axis([-20, 20, 0, 0.5])
        plt.xticks(np.arange(-20, 21, 5), fontsize=6)
        plt.yticks(np.arange(0, 0.55, 0.05), fontsize=6)
        plt.tick_params(bottom=True, top=True, left=True, right=True)
        if v == 0:
            plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        else:
            plt.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False)
        plt.tick_params(direction="in")

plt.show()
plt.close()
