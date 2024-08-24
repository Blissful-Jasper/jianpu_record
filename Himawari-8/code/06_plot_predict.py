#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:58:17 2024

@author: Jianpu

@blog:  https://blog.csdn.net/weixin_44237337?spm=1000.2115.3001.5343

@email: Xpji@hhu.edu.cn

@introduction: keep learning although slowly

"""

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="plot the netcdf of prediction.")
parser.add_argument("--input_file", type=str, help="Path to the input NetCDF file")

args = parser.parse_args()
input_file = args.input_file

path  = input_file

figpath = r'/DatadiskExt/xpji/jxp_deeplearn_model_code/figure'

output_folder = os.path.basename(path).split('_')[2]

out_path =  os.path.join(figpath,output_folder)

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(out_path, exist_ok=True)

ds    = xr.open_dataset(path)
lon   = ds.longitude
lat   = ds.latitude
cmap  = cmaps.WhiteBlueGreenYellowRed
time  = np.array( ds.time)

formatted_times = [str(np.datetime_as_string(t, unit='h')) for t in time]


output_name = 'Precip_prediction_from_' + formatted_times[0] + '_to_' + formatted_times[-1] + '.png'


# 创建一个包含6个子图的图形
fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
cax   = fig.add_axes([0.26,0.05,0.5,0.02])
# 将 axs 展开为一维数组，方便后续操作
axs = axs.ravel()

# 循环绘制每个子图
for i in range(6):
    ax = axs[i]
    f  = ds.precip[:, :, i].plot.contourf(ax=ax, cmap=cmap,
                            add_colorbar=False,
                            # cbar_kwargs={'orientation': 'vertical', 
                            #               'shrink':0.6, 
                            #        'aspect':40,
                            #       'label': 'Precipitation (mm/hour)'}
                            )

    # 添加地图特征
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # 设置地图范围
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # 添加网格线
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # 设置坐标轴标签
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

# 调整子图布局
# plt.tight_layout()

f = plt.colorbar(f,cax=cax,orientation='horizontal',pad=0.1)



fig.savefig(os.path.join(out_path, output_name),dpi=300)





