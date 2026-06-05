#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics Module - 诊断工具模块
==================================

提供大气动力学和热力学诊断工具，包括总体湿稳定性(GMS)计算等。

主要功能：
---------
1. 水平总体湿稳定性计算
2. 垂直总体湿稳定性计算  
3. 干静力能(DSE)和湿静力能(MSE)计算

依赖：
-----
- metpy: 气象学计算库
- geocat-comp: 地球科学计算库

安装：
-----
    pip install metpy geocat-comp

作者: xpji
创建: 2025-05-15
"""

from typing import Tuple, Dict, Union, Optional, List, Any
import time
import os
import glob
import numpy as np
import xarray as xr

# 从 utils 模块导入工具函数
from .utils import load_data

# 可选依赖
try:
    from metpy.calc import mixing_ratio_from_specific_humidity
    from metpy.units import units
    HAS_METPY = True
except ImportError:
    HAS_METPY = False
    # 提供简单的近似实现
    def mixing_ratio_from_specific_humidity(specific_humidity):
        """
        将比湿转换为混合比 (简化版本)
        r ≈ q / (1 - q)
        
        Parameters
        ----------
        specific_humidity : array-like
            比湿 (kg/kg)
            
        Returns
        -------
        mixing_ratio : array-like
            混合比 (kg/kg)
        """
        q = np.asarray(specific_humidity)
        return q / (1.0 - q)

try:
    # geocat.comp 的 delta_pressure 可能不存在，改用其他方法
    try:
        from geocat.comp import delta_pressure
    except ImportError:
        # 如果 delta_pressure 不存在，我们自己实现
        def delta_pressure(pressure_levels):
            """计算压力层之间的差值"""
            return np.diff(pressure_levels)
    HAS_GEOCAT = True
except ImportError:
    HAS_GEOCAT = False
    def delta_pressure(pressure_levels):
        """计算压力层之间的差值"""
        return np.diff(pressure_levels)

# Physical constants
Cp = 1004.0  # Specific heat at constant pressure (J/kg/K)
g = 9.81     # Gravitational acceleration (m/s^2)
L = 2.5e6    # Latent heat of vaporization (J/kg)
T_ref = 300.0  # Reference temperature (K)
R_earth = 6.371e6  # Earth radius (m)

def compute_dx_dy(lat: xr.DataArray, lon: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算经纬度网格的 dx 和 dy (单位: 米)
    
    Parameters
    ----------
    lat : xr.DataArray
        纬度数组 (度)
    lon : xr.DataArray
        经度数组 (度)
        
    Returns
    -------
    dx, dy : Tuple[np.ndarray, np.ndarray]
        经向和纬向网格间距 (米)
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # dy: 纬向间距 (恒定)
    dlat = np.abs(np.diff(lat.values).mean())
    dy = R_earth * np.deg2rad(dlat)
    
    # dx: 经向间距 (随纬度变化)
    dlon = np.abs(np.diff(lon.values).mean())
    dx = R_earth * np.cos(lat_rad) * np.deg2rad(dlon)
    
    # 将 dx 广播到正确的形状
    if len(lat.shape) == 1 and len(lon.shape) == 1:
        # 1D lat, lon -> 2D grid
        dx = np.broadcast_to(dx[:, np.newaxis], (len(lat), len(lon)))
        dy = np.full((len(lat), len(lon)), dy)
    
    return dx, dy

def calc_dse(ta: xr.DataArray, zg: xr.DataArray, plev: xr.DataArray) -> xr.DataArray:
    """
    Calculate Dry Static Energy (DSE).
    
    DSE = Cp * T + g * Z
    
    Parameters
    ----------
    ta : xr.DataArray
        Air temperature (K)
    zg : xr.DataArray
        Geopotential height (m)
    plev : xr.DataArray
        Pressure levels (Pa)
        
    Returns
    -------
    xr.DataArray
        Dry static energy (J/kg)
    """
    return Cp * ta + g * zg

def extract_model_name(path: str) -> str:
    return os.path.basename(path).split("_")[2]

def get_unique_file_or_raise(path_pattern: str, varname: str, model: str) -> str:
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"❌ 未找到 {varname} 文件，模型：{model}")
    if len(files) > 1:
        print(f"⚠️ 警告：找到多个 {varname} 文件，使用第一个：{files[0]}")
    return files[0]

def vertically_integrated_moist_flux_divergence(
    hus: xr.DataArray, ua: xr.DataArray, va: xr.DataArray,
    lat: xr.DataArray, lon: xr.DataArray
) -> xr.DataArray:
    """
    返回单位为 W/m²（能量通量散度，L × ∇·(rV)）
    """
    r = mixing_ratio_from_specific_humidity(hus)
    r = xr.DataArray(r, coords=hus.coords, dims=hus.dims)

    r_u = r * ua
    r_v = r * va

    dx, dy = compute_dx_dy(lat, lon)

    dr_u_dx = xr.DataArray(np.gradient(r_u, axis=-1) / dx, coords=r_u.coords, dims=r_u.dims)
    dr_v_dy = xr.DataArray(np.gradient(r_v, axis=-2) / dy, coords=r_v.coords, dims=r_v.dims)

    div_rV = dr_u_dx + dr_v_dy

    # dp = np.gradient(hus.plev.values * 100)  # Pa
    # weights = (np.abs(dp) / g).reshape(1, -1, 1, 1)

    return L *( div_rV.integrate("plev"))

def calc_horizontal_GMS(
    ta: xr.DataArray, zg: xr.DataArray, plev: np.ndarray,
    lon: xr.DataArray, lat: xr.DataArray,
    ua: xr.DataArray, va: xr.DataArray, hus: xr.DataArray
) -> xr.DataArray:
    """
    输出：水平 GMS，单位：无量纲（单位化的能量通量比）
    """
    dse = calc_dse(ta, zg, plev)  # J/kg

    dx, dy = compute_dx_dy(lat, lon)
    ds_dx = xr.DataArray(np.gradient(dse, axis=-1) / dx, coords=dse.coords, dims=dse.dims)
    ds_dy = xr.DataArray(np.gradient(dse, axis=-2) / dy, coords=dse.coords, dims=dse.dims)

    v_grad_s = ua * ds_dx + va * ds_dy

    iv_dot_grad_s = (v_grad_s).integrate("plev")
    i_r = vertically_integrated_moist_flux_divergence(hus, ua, va, lat, lon)

    return -T_ref * iv_dot_grad_s / i_r

def calc_vertical_GMS(
    ta: xr.DataArray, zg: xr.DataArray, plev: np.ndarray,
    wa: xr.DataArray, hus: xr.DataArray, ua: xr.DataArray, va: xr.DataArray,
    lat: xr.DataArray, lon: xr.DataArray
) -> xr.DataArray:
    """
    输出：垂直 GMS，单位：无量纲
    """
    dse = calc_dse(ta, zg, plev)  # J/kg
    ds_dp = dse.differentiate("plev")  # J/kg/Pa，注意 dse 必须按 plev 定义
    ids_dp = (wa * ds_dp).integrate('plev')  # J/m²/s
    i_r = vertically_integrated_moist_flux_divergence(hus, ua, va, lat, lon)

    return -T_ref * ids_dp / i_r
    
def gross_moist_stability(
    ta_path: str, zg_path: str, ua_path: str, va_path: str, wa_path: str, hus_path: str
) -> tuple[xr.DataArray, xr.DataArray]:
    ta, lon, lat = load_data(ta_path, 'ta')
    zg, _, _     = load_data(zg_path, 'zg')
    ua, _, _     = load_data(ua_path, 'ua')
    va, _, _     = load_data(va_path, 'va')
    wa, _, _     = load_data(wa_path, 'wap')
    hus,_,_      = load_data(hus_path, 'hus')
    plev = ta.plev.values


    h_gms = calc_horizontal_GMS(ta, zg, plev, lon, lat, ua, va, hus)
    v_gms = calc_vertical_GMS(ta, zg, plev, wa, hus, ua, va, lat, lon)

    return h_gms, v_gms
