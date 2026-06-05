#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:55:03 2025

@author: xpji
"""
import xarray as xr
import numpy as np
from joblib import Parallel, delayed
from typing import Optional, List, Tuple
from scipy.signal import butter, filtfilt
import time


def butter_lowpass_filter(data: np.ndarray, cutoff_freq: float, fs: float, order: int = 4) -> np.ndarray:
    """
    应用 Butterworth 低通滤波器，滤除高于截止频率的信号成分。
    
    Parameters
    ----------
    data : np.ndarray
        一维时间序列数据。
    cutoff_freq : float
        截止频率 (Hz)。例如 1/10 表示滤除周期小于 10 天的信号。
    fs : float
        采样频率 (Hz)。例如逐日数据为 1。
    order : int, optional
        滤波器阶数，默认值为 4。
    
    Returns
    -------
    np.ndarray
        滤波后的时间序列。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normalized_cut = cutoff_freq / nyq
    b, a = butter(order, normalized_cut, btype='lowpass')
    return filtfilt(b, a, data)


def remove_10d_from_daily_data(data: np.ndarray, fs: float = 1.0, cutoff: float = 1/10, order: int = 4) -> np.ndarray:
    """
    对三维数组 (time, lat, lon) 应用低通滤波，滤除周期小于10天的信号。
    
    Parameters
    ----------
    data : np.ndarray
        三维数组 (time, lat, lon)。
    fs : float
        采样频率 (默认 1/day)。
    cutoff : float
        截止频率，默认 1/10。
    order : int
        Butterworth滤波器阶数。
    
    Returns
    -------
    np.ndarray
        滤波后的三维数组。
    """
    nt, nlat, nlon = data.shape
    filtered = np.empty_like(data)
    for i in range(nlat):
        for j in range(nlon):
            filtered[:, i, j] = butter_lowpass_filter(data[:, i, j], cutoff_freq=cutoff, fs=fs, order=order)
    return filtered


def remove_clm(data: xr.DataArray, fs: float = 1.0, cutoff: float = 1/10, order: int = 4) -> xr.DataArray:
    """
    去除季节循环气候态并对异常进行低通滤波，提取长于10天的信号分量。

    Parameters
    ----------
    data : xr.DataArray
        输入的三维逐日数据，需包含 'time', 'lat', 'lon' 三个维度。
    fs : float
        采样频率，默认每日为 1。
    cutoff : float
        截止频率，默认周期为 10 天。
    order : int
        Butterworth 滤波器阶数。

    Returns
    -------
    xr.DataArray
        滤波后的异常数据（时间序列滤除了 10 天以下信号）。
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("输入数据必须是 xarray.DataArray 类型")
    if 'time' not in data.dims:
        raise ValueError("输入数据必须包含 'time' 维度")

    # 去除气候态（季节循环）
    climatology = data.mean('time')
    anomalies = data - climatology

    # 低通滤波（保留10天以上周期）
    # filtered_data = remove_10d_from_daily_data(anomalies.values, fs=fs, cutoff=cutoff, order=order)

    return xr.DataArray(
        data=anomalies,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs,
        name='lowpass_filtered_anomaly'
    )


def find_local_extrema(V: np.ndarray) -> np.ndarray:
    """
    查找二维时序数据中的局部最大值和最小值点。
    
    参数:
    ----------
    V : np.ndarray
        输入二维数组，形状为 [time, lon]

    返回:
    ----------
    np.ndarray
        与输入同形状的数组，其中：
        - 1 表示局部最小值
        - -1 表示局部最大值
        - np.nan 表示其他位置
    """
    nt, nlon = V.shape
    local_min_max_id = np.full((nt, nlon), np.nan, dtype=np.float32)

    prev = V[:-2, :]
    curr = V[1:-1, :]
    next_ = V[2:, :]

    is_local_min = (curr <= prev) & (curr <= next_)
    is_local_max = (curr >= prev) & (curr >= next_)

    local_min_max_id[1:-1, :] = np.where(is_local_min, 1, local_min_max_id[1:-1, :])
    local_min_max_id[1:-1, :] = np.where(is_local_max, -1, local_min_max_id[1:-1, :])

    return local_min_max_id


def find_peak_influence_range(
    peak_idx: int,
    peak_value: float,
    zero_idx: np.ndarray,
    peak_indices: np.ndarray,
    V: np.ndarray,
    V_std: float,
    Nstd: float
) -> Tuple[int, int]:
    """
    确定某个峰值在时间序列中的影响范围。

    参数:
    ----------
    peak_idx : int
        峰值的时间索引。
    peak_value : float
        峰值对应的值。
    zero_idx : np.ndarray
        零点索引数组。
    peak_indices : np.ndarray
        所有峰值索引。
    V : np.ndarray
        当前经度对应的原始数据 (1D 时间序列)。
    V_std : float
        标准差阈值。
    Nstd : float
        判断显著峰值的标准差倍数。

    返回:
    ----------
    Tuple[int, int]
        峰值影响范围的左右时间索引边界 (inclusive)。
    """
    if np.abs(peak_value) < V_std * Nstd:
        return peak_idx, peak_idx

    # 与零点距离（正数 = 零点在右边，负数 = 零点在左边）
    dpeak_zero = zero_idx - peak_idx

    # 右边界
    pos_dist = np.where(dpeak_zero >= 0, dpeak_zero, np.inf)
    if np.all(np.isinf(pos_dist)):
        id_r = peak_idx
    else:
        id_r = zero_idx[np.argmin(pos_dist)]
        next_peaks = peak_indices[peak_indices > peak_idx]
        if next_peaks.size > 0:
            id_r = min(id_r, next_peaks.min() - 1)

    # 左边界
    neg_dist = np.where(dpeak_zero < 0, dpeak_zero, -np.inf)
    if np.all(neg_dist == -np.inf):
        id_l = peak_idx
    else:
        id_l = zero_idx[np.argmax(neg_dist)] + 1
        prev_peaks = peak_indices[peak_indices < peak_idx]
        if prev_peaks.size > 0:
            id_l = max(id_l, prev_peaks.max() + 1)

    return int(id_l), int(id_r)


   
def process_single_longitude(
    ilon: int,
    V: np.ndarray,
    local_min_max_id: np.ndarray,
    V_std: float,
    Nstd: float
) -> np.ndarray:
    """
    处理某个经度的时间序列，提取显著峰值并标记其影响范围。

    参数:
    ----------
    ilon : int
        经度索引。
    V : np.ndarray
        原始二维数据数组 [time, lon]。
    local_min_max_id : np.ndarray
        局部极值标记数组 [time, lon]，值为 1 (min), -1 (max), or NaN。
    V_std : float
        当前数据的标准差。
    Nstd : float
        显著性阈值，标准差倍数。

    返回:
    ----------
    np.ndarray
        一维数组 [time]，表示该经度上每个时刻对应的显著峰值或 NaN。
    """
    nt = V.shape[0]
    V_peak_lon = np.full(nt, np.nan, dtype=np.float32)

    # 找零交叉点（符号变化），时间维度上
    zero_idx = np.where(V[:-1, ilon] * V[1:, ilon] <= 0)[0]

    # 查找局部极值点索引
    peak_idx = np.where(np.abs(local_min_max_id[:, ilon]) == 1)[0]
    if len(peak_idx) == 0:
        return V_peak_lon

    # 遍历所有峰值点，判断是否显著，确定左右影响范围，并赋值
    for idx in peak_idx:
        peak_value = V[idx, ilon]
        if np.abs(peak_value) < V_std * Nstd:
            continue

        id_l, id_r = find_peak_influence_range(
            peak_idx=idx,
            peak_value=peak_value,
            zero_idx=zero_idx,
            peak_indices=peak_idx,
            V=V[:, ilon],
            V_std=V_std,
            Nstd=Nstd
        )

        V_peak_lon[id_l:id_r + 1] = peak_value

    return V_peak_lon    


def optimize_peak_detection(
    V: np.ndarray,
    kelvin_ref: xr.DataArray,
    V_std: float,
    Nstd: float = 1.0,
    use_parallel: bool = True,
    n_jobs: int = -1
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    主函数：执行局部极值查找与显著峰值影响范围分配

    参数:
        V (np.ndarray): 输入二维数组，形状为 [时间, 经度]，单位任意
        kelvin_ref (xr.DataArray): xarray 结构的参考数据，提供坐标与维度信息
        V_std (float): 输入数据的标准差（单位同V）
        Nstd (float, 可选): 判定显著性峰值的标准差倍数阈值，默认1
        use_parallel (bool, 可选): 是否使用并行处理（基于joblib），默认True
        n_jobs (int, 可选): 并行核数，-1为全部核心

    返回:
        V_peak_da (xr.DataArray): 峰值影响范围值分配结果，单位同V
        local_min_max_da (xr.DataArray): 局部极值标识结果，1为最小值，-1为最大值，非极值为NaN
    """
    nt, nlon = V.shape

    # Step 1: 查找局部极值
    t0 = time.time()
    local_min_max_id = find_local_extrema(V)
    print(f"[√] 局部极值检测耗时: {time.time() - t0:.2f} 秒")

    # 包装为 DataArray
    local_min_max_da = xr.DataArray(
        data=local_min_max_id,
        dims=kelvin_ref.dims,
        coords=kelvin_ref.coords,
        name='local_extrema'
    )

    # Step 2: 每个经度上进行峰值影响分配
    t1 = time.time()

    if use_parallel:
        # 并行处理每个经度
        V_peak_cols = Parallel(n_jobs=n_jobs)(
            delayed(process_single_longitude)(ilon, V, local_min_max_id, V_std, Nstd)
            for ilon in range(nlon)
        )
        V_peak = np.column_stack(V_peak_cols)
    else:
        # 顺序处理
        V_peak = np.full((nt, nlon), np.nan, dtype=np.float32)
        for ilon in range(nlon):
            V_peak[:, ilon] = process_single_longitude(ilon, V, local_min_max_id, V_std, Nstd)

    print(f"[√] 峰值影响范围赋值耗时: {time.time() - t1:.2f} 秒")

    # 包装为 DataArray
    V_peak_da = xr.DataArray(
        data=V_peak,
        dims=kelvin_ref.dims,
        coords=kelvin_ref.coords,
        name='peak_influence'
    )

    return V_peak_da, local_min_max_da


def meridional_projection(
    inputdata: xr.DataArray,
    lat: np.ndarray,
    lat_0: float = 9.0,
    lat_tropics: float = 10.0,
    omega: int = 0
) -> xr.DataArray:
    """
    对输入变量进行纬向投影，使用高斯加权和可选的纬向窗口（omega）限制。

    Parameters
    ----------
    inputdata : xr.DataArray
        输入数据，维度为 (time, lat, lon)
    lat : np.ndarray
        纬度数组，与 inputdata.lat 对应
    lat_0 : float, default=9.0
        高斯加权中控制尺度的参数
    lat_tropics : float, default=10.0
        omega 限定热带范围
    omega : int, default=0
        如果为0，限制在 |lat| <= lat_tropics 内；如果为1，则相反

    Returns
    -------
    xr.DataArray
        纬向投影后的数据，维度为 (time, lon)

    Examples
    --------
    >>> kelvin_eq = meridional_projection(kelvin_data, lat)
    >>> print(kelvin_eq.shape)  # (time, lon)
    """
    KW_coef = np.exp(-(lat / (2 * lat_0)) ** 2)

    if omega == 0:
        omega_mask = np.where(np.abs(lat) > lat_tropics, 0, 1)
    else:
        omega_mask = np.where(np.abs(lat) > lat_tropics, 1, 0)

    KW_filt = KW_coef * omega_mask
    SUM = np.sum(KW_filt)

    V = inputdata
    if V.ndim != 3 or V.dims != ('time', 'lat', 'lon'):
        raise ValueError("inputdata 必须是 (time, lat, lon) 的 xarray.DataArray")

    nt, nlat, nlon = V.shape
    V_T = np.transpose(V.data, (0, 2, 1))  # shape: (time, lon, lat)

    if np.sum(np.isnan(V_T)) == 0:
        # 无缺失值
        V_projected = np.inner(V_T, KW_filt) / SUM  # shape: (time, lon)
    else:
        # 有缺失值
        KW_filt_large = np.tile(KW_filt, (nt, nlon, 1))  # shape: (nt, nlon, nlat)
        KW_filt_masked = np.ma.array(KW_filt_large, mask=np.isnan(V_T))
        Vsum = np.nansum(V_T * KW_filt_masked, axis=2)
        KWsum = np.nansum(KW_filt_masked, axis=2)
        V_projected = Vsum / KWsum

    # 构建输出 DataArray
    return xr.DataArray(
        data=V_projected,
        dims=("time", "lon"),
        coords={
            "time": inputdata.time,
            "lon": inputdata.lon
        },
        name="meridional_projection"
    )


def calculate_kelvin_phase(
    kelvin_filtered: xr.DataArray,
    V_peak: xr.DataArray,
    correct_phase: bool = True
) -> xr.DataArray:
    """
    计算 Kelvin 波的相位。

    Parameters
    ----------
    kelvin_filtered : xr.DataArray
        滤波后的 Kelvin 波数据 (time, lon)
    V_peak : xr.DataArray
        峰值标记数据 (time, lon)
    correct_phase : bool, default=True
        是否进行相位修正（增强/衰减）

    Returns
    -------
    xr.DataArray
        相位数据 (time, lon)，范围 [-π, π]

    Examples
    --------
    >>> phase = calculate_kelvin_phase(kelvin_eq, V_peak)
    >>> print(f"Phase range: [{phase.min().values:.2f}, {phase.max().values:.2f}]")
    """
    V = kelvin_filtered.data
    V_peak_data = V_peak.data

    # 标记增强/衰减阶段
    nt, nlon = V.shape
    enh_dec = np.full((nt, nlon), np.nan)
    enh_dec[1:-1, :] = np.where(
        (V[1:-1, :] > V[0:-2, :]) & (V[1:-1, :] < V[2:, :]) & (~np.isnan(V_peak_data[1:-1, :])),
        0, np.nan
    )
    enh_dec[1:-1, :] = np.where(
        (V[1:-1, :] < V[0:-2, :]) & (V[1:-1, :] > V[2:, :]) & (~np.isnan(V_peak_data[1:-1, :])),
        1, enh_dec[1:-1, :]
    )

    # 计算相位
    V_norm = V / np.abs(V_peak_data)
    phase = np.arcsin(V_norm)
    phase = np.where(np.isfinite(V_peak_data), phase, np.nan)

    if correct_phase:
        # 相位修正
        phase_corrected = phase.copy()
        dec_neg = (enh_dec == 1) & (V_peak_data <= 0)
        dec_pos = (enh_dec == 1) & (V_peak_data >= 0)
        phase_corrected[dec_neg] = -np.pi - phase[dec_neg]
        phase_corrected[dec_pos] = np.pi - phase[dec_pos]
        phase_corrected = -phase_corrected
        phase = phase_corrected

    return xr.DataArray(
        data=phase,
        dims=kelvin_filtered.dims,
        coords=kelvin_filtered.coords,
        name="kelvin_phase"
    )


def phase_composite(
    data: xr.DataArray,
    phase: xr.DataArray,
    n_bins: int = None,
    phase_range: Tuple[float, float] = (-np.pi, np.pi)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对数据进行相位合成分析。

    Parameters
    ----------
    data : xr.DataArray
        需要合成的数据 (time, lon) 或 (time, lat, lon)
    phase : xr.DataArray
        相位数据 (time, lon)
    n_bins : int, optional
        相位分bin数量，默认根据 phase_range 自动计算
    phase_range : tuple, default=(-π, π)
        相位范围

    Returns
    -------
    phase_bin : np.ndarray
        相位bin中心值
    composite_mean : np.ndarray
        各bin的平均值
    composite_count : np.ndarray
        各bin的样本数

    Examples
    --------
    >>> phase_bin, composite_mean, count = phase_composite(pr_data, phase)
    >>> plt.plot(phase_bin, composite_mean)
    """
    from scipy import stats as stat

    if n_bins is None:
        dph = 1 / np.pi
        n_bins = int((phase_range[1] - phase_range[0]) / dph)
    else:
        dph = (phase_range[1] - phase_range[0]) / n_bins

    mybin = np.arange(phase_range[0], phase_range[1] + dph * 2, dph) - dph / 2
    phase_bin = mybin[:-1] + dph / 2

    # 展平数据
    phase_flat = phase.values.flatten()
    data_flat = data.values.flatten()

    # 移除NaN
    mask = ~np.isnan(phase_flat) & ~np.isnan(data_flat)
    phase_clean = phase_flat[mask]
    data_clean = data_flat[mask]

    # 统计
    composite_count = stat.binned_statistic(
        phase_clean, data_clean, statistic='count', bins=mybin
    ).statistic

    composite_mean = stat.binned_statistic(
        phase_clean, data_clean, statistic='mean', bins=mybin
    ).statistic

    return phase_bin, composite_mean, composite_count


def lag_composite(
    data: xr.DataArray,
    phase: xr.DataArray,
    lon: np.ndarray,
    lon_ref: float = 180.0,
    nlag: int = 10,
    phase_threshold: float = -np.pi / 2,
    tolerance: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对数据进行滞后合成分析。

    Parameters
    ----------
    data : xr.DataArray
        需要合成的数据 (time, lon)
    phase : xr.DataArray
        相位数据 (time, lon)
    lon : np.ndarray
        经度数组
    lon_ref : float, default=180.0
        参考经度
    nlag : int, default=10
        滞后步数
    phase_threshold : float, default=-π/2
        用于选择参考时刻的相位阈值
    tolerance : float, default=0.001
        相位匹配容差

    Returns
    -------
    tlag : np.ndarray
        滞后时间数组
    composite : np.ndarray
        合成数据 (2*nlag+1, nlon)
    it_max : np.ndarray
        参考时刻索引

    Examples
    --------
    >>> tlag, composite, _ = lag_composite(pr_data, phase, lon, lon_ref=180)
    >>> plt.contourf(lon, tlag, composite)
    """
    nt, nlon = data.shape

    # 找到参考经度
    ilon = np.argwhere(lon == lon_ref).squeeze()
    phase_ref = phase.data[:, ilon]

    # 找到满足相位条件的时刻
    it_max = np.argwhere(np.abs(phase_ref - phase_threshold) < tolerance).squeeze()
    it_max = np.where(((it_max <= nlag) | (it_max >= nt - nlag)), np.nan, it_max)
    it_max = np.delete(it_max, np.isnan(it_max) == 1)
    it_max = it_max.astype(dtype='int')

    # 初始化合成数组
    composite = np.full([2 * nlag + 1, nlon], np.nan)

    # lag=0
    composite[nlag, :] = np.nanmean(data.data[it_max, :], 0)

    # lag != 0
    for ilag in range(1, nlag + 1):
        composite[nlag - ilag, :] = np.nanmean(data.data[it_max - ilag, :], 0)
        composite[nlag + ilag, :] = np.nanmean(data.data[it_max + ilag, :], 0)

    tlag = np.arange(-nlag, nlag + 1, 1)

    return tlag, composite, it_max


def save_composite_to_netcdf(
    output_path: str,
    pr_ano_comp: np.ndarray,
    pr_kw_comp: np.ndarray,
    lon: np.ndarray,
    nlag: int,
):
    """
    Save Kelvin wave composite precipitation data to a NetCDF file.

    Parameters
    ----------
    output_path : str
        Full path to save the NetCDF file.
    pr_ano_comp : np.ndarray
        Precipitation anomaly composite, shape (2*nlag+1, nlon)
    pr_kw_comp : np.ndarray
        Kelvin wave filtered precipitation composite, shape (2*nlag+1, nlon)
    lon : np.ndarray
        Longitude array, shape (nlon,)
    nlag : int
        Lag window size (before/after maximum), total time dimension is 2*nlag+1
    """
    from netCDF4 import Dataset

    outdir = os.path.dirname(output_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    nlon = len(lon)
    ntlag = 2 * nlag + 1
    tlag = np.arange(-nlag, nlag + 1)
    lont, tlon = np.meshgrid(lon, tlag)

    with Dataset(output_path, 'w', format='NETCDF4') as nc:
        # Dimensions
        nc.createDimension('lon', nlon)
        nc.createDimension('tlag', ntlag)

        # Variables
        tlag_var = nc.createVariable('tlag', 'f4', ('tlag',))
        lon_var = nc.createVariable('lon', 'f4', ('lon',))

        prano_var = nc.createVariable('pr_ano_comp', 'f4', ('tlag', 'lon'))
        prkw_var = nc.createVariable('pr_kw_comp', 'f4', ('tlag', 'lon'))

        lont_var = nc.createVariable('lont', 'f4', ('tlag', 'lon'))
        tlon_var = nc.createVariable('tlon', 'f4', ('tlag', 'lon'))

        # Metadata
        tlag_var.standard_name = 'lag_time'
        tlag_var.units = 'lag (time steps from maximum)'

        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'

        prano_var.units = 'mm/day'
        prano_var.long_name = 'Composite of precipitation anomaly (climatology removed)'

        prkw_var.units = 'mm/day'
        prkw_var.long_name = 'Composite of Kelvin-wave-filtered precipitation'

        lont_var.long_name = 'Longitude meshgrid for plotting'
        tlon_var.long_name = 'Time lag meshgrid for plotting'

        # Write
        tlag_var[:] = tlag
        lon_var[:] = lon

        prano_var[:, :] = pr_ano_comp
        prkw_var[:, :] = pr_kw_comp

        lont_var[:, :] = lont
        tlon_var[:, :] = tlon

    print(f"✅ NetCDF saved to: {output_path}")


def composite_kw_phase(
    kelvin: xr.DataArray,
    pr_ori: xr.DataArray,
    lon: np.ndarray,
    model_name: str,
    output_dir: str,
    lon_ref: float = 180.0,
    nlag: int = 10,
    Nstd: float = 1.0,
    debug_plot: bool = False,
):
    """
    Perform phase composite analysis for Kelvin-wave-filtered precipitation and save results.

    This function wraps existing helpers in this module: meridional_projection,
    remove_clm, optimize_peak_detection, calculate_kelvin_phase, phase_composite,
    lag_composite, and save_composite_to_netcdf.
    """
    import os
    from sklearn import linear_model

    lat = kelvin.lat.values

    # meridional projection
    kelvin_ref = meridional_projection(kelvin, lat)
    pr_ori_eq = meridional_projection(pr_ori, lat)

    # remove climatology / anomalies
    pr_ano = remove_clm(pr_ori)
    pr_ano_eq = meridional_projection(pr_ano, lat)

    # peak detection
    V = kelvin_ref.data
    V_std = np.nanstd(V)
    V_peak, _ = optimize_peak_detection(V, kelvin_ref, V_std, Nstd=Nstd)

    print('Std of variable:', V_std)

    # compute phase
    phase = calculate_kelvin_phase(kelvin_ref, V_peak)

    # phase composite
    phase_bin, pr_kw_phase_mean, counts = phase_composite(kelvin_ref, phase)

    # save simple phase result
    np.savez(
        os.path.join(output_dir, f'{model_name}_precip_kw_{lon_ref}.npz'),
        KW_filtered_pr=kelvin_ref,
        lon=lon,
        phase_bin=phase_bin,
        phase_correct=phase,
        pr_kw=pr_kw_phase_mean,
    )

    # lag composite
    tlag, pr_kw_comp, it_max = lag_composite(kelvin_ref, phase, lon, lon_ref=lon_ref, nlag=nlag)
    pr_ano_comp = np.full_like(pr_kw_comp, np.nan)
    # For anomaly composite we need pr_ano_eq
    if pr_ano_eq is not None:
        # align pr_ano_eq shape
        pr_ano_comp = np.full_like(pr_kw_comp, np.nan)
        # compute center composite
        if it_max.size > 0:
            pr_ano_comp[nlag, :] = np.nanmean(pr_ano_eq.data[it_max, :], axis=0)
            pr_kw_comp[nlag, :] = np.nanmean(pr_kw_comp[nlag, :]) if pr_kw_comp is not None else pr_kw_comp[nlag, :]

    # save netcdf composite
    out_nc = os.path.join(output_dir, f'{model_name}_kw_composite_lag_lon_prano_prkw_history_{lon_ref}.nc')
    save_composite_to_netcdf(out_nc, pr_ano_comp, pr_kw_comp, lon, nlag)

    # compute simple metrics: zonal wavenumber and frequency via lag regression on strong points
    if model_name != 'GPCP':
        pr_kw_plot = pr_kw_comp * 86400
    else:
        pr_kw_plot = pr_kw_comp

    # thresholding
    pr_strong = np.where(pr_kw_plot >= 1.5, pr_kw_plot, np.nan)
    istrong = np.argwhere(~np.isnan(pr_strong))
    if istrong.size == 0:
        print('No strong points found for regression')
        return

    itlag_strong = istrong[:, 0]
    ilon_strong = istrong[:, 1]

    # regression lon ~ tlag
    X = tlag[itlag_strong].reshape(-1, 1)
    y = lon[ilon_strong]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    b1 = regr.coef_[0]
    Cp_ave = b1 * 111000.0 / (24 * 3600.0)

    np.savez(
        os.path.join(output_dir, f'{model_name}_kw_zwnum_freq_Cp_ave_from_precip_lag_regression_history_{lon_ref}.npz'),
        Cp_ave=Cp_ave,
    )

    # plotting (minimal)
    if debug_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(phase_bin, pr_kw_phase_mean)
        plt.title(f'{model_name} KW phase composite')
        plt.show()


if __name__ == "__main__":
    print("Phase Analysis Module for Wave Tools")
    print("====================================")
    print("\nAvailable functions:")
    print("  - remove_clm: 去除季节循环气候态")
    print("  - optimize_peak_detection: 峰值检测")
    print("  - meridional_projection: 纬向投影")
    print("  - calculate_kelvin_phase: 计算Kelvin波相位")
    print("  - phase_composite: 相位合成分析")
    print("  - lag_composite: 滞后合成分析")
    print("\nExample usage:")
    print("  >>> from wave_tools.phase import meridional_projection, calculate_kelvin_phase")
    print("  >>> kelvin_eq = meridional_projection(kelvin_data, lat)")
    print("  >>> phase = calculate_kelvin_phase(kelvin_eq, V_peak)")





























