"""
Cross-Spectrum Analysis Module
================================

交叉谱分析工具，用于计算两个变量之间的波数-频率交叉谱。

主要功能：
---------
1. 交叉功率谱计算
2. 相干性和相位谱计算
3. 对称/反对称分解
4. 矢量场可视化支持

作者: Jianpu
邮箱: xianpuji@hhu.edu.cn
"""

import numpy as np
import xarray as xr
from scipy import signal, fft
from typing import Dict, Optional, Tuple, Union
import warnings


class CrossSpectrumConfig:
    """交叉谱分析配置参数"""
    
    # 分析参数
    SEGMENT_LENGTH = 96  # 时间段长度（天）
    SEGMENT_OVERLAP = -65  # 时间段重叠（负数表示间隔）
    
    # 窗函数参数
    TAPER_FRACTION = 0.10  # Tukey窗的taper百分比
    
    # 平滑参数
    SMOOTH_FACTOR = 2.667  # 1-2-1平滑器的自由度系数
    
    # 滤波参数
    FREQ_CUTOFF = 1.0 / 365.0  # 去除年循环的频率阈值
    SAMPLES_PER_DAY = 1
    
    # 显著性检验
    PROB_LEVELS = np.array([0.80, 0.85, 0.90, 0.925, 0.95, 0.99])


def nan_to_value_by_interp_3D(V: np.ndarray) -> np.ndarray:
    """
    对3维数据 (time, lat, lon) 进行NaN插值
    
    采用周围点的均值填充NaN
    
    Parameters
    ----------
    V : np.ndarray
        输入数据，形状为 (time, lat, lon)
    
    Returns
    -------
    np.ndarray
        插值后的数据
    """
    V_nonan = np.where(np.isnan(V), np.nan, V)
    
    nanloc = np.argwhere(np.isnan(V))
    nannum = np.size(nanloc, 0)

    if nannum == 0:
        return V_nonan

    # 获取数据维度
    nt, nlat, nlon = V.shape

    for i in range(nannum):
        t, lat, lon = nanloc[i]

        # 时间维度插值
        V_previous = V[t-1, lat, lon] if t > 0 else np.nan
        V_later = V[t+1, lat, lon] if t < nt - 1 else np.nan

        # 纬度维度插值
        V_N = V[t, lat+1, lon] if lat < nlat - 1 else np.nan
        V_S = V[t, lat-1, lon] if lat > 0 else np.nan

        # 经度维度插值（经度通常是环绕的）
        V_E = V[t, lat, lon+1] if lon < nlon - 1 else V[t, lat, 0]
        V_W = V[t, lat, lon-1] if lon > 0 else V[t, lat, -1]

        # 计算周围点的均值（忽略 NaN）
        V_nonan[t, lat, lon] = np.nanmean(
            np.array([V_E, V_W, V_N, V_S, V_previous, V_later])
        )

    return V_nonan


def remove_annual_cycle(
    data: Union[xr.DataArray, np.ndarray],
    spd: int = 1,
    # fCrit: float = 1.0/365.0
) -> Union[xr.DataArray, np.ndarray]:
    """
    去除数据的年循环和趋势
    
    步骤1: 去除线性趋势
    步骤2: 去除频率小于fCrit的成分
    
    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        输入数据，形状为 (time, lat, lon)
    spd : int
        每天的样本数
    fCrit : float
        临界频率（低于此频率的成分将被去除）
    
    Returns
    -------
    xr.DataArray or np.ndarray
        处理后的数据
    """
    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        dims = data.dims
        coords = data.coords
        data_values = data.values
    else:
        data_values = data
    
    ntim, nlat, nlon = data_values.shape
    
    # 步骤1: 去趋势
    detrend = signal.detrend(data_values, axis=0)
    
    # 步骤2: FFT并去除低频成分
    rf = fft.rfft(detrend, axis=0)
    freq = fft.rfftfreq(ntim, d=1.0 / float(spd))
    harmonics = [1/365.0, 2/365.0, 3/365.0]
    for hf in harmonics:
        idx = np.argmin(np.abs(freq - hf))
        rf[idx, ...] = 0.0
    
    # 逆FFT
    datain = fft.irfft(rf, axis=0, n=ntim)
    
    if is_xarray:
        return xr.DataArray(datain, dims=dims, coords=coords)
    else:
        return datain


def _smooth121_1D(array_in: np.ndarray) -> np.ndarray:
    """
    1-2-1平滑器（一维）
    
    Parameters
    ----------
    array_in : np.ndarray
        输入一维数组
    
    Returns
    -------
    np.ndarray
        平滑后的数组
    """
    temp = np.copy(array_in)
    array_out = np.copy(temp) * 0.0

    for i in range(len(temp)):
        if np.isnan(temp[i]):
            array_out[i] = np.nan
        elif i == 0 or np.isnan(temp[i-1]):
            array_out[i] = (3*temp[i] + temp[i+1]) / 4
        elif i == (len(temp)-1) or np.isnan(temp[i+1]):
            array_out[i] = (3*temp[i] + temp[i-1]) / 4
        else:
            array_out[i] = (temp[i+1] + 2*temp[i] + temp[i-1]) / 4

    return array_out


def _smooth121_frequency(STC: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """
    在频率维度上应用1-2-1平滑器
    
    Parameters
    ----------
    STC : np.ndarray
        谱数组，形状为 (nvar, nfreq, nwave)
    freq : np.ndarray
        频率数组
    
    Returns
    -------
    np.ndarray
        平滑后的谱数组
    """
    nvar, nfreq, nwave = STC.shape
    
    # 找到零频率的索引
    indfreqzero = np.where(freq == 0)[0]
    
    if len(indfreqzero) > 0:
        indfreqzero = indfreqzero[0]
        
        # 对前4个变量（功率谱和交叉谱）在频率维度上平滑
        for wv in range(nwave):
            for var_idx in range(4):
                STC[var_idx, indfreqzero + 1:, wv] = _smooth121_1D(
                    STC[var_idx, indfreqzero + 1:, wv]
                )
    
    return STC


def _get_symm_asymm(
    X: np.ndarray,
    lat: np.ndarray,
    mode: str = 'symm'
) -> np.ndarray:
    """
    将数据分解为对称或反对称分量
    
    Parameters
    ----------
    X : np.ndarray
        输入数据，形状为 (time, lat, lon)
    lat : np.ndarray
        纬度数组
    mode : str
        'symm' 或 'asymm'
    
    Returns
    -------
    np.ndarray
        对称或反对称分量
    """
    NT, NM, NL = X.shape
    
    if mode == 'symm':
        x = X[:, lat[:] >= 0, :]
        if len(lat) % 2 == 1:
            for ll in range(NM // 2 + 1):
                x[:, ll, :] = 0.5 * (X[:, ll, :] + X[:, NM - ll - 1, :])
        else:
            for ll in range(NM // 2):
                x[:, ll, :] = 0.5 * (X[:, ll, :] + X[:, NM - ll - 1, :])
    elif mode in ['asymm', 'anti-symm']:
        x = X[:, lat[:] > 0, :]
        if len(lat) % 2 == 1:
            for ll in range(NM // 2):
                x[:, ll, :] = 0.5 * (X[:, ll, :] - X[:, NM - ll - 1, :])
        else:
            for ll in range(NM // 2):
                x[:, ll, :] = 0.5 * (X[:, ll, :] - X[:, NM - ll - 1, :])
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'symm' or 'asymm'")
    
    return x


def _cross_spectrum_segment(XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """
    计算一个时间段的FFT交叉谱
    
    Parameters
    ----------
    XX, YY : np.ndarray
        输入数据段，形状为 (time, lat, lon)
    
    Returns
    -------
    np.ndarray
        谱数组，形状为 (8, nfreq, nwave)
        包含: [PX, PY, CXY, QXY, COH2, PHAS, V1, V2]
    """
    NT, NM, NL = XX.shape
    
    # 转置为 (lat, lon, time)
    XX = np.transpose(XX, axes=[1, 2, 0])
    YY = np.transpose(YY, axes=[1, 2, 0])
    
    # 在时间和经度维度上进行2D FFT
    Xfft = np.fft.rfft2(XX, axes=(1, 2))
    Yfft = np.fft.rfft2(YY, axes=(1, 2))
    
    # 转置回 (time, lat, lon)
    Xfft = np.transpose(Xfft, axes=[2, 0, 1])
    Yfft = np.transpose(Yfft, axes=[2, 0, 1])
    
    # 归一化
    Xfft = Xfft / (NT * NL)
    Yfft = Yfft / (NT * NL)
    
    # 将零频率和零波数移到中心
    Xfft = np.fft.fftshift(Xfft, axes=(2,))
    Yfft = np.fft.fftshift(Yfft, axes=(2,))
    
    # 计算功率谱（在纬度上平均）
    PX = np.average(np.square(np.abs(Xfft)), axis=1)
    PY = np.average(np.square(np.abs(Yfft)), axis=1)
    
    # 计算交叉谱（共谱和正交谱）
    PXY = np.conj(Xfft) * Yfft
    CXY = np.average(np.real(PXY), axis=1)
    QXY = np.average(np.imag(PXY), axis=1)
    
    # 翻转波数维度
    PX = PX[:, ::-1]
    PY = PY[:, ::-1]
    CXY = CXY[:, ::-1]
    QXY = QXY[:, ::-1]
    
    # 确定频率和波数的维度
    if NT % 2 == 1:  # 奇数时间点
        nfreq = (NT + 1) // 2
    else:
        nfreq = NT // 2 + 1
    
    if NL % 2 == 1:
        nwave = NL
    else:
        nwave = NL + 1
    
    # 初始化谱数组
    STC = np.zeros([8, nfreq, nwave], dtype='double')
    
    # 存储功率谱和交叉谱
    if NL % 2 == 1:
        STC[0, :nfreq, :NL] = PX
        STC[1, :nfreq, :NL] = PY
        STC[2, :nfreq, :NL] = CXY
        STC[3, :nfreq, :NL] = QXY
    else:
        STC[0, :nfreq, 1:NL + 1] = PX
        STC[1, :nfreq, 1:NL + 1] = PY
        STC[2, :nfreq, 1:NL + 1] = CXY
        STC[3, :nfreq, 1:NL + 1] = QXY
        STC[:, :, 0] = STC[:, :, NL]
    
    return STC


def _compute_coherence_phase(STC: np.ndarray, normalize_by_reference: bool = False) -> np.ndarray:
    """
    从平均功率谱和交叉谱计算相干性平方和相位谱
    
    Parameters
    ----------
    STC : np.ndarray
        谱数组，形状为 (8或10, nfreq, nwave)
    normalize_by_reference : bool
        是否计算标准化谱
    
    Returns
    -------
    np.ndarray
        更新后的谱数组，包含相干性和相位
    """
    PX = STC[0, :, :]
    PY = STC[1, :, :]
    CXY = STC[2, :, :]
    QXY = STC[3, :, :]
    
    # 避免除零
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PY_safe = np.where(PY == 0, np.nan, PY)
        
        # 相干性平方
        COH2 = (np.square(CXY) + np.square(QXY)) / (PX * PY_safe)
        
        # 相位
        PHAS = np.arctan2(QXY, CXY)
        
        # 矢量分量（用于矢量场绘图）
        norm = np.sqrt(np.square(QXY) + np.square(CXY))
        norm_safe = np.where(norm == 0, np.nan, norm)
        V1 = -QXY / norm_safe  # cos component
        V2 = CXY / norm_safe   # sin component
    
    STC[4, :, :] = COH2
    STC[5, :, :] = PHAS
    STC[6, :, :] = V1
    STC[7, :, :] = V2
    
    return STC


def calculate_cross_spectrum(
    X: Union[xr.DataArray, np.ndarray],
    Y: Union[xr.DataArray, np.ndarray],
    segLen: int = 96,
    segOverLap: int = -65,
    symmetry: str = 'symm',
    return_xarray: bool = True,
    normalize_by_reference: bool = False,
    latent_heat: float = 2.5e6
) -> Dict:
    """
    计算两个变量的交叉谱
    
    基于Yasunaga et al. (2019, J. Climate) 的方法实现。
    
    Parameters
    ----------
    X, Y : xr.DataArray or np.ndarray
        输入数据，形状为 (time, lat, lon)
        如果是DataArray，必须包含 'lat' 坐标
        通常X是参考变量（如降水），Y是待分析变量
    segLen : int
        时间段长度（天）
    segOverLap : int
        时间段重叠（负数表示间隔）
    symmetry : str
        对称性选择: 'symm' 或 'asymm'
    return_xarray : bool
        是否返回xarray格式的结果
    normalize_by_reference : bool
        是否用参考变量X的功率谱标准化交叉谱（Yasunaga et al. 2019方法）
        如果True，返回标准化的交叉谱：(X* · Y) / (L · |X* · X|)
        这使得结果可解释为"每单位参考变量变化对应的Y变化"
    latent_heat : float
        潜热（J/kg），仅在normalize_by_reference=True时使用
        默认值: 2.5e6 J/kg
    
    Returns
    -------
    dict
        包含以下键的字典:
        - 'STC': 谱数组 (8 或 9, nfreq, nwave) 或 xr.DataArray
          [0]: PX - X的功率谱
          [1]: PY - Y的功率谱
          [2]: CXY - 共谱（co-spectrum）
          [3]: QXY - 正交谱（quadrature spectrum）
          [4]: COH2 - 相干性平方
          [5]: PHAS - 相位
          [6]: V1 - 矢量场cos分量
          [7]: V2 - 矢量场sin分量
          [8]: NORM_CXY_REAL - 标准化共谱实部（仅当normalize_by_reference=True）
          [9]: NORM_CXY_IMAG - 标准化共谱虚部（仅当normalize_by_reference=True）
        - 'freq': 频率数组
        - 'wave': 波数数组
        - 'nseg': 时间段数量
        - 'dof': 自由度
        - 'p': 显著性水平
        - 'prob_coh2': 相干性平方的显著性阈值
    
    Raises
    ------
    ValueError
        如果X和Y的形状不匹配
    
    Example
    -------
    >>> # 基本交叉谱
    >>> result = calculate_cross_spectrum(pr_data, div_data, segLen=96, segOverLap=-65)
    >>> coherence_squared = result['STC'][4]
    >>> phase = result['STC'][5]
    
    >>> # Yasunaga et al. (2019) 标准化方法
    >>> result = calculate_cross_spectrum(
    ...     pr_data, mse_advection, segLen=96, segOverLap=-65,
    ...     normalize_by_reference=True, latent_heat=2.5e6
    ... )
    >>> normalized_cospectrum = result['STC'][8]  # 实部：振幅放大/衰减
    >>> normalized_quadspectrum = result['STC'][9]  # 虚部：相位超前/滞后
    
    References
    ----------
    Yasunaga, K., S. Yokoi, K. Inoue, and B. E. Mapes, 2019: 
    Space–Time Spectral Analysis of the Moist Static Energy Budget Equation.
    J. Climate, 32, 501-529, https://doi.org/10.1175/JCLI-D-18-0334.1
    """
    # 检查输入类型并提取数据
    is_xarray = isinstance(X, xr.DataArray)
    if is_xarray:
        if not isinstance(Y, xr.DataArray):
            raise ValueError("X和Y必须都是xarray.DataArray或numpy.ndarray")
        lat = X.lat.values
        x_values = X.values
        y_values = Y.values
    else:
        if isinstance(Y, xr.DataArray):
            raise ValueError("X和Y必须都是xarray.DataArray或numpy.ndarray")
        x_values = X
        y_values = Y
        # 假设纬度是对称的
        ntim, nlat, mlon = x_values.shape
        lat = np.linspace(-14, 14, nlat)  # 默认值，可能需要调整
    
    # 检查形状
    ntim, nlat, mlon = x_values.shape
    ntim1, nlat1, mlon1 = y_values.shape
    
    if ntim != ntim1 or nlat != nlat1 or mlon != mlon1:
        raise ValueError(
            f"X和Y的形状必须相同: X{(ntim, nlat, mlon)} vs Y{(ntim1, nlat1, mlon1)}"
        )
    
    # 对称/反对称分解
    y = _get_symm_asymm(y_values, lat, symmetry)
    x = _get_symm_asymm(x_values, lat, symmetry)
    
    # 去趋势
    x = signal.detrend(x, 0)
    y = signal.detrend(y, 0)
    
    # 生成Tukey窗
    window = signal.windows.tukey(segLen, CrossSpectrumConfig.TAPER_FRACTION, True)
    
    # 设置频率和波数维度
    if segLen % 2 == 1:
        nfreq = (segLen + 1) // 2
    else:
        nfreq = segLen // 2 + 1
    
    if mlon % 2 == 1:
        nwave = mlon
    else:
        nwave = mlon + 1
    
    # 初始化谱数组（始终用8个分量累加，标准化在最后添加）
    STC = np.zeros([8, nfreq, nwave], dtype='double')
    
    # 计算波数和频率数组
    wave = np.arange(-int(nwave / 2), int(nwave / 2) + 1, 1.0)
    if segLen % 2 == 1:
        freq = np.linspace(0, 0.5*(segLen-1)/segLen, num=nfreq)
    else:
        freq = np.linspace(0, 0.5, num=nfreq)
    
    # 找到零频率索引
    indfreq0 = np.where(freq == 0.0)[0]
    
    # 循环处理时间段
    kseg = 0
    ntStrt = 0
    
    while ntStrt + segLen <= ntim:
        ntLast = ntStrt + segLen
        
        # 应用窗函数
        XX = x[ntStrt:ntLast, :, :] * window[:, np.newaxis, np.newaxis]
        YY = y[ntStrt:ntLast, :, :] * window[:, np.newaxis, np.newaxis]
        
        # 计算该段的交叉谱
        STCseg = _cross_spectrum_segment(XX, YY)
        
        # 将零频率设为NaN
        STCseg[:, indfreq0, :] = np.nan
        
        # 在频率维度上平滑
        STCseg = _smooth121_frequency(STCseg, freq)
        
        # 累加谱
        STC = STC + STCseg
        
        kseg += 1
        ntStrt = ntLast + segOverLap - 1
    
    if kseg == 0:
        raise ValueError(f"数据长度不足以计算交叉谱: ntim={ntim}, segLen={segLen}")
    
    # 平均谱
    STC = STC / kseg
    
    # 计算相干性和相位
    STC = _compute_coherence_phase(STC, normalize_by_reference=normalize_by_reference)
    
    # 如果需要标准化，按Yasunaga et al. (2019)方法处理
    if normalize_by_reference:
        # 扩展STC数组以包含标准化分量
        STC_extended = np.zeros([10, nfreq, nwave], dtype='double')
        STC_extended[:8, :, :] = STC  # 复制前8个分量
        
        PX = STC[0, :, :]  # X的功率谱（通常是降水）
        CXY = STC[2, :, :]  # 共谱（实部）
        QXY = STC[3, :, :]  # 正交谱（虚部）
        
        # 避免除零
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PX_safe = np.where(PX == 0, np.nan, PX)
            
            # 按公式标准化：(X* · Y) / (L · |X* · X|)
            # 注意：CXY + i*QXY 就是 X* · Y
            # PX 就是 |X* · X|
            STC_extended[8, :, :] = CXY / (latent_heat * PX_safe)  # 标准化实部
            STC_extended[9, :, :] = QXY / (latent_heat * PX_safe)  # 标准化虚部
        
        STC = STC_extended  # 使用扩展的数组
    
    # 计算自由度和显著性水平
    dof = CrossSpectrumConfig.SMOOTH_FACTOR * kseg
    prob = CrossSpectrumConfig.PROB_LEVELS
    prob_coh2 = 1 - np.power(1 - prob, 1.0 / (0.5 * dof - 1))
    
    # 构建结果字典
    result = {
        'freq': freq,
        'wave': wave,
        'nseg': kseg,
        'dof': dof,
        'p': prob,
        'prob_coh2': prob_coh2
    }
    
    # 转换为xarray格式（如果需要）
    if return_xarray:
        if normalize_by_reference:
            component_names = ['PX', 'PY', 'CXY', 'QXY', 'COH2', 'PHAS', 'V1', 'V2', 
                             'NORM_CXY_REAL', 'NORM_CXY_IMAG']
            description = (
                'PX: X power spectrum, '
                'PY: Y power spectrum, '
                'CXY: Co-spectrum, '
                'QXY: Quadrature spectrum, '
                'COH2: Coherence squared, '
                'PHAS: Phase, '
                'V1: Vector cos component, '
                'V2: Vector sin component, '
                'NORM_CXY_REAL: Normalized co-spectrum (real part), '
                'NORM_CXY_IMAG: Normalized quadrature spectrum (imaginary part)'
            )
        else:
            component_names = ['PX', 'PY', 'CXY', 'QXY', 'COH2', 'PHAS', 'V1', 'V2']
            description = (
                'PX: X power spectrum, '
                'PY: Y power spectrum, '
                'CXY: Co-spectrum, '
                'QXY: Quadrature spectrum, '
                'COH2: Coherence squared, '
                'PHAS: Phase, '
                'V1: Vector cos component, '
                'V2: Vector sin component'
            )
        
        result['STC'] = xr.DataArray(
            STC,
            dims=('component', 'frequency', 'wavenumber'),
            coords={
                'component': component_names,
                'frequency': freq,
                'wavenumber': wave
            },
            attrs={
                'long_name': 'Cross-spectrum components',
                'description': description,
                'nseg': kseg,
                'dof': dof,
                'normalized': normalize_by_reference
            }
        )
    else:
        result['STC'] = STC
    
    return result


# ===== 便捷函数 =====

def quick_cross_spectrum(
    X: xr.DataArray,
    Y: xr.DataArray,
    remove_annual: bool = True,
    **kwargs
) -> Dict:
    """
    快速计算交叉谱的便捷函数
    
    自动处理年循环去除和异常计算
    
    Parameters
    ----------
    X, Y : xr.DataArray
        输入数据，形状为 (time, lat, lon)
    remove_annual : bool
        是否去除年循环
    **kwargs : dict
        传递给 calculate_cross_spectrum 的其他参数
    
    Returns
    -------
    dict
        交叉谱结果
    """
    # 去除年循环（如果需要）
    if remove_annual:
        X_ano = X.groupby('time.dayofyear') - X.groupby('time.dayofyear').mean()
        Y_ano = Y.groupby('time.dayofyear') - Y.groupby('time.dayofyear').mean()
    else:
        X_ano = X
        Y_ano = Y
    
    # 计算交叉谱
    return calculate_cross_spectrum(X_ano, Y_ano, **kwargs)


__all__ = [
    'CrossSpectrumConfig',
    'calculate_cross_spectrum',
    'quick_cross_spectrum',
    'remove_annual_cycle',
    'nan_to_value_by_interp_3D',
]
