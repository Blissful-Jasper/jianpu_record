"""
Spectral Analysis Module
========================

热带大气波动的频谱分析工具，实现Wheeler-Kiladis波数-频率谱分析。

主要功能：
---------
1. 对称/反对称分解
2. 波数-频率功率谱计算
3. 背景谱平滑
4. 频谱归一化和可视化

作者: Jianpu
邮箱: xianpuji@hhu.edu.cn
"""

import numpy as np
import xarray as xr
import scipy.signal as signal
from scipy import fft
import matplotlib.pyplot as plt
import time
from typing import Optional, Tuple

try:
    import cmaps
    DEFAULT_COLORMAP = cmaps.NCV_blu_red
except ImportError:
    DEFAULT_COLORMAP = 'RdBu_r'


class SpectralConfig:
    """频谱分析配置参数"""
    
    # 分析参数
    WINDOW_SIZE_DAYS = 96
    WINDOW_SKIP_DAYS = 30
    SAMPLES_PER_DAY = 1
    
    # 滤波参数
    FREQ_CUTOFF = 1.0 / WINDOW_SIZE_DAYS
    
    # 绘图参数
    CONTOUR_LEVELS = np.array([0.0, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 4.0])
    COLORMAP = DEFAULT_COLORMAP
    WAVENUMBER_LIMIT = 15


# ============= 工具函数 =============

def smooth_121(array: np.ndarray) -> np.ndarray:
    """应用1-2-1平滑滤波器"""
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        return array
    ok = np.isfinite(array)
    if not np.any(ok):
        return np.full_like(array, np.nan)
    if not np.all(ok):
        x = np.arange(array.size)
        array = np.interp(x, x[ok], array[ok])
    weight = np.array([1., 2., 1.]) / 4.0
    return np.convolve(np.r_[array[0], array, array[-1]], weight, 'valid')


def remove_annual_cycle(data: xr.DataArray, samples_per_day: float, freq_cutoff: float) -> xr.DataArray:
    """去除年循环信号"""
    n_time, _, _ = data.shape
    
    # 去趋势
    detrended_data = signal.detrend(data, axis=0)
    
    # FFT
    fourier_transform = fft.rfft(detrended_data, axis=0)
    frequencies = fft.rfftfreq(n_time, d=1. / float(samples_per_day))
    
    # 低频滤波
    cutoff_index = np.argwhere(frequencies <= freq_cutoff).max()
    if cutoff_index > 1:
        fourier_transform[1:cutoff_index + 1, ...] = 0.0
    
    # 逆FFT
    filtered_data = fft.irfft(fourier_transform, axis=0, n=n_time)
    
    return xr.DataArray(filtered_data, dims=data.dims, coords=data.coords)


def decompose_symmetric_antisymmetric(data_array: xr.DataArray) -> xr.DataArray:
    """
    对称/反对称分解（完全按照Wheeler-Kiladis原始方法）
    
    注意：
    -----
    - symmetric = 0.5 * (data - flip(data))     # 南北反相
    - antisymmetric = 0.5 * (data + flip(data)) # 南北同相
    - 结果数组：南半球存储symmetric，北半球存储antisymmetric
    
    参数:
    -----
    data_array : xr.DataArray
        输入数据，维度为 (time, lat, lon)
    
    返回:
    -----
    result : xr.DataArray
        处理后的数据，南半球=对称分量，北半球=反对称分量
    """
    lat_dim = data_array.dims.index('lat')
    nlat = data_array.shape[lat_dim]
    
    # 计算对称和反对称分量（原始公式）
    symmetric = 0.5 * (data_array.values - np.flip(data_array.values, axis=lat_dim))
    antisymmetric = 0.5 * (data_array.values + np.flip(data_array.values, axis=lat_dim))
    
    # 转为DataArray
    symmetric = xr.DataArray(symmetric, dims=data_array.dims, coords=data_array.coords)
    antisymmetric = xr.DataArray(antisymmetric, dims=data_array.dims, coords=data_array.coords)
    
    # 组合结果：南半球=对称，北半球=反对称
    result = data_array.copy()
    half = nlat // 2
    
    if nlat % 2 == 0:
        # 偶数纬度
        result.values[:, :half, :] = symmetric.values[:, :half, :]
        result.values[:, half:, :] = antisymmetric.values[:, half:, :]
    else:
        # 奇数纬度（包含赤道）
        result.values[:, :half, :] = symmetric.values[:, :half, :]
        result.values[:, half+1:, :] = antisymmetric.values[:, half+1:, :]
        result.values[:, half, :] = symmetric.values[:, half, :]  # 赤道使用对称分量
    
    return result


# ============= 主类 =============

class WKSpectralAnalysis:
    """Wheeler-Kiladis频谱分析类"""
    
    def __init__(self, config: Optional[SpectralConfig] = None):
        """
        初始化频谱分析
        
        参数:
        ----
        config : SpectralConfig, optional
            配置参数对象
        """
        self.config = config or SpectralConfig()
        self.raw_data = None
        self.processed_data = None
        self.power_symmetric = None
        self.power_antisymmetric = None
        self.background = None
        self.frequency = None
        self.wavenumber = None
        
    def load_data(self, 
                  data: Optional[xr.DataArray] = None,
                  data_path: Optional[str] = None,
                  variable: str = 'olr',
                  lat_range: Tuple[float, float] = (-15, 15),
                  time_range: Optional[Tuple[str, str]] = None) -> 'WKSpectralAnalysis':
        """
        加载数据
        
        参数:
        ----
        data : xr.DataArray, optional
            直接提供的数据数组
        data_path : str, optional
            NetCDF文件路径
        variable : str
            变量名
        lat_range : tuple
            纬度范围
        time_range : tuple, optional
            时间范围
            
        返回:
        ----
        self
        """
        if data is not None:
            self.raw_data = data
        elif data_path is not None:
            ds = xr.open_dataset(data_path).sortby('lat')
            self.raw_data = ds[variable].sel(lat=slice(*lat_range))
            if time_range:
                self.raw_data = self.raw_data.sel(time=slice(*time_range))
        else:
            raise ValueError("必须提供data或data_path")
        
        # 确保维度顺序
        self.raw_data = self.raw_data.transpose('time', 'lat', 'lon')
        print(f"数据已加载: {self.raw_data.shape} (time, lat, lon)")
        return self
    
    def preprocess(self) -> 'WKSpectralAnalysis':
        """
        预处理数据：去趋势、去年循环、对称/反对称分解
        
        返回:
        ----
        self
        """
        print("预处理中...")
        start_time = time.time()
        
        # 去趋势
        mean_value = self.raw_data.mean(dim='time')
        detrended = signal.detrend(self.raw_data, axis=0, type='linear')
        detrended = xr.DataArray(detrended, dims=self.raw_data.dims, coords=self.raw_data.coords) + mean_value
        
        # 去年循环
        filtered = remove_annual_cycle(detrended, self.config.SAMPLES_PER_DAY, self.config.FREQ_CUTOFF)
        
        # 对称/反对称分解（原始方法）
        self.processed_data = decompose_symmetric_antisymmetric(filtered)
        
        print(f"预处理完成，耗时 {time.time() - start_time:.1f} 秒")
        print(f"  数据形状: {self.processed_data.shape}")
        print(f"  纬度范围: {self.processed_data.lat.values}")
        return self
    
    def compute_spectrum(self) -> 'WKSpectralAnalysis':
        """
        计算波数-频率功率谱
        
        返回:
        ----
        self
        """
        print("计算功率谱...")
        start_time = time.time()
        
        ntim, nlat, nlon = self.processed_data.shape
        spd = self.config.SAMPLES_PER_DAY
        nDayWin = self.config.WINDOW_SIZE_DAYS
        nDaySkip = self.config.WINDOW_SKIP_DAYS
        nSampWin = nDayWin * spd
        nSampSkip = nDaySkip * spd
        nWindow = int((ntim - nSampWin) / (nSampSkip + nSampWin)) + 1
        
        print(f"窗口大小: {nDayWin}天, 跳跃: {nDaySkip}天, 总窗口数: {nWindow}")
        
        # 累积功率
        sumpower = np.zeros((nSampWin, nlat, nlon))
        ntStrt, ntLast = 0, nSampWin
        
        for nw in range(nWindow):
            data_win = self.processed_data[ntStrt:ntLast, :, :]
            data_win = signal.detrend(data_win, axis=0)
            
            # Tukey窗
            window = signal.windows.tukey(nSampWin, 0.1, True)
            data_win *= window[:, np.newaxis, np.newaxis]
            
            # 2D FFT
            power = fft.fft2(data_win, axes=(0, 2)) / (nlon * nSampWin)
            sumpower += np.abs(power) ** 2
            
            ntStrt = ntLast + nSampSkip
            ntLast = ntStrt + nSampWin
        
        sumpower /= nWindow
        
        # 设置频率和波数轴
        if nlon % 2 == 0:
            self.wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)[1:]
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, nlon:0:-1]
        else:
            self.wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, ::-1]
        
        self.frequency = fft.fftshift(fft.fftfreq(nSampWin, d=1./spd))[nSampWin//2:]
        sumpower = fft.fftshift(sumpower, axes=0)[nSampWin//2:, :, :]
        
        # 分离对称/反对称功率
        power_symmetric = np.array(
            2.0 * sumpower[:, nlat//2:, :].sum(axis=1), copy=True
        )
        power_antisymmetric = np.array(
            2.0 * sumpower[:, :nlat//2, :].sum(axis=1), copy=True
        )
        background = np.array(sumpower.sum(axis=1), copy=True)

        # 屏蔽零频率
        power_symmetric[0, :] = np.nan
        power_antisymmetric[0, :] = np.nan
        background[0, :] = np.nan

        # 转为DataArray
        self.power_symmetric = xr.DataArray(
            power_symmetric,
            dims=("frequency", "wavenumber"),
            coords={"wavenumber": self.wavenumber, "frequency": self.frequency}
        )
        self.power_antisymmetric = xr.DataArray(
            power_antisymmetric,
            dims=("frequency", "wavenumber"),
            coords={"wavenumber": self.wavenumber, "frequency": self.frequency}
        )
        
        # 背景谱
        self.background = background
        
        print(f"功率谱计算完成，耗时 {time.time() - start_time:.1f} 秒")
        return self
    
    def smooth_background(self, wave_limit: int = 27) -> 'WKSpectralAnalysis':
        """
        平滑背景谱
        
        参数:
        ----
        wave_limit : int
            平滑的波数限制
            
        返回:
        ----
        self
        """
        print("平滑背景谱...")
        wave_indices = np.where(np.abs(self.wavenumber) <= wave_limit)[0]
        
        for idx, freq in enumerate(self.frequency):
            # 根据频率调整平滑次数
            if freq < 0.1:
                n_smooth = 5
            elif freq < 0.2:
                n_smooth = 10
            elif freq < 0.3:
                n_smooth = 20
            else:
                n_smooth = 40
            
            for _ in range(n_smooth):
                self.background[idx, wave_indices] = smooth_121(self.background[idx, wave_indices])
        
        # 频率方向平滑
        for wn_idx in wave_indices:
            for _ in range(10):
                self.background[:, wn_idx] = smooth_121(self.background[:, wn_idx])
        
        print("背景谱平滑完成")
        return self
    
    def save(self, output_path: str) -> 'WKSpectralAnalysis':
        """
        保存频谱到NetCDF
        
        参数:
        ----
        output_path : str
            输出文件路径
            
        返回:
        ----
        self
        """
        ds = xr.Dataset({
            "power_symmetric": self.power_symmetric,
            "power_antisymmetric": self.power_antisymmetric,
            "background": xr.DataArray(
                self.background,
                dims=("frequency", "wavenumber"),
                coords={"frequency": self.frequency, "wavenumber": self.wavenumber}
            )
        })
        ds.to_netcdf(output_path)
        print(f"频谱已保存至: {output_path}")
        return self


# ============= 便捷函数 =============

def calculate_wk_spectrum(data: xr.DataArray,
                          window_days: int = 96,
                          skip_days: int = 30,
                          output_path: Optional[str] = None) -> Tuple[xr.DataArray, xr.DataArray, np.ndarray]:
    """
    便捷函数：一步完成WK频谱计算
    
    参数:
    ----
    data : xr.DataArray
        输入数据
    window_days : int
        窗口大小（天）
    skip_days : int
        窗口跳跃（天）
    output_path : str, optional
        保存路径
        
    返回:
    ----
    power_symmetric, power_antisymmetric, background
    """
    config = SpectralConfig()
    config.WINDOW_SIZE_DAYS = window_days
    config.WINDOW_SKIP_DAYS = skip_days
    
    analysis = WKSpectralAnalysis(config)
    analysis.load_data(data=data)
    analysis.preprocess()
    analysis.compute_spectrum()
    analysis.smooth_background()
    
    if output_path:
        analysis.save(output_path)
    
    return analysis.power_symmetric, analysis.power_antisymmetric, analysis.background
