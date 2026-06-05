# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

@email : xianpuji@hhu.edu.cn
"""
from math import pi, acos, sqrt, floor, ceil
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fft
from joblib import Parallel, delayed
from typing import Tuple, Dict, Union, Optional, List, Any
import os
import sys


# ================================================================================================
# Author: %(Jianpu)s | Affiliation: Hohai
# email : xianpuji@hhu.edu.cn
# Last modified:  %(date)s
# Filename: 
# =================================================================================================
class WaveFilter:
    """
    气候波动滤波与分析工具类
    
    用于从气候数据中提取各种波动信号，如Kelvin波、MJO等，
    并进行谐波分析和滤波处理。
    """
    
    def __init__(self):
        """初始化波动滤波器，定义各种波动的参数"""
        # 定义各种波动的参数
        self.wave_params = {
            'kelvin': {
                'freq_range': (3, 20),       # 周期范围（天）
                'wnum_range': (2, 14),       # 波数范围
                'equiv_depth': (8, 90)       # 等效深度范围（米）
            },
            'er': {
                'freq_range': (9, 72),
                'wnum_range': (-10, -1),
                'equiv_depth': (8, 90)
            },
            'mrg': {
                'freq_range': (3, 10),
                'wnum_range': (-10, -1),
                'equiv_depth': (8, 90)
            },
            'ig': {
                'freq_range': (1, 14),
                'wnum_range': (1, 5),
                'equiv_depth': (8, 90)
            },
            'mjo': {
                'freq_range': (20, 100),
                'wnum_range': (1, 5),
                'equiv_depth': (np.nan, np.nan)
            },
            'td': {
                'freq_range': (2.5, 5),
                'wnum_range': (-20, -6),
                'equiv_depth': (np.nan, np.nan)
            },
        }
        
        # 物理常数
        self.beta = 2.28e-11  # 地球自转参数（单位：1/s/m）
        self.a = 6.37e6       # 地球半径（单位：m）
        
    def extract_low_harmonics(self, 
                              data: xr.DataArray, 
                              n_harm: int = 3, 
                              dim: str = 'dayofyear') -> xr.DataArray:
        """
        从逐日气候态中提取指定阶数的谐波并重构信号。

        参数：
            data: 输入的xarray.DataArray（时间维度应为dayofyear的气候态）
            n_harm: 要保留的最高谐波阶数（保留 0~n_harm-1 的谐波，第 n_harm 的系数减半）
            dim: 要进行 FFT 的维度（默认是 'dayofyear'）

        返回：
            仅包含低阶谐波的重建数据，类型为 xarray.DataArray
        """
        # 傅里叶变换
        z_fft = np.fft.rfft(data, axis=data.get_axis_num(dim))
        # 设置频率
        freqs = np.fft.rfftfreq(data.sizes[dim])
        
        # 保留低阶谐波并处理第 n_harm 阶的振幅
        z_fft_n = z_fft.copy()
        z_fft_n[n_harm,:,:] *= 0.5  # 第 n_harm 阶振幅减半
        z_fft_n[(n_harm+1):,:,:] = 0
      
        # 反傅里叶变换，保留实数部分
        clim_low_harm = np.fft.irfft(z_fft_n, n=data.sizes[dim], axis=data.get_axis_num(dim)).real
        
        # 保持 xarray 格式和原数据一致
        coords = {k: v for k, v in data.coords.items()}
        dims = data.dims
        attrs = {
            "smoothing"     : f"FFT: {n_harm} harmonics were retained.",
            "information"   : "Smoothed daily climatological averages",
            "units"         : data.attrs.get("units", "W/m^2"),
            "long_name"     : f"Daily Climatology: {n_harm} harmonics retained",
        }
        
        return xr.DataArray(clim_low_harm, coords=coords, dims=dims, attrs=attrs)
    
    def _kf_filter(self, 
                  in_data: Union[xr.DataArray, np.ndarray], 
                  lon: np.ndarray, 
                  obs_per_day: int, 
                  t_min: float, 
                  t_max: float, 
                  k_min: int, 
                  k_max: int, 
                  h_min: float, 
                  h_max: float, 
                  wave_name: str) -> Union[xr.DataArray, np.ndarray]:
        """
        应用WK99滤波方法对2D时间-经度数据进行特定波动的滤波。
        
        参数：
            in_data: 输入数据，dims=("time", "lon")
            lon: 经度坐标数组
            obs_per_day: 每天的观测次数（例如，6小时数据为4）
            t_min, t_max: 滤波周期范围（天）
            k_min, k_max: 波数范围
            h_min, h_max: 等效深度范围（米）
            wave_name: 波动类型名称
            
        返回：
            与输入相同形状的已滤波数据
        """
        is_xarray = isinstance(in_data, xr.DataArray)

        if is_xarray:
            data_np = in_data.values
            time_dim, lon_dim = in_data.sizes["time"], in_data.sizes["lon"]
        else:
            data_np = in_data
            time_dim, lon_dim = data_np.shape
            
        # 检查经度是否包裹（首尾相连）
        wrap_flag = np.isclose((lon[0] + 360) % 360, lon[-1] % 360)

        if wrap_flag:
            data = in_data.isel(lon=slice(1, None)) if is_xarray else in_data[:, 1:]  # 丢掉第一个点
        else:
            data = in_data
        
        if is_xarray:
            data_np = data.values
        else:
            data_np = data
        
        # 去趋势和加窗处理
        data_np = signal.detrend(data_np, axis=0)
        data_np = signal.windows.tukey(time_dim, alpha=0.05)[:, np.newaxis] * data_np

        # 二维FFT: timexlon
        fft_data = fft.rfft2(data_np, axes=(1, 0))
        fft_data[:, 1:] = fft_data[:, -1:0:-1]

        # 频率/波数轴: 找到周期截止的索引
        freq_dim = fft_data.shape[0]
        k_dim = fft_data.shape[1]
        j_min = int(time_dim / (t_max * obs_per_day))
        j_max = int(time_dim / (t_min * obs_per_day))
        j_max = min(j_max, freq_dim)
        
        # 找到波数截止的索引
        if k_min < 0:
            i_min = max(k_dim + k_min, k_dim // 2)
        else:
            i_min = min(k_min, k_dim // 2)
        if k_max < 0:
            i_max = max(k_dim + k_max, k_dim // 2)
        else:
            i_max = min(k_max, k_dim // 2)

        # 按频率进行带通滤波: 设置相应系数为零
        if j_min > 0:
            fft_data[:j_min, :] = 0
            
        if j_max < freq_dim - 1:
            fft_data[j_max + 1:, :] = 0
            
        if i_min < i_max:
            if i_min > 0:
                fft_data[:, :i_min] = 0
                
            if i_max < k_dim - 1:
                fft_data[:, i_max + 1:] = 0
                
        # 色散滤波（波动类型）
        spc = 24 * 3600 / (2 * np.pi * obs_per_day)
        c = np.sqrt(9.8 * np.array([h_min, h_max]))
        
        for i in range(k_dim):
            k = (i - k_dim if i > k_dim // 2 else i) / self.a  # 调整地球周长
            
            freq = np.array([0, freq_dim]) / spc
            j_min_wave = 0
            j_max_wave = freq_dim
            
            if wave_name.lower() == "kelvin":
                freq = k * c
            elif wave_name.lower() == "er":
                freq = -self.beta * k / (k**2 + 3 * self.beta / c)
            elif wave_name.lower() in ["mrg", "ig0"]:
                if k == 0:
                    freq = np.sqrt(self.beta * c)
                elif k > 0:
                    freq = k * c * (0.5 + 0.5 * np.sqrt(1 + 4 * self.beta / (k**2 * c)))
                else:
                    freq = k * c * (0.5 - 0.5 * np.sqrt(1 + 4 * self.beta / (k**2 * c)))
            elif wave_name.lower() == "ig1":
                freq = np.sqrt(3 * self.beta * c + (k**2 * c**2))
            elif wave_name.lower() == "ig2":
                freq = np.sqrt(5 * self.beta * c + (k**2 * c**2))
            else:
                continue

            j_min_wave = int(np.floor(freq[0] * spc * time_dim)) if not np.isnan(h_min) else 0
            j_max_wave = int(np.ceil(freq[1] * spc * time_dim)) if not np.isnan(h_max) else freq_dim
            j_min_wave = min(j_min_wave, freq_dim)
            j_max_wave = max(j_max_wave, 0)
            
            # 设置相应系数为零
            fft_data[:j_min_wave, i] = 0
            if j_max_wave < freq_dim:
                fft_data[j_max_wave + 1:, i] = 0

        # 逆FFT
        fft_data[:, 1:] = fft_data[:, -1:0:-1]
        temp_data = np.real(fft.irfft2(fft_data, axes=(1, 0), s=(lon_dim, time_dim)))

        # 重构完整场
        if is_xarray:
            out = in_data.copy(data=temp_data)
            if "dayofyear" in out.coords:
                out = out.drop_vars("dayofyear")
            out.attrs.update({
                "wavenumber": (k_min, k_max),
                "period": (t_min, t_max),
                "depth": (h_min, h_max),
                "waveName": wave_name
            })
            return out.transpose("time", "lon")
        else:
            return temp_data
      
    def extract_wave_signal(self, 
                           ds: xr.DataArray, 
                           wave_name: str = 'kelvin', 
                           obs_per_day: int = 1, 
                           use_parallel: bool = True, 
                           n_jobs: int = -1,
                           n_harm: int = 3) -> xr.DataArray:
        """
        对气候数据进行年循环去除，并滤波提取特定波动成分
        
        参数：
            ds: 输入数据，xr.DataArray类型，维度应包含('time', 'lat', 'lon')
            wave_name: 波动类型名称，可选值: 'kelvin', 'er', 'mrg', 'ig', 'mjo', 'td'
            obs_per_day: 每天的观测次数（例如，6小时数据为4）
            use_parallel: 是否使用并行计算
            n_jobs: 并行计算的作业数量，-1表示使用所有可用核心
            n_harm: 年循环谐波提取时保留的谐波数
            
        返回：
            提取的波动信号，xr.DataArray类型
        """
        # 检查波动类型是否有效
        assert wave_name in self.wave_params, f"wave_name必须是以下之一: {list(self.wave_params.keys())}"

        # 步骤1: 年循环去除
        clim = ds.groupby('time.dayofyear').mean(dim='time')
        clim_fit = self.extract_low_harmonics(clim, n_harm=n_harm)
        anomaly = ds.groupby('time.dayofyear') - clim_fit
        
        # 步骤2: 参数提取
        params = self.wave_params[wave_name]
        t_min, t_max = params['freq_range']
        k_min, k_max = params['wnum_range']
        h_min, h_max = params['equiv_depth']
        lon = ds.lon.values
        
        # 步骤3: 滤波主逻辑（逐纬度调用 kf_filter）
        def _filter_lat(lat_idx):
            in_data = anomaly.isel(lat=lat_idx)
            return self._kf_filter(
                in_data.values if use_parallel else in_data,
                lon=lon,
                obs_per_day=obs_per_day,
                t_min=t_min, t_max=t_max,
                k_min=k_min, k_max=k_max,
                h_min=h_min, h_max=h_max,
                wave_name=wave_name
            )

        if use_parallel:
            # Use threads here to avoid pickling notebook-defined import state
            # into loky workers. The heavy lifting is NumPy/SciPy-based, so
            # threaded parallelism still works without requiring module re-imports.
            filtered = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_filter_lat)(i) for i in range(len(ds.lat))
            )
        else:
            filtered = [_filter_lat(i) for i in range(len(ds.lat))]

        # 组合结果
        filtered = np.stack(filtered, axis=1)
        
        # 步骤4: 构造新的 DataArray
        da_filtered = xr.DataArray(
            filtered,
            coords=ds.coords,
            dims=ds.dims,
            attrs={
                'long_name': f'{wave_name.title()} Wave Component',
                'units': ds.attrs.get('units', 'unknown'),
                'wavenumber': (k_min, k_max),
                'period': (t_min, t_max),
                'depth': (h_min, h_max),
                'waveName': wave_name
            }
        )
        
        return da_filtered
    
    def check_filter_wave(self, 
                         python_result: xr.DataArray, 
                         ncl_path: str, 
                         wave_name: str,
                         sample_size: int = None,
                         random_seed: int = None):
        """
        比较Python结果与NCL结果
        
        参数：
            python_result: Python滤波结果
            ncl_path: NCL结果文件路径
            wave_name: 波动类型名称
            sample_size: 用于比较的样本大小，默认使用全部数据
            random_seed: 随机种子，用于可重复的随机抽样
        """
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # 打开NCL结果
        ds_ncl = xr.open_dataset(ncl_path)
        clm_ncl = ds_ncl[wave_name]
        
        # 如果指定了样本大小，则使用随机索引
        if sample_size is None or sample_size >= len(python_result.time):
            sample_size = len(python_result.time)
            random_index = sample_size
        else:
            random_index = np.random.randint(sample_size, len(python_result.time))
        
        # 计算空间标准差
        clm_ncl_mean = clm_ncl[:random_index].std(['lon', 'lat'])
        clim_py_mean = python_result[:random_index].std(['lon', 'lat'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Python vs NCL {wave_name.upper()} Wave Comparison", fontsize=16)
        
        # 1. 平均气候态曲线对比
        ax = axes[0, 0]
        clim_py_mean.plot(ax=ax, label='Python', linewidth=2)
        clm_ncl_mean.plot(ax=ax, label='NCL', linewidth=2, linestyle='--')
        ax.set_title('(a) Temporal Mean Curve')
        ax.legend()
        ax.grid(True)
        
        # 2. Python 版本空间图（标准差沿 time）
        ax = axes[0, 1]
        python_result.std('time').plot.contourf(ax=ax, cmap='jet', levels=21, extend='both')
        ax.set_title('(b) Python: Spatial STD')
        
        # 3. NCL 版本空间图（标准差沿 time）
        ax = axes[1, 0]
        clm_ncl.std('time').plot.contourf(ax=ax, cmap='jet', levels=21, extend='both')
        ax.set_title('(c) NCL: Spatial STD')
        
        # 4. 差值图（Python - NCL）
        ax = axes[1, 1]
        diff = python_result.std('time') - clm_ncl.std('time')
        diff.plot.contourf(ax=ax, cmap='RdBu_r', levels=21, center=0)
        ax.set_title('(d) Difference: Python - NCL')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 返回图表对象，方便进一步自定义
        return fig, axes
    
    def add_wave_param(self, 
                      wave_name: str, 
                      freq_range: Tuple[float, float], 
                      wnum_range: Tuple[int, int], 
                      equiv_depth: Tuple[float, float] = (np.nan, np.nan)):
        """
        添加新的波动参数
        
        参数：
            wave_name: 波动类型名称
            freq_range: 周期范围（天），(min, max)
            wnum_range: 波数范围，(min, max)
            equiv_depth: 等效深度范围（米），(min, max)，默认为(nan, nan)
        """
        self.wave_params[wave_name.lower()] = {
            'freq_range': freq_range,
            'wnum_range': wnum_range,
            'equiv_depth': equiv_depth
        }
        
    def get_available_waves(self) -> List[str]:
        """获取可用的波动类型列表"""
        return list(self.wave_params.keys())
    
    def get_wave_params(self, wave_name: str = None) -> Dict:
        """
        获取波动参数
        
        参数：
            wave_name: 波动类型名称，如果为None则返回所有波动的参数
            
        返回：
            波动参数字典
        """
        if wave_name is None:
            return self.wave_params
        else:
            wave_name = wave_name.lower()
            if wave_name in self.wave_params:
                return self.wave_params[wave_name]
            else:
                raise ValueError(f"未知的波动类型: {wave_name}，可用的波动类型: {self.get_available_waves()}")


# ================================================================================================
# Convectively Coupled Kelvin Wave Filter with Dask
# ================================================================================================

class CCKWFilter:
    """
    对流耦合赤道波滤波器。

    当前实现保持原有 API，但内部流程重构为更接近 NCL `kf_filter`：
    1. 先计算日气候态并保留低阶年循环谐波
    2. 在 anomaly 上逐纬度调用 NCL-aligned kf-filter
    3. 对无法由当前采样频率可靠解析的波段直接返回零场
    """

    WAVE_SPECS: Dict[str, Dict[str, Any]] = {
        "kelvin": {"period_days": (3.0, 20.0), "wavenumber": (2, 14), "equiv_depth": (8.0, 90.0), "meridional_mode": None, "dispersion_family": "kelvin"},
        "er": {"period_days": (9.0, 72.0), "wavenumber": (-10, -1), "equiv_depth": (8.0, 90.0), "meridional_mode": 1, "dispersion_family": "er"},
        "ig": {"period_days": (None, None), "wavenumber": (-15, -1), "equiv_depth": (12.0, 90.0), "meridional_mode": 1, "dispersion_family": "ig"},
        "eig0": {"period_days": (None, 1.0 / 0.55), "wavenumber": (0, 15), "equiv_depth": (12.0, 50.0), "meridional_mode": 0, "dispersion_family": "eig0_mrg"},
        "mrg": {"period_days": (2.5, 10.0), "wavenumber": (-10, -1), "equiv_depth": (8.0, 90.0), "meridional_mode": 0, "dispersion_family": "eig0_mrg"},
        "td": {"period_days": (2.0, 8.5), "wavenumber": (-15, -6), "equiv_depth": (None, None), "meridional_mode": None, "dispersion_family": "none"},
        "mjo": {"period_days": (30.0, 100.0), "wavenumber": (1, 5), "equiv_depth": (None, None), "meridional_mode": None, "dispersion_family": "none"},
    }

    def __init__(self, ds=None, var=None, sel_dict=None, wave_name=None,
                 units=None, spd=1, n_workers=4, verbose=True, n_harm=3):
        self.sel_dict = sel_dict
        self.wave_name = wave_name
        self.n_workers = n_workers
        self.spd = spd
        self.data = None
        self.units = units
        self.filtered_data = None
        self.fftdata = None
        self.var = var
        self.verbose = verbose
        self.n_harm = n_harm
        self.anomaly = None
        self.filter_note = None
        self.mask = None
        self._resolved_spec = None

        if isinstance(ds, str):
            self.ds = xr.open_dataset(ds, chunks={'time': 'auto'})
        elif isinstance(ds, (xr.Dataset, xr.DataArray)):
            self.ds = ds
        else:
            raise ValueError("`ds` 必须是文件路径(str)或xarray.Dataset/DataArray")

    def __repr__(self):
        lat_sel = self.sel_dict.get('lat') if self.sel_dict and 'lat' in self.sel_dict else 'N/A'
        time_sel = self.sel_dict.get('time') if self.sel_dict and 'time' in self.sel_dict else 'N/A'
        lines = [
            "📡 CCKWFilter Summary:",
            f"  • Wave Type     : {self.wave_name or 'N/A'}",
            f"  • Variable      : {self.var or 'N/A'}",
            f"  • Time Range    : {time_sel}",
            f"  • Latitude Range: {lat_sel}",
            f"  • Units         : {self.units or 'N/A'}",
            f"  • Workers       : {self.n_workers}",
            f"  • Sampling/day  : {self.spd}",
            f"  • Harmonics     : {self.n_harm}",
        ]
        if self.data is not None:
            lines.append(f"  • Data Shape    : {self.data.shape}")
            lines.append(f"  • Data Dims     : {tuple(self.data.dims)}")
        else:
            lines.append("  • Data          : Not loaded")
        return "\n".join(lines)


    def print_diagnostic_info(self, variable, name):
        if not self.verbose:
            return
        try:
            print(f"\n{'='*20} {name} Information {'='*20}")

            print(f"Type: {type(variable)}")
            print(f"Shape: {variable.shape}")
            if hasattr(variable, 'dtype'):
                print(f"Data type: {variable.dtype}")
            print(f"First few values: {variable[:5]}")
            print("=" * 60)
        except Exception as e:
            print(f"Error printing info for {name}: {e}")

    @classmethod
    def get_wave_specs(cls) -> Dict[str, Dict[str, Any]]:
        return {name: spec.copy() for name, spec in cls.WAVE_SPECS.items()}

    def load_data(self):
        if isinstance(self.ds, xr.Dataset):
            if self.var is None:
                raise ValueError("ds是Dataset时必须指定var参数")
            self.data = self.ds[self.var].sortby('lat')
        elif isinstance(self.ds, xr.DataArray):
            self.data = self.ds.sortby('lat')

        if self.sel_dict:
            self.data = self.data.sel(**self.sel_dict)

        self.data = self.data.sortby('lat').transpose('time', 'lat', 'lon')
        if self.verbose:
            self.print_diagnostic_info(self.data, 'Loaded Data')

    def _resolve_wave_spec(self) -> Dict[str, Any]:
        if self.wave_name is None:
            raise ValueError("必须指定 wave_name")
        wave_key = self.wave_name.lower()
        if wave_key not in self.WAVE_SPECS:
            raise ValueError(f"不支持的波动类型: {self.wave_name}，支持: {list(self.WAVE_SPECS)}")

        spec = self.WAVE_SPECS[wave_key]
        self.wave_name = wave_key
        self.tMin, self.tMax = spec["period_days"]
        self.kmin, self.kmax = spec["wavenumber"]
        self.hmin, self.hmax = spec["equiv_depth"]
        self.mode_n = spec["meridional_mode"]
        self.dispersion_family = spec["dispersion_family"]
        self.fmin = None if self.tMax is None else 1.0 / self.tMax
        self.fmax = None if self.tMin is None else 1.0 / self.tMin
        self._resolved_spec = spec
        return spec

    def _backend_wave_name(self) -> str:
        if self.wave_name == 'eig0':
            return 'ig0'
        return self.wave_name

    def _legacy_filter(self) -> 'WaveFilter':
        wf = WaveFilter()
        backend_name = self._backend_wave_name()
        wf.wave_params[backend_name] = {
            'freq_range': (self.tMin, self.tMax),
            'wnum_range': (self.kmin, self.kmax),
            'equiv_depth': (
                np.nan if self.hmin is None else self.hmin,
                np.nan if self.hmax is None else self.hmax,
            ),
        }
        return wf

    def _nyquist_frequency(self) -> float:
        return 0.5 * float(self.spd)

    def _is_resolvable(self) -> bool:
        nyquist = self._nyquist_frequency()
        if self.fmin is not None and self.fmin >= nyquist - 1.0e-12:
            return False
        if self.fmax is not None and self.fmax > nyquist + 1.0e-12:
            return False
        return True

    def detrend_data(self):
        self._resolve_wave_spec()
        wf = WaveFilter()
        clim = self.data.groupby('time.dayofyear').mean(dim='time')
        clim_fit = wf.extract_low_harmonics(clim, n_harm=self.n_harm)
        self.anomaly = (self.data.groupby('time.dayofyear') - clim_fit).transpose('time', 'lat', 'lon')
        self.detrend = self.anomaly.data
        self.filter_note = None
        if not self._is_resolvable():
            self.filter_note = (
                f"Wave {self.wave_name} is not resolvable for SPD={self.spd}; "
                "returning zeros to avoid Nyquist-edge artefacts."
            )

    def fft_transform(self):
        self.wavenumber = np.fft.fftfreq(self.data.shape[2]) * self.data.shape[2]
        self.frequency = np.fft.rfftfreq(self.data.shape[0], d=1. / float(self.spd))
        self.knum_ori, self.freq_ori = np.meshgrid(self.wavenumber, self.frequency)
        self.knum = self.knum_ori.copy()
        self.freq = np.abs(self.freq_ori)

    def _filter_single_latitude(self, lat_idx: int) -> np.ndarray:
        if not self._is_resolvable():
            return np.zeros((self.anomaly.sizes['time'], self.anomaly.sizes['lon']), dtype=np.float64)

        wf = self._legacy_filter()
        backend_name = self._backend_wave_name()
        lat_slice = self.anomaly.isel(lat=lat_idx)
        return wf._kf_filter(
            lat_slice.values,
            lon=self.anomaly.lon.values,
            obs_per_day=self.spd,
            t_min=self.tMin,
            t_max=self.tMax,
            k_min=self.kmin,
            k_max=self.kmax,
            h_min=np.nan if self.hmin is None else self.hmin,
            h_max=np.nan if self.hmax is None else self.hmax,
            wave_name=backend_name,
        )

    def apply_filter(self):
        if self.anomaly is None:
            self.detrend_data()
        self.fftdata = None
        self.mask = None

        nlat = self.anomaly.sizes['lat']
        if self.n_workers == 1:
            filtered = [self._filter_single_latitude(i) for i in range(nlat)]
        else:
            # Avoid loky process pickling issues when this package is imported
            # from notebooks or ad-hoc paths.
            filtered = Parallel(n_jobs=self.n_workers, prefer="threads")(
                delayed(self._filter_single_latitude)(i) for i in range(nlat)
            )
        self.filtered_data = np.stack(filtered, axis=1)

    def _apply_dispersion_relation(self):
        return None

    def inverse_fft(self):
        if self.filtered_data is None:
            raise ValueError("请先执行 apply_filter()")
        return self.filtered_data

    def create_output(self):
        if self.filtered_data is None:
            raise ValueError("请先执行 apply_filter()")

        values = self.filtered_data.compute() if hasattr(self.filtered_data, 'compute') else np.asarray(self.filtered_data)
        self.wave_data = xr.DataArray(
            values,
            coords={'time': self.data.time, 'lat': self.data.lat, 'lon': self.data.lon},
            dims=['time', 'lat', 'lon']
        )
        attrs = {
            'long_name': f'{self.wave_name} wave filtered data',
            'min_equiv_depth': self.hmin,
            'max_equiv_depth': self.hmax,
            'min_wavenumber': self.kmin,
            'max_wavenumber': self.kmax,
            'min_period': self.tMin,
            'max_period': self.tMax,
            'min_frequency': self.fmin,
            'max_frequency': self.fmax,
            'units': self.units,
            'filter_method': 'NCL-aligned Wheeler-Kiladis kf_filter',
            'processing_date': str(np.datetime64('today')),
            'samples_per_day': self.spd,
            'annual_cycle_harmonics': self.n_harm,
            'backend_wave_name': self._backend_wave_name(),
        }
        if self.filter_note is not None:
            attrs['note'] = self.filter_note
        self.wave_data.attrs.update({key: value for key, value in attrs.items() if value is not None})
        return self.wave_data

    def process(self):
        if self.verbose:
            print(f"{'='*70}")
            print(f"🌊 Processing {self.wave_name.upper()} wave filter")
            print(f"{'='*70}")

        self.load_data()
        if self.verbose:
            print("⏳ Building anomaly field...")
        self.detrend_data()

        if self.verbose:
            print("⏳ Preparing spectral coordinates...")
        self.fft_transform()

        if self.verbose:
            print("⏳ Applying NCL-aligned kf_filter...")
        self.apply_filter()

        if self.verbose:
            print("⏳ Finalizing output...")
        self.inverse_fft()
        output = self.create_output()

        if self.verbose:
            print(f"✅ {self.wave_name.upper()} wave filtering completed!")
            print(f"{'='*70}\n")

        return output

