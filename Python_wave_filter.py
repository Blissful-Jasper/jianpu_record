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


import time
import sys
import numpy as np
from scipy import signal
import xarray as xr
import matplotlib.pyplot as plt
import dask.array as da
import os
# 切换到脚本所在的目录

"""
Created on Mon Mar  3 15:25:08 2025

@author: xpji

connect me : xianpuji@hhu.edu.cn

This script is used to filter waves in the OLR data using dask for parallel processing. Only support Kelvin and ER waves.

The script is based on the NCL example: 
- https://www.ncl.ucar.edu/Document/Functions/User_contributed/kf_filter.shtml
and
- https://github.com/Blissful-Jasper/mcclimate/blob/master/kf_filter.py

The script is tested on the OLR data from the NCL example and the results are validated against the NCL output.

"""



class WaveFilter:
    def __init__(self, ds=None, time_range=None, lat_range=None, wave_name=None, units=None, n_workers=4):
        
        self.time_range = time_range
        self.lat_range = lat_range
        self.wave_name = wave_name
        self.n_workers = n_workers
        
        self.data = None
        self.units = units
        self.filtered_data = None
        self.fftdata = None
        
        if isinstance(ds, str):  # 如果是文件路径，读取 NetCDF 文件
            self.ds = xr.open_dataset(ds, chunks={'time': 'auto'})
        elif isinstance(ds, (xr.Dataset, xr.DataArray)):  # 直接传入 xarray 数据
            self.ds = ds
        else:
            raise ValueError("`data` 必须是文件路径 (str) 或 `xarray.Dataset` / `xarray.DataArray`")
            
    def print_diagnostic_info(self, variable, name):
        """打印变量的诊断信息"""
        try:
            print(f"\n=== {name} Information ===")
            print(f"Type: {type(variable)}")
            print(f"Shape: {variable.shape}")
            if isinstance(variable, (da.Array, np.ndarray, xr.DataArray)):
                print(f"Data type: {variable.dtype}")
            if isinstance(variable, da.Array):
                print(f"Chunks: {variable.chunks}")
            print(f"First few values: {variable[:5]}")  # 只打印前5个值
            print("========================\n")
        except Exception as e:
            print(f"Error printing info for {name}: {e}")
            
    def load_data(self):
        """Load and preprocess data."""
        if isinstance(self.ds, xr.Dataset):  # 确保 ds 是 Dataset
            self.data = self.ds[list(self.ds.data_vars)[0]]  # 选取第一个变量

        elif isinstance(self.ds, xr.DataArray):  # 如果传入的是 DataArray，直接赋值
            self.data = self.ds
            
        if self.time_range:
            self.data = self.data.sel(time=slice(*self.time_range))
        if self.lat_range:
            self.data = self.data.sel(lat=slice(*self.lat_range))
            
        self.data = self.data.sortby('lat').transpose('time', 'lat', 'lon')
        # print(f"Data chunks before rechunk: {self.data.chunks}")
        self.data = self.data.chunk({'time': -1})
        # print(f"Data chunks after rechunk: {self.data.chunks}")
        # 计算日异常值
        # self.anomaly = self.data.groupby('time.dayofyear') - self.data.groupby('time.dayofyear').mean()
        
    def detrend_data(self):
        """Detrend the data using dask for parallel processing."""
        ntim, nlat, nlon = self.data.shape
        spd = 1 # obs_per_day, if one day have four times sample ,then spd=4
       
        data_rechunked = self.data.data.rechunk({0: -1})
        if ntim >  365*spd/3:

            # 进行并行 FFT
            rf = da.fft.rfft(data_rechunked, axis=0)
            freq = da.fft.rfftfreq(ntim * spd, d=1. / float(spd))
            rf[(freq <= 3. / 365) & (freq >= 1. / 365), :, :] = 0.0
            datain = da.fft.irfft(rf, axis=0, n=ntim)
 
        # 使用 dask 并行去趋势
        self.detrend = da.apply_along_axis(signal.detrend, 0, datain)
    
        window = signal.windows.tukey(self.data.shape[0],0.05,True)
        
        self.detrend = self.detrend * window[:, np.newaxis, np.newaxis]   
        
    def fft_transform(self):
        """Perform 2D FFT on the detrended data using dask."""
        
        # data.shape[2] is the length of lon
        self.wavenumber = -da.fft.fftfreq(self.data.shape[2]) * self.data.shape[2]   # shape: (lon,)
        self.frequency = da.fft.fftfreq(self.data.shape[0], d=1. / float(1))    # shape: (time,)
        # print("\nFrequency values:")
        # print(self.frequency.compute())
        # print("\nFirst 10 frequency values:")
        # print(self.frequency[:10].compute())
        # print("\nWavenumber values:")
        # print(self.wavenumber.compute())
        self.knum_ori, self.freq_ori = da.meshgrid(self.wavenumber, self.frequency)   # shape: (time, lon)
        self.knum = self.knum_ori.copy()
        # print(self.knum_ori.compute().shape)
        self.knum = da.where(self.freq_ori < 0, -self.knum_ori, self.knum_ori)   # shape: (time, lon)
        plt.title(f'{self.wave_name}')
        
        
        self.freq = da.abs(self.freq_ori)    # shape: (time, lon)
    
    def apply_filter(self):
        """Apply filter based on wave type."""
        if self.wave_name.lower() == "kelvin":
            self.tMin, self.tMax = 3, 20
            self.kmin, self.kmax = 2, 14
            self.hmin, self.hmax = 8, 90
        elif self.wave_name.lower() == "er":
            self.tMin, self.tMax = 9, 72
            self.kmin, self.kmax = -10, -1
            self.hmin, self.hmax = 8, 90
        self.fmin, self.fmax = 1 / self.tMax, 1 / self.tMin
        self.mask =  da.zeros((self.data.shape[0], self.data.shape[2]), dtype=bool)

     
        
        if self.kmin is not None:
            self.mask = self.mask | (self.knum < self.kmin)
        if self.kmax is not None:
            self.mask = self.mask | (self.kmax < self.knum)

        if self.fmin is not None:
            self.mask = self.mask | (self.freq < self.fmin)
        if self.fmax is not None:
            self.mask = self.mask | (self.fmax < self.freq)

        if self.wave_name.lower() == 'kelvin':
            self.apply_wave_filter(self.wave_name)
        elif self.wave_name.lower() == 'er':
            self.apply_wave_filter(self.wave_name)
            
        plt.contourf(self.mask[:,:].compute())
        self.fftdata = da.fft.fft2(self.detrend, axes=(0, 2)) # shape: (time, lat, lon)
        self.mask = da.repeat(self.mask[:, np.newaxis, :], self.data.shape[1], axis=1)
     
        # self.fftdata[self.mask] = 0.0
        self.fftdata = da.where(self.mask, 0.0, self.fftdata)
        
    def apply_wave_filter(self,wave_name):
        """Apply Kelvin wave filter."""
        g = 9.8
        beta = 2.28e-11
        a = 6.37e6
        n = 1 
        if self.wave_name.lower() == "kelvin":
            
            if self.hmin is not None:
                c = da.sqrt(g * self.hmin)
                omega = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega - k < 0)
            if self.hmax is not None:
                c = da.sqrt(g * self.hmax)
                omega = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega - k > 0)
    
        if self.wave_name.lower() == "er":
    
            if self.hmin is not None:
                c = da.sqrt(g * self.hmin)
                omega = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega * (k ** 2 + (2 * n + 1)) + k < 0)
            if self.hmax is not None:
                c = da.sqrt(g * self.hmax)
                omega = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega * (k ** 2 + (2 * n + 1)) + k > 0)
         # if wavename.lower() == "er":
         
         
         
         
    def inverse_fft(self):
        """Perform inverse FFT to get filtered data."""
        self.filtered_data = da.fft.ifft2(self.fftdata, axes=(0, 2)).real

    def create_output(self):
        """Create xarray DataArray for filtered data."""
        self.wave_data = xr.DataArray(self.filtered_data.compute(),  # 将 dask 数组转换为 numpy 数组
                                      coords = {'time': self.data.time,
                                                'lat': self.data.lat,
                                                'lon': self.data.lon},
                                          
                                      dims=['time', 'lat', 'lon'])
        self.wave_data.attrs.update({
            'long_name': self.wave_name,
            'min_equiv_depth': self.hmin,
            'max_equiv_depth': self.hmax,
            'min_wavenumber': self.kmin,
            'max_wavenumber': self.kmax,
            'min_period': self.tMin,
            'max_period': self.tMax,
            'min_frequency': self.fmin,
            'max_frequency': self.fmax,
            
            'units': self.units,

        })
        
        print(self.wave_data.compute())
        
        return self.wave_data.compute()


