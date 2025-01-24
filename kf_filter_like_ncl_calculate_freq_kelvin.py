#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Jianpu)s

@email: xianpuji@hhu.edu.cn
"""

import matplotlib.ticker as mticker
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import pandas as pd
import datetime
import cftime
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
import scipy.signal as signal


def _taper2(x, p, option=0):

    # Convert x to an array (if it isn't already) for safe manipulation
    x = np.asanyarray(x)

    # Check that p is in [0, 1]
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be between 0.0 and 1.0 inclusive.")

    # Length of the rightmost dimension
    n = x.shape[0]
    
    # Number of points to be tapered on each end
    # int(...) floors by default in Python 3
    m = int(np.floor(p * n / 2.0))

    # Construct the 1D taper window (size = n)
    # Initialize everything to 1.0 (no taper in the center region)
    taper_window = np.ones(n, dtype=x.dtype)

    # If m > 0, apply the split-cosine-bell shape at both ends
    if m > 0:
        # Indices for the tapered region
        i = np.arange(m)

        # Half-cosine shape that smoothly ramps from 0 to 1
        # You will see variations of this formula; this one
        # gives a smooth ramp from 0 up to 1 across m points.
        # (Feel free to adjust the exact formula if you need
        # to match NCL's internals exactly.)
        taper_vals = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (m + 1)))

        # Left edge taper
        taper_window[i] = taper_vals
        # Right edge taper (mirror it)
        taper_window[-m:] = taper_vals[::-1]

    # Reshape taper_window for broadcasting along the rightmost dimension
    # e.g. if x.shape is (n1, n2, ..., n_{k-1}, n_{k}),
    # we want taper_window.shape to be (1, 1, ..., 1, n_{k}).
    # This ensures multiplication is done only along the last dimension.
    shape_ones = (n,)+(1,) * (x.ndim - 1) 
    taper_window = taper_window.reshape(shape_ones)
    print(taper_window.shape)
    return taper_window


data = xr.open_dataset('/Users/xpji/spacetime_filter/filter_wave/olr.day.mean.nc').olr.sel(time=slice('1980-01-01', 
                                                                                                      '1999-12-31'),lat=slice(15, 
                                                                                                    -15)).sortby('lat').transpose('time', 'lat', 'lon')

anomaly =  data.groupby('time.dayofyear') - data.groupby('time.dayofyear').mean()



nt,nlat,nlon = anomaly.shape

V1 = signal.detrend(anomaly,axis=0)#*HANN 
taper = _taper2(V1, p=0.05)

v1 = V1*taper

plt.plot(taper[:,0,0])

del data

FFT_V1 = np.zeros([nt,nlat,nlon],dtype=complex)

for ilat in range(0,nlat):
    FFT_V1[:,ilat,:] = np.fft.fft2(v1[:,ilat,:])#/nlon   



Fs_t = 1
Fs_lon = 1/2.5
freq_min = np.array([1/2])
freq_max = np.array([1/3])
wnum_min = np.array([2])
wnum_max = np.array([14])
h_min     = np.array([8]) #m
h_max     = np.array([90])# #m
g = 9.8       # m/s^2
pi = np.pi
spi = '\u03C0'
re = 6371*1000 #earth radius (m)

V1_shift = np.fft.fftshift( np.fft.fftshift(FFT_V1,axes=2),axes=0 )  
V1_shift2 = np.zeros([nt,nlat,nlon],dtype=complex)

# freq filter
freq = np.arange(-nt/2,nt/2)*Fs_t/nt #(1/day)
freq_1 = 1/20 #1/20  
freq_2 = 1/3 #1/2.5 

# zwnum filter
zwnum = np.arange(-nlon/2,nlon/2)*Fs_lon/nlon*360 #zonal wavenum
wnum_1 = 2#1
wnum_2 = 14#15


ifreq_1 = np.abs(freq-freq_1).argmin() 
ifreq_2 = np.abs(freq-freq_2).argmin()
ifreq_1_neg = np.abs(freq-(-freq_1)).argmin() 
ifreq_2_neg = np.abs(freq-(-freq_2)).argmin() 


iwnum_1 = np.abs(zwnum-wnum_1).argmin() 
iwnum_2 = np.abs(zwnum-wnum_2).argmin() 
iwnum_1_neg = np.abs(zwnum+wnum_1).argmin() 
iwnum_2_neg = np.abs(zwnum+wnum_2).argmin()                 



c_min = np.sqrt(g * h_min)  # minimum phase speed
c_max = np.sqrt(g * h_max)  # maximum phase speed

for ilat in range(nlat):
    
    # Process positive frequencies and negative wavenumbers
    for ifreq in range(ifreq_1, ifreq_2+1):
        for iwnum in range(iwnum_2_neg, iwnum_1_neg+1):
            k = zwnum[iwnum]/(2*6371*1000*np.pi)  # Convert to rad/m
            if k == 0:  # Avoid division by zero
                continue
            
            # Calculate allowable frequency range for this wavenumber
            f_min = abs(k * c_min)  # Hz
            f_max = abs(k * c_max)  # Hz
            
            # Convert observed frequency to Hz
            f_obs = abs(freq[ifreq]/86400)
            
            # Check if observed frequency is within bounds
            if f_min <= f_obs <= f_max:
                V1_shift2[ifreq, ilat, iwnum] = V1_shift[ifreq, ilat, iwnum]
    
    # Process negative frequencies and positive wavenumbers
    for ifreq in range(ifreq_2_neg, ifreq_1_neg+1):
        for iwnum in range(iwnum_1, iwnum_2+1):
            k = zwnum[iwnum]/(2*6371*1000*np.pi)
            if k == 0:
                continue
            
          
            
            f_min = abs(k * c_min)
            f_max = abs(k * c_max)
            
            f_obs = abs(freq[ifreq]/86400)
            
            if f_min <= f_obs <= f_max:
                V1_shift2[ifreq, ilat, iwnum] = V1_shift[ifreq, ilat, iwnum]
  
                
V1_shift2 = np.fft.ifftshift( np.fft.ifftshift(V1_shift2,axes=2),axes=0 )

pr_kw = np.zeros([nt,nlat,nlon])

for ilat in range(0,nlat):
    pr_kw[:,ilat,:] = np.fft.ifft2(V1_shift2[:,ilat,:])


    
    
wave_data= xr.DataArray(pr_kw, coords=[anomaly['time'], anomaly['lat'], anomaly['lon']], dims=['time', 'lat', 'lon'])

ncl_kelvin = xr.open_dataset("/Users/xpji/spacetime_filter/olr.kelvin.15_wm_2_14.nc").kelvin
    

dif = wave_data - ncl_kelvin


plt.figure(dpi=200,figsize=(15,3))
plt.subplot(131)    

wave_data.std('time').plot.contourf(levels=np.linspace(0,15,16),cmap='bwr')
plt.title('Python')
plt.subplot(132)    
ncl_kelvin.std('time').plot.contourf(levels=np.linspace(0,15,16),cmap='bwr')  
plt.title('Ncl')
plt.subplot(133)    

dif.std('time').plot.contourf(cmap='BrBG',levels=np.linspace(-2,2,9))  
plt.title('difference(Python-Ncl)')




plt.figure(dpi=200,figsize=(6,3))
  

wave_data.std(['time','lat']).plot(label='Python')
plt.title('Python')
  
ncl_kelvin.std(['time','lat']).plot(label='Ncl')

plt.legend()




plt.figure(dpi=200,figsize=(6,3))
  

wave_data.std(['time','lon']).plot(label='Python')
plt.title('Python')
  
ncl_kelvin.std(['time','lon']).plot(label='Ncl')

plt.legend()




plt.figure(dpi=200,figsize=(6,3))
  

wave_data[:].std(['lat','lon']).plot(label='Python')
plt.title('Python')
  
ncl_kelvin[:].std(['lat','lon']).plot(label='Ncl')

plt.legend()

    
    
    
    
    
    
    
    
    
    
    