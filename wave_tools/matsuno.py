# -*- coding: utf-8 -*-
"""
Matsuno Theory Module - Matsuno理论模态模块
==========================================

实现Matsuno (1966) 赤道波动理论的各种模态计算。

主要功能：
---------
1. Kelvin波模态计算
2. 赤道Rossby波(ER)模态计算
3. 混合Rossby-重力波(MRG)模态计算
4. 惯性重力波(IG)模态计算
5. Westward惯性重力波(WIG)模态计算
6. Wheeler-Kiladis图的理论色散曲线生成

理论背景：
---------
Matsuno (1966) 在β平面近似下求解了浅水方程，得到了一系列赤道波动解。
这些波动在Wheeler-Kiladis波数-频率谱中表现为明显的能量峰。

主要波型：
- Kelvin波: 沿赤道向东传播，无南北风分量
- ER波 (n=1,2,...): 向西传播的大尺度波动  
- MRG波: 混合型波动，在低频时表现为重力波，高频时表现为Rossby波
- IG波 (n=1,2,...): 惯性重力波

参考文献：
---------
Matsuno, T., 1966: Quasi-geostrophic motions in the equatorial area. 
    J. Meteor. Soc. Japan, 44, 25–43.

Wheeler, M., and G. N. Kiladis, 1999: Convectively coupled equatorial waves: 
    Analysis of clouds and temperature in the wavenumber–frequency domain. 
    J. Atmos. Sci., 56, 374–399.

- https://github.com/Blissful-Jasper/wk_spectra/blob/master/wk_spectra/wk_analysis.py

作者: Jianpu
邮箱: xianpuji@hhu.edu.cn
机构: Hohai University
"""

import numpy as np
from scipy.optimize import fsolve
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd

# ===== 物理常数 =====
pi = np.pi
re    = 6.371008e6  # 地球半径 (m)
g     = 9.80665     # 重力加速度 (m/s²)
omega = 7.292e-05   # 地球自转角速度 (rad/s)
deg2rad = pi/180    # 度转弧度
sec2day = 1./(24.*60.*60.)  # 秒转天

def beta_parameters(latitude):
    """
    计算给定纬度的β平面参数
    
    在赤道β平面近似中，科里奥利参数f随纬度线性变化：
    f = β * y，其中 y 是离赤道的距离，β = df/dy
    
    参数
    ----
    latitude : float
        纬度 (度)，可以是正值(北半球)或负值(南半球)
    
    返回
    ------
    beta : float
        β平面参数 (m⁻¹ s⁻¹)
        计算公式: β = 2Ω cos(φ) / R
        其中 Ω是地球自转角速度，φ是纬度，R是地球半径
    perimeter : float
        该纬度圈的周长 (m)
        计算公式: L = 2πR cos(φ)
    
    示例
    ----
    >>> beta, perimeter = beta_parameters(0)  # 赤道
    >>> print(f"β = {beta:.2e} m⁻¹s⁻¹")
    >>> print(f"周长 = {perimeter/1e6:.2f} km")
    """
    beta  = 2.*omega*np.cos(abs(latitude)*deg2rad)/re
    perimeter = 2.*pi*re*np.cos(abs(latitude)*deg2rad)
    return (beta,perimeter)

def wn_array(max_wn,n_wn):
    """
    Creates an array with wavenumbers in the range (-max_wn,max_wn).
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: Array of Global Wavenumbers
    :rtype: Numpy Array
    """
    maxwn = abs(int(max_wn)) # maxwn is Positive Integer (maxwn > 0)
    n_wn = abs(int(n_wn))
    wn = np.linspace(-max_wn,max_wn,n_wn)
    return wn

def wn2k(wn,perimeter):
    """
    Converts an array of Global wave numbers to wavenumber in [rad m^{-1}].
    :param wn:
        Array of Global wavenumbers.
    :param perimeter:
        Perimeter in meters of the Earths's circunference at the given latitude
    :type wn: Numpy Array
    :type perimeter: Float
    :return: Array of Wavenumbers in [rad m^{-1}]
    :rtype: Numpy Array
    """
    wavelength = perimeter/wn # Wavekength[m]
    k  = 2.*pi/wavelength # Wavenumber[rad m^{-1}]
    return k

def afreq2freq(angular_frequency):
    """
    Convert angular frequency in [rad s^{-1}] to frequency in Cycles per
    Day(CPD).
    :param angular_frequency:
        Angular Frequency
    :type angular_frequency: Numpy Array
    :return: (Period,Frequency)
        Period in [days/cycle]
        Frequency in [cycles/day] Cycles per Day(CPD)
    :rtype: tuple
    """
    period = (2.*pi/angular_frequency)*sec2day # Period in [days/cycle]
    frequency = 1./period #[cycles/day] Cycles per Day(CPD)
    return (period,frequency)

def kelvin_mode(he, latitude=0, max_wn=50, n_wn=500):
    """
    计算赤道Kelvin波的色散曲线
    
    Kelvin波是赤道地区特有的向东传播波动，无南北风分量，
    常与对流耦合，是MJO的重要组成部分。
    
    理论：
    -----
    Kelvin波满足色散关系：ω = √(gH) * k
    其中 ω是角频率，g是重力加速度，H是等效深度，k是波数
    
    参数
    ----
    he : float
        等效深度 (m)，典型值 8-90m 对应不同的垂直模态
        he=12m 约对应第一斜压模态
    latitude : float, optional
        纬度 (度)，默认为 0 (赤道)
    max_wn : int, optional
        最大波数，波数范围为 [-max_wn, max_wn]，默认 50
    n_wn : int, optional
        波数数组的长度，默认 500
    
    返回
    ------
    df : pandas.DataFrame
        包含波数(index)和频率(列)的数据框
        - Index: 波数 (无量纲)
        - Column: 频率 (cycles/day，cpd)
    
    示例
    ----
    >>> import matplotlib.pyplot as plt
    >>> df = kelvin_mode(he=12, max_wn=15)
    >>> df.plot()
    >>> plt.xlabel('波数')
    >>> plt.ylabel('频率 (cpd)')
    >>> plt.title('Kelvin波色散曲线')
    
    注释
    ----
    - 只有正波数(k>0)有意义，因为Kelvin波仅向东传播
    - 负波数部分会被设置为NaN
    """
    (beta, perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)  # 全球波数
    k = wn2k(wn, perimeter)  # 波数 [rad/m]
    k[k <= 0] = np.nan  # Kelvin波只向东传播
    angular_frequency = np.sqrt(g*he) * k  # 角频率 [rad/s]
    (period, frequency) = afreq2freq(angular_frequency)
    # 周期 [days/cycle]，频率 [cycles/day]
    name = f'Kelvin(he={he}m)'
    df = pd.DataFrame(data={name: frequency}, index=wn)
    df.index.name = 'Wavenumber'
    return df

def mrg_mode(he,latitude=0,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Mixed
    Rossby Gravity Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    #wn = wn[wn<0] # Extract only wn < 0
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    k[k>=0] = np.nan
    angular_frequency = np.sqrt(g*he)*k*(0.5-0.5*np.sqrt(1.+\
                        (4*beta/(k*k*np.sqrt(g*he))))) # [rad s^{-1}]
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'MRG(he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df
def eig_n_0(he,latitude=0,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Mixed
    Rossby Gravity Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    k[k<=0] = np.nan
    angular_frequency = np.sqrt(g*he)*k*(0.5+0.5*np.sqrt(1.+\
                        (4*beta/(k*k*np.sqrt(g*he))))) # [rad s^{-1}]
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'EIG(n=0,he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def er_n(he,n,latitude=0.,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Mixed
    Rossby Gravity Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param n:
        Meridional Mode Number
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type n: Integer
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    # Use the Approximation to the Equatorial Rossby dispersion relationship as
    # a seed for the solver function
    angular_frequency = -beta*k/((k*k)+(2.*n+1.)*(beta/np.sqrt(g*he)))
    angular_frequency[k>=0] = np.nan
    angular_frequency[k<0] = fsolve(dispersion,angular_frequency[k<0],\
                                    args=(k[k<0],n,he,beta))
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'ER(n='+str(n)+',he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def eig_n(he,n,latitude=0.,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Easterly
    Gravity Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param n:
        Meridional Mode Number
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type n: Integer
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    # Use the Approximation to the EIG dispersion relationship as
    # a seed for the solver function
    angular_frequency = np.sqrt((2.*n+1.)*beta*np.sqrt(g*he)+(k**2)*g*he)
    angular_frequency[k<=0] = np.nan
    angular_frequency[k>0] = fsolve(dispersion,angular_frequency[k>0],\
                                    args=(k[k>0],n,he,beta))
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'EIG(n='+str(n)+',he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def wig_n(he,n,latitude=0.,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Westerly
    Gravity Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param n:
        Meridional Mode Number
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type n: Integer
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    # Use the Approximation to the WIG dispersion relationship as
    # a seed for the solver function
    angular_frequency = np.sqrt((2.*n+1.)*beta*np.sqrt(g*he)+(k**2)*g*he)
    angular_frequency[k>=0] = np.nan
    angular_frequency[k<0] = fsolve(dispersion,angular_frequency[k<0],\
                                    args=(k[k<0],n,he,beta))
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'WIG(n='+str(n)+',he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def dispersion(w,k,n,he,beta):
    """
    Dispersion relationship for Matsuno Modes(See Wheeler and Nguyen (2015)
    , Eq (13)).
    The roots of this function correspond to the angular frequencies of the
    Matsuno modes for a given k.
    :param w:
        Angular Frequency
    :param k:
        Longitudinal Wavenumber
    :param n:
        Meridional Mode Number
    :param he:
        Equivalent Depth
    :param beta:
        Beta-Plane Parameter
    :type w: Float
    :type k: Float
    :type n: Integer
    :type he: Float
    :type beta: Float
    :return: Zero if w and k corresponds to a Matsuno Mode.
    :rtype: Float
    """
    disp = w**3-g*he*(k**2+(beta*(2.*n+1.)/np.sqrt(g*he)))*w-k*beta*g*he
    return disp

def matsuno_dataframe(he,n=[1,2,3],latitude=0.,max_wn=50,n_wn=500):
    """
    Creates a dataframe with all Matsuno modes for a given set of meridional
    mode numbers given in a list.
    :param he:
        Equivalent Depth
    :param n:
        Meridional Mode Number
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type n: List of integers (e.g. [1,2,3])
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    df = []
    df.append(kelvin_mode(he,latitude,max_wn,n_wn))
    df.append(mrg_mode(he,latitude,max_wn,n_wn))
    df.append(eig_n_0(he,latitude,max_wn,n_wn))

    for nn in n:
        df.append(er_n(he,nn,latitude,max_wn,n_wn))
        df.append(eig_n(he,nn,latitude,max_wn,n_wn))
        df.append(wig_n(he,nn,latitude,max_wn,n_wn))

    df = reduce(lambda left,right: pd.merge(left,right,on='Wavenumber'), df)
    return df

def standar_plot(he,size=12,figsize=(8, 8),mx_wn=20,mx_freq=1.,labels='on'):
    """
    Creates a standard plot with all Matsuno modes for a given Equivalent Depth
    he.  This function plots the dispersion curves for the meridional mode
    numbers n = [1,2,3]. Some configuration is possible although with several
    limitations.
    :param he:
        Equivalent Depth
    :param size(optional):
        Text size
    :param figsize(optional):
        Figure size
    :param mx_wn(optional):
        Upper Limit of the Zonal Wave number range. The range is given
        by (-mx_wn,mx_wn)
    :param mx_freq(optional):
        Upper Limit for the y axis given as frequency in CPD.
    :param labels(optional):
        Plot the name of the Matsuno Modes in the figureself.
    :type he: Float
    :type size: Integer
    :type figsize: Tuple
    :type mx_wn: Integer
    :type mx_freq: Float
    :type mx_freq: String
    :return: Plot with Matsuno Modes
    :rtype: Matplotlib Figure
    """

    plt.rc('font', size=size)          # controls default text sizes
    plt.rc('axes', titlesize=size)     # fontsize of the axes title
    plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size)    # legend fontsize
    plt.rc('figure', titlesize=size)  # fontsize of the figure title

    df = matsuno_dataframe(he,n=[1,2,3])
    wn = df.index.values
    fig,ax = plt.subplots(figsize=figsize)

    for column in df:
        ax.plot(wn,df[column].values,color='k')

    ax.set_xlim(-mx_wn,mx_wn)
    ax.set_ylim(0,mx_freq)
    ax.set_xlabel('ZONAL WAVENUMBER')
    ax.set_ylabel('FREQUENCY (CPD)')
    plt.text(mx_wn-2*0.25*mx_wn,-0.06,'EASTWARD',fontsize=size-2)
    plt.text(-mx_wn+0.25*mx_wn,-0.06,'WESTWARD',fontsize=size-2)

    if labels=='on':
        # Print Kelvin Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int((len(p_wn)/2)+0.3*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][0],'Kelvin', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print MRG Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int(0.7*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][1],'MRG', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print EIG(n=0) Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int((len(p_wn)/2)+0.1*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][2],'EIG(n=0)', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print ER Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int(0.7*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][3]+0.01,'ER', \
        bbox={'facecolor':'none','edgecolor':'none'},fontsize=size+1)

        # Print EIG Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int((len(p_wn)/2)+0.3*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][7],'EIG', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print WIG Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int(0.55*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][8],'WIG', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print n Labels
        p_wn = wn[wn>=0]
        i, = np.where(wn == p_wn[0])[0]
        plt.text(-1,df.iloc[i][4],'n=1', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)
        plt.text(-1,df.iloc[i][7],'n=2', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)
        plt.text(-1,df.iloc[i][10],'n=3', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

    plt.show()
    return fig

def matsuno_modes_wk(he=[12,25,50],n=[1,],latitude=0.,max_wn=20,n_wn=500):
    """
    Creates a dataframe with all Matsuno modes for a given set of meridional
    mode numbers given in a list.
    :param he:
        Equivalent Depth
    :param n:
        Meridional Mode Number
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type n: List of integers (e.g. [1,2,3])
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    matsuno_modes = {}

    for h in he:
        df = []
        df.append(kelvin_mode(h,latitude,max_wn,n_wn))
        df.append(mrg_mode(h,latitude,max_wn,n_wn))
        df.append(eig_n_0(h,latitude,max_wn,n_wn))

        for nn in n:
            df.append(er_n(h,nn,latitude,max_wn,n_wn))
            df.append(eig_n(h,nn,latitude,max_wn,n_wn))
            df.append(wig_n(h,nn,latitude,max_wn,n_wn))

        df = reduce(lambda left,right: pd.merge(left,right,on='Wavenumber'), df)
        matsuno_modes[h] = df
    return matsuno_modes


