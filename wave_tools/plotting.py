"""
Plotting Module
===============

所有绘图功能的统一模块，包括：
1. Wheeler-Kiladis频谱图
2. 地图绘制
3. 泰勒图
4. CCKW包络图
5. 波动趋势图

作者: Jianpu
邮箱: xianpuji@hhu.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib import gridspec
from typing import Optional, List, Tuple
import xarray as xr
import os
from wave_tools import matsuno as mp
try:
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available, map plotting disabled")

try:
    import cmaps
    DEFAULT_CMAP = getattr(cmaps, 'cmocean_balance', getattr(cmaps, 'NCV_blu_red', 'RdBu_r'))
except ImportError:
    DEFAULT_CMAP = 'RdBu_r'


# ==================== CCKW包络绘图 ====================

def get_cckw_envelope_curve(he: Optional[List[float]] = None,
                            fmax: Optional[List[float]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    计算CCKW包络曲线坐标
    
    参数:
    ----
    he : list of float, optional
        等效深度列表（米），默认[8, 25, 90]
    fmax : list of float, optional
        最大频率列表，默认[1/3, 1/2.25, 0.5]
        
    返回:
    ----
    kw_x, kw_y : 曲线的x和y坐标列表
    """
    if he is None:
        he = [8, 25, 90]
    if fmax is None:
        fmax = [1/3, 1/2.25, 0.5]
    
    g = 9.8
    re = 6371e3
    s2d = 86400
    
    kw_x, kw_y = [], []
    
    for v in range(len(he)):
        s_min = (g * he[0]) ** 0.5 / (2 * np.pi * re) * s2d
        s_max = (g * he[-1]) ** 0.5 / (2 * np.pi * re) * s2d
        kw_tmax = 20
        
        kw_x.append(np.array([2, 1/kw_tmax/s_min, 14, 14, fmax[0]/s_max, 2, 2]))
        kw_y.append(np.array([1/kw_tmax, 1/kw_tmax, 14*s_min, fmax[0], fmax[0], 2*s_max, 1/20]))
    
    return kw_x, kw_y


def plot_cckw_envelope(he: Optional[List[float]] = None,
                       fmax: Optional[List[float]] = None,
                       save_path: Optional[str] = None,
                       dpi: int = 200) -> None:
    """
    绘制CCKW包络示意图
    
    参数:
    ----
    he : list, optional
        等效深度
    fmax : list, optional
        最大频率
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    """
    kw_x, kw_y = get_cckw_envelope_curve(he=he, fmax=fmax)
    
    if he is None:
        he_all = np.array([8, 25, 90])
    else:
        he_all = np.array(he)
    
    g = 9.8
    re = 6371e3
    s2d = 86400
    cp = (g * he_all) ** 0.5
    zwnum_goal = np.pi * re / cp / s2d
    
    plt.rcParams.update({'font.size': 6.5})
    fig, axs = plt.subplots(2, 3, figsize=(5.8, 3.9), dpi=dpi)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.15, hspace=0.22)
    
    for idx, ax in enumerate(axs.flat):
        # 绘制周期线
        for dd, d in enumerate([3, 6, 20]):
            ax.plot([-20, 20], [1/d, 1/d], 'k', linewidth=0.5, linestyle=':')
            ax.text(-14.8, 1/d + 0.01, f'{d}d', fontsize=6)
        
        # 绘制等效深度线
        for hh in range(len(he_all)):
            ax.plot([0, zwnum_goal[hh]], [0, 0.5], 'grey', linewidth=0.5, linestyle='dashed')
        
        ax.plot([0, 0], [0, 0.5], 'k', linewidth=0.5, linestyle=':')
        ax.plot(kw_x[0], kw_y[0], 'purple', linewidth=1.2, linestyle='solid')
        
        ax.set_xlim([-20, 20])
        ax.set_ylim([0, 0.5])
        ax.set_xticks(np.arange(-20, 21, 5))
        ax.set_yticks(np.arange(0, 0.55, 0.05))
        ax.tick_params(labelsize=6, direction='in', top=True, right=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(f"{save_path}.pdf", dpi=dpi, bbox_inches='tight')
        print(f'保存至: {save_path}.pdf')
    
    plt.show()
    plt.close()


# ==================== Wheeler-Kiladis频谱绘图 ====================

def plot_wk_spectrum(power_symmetric: xr.DataArray,
                     power_antisymmetric: xr.DataArray,
                     background: np.ndarray,
                     wavenumber: np.ndarray,
                     frequency: np.ndarray,
                     max_wn: int = 15,
                     max_freq: float = 0.5,
                     add_matsuno_lines: bool = True,
                     he: List[float] = [8, 25, 90],
                     cpd_lines: List[float] = [3, 6, 30],
                     save_path: Optional[str] = None,
                     cmap: str = 'RdBu_r',
                     levels: Optional[np.ndarray] = None) -> None:
    """
    绘制Wheeler-Kiladis频谱图（对称和反对称分量）
    
    参数:
    ----
    power_symmetric : xr.DataArray
        对称功率谱
    power_antisymmetric : xr.DataArray
        反对称功率谱
    background : np.ndarray
        背景谱
    wavenumber : np.ndarray
        波数数组
    frequency : np.ndarray
        频率数组
    max_wn : int
        最大波数
    max_freq : float
        最大频率
    add_matsuno_lines : bool
        是否添加Matsuno理论曲线
    he : list
        等效深度列表
    cpd_lines : list
        标注的周期线（天）
    save_path : str, optional
        保存路径
    cmap : str
        色标
    levels : np.ndarray, optional
        等值线水平
    """
    if levels is None:
        levels = np.array([1, 1.2, 1.4, 1.6, 1.8, 2.0])
    
    # 归一化
    sym_norm = power_symmetric / background
    asym_norm = power_antisymmetric / background
    
    # 选择绘图区域
    sym_plot = sym_norm.sel(frequency=slice(0, max_freq), wavenumber=slice(-max_wn, max_wn))
    asym_plot = asym_norm.sel(frequency=slice(0, max_freq), wavenumber=slice(-max_wn, max_wn))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    
    # 对称分量
    im1 = sym_plot.plot.contourf(ax=axes[0], cmap=cmap, levels=levels, 
                                  extend='neither', add_colorbar=False)
    sym_plot.plot.contour(ax=axes[0], levels=levels[levels>=1.1], 
                          colors='k', linewidths=0.5, add_labels=False)
    axes[0].set_title('Symmetric Component')
    
    # 反对称分量
    im2 = asym_plot.plot.contourf(ax=axes[1], cmap=cmap, levels=levels,
                                   extend='neither', add_colorbar=False)
    asym_plot.plot.contour(ax=axes[1], levels=levels[levels>=1.1],
                           colors='k', linewidths=0.5, add_labels=False)
    axes[1].set_title('Antisymmetric Component')
    
    # 设置坐标轴
    for ax in axes:
        ax.axvline(0, linestyle='--', color='k', linewidth=0.5)
        ax.set_xlim([-max_wn, max_wn])
        ax.set_ylim([0, max_freq])
        ax.set_xlabel('Zonal Wavenumber')
        ax.set_ylabel('Frequency (CPD)')
        
        # 周期线
        for cpd in cpd_lines:
            if 1/cpd <= max_freq:
                ax.axhline(1/cpd, color='k', linestyle=':', linewidth=0.5)
                ax.text(-max_wn+1, 1/cpd+0.01, f'{cpd}d', fontsize=8,
                       bbox=dict(facecolor='w', alpha=0.7, edgecolor='none'))
    
    # 添加Matsuno理论曲线
    if add_matsuno_lines:
        from .matsuno import matsuno_modes_wk
        matsuno_modes = matsuno_modes_wk(he=he, n=[1], max_wn=max_wn)
        
        kw_x, kw_y = get_cckw_envelope_curve()
        
        for ax in axes:
            for key in matsuno_modes:
                ax.plot(matsuno_modes[key]['Kelvin(he={}m)'.format(key)], 
                       color='k', linestyle='-', linewidth=0.8)
                ax.plot(matsuno_modes[key]['ER(n=1,he={}m)'.format(key)],
                       color='k', linestyle='-', linewidth=0.8)
            
            ax.plot(kw_x[0], kw_y[0], 'g', linewidth=1.2, linestyle='-', zorder=5)
    
    # 色标
    fig.colorbar(im1, ax=axes, orientation='horizontal', 
                 shrink=0.6, aspect=30, pad=0.08)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'保存至: {save_path}')
    
    plt.show()


# ==================== 地图绘图 ====================

def setup_map_axes(ax, title: str, box: List[float] = [0, 360, -20, 20]):
    """
    设置地图坐标轴
    
    参数:
    ----
    ax : matplotlib.axes.Axes
        坐标轴对象
    title : str
        标题
    box : list
        [lon_min, lon_max, lat_min, lat_max]
        
    返回:
    ----
    ax : 设置后的坐标轴
    """
    if not HAS_CARTOPY:
        raise ImportError("需要安装cartopy进行地图绘制")
    
    ax.set_xticks(np.linspace(box[0], box[1], 7), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(box[2], box[3], 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.xaxis.set_major_locator(ticker.MultipleLocator((box[1]-box[0])/6))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(6))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.set_title(title, loc='left')
    ax.tick_params(which='major', length=5)
    
    return ax


def plot_spatial_field(data: xr.DataArray,
                       ax,
                       cmap: str = 'RdBu_r',
                       title: str = '',
                       box: List[float] = [0, 360, -20, 20],
                       levels: Optional[np.ndarray] = None,
                       add_coastlines: bool = True) -> any:
    """
    绘制空间场
    
    参数:
    ----
    data : xr.DataArray
        数据
    ax : matplotlib.axes.Axes
        坐标轴
    cmap : str
        色标
    title : str
        标题
    box : list
        地图范围
    levels : np.ndarray, optional
        等值线水平
    add_coastlines : bool
        是否添加海岸线
        
    返回:
    ----
    contourf对象
    """
    if not HAS_CARTOPY:
        raise ImportError("需要安装cartopy")
    
    if levels is None:
        levels = np.linspace(data.min().values, data.max().values, 21)
    
    f = data.plot.contourf(ax=ax, cmap=cmap, levels=levels,
                           add_labels=False, add_colorbar=False,
                           transform=ccrs.PlateCarree())
    
    if add_coastlines:
        ax.coastlines()
    
    ax.set_extent(box, crs=ccrs.PlateCarree())
    setup_map_axes(ax, title, box)
    
    return f

def set_axis_for_wave(ax, text_size, open_ma = True, wig=True, kelvin_box=True,freq_lines=True, depth=True, \
                      cpd_lines=[3, 6, 30], max_wn_plot=15, max_freq_plot=0.5, labels=True,wig_text=True):
    ax.axvline(x=0, color='k', linestyle='--')
    ax.set_xlim((-max_wn_plot,max_wn_plot))
    ax.set_ylim((0.02,max_freq_plot))
    kw_x,kw_y = get_cckw_envelope_curve()
    if max_wn_plot is not None and max_freq_plot is not None:
        ax.set_xlim((-max_wn_plot, max_wn_plot))
        ax.set_ylim((0.02, max_freq_plot))
        
        
    if kelvin_box:
            ax.plot(kw_x[0], kw_y[0], 'purple', linewidth=1.2, linestyle='solid',zorder=5)
            
    if freq_lines:
        # Assuming self.freq_lines and self.cpd_lines are defined elsewhere
        for d in cpd_lines:
            if (1./d) <= max_freq_plot:
                ax.axhline(y=1./d, color='k', linestyle='--',linewidth=0.5)
                ax.text(-max_wn_plot+0.8, (1./d+0.01), str(d)+' days', color='k',
                size=text_size-6, bbox={'facecolor': 'w', 'alpha': 0.9, 'edgecolor': 'none'})
    if depth:
       
        # Define the range for filtering
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        ax.text(right+10, 0.29  * (bottom+top),'Kelvin', ha="center",va="center",size=text_size-6,
                bbox={'facecolor':'w','alpha':0.9,'edgecolor':'none'})
        ax.text(right+5.6, 0.4 * (bottom+top),'h=90', ha="center",va="center",size=text_size-6,
                bbox={'facecolor':'w','alpha':0.9,'edgecolor':'none'})
        ax.text(right+10.5, 0.38 * (bottom+top),'h=25', ha="center",va="center",size=text_size-6,
                bbox={'facecolor':'w','alpha':0.9,'edgecolor':'none'})
        ax.text(right+9.9, 0.2 * (bottom+top),'h=8', ha="center",va="center",size=text_size-6,
                bbox={'facecolor':'w','alpha':0.9,'edgecolor':'none'})
    def filter_series(series, min_wn, max_wn):
        return series[(series.index >= min_wn) & (series.index <= max_wn)]
    
    if open_ma:
    
        matsuno_modes = mp.matsuno_modes_wk(he=[90,25,8],n=[1],max_wn=max_wn_plot)
        # kelvin_series_90 = filter_series(matsuno_modes[90]['Kelvin(he=90m)'], 2, 5.2)
        # kelvin_series_8 = filter_series(matsuno_modes[8]['Kelvin(he=8m)'], 2.6, 14)
        
        for key in matsuno_modes:
            # print(key)
            ax.plot(matsuno_modes[key]['Kelvin(he={}m)'.format(key)],color='k',linestyle='-')
            ax.plot(matsuno_modes[key]['ER(n=1,he={}m)'.format(key)],color='k',linestyle='-')
            ax.plot(matsuno_modes[key]['Kelvin(he={}m)'.format(key)],color='k',linestyle='-')
            ax.plot(matsuno_modes[key]['ER(n=1,he={}m)'.format(key)],color='k',linestyle='-')
            ax.plot(matsuno_modes[key]['Kelvin(he={}m)'.format(key)],color='k',linestyle='-')
            ax.plot(matsuno_modes[key]['ER(n=1,he={}m)'.format(key)],color='k',linestyle='-')
            if  wig==True:
                
                ax.plot(matsuno_modes[key]['WIG(n=1,he={}m)'.format(key)],color='k',linestyle='-')
                ax.plot(matsuno_modes[key]['WIG(n=1,he={}m)'.format(key)],color='k',linestyle='-')
                ax.plot(matsuno_modes[key]['WIG(n=1,he={}m)'.format(key)],color='k',linestyle='-')
            
                ax.plot(matsuno_modes[key]['EIG(n=1,he={}m)'.format(key)],color='k',linestyle='-')
                ax.plot(matsuno_modes[key]['EIG(n=1,he={}m)'.format(key)],color='k',linestyle='-')
                if kelvin_box:
                    kx,ky = get_cckw_envelope_curve()
                    ax.plot(kx[0], ky[0], 'purple', linewidth=1.2, linestyle='solid')
                    # ax.plot(kelvin_series_90, color='purple', linestyle='-')
                    # ax.plot(kelvin_series_8, color='purple', linestyle='-')

        if labels:
            key = list(matsuno_modes.keys())[len(list(matsuno_modes.keys()))//2] 
            wn = matsuno_modes[key].index.values
            k = int((len(wn)/2)+0.3*(len(wn)/2))
            k, = np.where(wn == wn[k])[0]

            k = int(0.7*(len(wn)/2))
            k = np.where(wn == wn[k])[0]
            ax.text(wn[k]+0.4,matsuno_modes[key]['ER(n=1,he={}m)'.format(key)].iloc[k]+0.02,'ER', \
            bbox={'facecolor':'w','alpha':0.9,'edgecolor':'none'},fontsize=text_size-6)
            if wig_text:
                
                ax.text(wn[k]+0.4,matsuno_modes[key]['WIG(n=1,he={}m)'.format(key)].iloc[k]+0.02,'n=1 WIG', \
                        bbox={'facecolor':'w','alpha':1,'edgecolor':'none'},fontsize=text_size-6)

# ==================== 泰勒图 ====================

class TaylorDiagram:
    """泰勒图类"""
    
    def __init__(self, refstd: float, fig=None, rect=111,
                 label='_', srange=(0, 1.5), extend=False):
        """
        初始化泰勒图
        
        参数:
        ----
        refstd : float
            参考标准差
        fig : matplotlib.figure.Figure, optional
            图形对象
        rect : int
            子图位置
        label : str
            参考标签
        srange : tuple
            标准差范围
        extend : bool
            是否扩展到负相关
        """
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF
        
        self.refstd = refstd
        tr = PolarAxes.PolarTransform()
        
        # 相关系数标签
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            self.tmax = np.pi/2
        
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd
        
        ghelper = FA.GridHelperCurveLinear(
            tr, extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)
        
        if fig is None:
            fig = plt.figure()
        
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        
        # 调整坐标轴
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].set_axis_direction("top" if extend else "left")
        
        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)
        
        self._ax = ax
        self.ax = ax.get_aux_axes(tr)
        
        # 添加参考点
        self.ax.plot([0], self.refstd, 'r*', ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'r--', label='_')
        
        self.samplePoints = []
    
    def add_sample(self, stddev: float, corrcoef: float, *args, **kwargs):
        """添加样本点"""
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)
        return l
    
    def add_contours(self, levels=5, **kwargs):
        """添加RMS等值线"""
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                            np.linspace(0, self.tmax))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours


# ==================== 工具函数 ====================

def save_figure(fig, filename: str, folder: Optional[str] = None,
                fmt: str = 'pdf', dpi: int = 300) -> None:
    """
    保存图形
    
    参数:
    ----
    fig : matplotlib.figure.Figure
        图形对象
    filename : str
        文件名
    folder : str, optional
        文件夹路径
    fmt : str
        文件格式
    dpi : int
        分辨率
    """
    if folder is None:
        folder = os.getcwd()
    
    os.makedirs(folder, exist_ok=True)
    outpath = os.path.join(folder, f"{filename}.{fmt}")
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight', format=fmt)
    print(f'图形已保存: {outpath}')
