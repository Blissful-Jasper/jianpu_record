"""
交叉谱分析工具模块
用于计算和可视化两个变量之间的交叉谱关系

Author: Refactored for generalization
Date: 2026-01-15
"""

import numpy as np
import xarray as xr
import os
import gc
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import psutil
import dask
import cmaps

from wave_tools import calculate_cross_spectrum, remove_annual_cycle
from wave_tools.utils import get_curve


# ============ 内存监控工具 ============
class MemoryMonitor:
    """内存监控类，用于跟踪程序运行时的内存使用情况"""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_info(self) -> Dict[str, float]:
        """
        获取当前内存使用信息
        
        Returns:
        --------
        dict: 包含以下键值:
            - rss_gb: 物理内存使用量 (GB)
            - vms_gb: 虚拟内存使用量 (GB)
            - percent: 进程内存占用百分比
            - available_gb: 系统可用内存 (GB)
            - total_gb: 系统总内存 (GB)
        """
        mem = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        virtual_mem = psutil.virtual_memory()
        
        return {
            'rss_gb': mem.rss / 1024**3,
            'vms_gb': mem.vms / 1024**3,
            'percent': mem_percent,
            'available_gb': virtual_mem.available / 1024**3,
            'total_gb': virtual_mem.total / 1024**3
        }
    
    def print_memory_status(self, label: str = "") -> Dict[str, float]:
        """
        打印内存状态
        
        Parameters:
        -----------
        label : str
            标签，用于标识当前检查点
            
        Returns:
        --------
        dict: 内存信息字典
        """
        info = self.get_memory_info()
        print(f"\n{'='*60}")
        print(f"💾 内存状态 {f'- {label}' if label else ''}")
        print(f"{'='*60}")
        print(f"  进程物理内存使用: {info['rss_gb']:.2f} GB")
        print(f"  进程虚拟内存使用: {info['vms_gb']:.2f} GB")
        print(f"  进程内存占比: {info['percent']:.1f}%")
        print(f"  系统可用内存: {info['available_gb']:.2f} GB / {info['total_gb']:.2f} GB")
        print(f"{'='*60}\n")
        
        if info['available_gb'] < 10:
            print("⚠️  警告: 系统可用内存不足10GB!")
        
        return info


# ============ 数据加载工具 ============
def load_netcdf_data(
    file_path: str,
    chunks: Optional[Dict[str, int]] = None,
    verbose: bool = True
) -> xr.DataArray:
    """
    加载NetCDF数据文件（支持lazy loading）
    
    Parameters:
    -----------
    file_path : str
        数据文件路径
    chunks : dict, optional
        数据分块参数，用于dask lazy loading
        例如: {'time': 5000}
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    xr.DataArray
        加载的数据（如果指定chunks则为lazy loading）
        
    Raises:
    -------
    FileNotFoundError
        如果文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if verbose:
        print(f"📂 加载数据: {file_path}")
    
    # 默认使用时间维度分块
    if chunks is None:
        chunks = {'time': 5000}
    
    # 尝试不同的引擎加载数据
    for engine in ['netcdf4', 'h5netcdf', None]:
        try:
            if engine:
                data = xr.open_dataarray(file_path, engine=engine, chunks=chunks)
            else:
                data = xr.open_dataarray(file_path, chunks=chunks)
            break
        except Exception as e:
            if engine is None:
                raise e
            continue
    
    if verbose:
        print(f"✅ 数据加载成功 (lazy loading)")
        print(f"   形状: {data.shape}")
        print(f"   维度: {data.dims}")
        if hasattr(data, 'chunks'):
            print(f"   数据块: {data.chunks}")
        if hasattr(data, 'time'):
            print(f"   时间范围: {data.time.values[0]} to {data.time.values[-1]}")
        data_size_gb = data.nbytes / 1024**3
        print(f"   估算数据大小: {data_size_gb:.2f} GB")
    
    return data


def load_multiple_experiments(
    variable_name: str,
    experiments: List[str],
    data_dir: str,
    file_pattern: str = "{var}_{exp}_2deg_interp.nc",
    chunks: Optional[Dict[str, int]] = None,
    scale_factor: float = 1.0,
    verbose: bool = True
) -> Dict[str, xr.DataArray]:
    """
    加载多个实验的同一变量数据
    
    Parameters:
    -----------
    variable_name : str
        变量名称（用于构建文件名）
    experiments : list of str
        实验名称列表，例如: ['cntl', 'p4k', '4co2']
    data_dir : str
        数据目录路径
    file_pattern : str
        文件名模板，使用{var}和{exp}作为占位符
    chunks : dict, optional
        数据分块参数
    scale_factor : float
        数据缩放因子（例如，潜热通量可能需要乘以-1）
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    dict
        键为实验名，值为对应的DataArray
        
    Example:
    --------
    >>> pr_data = load_multiple_experiments(
    ...     variable_name='pr',
    ...     experiments=['cntl', 'p4k', '4co2'],
    ...     data_dir='/path/to/data'
    ... )
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"加载变量: {variable_name.upper()}")
        print(f"{'='*60}")
    
    data_dict = {}
    
    for exp in experiments:
        try:
            file_path = os.path.join(
                data_dir,
                file_pattern.format(var=variable_name, exp=exp)
            )
            
            if verbose:
                print(f"\n加载 {exp.upper()}...")
            
            data = load_netcdf_data(file_path, chunks=chunks, verbose=verbose)
            
            # 应用缩放因子
            if scale_factor != 1.0:
                data = data * scale_factor
                if verbose:
                    print(f"   应用缩放因子: {scale_factor}")
            
            data_dict[exp] = data
            
        except Exception as e:
            print(f"❌ 加载 {exp} 失败: {e}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ 成功加载 {len(data_dict)} 个实验的数据")
        print(f"{'='*60}")
    
    return data_dict


# ============ 数据预处理工具 ============
def preprocess_data_with_mask(
    data1: xr.DataArray,
    data2: xr.DataArray,
    mask: Optional[xr.DataArray] = None,
    remove_annual: bool = True,
    fill_value: float = 0.0,
    verbose: bool = True
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    预处理两个数据数组：去除年循环、应用掩膜、清理无效值
    
    Parameters:
    -----------
    data1, data2 : xr.DataArray
        输入数据（支持dask lazy loading）
    mask : xr.DataArray, optional
        空间掩膜（例如海洋掩膜），True表示保留，False表示屏蔽
    remove_annual : bool
        是否去除年循环
    fill_value : float
        填充NaN的值
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    data1_processed, data2_processed : xr.DataArray
        预处理后的数据（延迟计算）
    """
    if verbose:
        print("\n" + "="*60)
        print("数据预处理")
        print("="*60)
    
    # 步骤1: 去除年循环
    if remove_annual:
        if verbose:
            print("  📊 步骤1: 去除年循环（延迟计算）...")
        data1_ano = data1.groupby('time.dayofyear') - data1.groupby('time.dayofyear').mean()
        data2_ano = data2.groupby('time.dayofyear') - data2.groupby('time.dayofyear').mean()
    else:
        data1_ano = data1
        data2_ano = data2
    
    # 步骤2: 应用掩膜
    if mask is not None:
        if verbose:
            print("  🌊 步骤2: 应用掩膜（延迟计算）...")
        mask_float = mask.astype(float)
        data1_ano = data1_ano * mask_float
        data2_ano = data2_ano * mask_float
    
    # 步骤3: 清理无效值
    if verbose:
        print("  🧹 步骤3: 清理无效值（延迟计算）...")
    
    # 处理Inf
    data1_ano = xr.where(np.isinf(data1_ano), np.nan, data1_ano)
    data2_ano = xr.where(np.isinf(data2_ano), np.nan, data2_ano)
    
    # 填充NaN
    data1_ano = data1_ano.fillna(fill_value)
    data2_ano = data2_ano.fillna(fill_value)
    
    # 步骤4: 再次去除年循环（使用wave_tools函数）
    if remove_annual:
        if verbose:
            print("  🔄 步骤4: 再次去除年循环（延迟计算）...")
        data1_ano = remove_annual_cycle(data1_ano)
        data2_ano = remove_annual_cycle(data2_ano)
    
    if verbose and hasattr(data1_ano, 'chunks'):
        print(f"     data1 chunks: {data1_ano.chunks}")
        print(f"     data2 chunks: {data2_ano.chunks}")
    
    return data1_ano, data2_ano


# ============ 交叉谱计算工具 ============
def compute_cross_spectrum_for_experiments(
    data1_dict: Dict[str, xr.DataArray],
    data2_dict: Dict[str, xr.DataArray],
    experiments: List[str],
    mask: Optional[xr.DataArray] = None,
    seg_length: int = 96,
    seg_overlap: int = -65,
    symmetry: str = 'symm',
    memory_monitor: Optional[MemoryMonitor] = None,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    为多个实验计算交叉谱
    
    Parameters:
    -----------
    data1_dict, data2_dict : dict
        键为实验名，值为DataArray的字典
    experiments : list of str
        要处理的实验名列表
    mask : xr.DataArray, optional
        空间掩膜
    seg_length : int
        分段长度（天数）
    seg_overlap : int
        分段重叠长度
    symmetry : str
        对称性设置: 'symm', 'asymm', 'latband'
    memory_monitor : MemoryMonitor, optional
        内存监控器实例
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    dict
        嵌套字典，结构为:
        {
            'exp_name': {
                'STC': xr.DataArray,  # 谱分量
                'freq': np.ndarray,    # 频率
                'wave': np.ndarray,    # 波数
                'nseg': int,           # 分段数
                'dof': int,            # 自由度
                'p': float,            # p值
                'prob_coh2': float     # coherence squared临界值
            }
        }
    """
    results = {}
    
    for exp_name in experiments:
        if verbose:
            print(f"\n{'='*60}")
            print(f"处理实验: {exp_name.upper()}")
            print(f"{'='*60}")
        
        if memory_monitor:
            memory_monitor.print_memory_status(f"开始处理 {exp_name}")
        
        # 检查数据是否存在
        if exp_name not in data1_dict or exp_name not in data2_dict:
            print(f"⚠️  警告: 实验 {exp_name} 的数据不完整，跳过")
            continue
        
        # 获取原始数据
        data1_raw = data1_dict[exp_name]
        data2_raw = data2_dict[exp_name]
        
        # 应用掩膜（如果提供）
        if mask is not None:
            data1_raw = data1_raw.where(mask, drop=True)
            data2_raw = data2_raw.where(mask, drop=True)
        
        if verbose and hasattr(data1_raw, 'chunks'):
            print(f"  📦 原始数据 chunks:")
            print(f"     data1: {data1_raw.chunks}")
            print(f"     data2: {data2_raw.chunks}")
        
        # 预处理数据
        data1_ano, data2_ano = preprocess_data_with_mask(
            data1_raw, data2_raw, mask=None, verbose=verbose
        )
        
        # 计算数据（Dask延迟计算）
        if verbose:
            print("\n  💻 执行Dask计算...")
        data1_computed, data2_computed = dask.compute(data1_ano, data2_ano)
        
        if memory_monitor:
            memory_monitor.print_memory_status(f"预处理完成 {exp_name}")
        
        # 计算交叉谱
        if verbose:
            print(f"\n  📊 计算交叉谱...")
            print(f"     分段长度: {seg_length}")
            print(f"     分段重叠: {seg_overlap}")
            print(f"     对称性: {symmetry}")
        
        result = calculate_cross_spectrum(
            data1_computed, data2_computed,
            segLen=seg_length,
            segOverLap=seg_overlap,
            symmetry=symmetry,
            return_xarray=True
        )
        
        # 检查结果
        if result is None:
            print(f"    ✗ 交叉谱计算失败")
            continue
        
        # 保存结果
        results[exp_name] = {
            'STC': result['STC'],
            'freq': result['freq'],
            'wave': result['wave'],
            'nseg': result['nseg'],
            'dof': result['dof'],
            'p': result['p'],
            'prob_coh2': result['prob_coh2']
        }
        
        if verbose:
            print(f"    ✓ 交叉谱计算完成")
            print(f"      分段数: {result['nseg']}")
            print(f"      自由度: {result['dof']}")
            print(f"      99%显著性阈值: {result['prob_coh2']}")
        
        if memory_monitor:
            memory_monitor.print_memory_status(f"完成 {exp_name}")
        
        # 清理内存
        del data1_ano, data2_ano, data1_computed, data2_computed
        gc.collect()
    
    return results


# ============ 可视化工具 ============
def plot_cross_spectrum_panel(
    results: Dict[str, Dict],
    experiments: List[str],
    exp_titles: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 8),
    dpi: int = 300,
    significance_level: float = 0.99,
    cmap: str = 'WhiteBlueGreenYellowRed',
    contour_levels: Optional[np.ndarray] = None,
    vector_scale: float = 30,
    vector_skip: int = 2,
    xlim: Tuple[float, float] = (-15, 15),
    ylim: Tuple[float, float] = (0, 0.5),
    add_dispersion_curves: bool = True,
    add_period_lines: bool = True,
    period_days: List[int] = [3, 6, 20],
    equivalent_depths: List[int] = [8, 25, 90],
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Figure, np.ndarray]:
    """
    绘制交叉谱分析的面板图
    
    Parameters:
    -----------
    results : dict
        compute_cross_spectrum_for_experiments的输出结果
    experiments : list of str
        要绘制的实验列表
    exp_titles : list of str, optional
        实验的显示标题，默认使用大写的实验名
    figsize : tuple
        图像大小 (width, height)
    dpi : int
        图像分辨率
    significance_level : float
        显著性水平（0-1之间），用于掩蔽不显著的结果
    cmap : str
        色标名称（cmaps包的色标）
    contour_levels : np.ndarray, optional
        等高线水平，默认为np.linspace(0.2, 0.8, 21)
    vector_scale : float
        矢量缩放因子
    vector_skip : int
        矢量场采样间隔
    xlim, ylim : tuple
        x和y轴范围
    add_dispersion_curves : bool
        是否添加Kelvin波色散关系曲线
    add_period_lines : bool
        是否添加周期参考线
    period_days : list of int
        要标注的周期（天）
    equivalent_depths : list of int
        Kelvin波等效深度（米）
    output_path : str, optional
        输出文件路径，如果提供则保存图像
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        图像对象
    axes : np.ndarray
        子图数组
    """
    # 参数检查
    n_exp = len(experiments)
    if exp_titles is None:
        exp_titles = [exp.upper() for exp in experiments]
    
    if len(exp_titles) != n_exp:
        raise ValueError("exp_titles长度必须与experiments一致")
    
    if contour_levels is None:
        contour_levels = np.linspace(0.2, 0.8, 21)
    
    # 创建子图
    fig, axes = plt.subplots(1, n_exp, figsize=figsize, dpi=dpi)
    if n_exp == 1:
        axes = np.array([axes])
    
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.15, wspace=0.25)
    plt.rcParams.update({'font.size': 10})
    
    # 常数
    s2d = 86400  # 秒/天
    earth_radius = 6371 * 1000  # 地球半径（米）
    
    # 绘制每个实验
    for idx, (exp_name, exp_title, ax) in enumerate(zip(experiments, exp_titles, axes)):
        if exp_name not in results:
            if verbose:
                print(f"⚠️  警告: 结果中不包含实验 {exp_name}，跳过")
            continue
        
        plt.sca(ax)
        
        # 获取数据
        stc = results[exp_name]['STC']
        wave = results[exp_name]['wave']
        freq = results[exp_name]['freq']
        prob_coh2 = results[exp_name]['prob_coh2']
        
        # 获取显著性阈值
        if isinstance(prob_coh2, np.ndarray) and prob_coh2.size > 1:
            threshold = float(prob_coh2.max())
        elif hasattr(prob_coh2, 'item'):
            threshold = float(prob_coh2.item())
        else:
            threshold = float(prob_coh2)
        
        if verbose:
            print(f"\n{exp_name.upper()}:")
            print(f"  显著性阈值 ({int(significance_level*100)}%): {threshold:.6f}")
        
        # 获取coherence squared
        coh2 = stc.sel(component='COH2')
        
        # 统计显著性
        n_significant = int((coh2 >= threshold).sum())
        if verbose:
            print(f"  显著点数: {n_significant}/{coh2.size}")
        
        # 掩蔽不显著的数据
        coh2_masked = coh2.where(coh2 >= threshold)
        
        # 绘制等高线
        try:
            contourf = coh2_masked.plot.contourf(
                ax=ax,
                cmap=getattr(cmaps, cmap) if hasattr(cmaps, cmap) else cmap,
                levels=contour_levels,
                add_colorbar=False,
                add_labels=False,
                extend='neither'
            )
        except Exception as e:
            print(f"⚠️  绘制等高线时出错: {e}")
            contourf = None
        
        # 准备矢量场
        wave_sub = wave[::vector_skip]
        freq_sub = freq[::vector_skip]
        u_sub = stc.sel(component='V1').values[::vector_skip, ::vector_skip]
        v_sub = stc.sel(component='V2').values[::vector_skip, ::vector_skip]
        coh2_sub = stc.sel(component='COH2').values[::vector_skip, ::vector_skip]
        
        # 掩蔽矢量场
        mask = coh2_sub < threshold
        u_masked = np.where(mask, np.nan, u_sub)
        v_masked = np.where(mask, np.nan, v_sub)
        
        n_valid = np.sum(~np.isnan(u_masked))
        if verbose:
            print(f"  有效矢量数: {n_valid}")
        
        # 绘制矢量
        if n_valid > 0:
            ax.quiver(
                wave_sub, freq_sub,
                u_masked, v_masked,
                scale=vector_scale, headwidth=4, headlength=5,
                width=0.004, alpha=0.8
            )
        
        # 设置标题
        ax.set_title(f'({chr(97 + idx)}) {exp_title}', fontsize=18, loc='left')
        ax.set_title(f'Sym', fontsize=10, loc='right')
        
        # 设置坐标轴
        ax.set_ylabel('Frequency (1/day)', fontsize=18)
        ax.set_xlabel('Zonal wavenumber', fontsize=18)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # 添加CCKW波段
        try:
            kw_x, kw_y = get_curve()
            ax.plot(kw_x[0], kw_y[0], 'red', linewidth=1.5, 
                   linestyle='solid', label='CCKW band')
        except:
            pass
        
        # 添加周期线
        if add_period_lines:
            ax.plot([0, 0], ylim, 'k', linewidth=1, linestyle=':')
            
            for day, label in zip(period_days, [f'{d}d' for d in period_days]):
                period_freq = 1 / day
                ax.plot(xlim, [period_freq, period_freq], 'k', 
                       linewidth=1, linestyle=':')
                ax.text(xlim[0] + 0.2, period_freq + 0.01, label, 
                       fontsize=15, color='k')
        
        # 添加色散关系
        if add_dispersion_curves:
            cp = (9.8 * np.array(equivalent_depths)) ** 0.5
            wave_goal = 0.5 / s2d / cp * 2 * np.pi * earth_radius
            
            for wg in wave_goal:
                ax.plot([0, wg], [0, ylim[1]], 'grey', 
                       linewidth=1, linestyle='dashed')
            
            ax.text(12, 0.35, 'kelvin', ha="center", va="center", size=9,
                   bbox={'facecolor': 'w', 'alpha': 0.9, 'edgecolor': 'none'})
        
        ax.tick_params(labelsize=18, which='both', top=True, right=True)
    
    # 添加色标
    if contourf is not None:
        cbar = fig.colorbar(contourf, ax=axes, orientation='horizontal',
                           pad=0.15, aspect=40, shrink=0.8)
        cbar.set_label('Coherence Squared', fontsize=14)
    
    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')
        if verbose:
            print(f"\n✅ 图像已保存: {output_path}")
    
    return fig, axes


# ============ 便捷函数 ============
def analyze_cross_spectrum(
    var1_name: str,
    var2_name: str,
    experiments: List[str],
    data_dir: str,
    mask: Optional[xr.DataArray] = None,
    file_pattern: str = "{var}_{exp}_2deg_interp.nc",
    var1_scale: float = 1.0,
    var2_scale: float = 1.0,
    chunks: Optional[Dict[str, int]] = None,
    seg_length: int = 96,
    seg_overlap: int = -65,
    symmetry: str = 'symm',
    output_dir: Optional[str] = None,
    plot_params: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Dict], Optional[Tuple[Figure, np.ndarray]]]:
    """
    一站式交叉谱分析：从数据加载到结果可视化
    
    Parameters:
    -----------
    var1_name, var2_name : str
        两个变量的名称
    experiments : list of str
        实验名称列表
    data_dir : str
        数据目录
    mask : xr.DataArray, optional
        空间掩膜
    file_pattern : str
        文件名模板
    var1_scale, var2_scale : float
        变量缩放因子
    chunks : dict, optional
        dask分块参数
    seg_length : int
        谱分析分段长度
    seg_overlap : int
        分段重叠
    symmetry : str
        对称性
    output_dir : str, optional
        输出目录（用于保存图像）
    plot_params : dict, optional
        传递给plot_cross_spectrum_panel的额外参数
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    results : dict
        交叉谱计算结果
    fig_axes : tuple or None
        (fig, axes)元组，如果指定了output_dir
        
    Example:
    --------
    >>> results, (fig, axes) = analyze_cross_spectrum(
    ...     var1_name='pr',
    ...     var2_name='olr',
    ...     experiments=['cntl', 'p4k', '4co2'],
    ...     data_dir='/path/to/data',
    ...     mask=ocean_mask,
    ...     output_dir='./figures/cross_spectrum'
    ... )
    """
    # 创建内存监控器
    mem_mon = MemoryMonitor()
    mem_mon.print_memory_status("分析开始")
    
    # 加载数据
    if verbose:
        print(f"\n{'='*60}")
        print(f"交叉谱分析: {var1_name.upper()} vs {var2_name.upper()}")
        print(f"{'='*60}")
    
    var1_data = load_multiple_experiments(
        var1_name, experiments, data_dir,
        file_pattern=file_pattern,
        chunks=chunks,
        scale_factor=var1_scale,
        verbose=verbose
    )
    
    var2_data = load_multiple_experiments(
        var2_name, experiments, data_dir,
        file_pattern=file_pattern,
        chunks=chunks,
        scale_factor=var2_scale,
        verbose=verbose
    )
    
    # 计算交叉谱
    results = compute_cross_spectrum_for_experiments(
        var1_data, var2_data,
        experiments=experiments,
        mask=mask,
        seg_length=seg_length,
        seg_overlap=seg_overlap,
        symmetry=symmetry,
        memory_monitor=mem_mon,
        verbose=verbose
    )
    
    # 可视化
    fig_axes = None
    if output_dir:
        if plot_params is None:
            plot_params = {}
        
        output_path = os.path.join(
            output_dir,
            f'cross_spectrum_{var1_name}_{var2_name}_{symmetry}.png'
        )
        
        fig, axes = plot_cross_spectrum_panel(
            results,
            experiments=experiments,
            output_path=output_path,
            verbose=verbose,
            **plot_params
        )
        fig_axes = (fig, axes)
    
    mem_mon.print_memory_status("分析完成")
    
    return results, fig_axes
