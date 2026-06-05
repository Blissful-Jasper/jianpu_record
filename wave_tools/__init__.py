"""
Wave Tools - 热带大气波动分析工具包
====================================

简洁、清晰、实用的Python工具包，专注于热带大气波动分析

主要功能:
---------
1. Matsuno理论模态计算
2. Wheeler-Kiladis频谱分析
3. 波动滤波与提取
4. 波动相位分析
5. 诊断工具（GMS等）
6. 专业科学绘图

作者: Jianpu
邮箱: xianpuji@hhu.edu.cn
机构: Hohai University
"""

__version__ = "1.0.0"
__author__ = "Jianpu"
__email__ = "xianpuji@hhu.edu.cn"

# ===== 核心分析模块 =====

# Matsuno理论
from .matsuno import (
    kelvin_mode,
    er_n,
    mrg_mode,
    eig_n,
    wig_n,
    matsuno_modes_wk,
    matsuno_dataframe,
)

# 频谱分析
from .spectral import (
    WKSpectralAnalysis,
    SpectralConfig,
    calculate_wk_spectrum,
)

# 交叉谱分析
from .cross_spectrum import (
    CrossSpectrumConfig,
    calculate_cross_spectrum,
    quick_cross_spectrum,
    remove_annual_cycle,
    nan_to_value_by_interp_3D,
)

# 交叉谱分析工具（重构版本，更通用）
from .cross_spectrum_analysis import (
    MemoryMonitor,
    load_netcdf_data,
    load_multiple_experiments,
    preprocess_data_with_mask,
    compute_cross_spectrum_for_experiments,
    plot_cross_spectrum_panel,
    analyze_cross_spectrum,
)

# 波动滤波
from .filters import WaveFilter, CCKWFilter

# EOF分析
from .eof import (
    EOFAnalyzer,
    quick_eof_analysis,
    eof_svd,
    eof_xeofs,
)

# 相位分析
from .phase import (
    optimize_peak_detection,
    remove_clm,
    find_local_extrema,
    butter_lowpass_filter,
    meridional_projection,
    calculate_kelvin_phase,
    phase_composite,
    lag_composite,
    save_composite_to_netcdf,
    composite_kw_phase,
)

# ===== 诊断工具 =====

# 注意: diagnostics 模块需要额外的依赖 (metpy, geocat)
# 如果需要使用 GMS 相关功能，请确保安装这些依赖
_diagnostics_available = False
try:
    from .diagnostics import (
        calc_horizontal_GMS,
        calc_vertical_GMS,
        gross_moist_stability,
    )
    _diagnostics_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"诊断模块导入失败: {e}. 如需使用 GMS 功能，请安装 metpy 和 geocat-comp", ImportWarning)
    # 提供占位函数
    def _diagnostics_unavailable(*args, **kwargs):
        raise ImportError("诊断模块不可用。请安装依赖: pip install metpy geocat-comp")
    
    calc_horizontal_GMS = _diagnostics_unavailable
    calc_vertical_GMS = _diagnostics_unavailable
    gross_moist_stability = _diagnostics_unavailable

# ===== 绘图功能 =====

from .plotting import (
    # CCKW包络
    get_cckw_envelope_curve,
    plot_cckw_envelope,
    
    # WK频谱
    plot_wk_spectrum,
    
    # 地图
    setup_map_axes,
    plot_spatial_field,
    
    # 泰勒图
    TaylorDiagram,
    
    # 工具
    save_figure,
)

# ===== 工具函数 =====

from .utils import (
    # 数据处理
    load_data,
    filter_series,
    filter_paths_by_models,
    extract_model_name,
    
    # HEALPix
    dataarray_healpix_to_equatorial_latlon,
    get_region_healpix_,
    
    # Radon
    calc_radon_angle,
    calc_c_from_theta,
    plot_radon_energy_distribution,
    
    # 其他
    create_cmap_from_string,
)


# ===== 导出列表 =====

__all__ = [
    # 核心类
    'WKSpectralAnalysis',
    'SpectralConfig',
    'CrossSpectrumConfig',
    'WaveFilter',
    'CCKWFilter',
    'TaylorDiagram',
    'EOFAnalyzer',
    'MemoryMonitor',
    
    # 频谱分析
    'calculate_wk_spectrum',
    'calculate_cross_spectrum',
    'quick_cross_spectrum',
    
    # 交叉谱分析（重构版本）
    'load_netcdf_data',
    'load_multiple_experiments',
    'preprocess_data_with_mask',
    'compute_cross_spectrum_for_experiments',
    'plot_cross_spectrum_panel',
    'analyze_cross_spectrum',  # 一站式函数
    
    # Matsuno模态
    'kelvin_mode',
    'er_n',
    'mrg_mode',
    'matsuno_modes_wk',
    
    # EOF分析
    'quick_eof_analysis',
    'eof_svd',
    'eof_xeofs',
    
    # 波动滤波（通过WaveFilter类）
    
    # 相位分析
    'optimize_peak_detection',
    'remove_clm',
    'meridional_projection',
    'calculate_kelvin_phase',
    'phase_composite',
    'lag_composite',
    'save_composite_to_netcdf',
    'composite_kw_phase',
    
    # 诊断
    'gross_moist_stability',
    'calc_horizontal_GMS',
    'calc_vertical_GMS',
    
    # 绘图
    'plot_wk_spectrum',
    'plot_cckw_envelope',
    'plot_spatial_field',
    'save_figure',
    'easyxp'
    
    # 工具
    'load_data',
    'filter_series',
    'calc_radon_angle',
    
    # 信息函数
    'get_version',
    'list_available_waves',
    'print_info',
]


# ===== 模块信息 =====

def get_version():
    """返回版本号"""
    return __version__

def list_available_waves():
    """列出可用的波动类型"""
    return ['kelvin', 'er', 'mrg', 'ig', 'mjo', 'td']

def print_info():
    """打印工具包信息"""
    info = f"""
    Wave Tools v{__version__}
    ========================
    热带大气波动分析工具包
    
    作者: {__author__}
    邮箱: {__email__}
    
    主要模块:
    - matsuno.py                 Matsuno理论模态
    - spectral.py                频谱分析
    - cross_spectrum.py          交叉谱分析（原始版本）
    - cross_spectrum_analysis.py 交叉谱分析（重构版本，推荐）
    - filters.py                 波动滤波
    - phase.py                   相位分析
    - eof.py                     EOF垂直模态分解
    - diagnostics.py             诊断工具
    - plotting.py                绘图功能
    - utils.py                   工具函数
    
    快速开始:
    >>> from wave_tools import *
    >>> help(WKSpectralAnalysis)
    >>> help(analyze_cross_spectrum)  # 新的一站式交叉谱分析
    >>> help(WaveFilter)
    >>> help(EOFAnalyzer)
    """
    print(info)
