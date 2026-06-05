# Wave Tools - 热带大气波动分析工具包

> **简洁、清晰、实用的Python工具包，专注于热带大气波动分析**

**作者**: Jianpu | **邮箱**: xianpuji@hhu.edu.cn | **机构**: Hohai University  
**版本**: v1.0.0 | **更新日期**: 2026-02-13

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 📑 目录

- [安装](#-安装)
- [核心功能](#-核心功能)
- [模块结构](#-模块结构)
- [快速开始](#-快速开始)
  - [示例1: Wheeler-Kiladis频谱分析](#示例1-wheeler-kiladis频谱分析)
  - [示例2: 提取Kelvin波 (CCKWFilter)](#示例25-提取kelvin波方法2cckwfilter---推荐)
  - [示例3: 交叉谱分析](#示例3-交叉谱分析新增重构版本)
  - [示例4: 绘制频谱图](#示例4-绘制频谱图)
  - [示例5: 风场图例](#示例5-简化的风场图例新增)
- [完整函数索引](#-完整函数索引)
- [完整工作流程](#完整工作流程)
- [依赖库](#-依赖库)
- [版本更新日志](#-版本更新日志)
- [进阶使用](#-进阶使用)
- [引用](#-引用)
- [联系方式](#-联系方式)

---

## 📦 安装

```bash
# 安装依赖
pip install -r wave_tools/requirements.txt

# 在Python中导入
from wave_tools import *
```

---

## 🎯 核心功能

本工具包提供**热带大气波动分析**的完整解决方案：

### 1. **数据预处理**
- **网格转换**: HEALPix → 经纬度网格 (`convert_icon_to_latlon_grid`)
- **批量转换**: 多实验数据批量处理 (`batch_convert_icon_to_latlon`)
- **背景场移除**: 使用Lanczos滤波器去除低频信号

### 2. **频谱分析**

1. **Matsuno理论模态计算** - 计算赤道浅水方程的理论色散关系
2. **Wheeler-Kiladis频谱分析** - 波数-频率功率谱诊断
3. **交叉谱分析** - 变量间的交叉谱和相干性分析 ⭐ **[新增重构版本，支持内存监控]**
4. **波动滤波与提取** - 时空滤波提取Kelvin、ER、MRG等波动（支持Dask并行）⭐
5. **波动相位分析** - 峰值检测和相位合成分析（支持并行加速）
6. **EOF分析** - 支持SVD和xeofs两种方法的经验正交函数分解
7. **诊断工具** - Radon变换、GMS计算等
8. **专业绘图** - WK频谱图、地图、泰勒图、CCKW包络图、风场图例等
9. **便捷工具** - 数据加载、内存监控、模型筛选等实用功能

---

## 📁 模块结构

```
wave_tools/
├── matsuno.py                  # Matsuno理论模态色散关系
├── spectral.py                 # Wheeler-Kiladis频谱分析
├── cross_spectrum.py           # 交叉谱分析（原始版本）
├── cross_spectrum_analysis.py  # 交叉谱分析（重构版本，推荐）⭐
├── filters.py                  # 波动滤波与信号提取（含CCKWFilter）⭐
├── phase.py                    # 相位分析与峰值检测
├── eof.py                      # EOF/PCA经验正交函数分解
├── diagnostics.py              # 大气动力学和热力学诊断
├── plotting.py                 # 所有绘图功能
├── easyxp.py                   # 简化的风场图例工具 ⭐
└── utils.py                    # 工具函数（HEALPix转换、Radon变换等）
```

---

## 🚀 快速开始

### 示例1: Wheeler-Kiladis频谱分析

```python
from wave_tools.spectral import calculate_wk_spectrum
import xarray as xr

# 加载数据
data = xr.open_dataset('olr.nc')['olr']

# 计算频谱（一行代码）
power_sym, power_asym, background = calculate_wk_spectrum(
    data, 
    window_days=96, 
    skip_days=30,
    output_path='spectrum.nc'
)
```

### 示例2: 提取Kelvin波（方法1：传统WaveFilter）

```python
from wave_tools.filters import WaveFilter

wf = WaveFilter()
kelvin = wf.extract_wave_signal(data, wave_name='kelvin', use_parallel=True)
kelvin.to_netcdf('kelvin_filtered.nc')
```

### 示例2.5: 提取Kelvin波（方法2：CCKWFilter - 推荐⭐）

```python
from wave_tools.filters import CCKWFilter
import xarray as xr

# 读取数据
pr_data = xr.open_dataarray('pr_data.nc')

# 初始化滤波器
wave_filter = CCKWFilter(
    ds=pr_data,
    sel_dict={'time': slice('1980-01-01', '1993-12-31'), 'lat': slice(-15, 15)},
    wave_name='kelvin',  # 或 'er' 提取ER波
    units='mm/day',
    spd=1,              # 每天采样次数，日数据为1
    n_workers=4,        # Dask并行工作进程数
    verbose=True        # 显示详细处理信息
)

# 方式1：逐步执行（适合调试和学习）
wave_filter.load_data()         # 加载数据
wave_filter.detrend_data()      # 去趋势处理
wave_filter.fft_transform()     # FFT变换
wave_filter.apply_filter()      # 应用波动滤波器
wave_filter.inverse_fft()       # 逆FFT变换
filtered_data = wave_filter.create_output()  # 创建输出

# 方式2：一键执行（推荐使用）✨
filtered_data = wave_filter.process()

# 计算标准差并保存
std_data = filtered_data.std(dim='time')
filtered_data.to_netcdf('kelvin_filtered.nc')
std_data.to_netcdf('kelvin_std.nc')

# 查看滤波器信息
print(wave_filter)
```

**CCKWFilter 特性**：
- ✅ 基于 Wheeler & Kiladis (1999) 频率-波数空间滤波方法
- ✅ 使用 Dask 进行大规模数据并行处理，支持内存优化
- ✅ 自动应用浅水波色散关系约束（Kelvin波和ER波）
- ✅ 支持 Kelvin 波和 ER 波，可轻松扩展
- ✅ 提供详细的处理进度和诊断信息
- ✅ 完整的元数据记录（滤波参数、日期等）

### 示例3: 交叉谱分析（新增重构版本）⭐

```python
from wave_tools import analyze_cross_spectrum
import xarray as xr

# 准备海洋掩膜
ocean_mask = xr.open_dataarray('land_mask_2deg.nc') == 0

# 一行代码完成交叉谱分析（从数据加载到可视化）
results, (fig, axes) = analyze_cross_spectrum(
    var1_name='pr',                       # 第一个变量（降水）
    var2_name='rlut',                     # 第二个变量（OLR）
    experiments=['cntl', 'p4k', '4co2'],  # 实验列表
    data_dir='/path/to/data',             # 数据目录
    mask=ocean_mask,                      # 海洋掩膜
    output_dir='./figures/cross_spectrum', # 输出目录
    var1_scale=86400,                     # 变量1缩放因子（例如kg/s → mm/day）
    var2_scale=1,                         # 变量2缩放因子
    latitudes=[-10, 10],                  # 纬度范围
    he_values=[12, 25, 50],               # 等效深度列表（米）
    fmax_values=[0.8, 0.8, 0.8],          # 最大频率列表
    coherence_threshold=0.5,              # 相干性阈值
    window_size=96,                       # 时间窗口大小（天）
    verbose=True                          # 详细输出
)

# 查看结果
for exp, res in results.items():
    print(f"\n实验: {exp}")
    print(f"  Coherence² 显著性阈值: {res['prob_coh2']:.4f}")
    print(f"  频率范围: {res['freq'].min():.3f} - {res['freq'].max():.3f} cpd")
    print(f"  波数范围: {res['wnum'].min():.0f} - {res['wnum'].max():.0f}")
    
    # 访问完整的交叉谱数据
    cross_power = res['cross_spectrum']  # 交叉功率谱
    coherence_sq = res['coherence_sq']   # 相干性平方
    phase_angle = res['phase']           # 相位角
```

**重构版交叉谱分析新特性**：
- ✅ **内存监控**：`MemoryMonitor` 类实时跟踪内存使用，防止内存溢出
- ✅ **批量实验分析**：一次性处理多个气候实验（cntl、p4k、4co2等）
- ✅ **Lazy Loading**：使用Dask分块加载大数据集，支持TB级数据
- ✅ **灵活的掩膜**：支持陆地/海洋掩膜，自动处理缺失值
- ✅ **自动化可视化**：生成多面板对比图，支持Kelvin波包络线叠加
- ✅ **完整的输出**：返回交叉谱、相干性、相位角等所有信息
- ✅ **单位转换**：支持变量缩放（如 kg/s → mm/day）

**便捷函数说明**：

| 函数 | 功能 | 返回 |
|------|------|------|
| `load_netcdf_data(file_path, chunks, verbose)` | 加载单个NetCDF文件（支持lazy loading） | xr.DataArray |
| `load_multiple_experiments(var_name, experiments, data_dir, ...)` | 批量加载多个实验数据 | Dict[str, xr.DataArray] |
| `preprocess_data_with_mask(data, latitudes, mask, scale, ...)` | 数据预处理（掩膜、纬度选择、缩放） | xr.DataArray |
| `compute_cross_spectrum_for_experiments(data_dict, ...)` | 为所有实验计算交叉谱 | Dict[str, Dict] |
| `plot_cross_spectrum_panel(results, var1_name, var2_name, ...)` | 绘制多面板对比图 | (fig, axes) |
| `analyze_cross_spectrum(...)` | **一站式交叉谱分析（推荐）** | (results_dict, (fig, axes)) |

**内存监控示例**：
```python
from wave_tools import MemoryMonitor

monitor = MemoryMonitor()

# 在关键步骤检查内存
monitor.print_memory_status("数据加载后")
# 输出:
# 💾 内存状态 - 数据加载后
#   进程物理内存使用: 2.34 GB
#   系统可用内存: 45.67 GB / 64.00 GB

info = monitor.get_memory_info()
if info['available_gb'] < 10:
    print("⚠️ 警告: 可用内存不足，建议增加数据分块")
```

### 示例4: 绘制频谱图

```python
from wave_tools.plotting import plot_wk_spectrum

plot_wk_spectrum(
    power_sym, power_asym, background,
    wavenumber, frequency,
    add_matsuno_lines=True,
    save_path='wk_spectrum.png'
)
```

### 示例5: 简化的风场图例（新增）⭐

```python
from wave_tools.easyxp import simple_quiver_legend
import matplotlib.pyplot as plt
import numpy as np

# 创建风场数据
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
u = np.random.randn(20, 30) * 5
v = np.random.randn(20, 30) * 5
lon = np.linspace(0, 360, 30)
lat = np.linspace(-15, 15, 20)

# 绘制风场
Q = ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree())

# 添加简洁的图例（仅需一行）
simple_quiver_legend(
    ax=ax,
    quiver=Q,
    reference_value=10.0,        # 参考风速
    unit='m/s',                  # 单位
    legend_location='lower right', # 图例位置
    box_width=0.11,              # 图例框宽度
    box_height=0.15,             # 图例框高度
    font_size=7,                 # 字体大小
    box_facecolor='white',       # 背景色
    box_edgecolor='k',           # 边框色
    zorder=10                    # 图层顺序
)

plt.savefig('wind_plot.png', dpi=300, bbox_inches='tight')
```

**`simple_quiver_legend` 特性**：
- ✅ 简洁紧凑的风场图例设计
- ✅ 支持四个角落位置：'lower right', 'lower left', 'upper right', 'upper left'
- ✅ 完全可自定义：框大小、字体、颜色、边框等
- ✅ 自动处理坐标转换，适配任何投影
- ✅ 一行代码即可添加专业图例

---

## 📚 完整函数索引

### 1. matsuno.py - Matsuno理论模态

#### 主要函数

| 函数 | 功能 | 参数 |
|------|------|------|
| `kelvin_mode(he, latitude, max_wn, n_wn)` | Kelvin波色散关系 | he: 等效深度(m)<br/>max_wn: 最大波数<br/>n_wn: 波数点数 |
| `er_n(he, n, latitude, max_wn, n_wn)` | 赤道Rossby波色散关系 | n: 经向模态数 |
| `mrg_mode(he, latitude, max_wn, n_wn)` | 混合Rossby重力波 | 同上 |
| `eig_n(he, n, latitude, max_wn, n_wn)` | 东传惯性重力波 | 同上 |
| `wig_n(he, n, latitude, max_wn, n_wn)` | 西传惯性重力波 | 同上 |
| `matsuno_modes_wk(he, n, max_wn)` | **批量计算所有模态** | he: 等效深度列表<br/>n: 经向模态数列表 |

**返回**: DataFrame，index为波数，columns为各模态的频率

**应用**: 生成理论框架，用于对比观测/模式的波动功率谱

---

### 2. spectral.py - Wheeler-Kiladis频谱分析

#### 主要类

**`WKSpectralAnalysis`** - 频谱分析主类

| 方法 | 功能 | 参数说明 |
|------|------|----------|
| `load_data(data, data_path, variable, lat_range, time_range)` | 加载数据 | data: xr.DataArray（可选）<br/>data_path: NetCDF路径（可选）<br/>lat_range: 纬度范围，如(-15,15) |
| `preprocess()` | 预处理 | 去趋势、去年循环、对称/反对称分解 |
| `compute_spectrum()` | 计算功率谱 | 波数-频率2D FFT |
| `smooth_background(wave_limit)` | 平滑背景谱 | wave_limit: 平滑的波数限制 |
| `save(output_path)` | 保存到NetCDF | output_path: 输出文件路径 |

**配置类**: `SpectralConfig`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `WINDOW_SIZE_DAYS` | 96 | 时间窗口大小（天） |
| `WINDOW_SKIP_DAYS` | 30 | 窗口跳跃间隔（天） |
| `SAMPLES_PER_DAY` | 1 | 每天采样次数 |

#### 便捷函数

| 函数 | 功能 | 返回 |
|------|------|------|
| `calculate_wk_spectrum(data, window_days, skip_days, output_path)` | **一步完成频谱计算** | power_symmetric, power_antisymmetric, background |

---

### 3. filters.py - 波动滤波

#### 主要类

**`WaveFilter`** - 波动滤波器（谐波分析方法）

| 方法 | 功能 | 参数说明 |
|------|------|----------|
| `extract_wave_signal(ds, wave_name, obs_per_day, use_parallel, n_jobs)` | **提取波动信号** | ds: 输入数据<br/>wave_name: 波动类型<br/>use_parallel: 是否并行 |
| `add_wave_param(wave_name, freq_range, wnum_range, equiv_depth)` | 添加自定义波动 | freq_range: 周期范围(天)<br/>wnum_range: 波数范围<br/>equiv_depth: 等效深度(m) |
| `get_available_waves()` | 获取可用波动列表 | 返回: ['kelvin', 'er', 'mrg', ...] |
| `get_wave_params(wave_name)` | 查看波动参数 | 返回: dict |

**`CCKWFilter`** - 对流耦合波动滤波器（频率-波数空间滤波，使用Dask并行）⭐

| 方法 | 功能 | 参数说明 |
|------|------|----------|
| `__init__(ds, var, sel_dict, wave_name, units, spd, n_workers, verbose)` | 初始化滤波器 | ds: 数据源（路径/Dataset/DataArray）<br/>var: 变量名（Dataset时需要）<br/>sel_dict: 选择字典<br/>wave_name: 'kelvin'或'er'<br/>units: 数据单位<br/>spd: 每天采样次数<br/>n_workers: 并行工作进程数<br/>verbose: 详细输出 |
| `load_data()` | 加载并预处理数据 | 自动应用sel_dict筛选，支持Dataset和DataArray |
| `detrend_data()` | 数据去趋势 | 移除年际变化和线性趋势，应用Tukey窗口 |
| `fft_transform()` | 2D FFT变换 | 计算波数-频率网格 |
| `apply_filter()` | 应用滤波器 | 根据波动类型设置mask，应用色散关系约束 |
| `inverse_fft()` | 逆FFT变换 | 获取滤波后的实空间数据 |
| `create_output()` | 创建输出 | 返回带完整元数据的xr.DataArray |
| `process()` | **一键执行完整流程** | 依次执行所有步骤，返回滤波后数据 |
| `__repr__()` | 打印滤波器信息 | 显示配置参数和数据状态 |
| `print_diagnostic_info(variable, name)` | 打印诊断信息 | 显示变量类型、形状、数据块等信息 |

**CCKWFilter特点**：
- ✅ 基于Wheeler-Kiladis频率-波数滤波方法
- ✅ 使用Dask进行大规模数据并行处理
- ✅ 自动应用浅水波色散关系约束
- ✅ 支持Kelvin波和ER波（可扩展至其他波动）
- ✅ 提供详细的处理信息和诊断输出
- ✅ 完整的元数据记录（滤波参数、处理日期等）

**波动参数对比**：

| 波动类型 | 周期范围(天) | 波数范围 | 等效深度(m) | 色散关系 |
|---------|-------------|---------|------------|---------|
| Kelvin | 3-20 | 2-14 | 8-90 | ω = ck |
| ER (n=1) | 9-72 | -10~-1 | 8-90 | ω(k²+3) + k = 0 |

#### 预定义波动类型

| 波动 | 周期(天) | 波数 | 等效深度(m) | 特征 |
|------|---------|------|------------|------|
| `'kelvin'` | 3-20 | 2-14 | 8-90 | 东传，赤道对称 |
| `'er'` | 9-72 | -10~-1 | 8-90 | 西传，低频 |
| `'mrg'` | 3-10 | -10~-1 | 8-90 | 西传，高频 |
| `'ig'` | 1-14 | 1-5 | 8-90 | 东传惯性重力波 |
| `'mjo'` | 20-100 | 1-5 | NaN | Madden-Julian振荡 |
| `'td'` | 2.5-5 | -20~-6 | NaN | 热带低压型扰动 |

#### WaveFilter vs CCKWFilter 对比

| 特性 | WaveFilter (传统方法) | CCKWFilter (推荐) ⭐ |
|------|---------------------|-------------------|
| **方法** | 谐波分析 + 时空滤波 | 频率-波数空间滤波 (WK99) |
| **并行处理** | joblib (多进程) | Dask (分布式) |
| **内存效率** | 中等 | 高（支持lazy loading） |
| **数据规模** | < 100GB | TB级 |
| **处理速度** | 快 | 非常快 |
| **色散关系** | 简化约束 | 完整浅水波方程 |
| **支持波动** | 6种预定义 + 自定义 | Kelvin、ER（可扩展） |
| **诊断输出** | 基础 | 详细（进度、内存、参数） |
| **适用场景** | 中小型数据集，需要多种波动类型 | 大型数据集，高精度Kelvin/ER波提取 |
| **易用性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**使用建议**:
- 🎯 **Kelvin波/ER波 + 大数据**: 优先使用 `CCKWFilter`
- 🎯 **多种波动类型 + 中小数据**: 使用 `WaveFilter`
- 🎯 **自定义波动参数**: 使用 `WaveFilter.add_wave_param()`
- 🎯 **需要最高精度**: 使用 `CCKWFilter`（完整色散关系）

---

### 4. phase.py - 波动相位分析

#### 主要函数

| 函数 | 功能 | 参数 | 返回 |
|------|------|------|------|
| `optimize_peak_detection(V, kelvin_ref, V_std, Nstd, use_parallel, n_jobs)` | **检测显著峰值及影响范围** | V: 数据数组[time, lon]<br/>kelvin_ref: 参考数据<br/>V_std: 标准差<br/>Nstd: 显著性阈值倍数<br/>use_parallel: 是否并行<br/>n_jobs: 并行核数 | V_peak: 峰值影响范围<br/>local_extrema: 极值标识(1=最小值，-1=最大值) |
| `remove_clm(data, fs, cutoff, order)` | 去除季节循环并低通滤波 | data: xr.DataArray<br/>fs: 采样频率(Hz)<br/>cutoff: 截止频率<br/>order: Butterworth滤波器阶数 | 滤波后的异常场 |
| `find_local_extrema(V)` | 查找二维场的局部极值 | V: [time, lon]数组 | 标识数组(1=局部最小值，-1=局部最大值，NaN=其他) |
| `butter_lowpass_filter(data, cutoff_freq, fs, order)` | Butterworth低通滤波 | data: 1D时间序列<br/>cutoff_freq: 截止频率(Hz)<br/>fs: 采样频率<br/>order: 滤波器阶数 | 滤波后的序列 |
| `find_peak_influence_range(peak_idx, peak_value, zero_idx, peak_indices, V, V_std, Nstd)` | 确定单个峰值的时间影响范围 | peak_idx: 峰值索引<br/>zero_idx: 零点索引数组 | (left_idx, right_idx): 左右边界 |
| `process_single_longitude(ilon, V, local_min_max_id, V_std, Nstd)` | 处理单个经度的峰值检测 | ilon: 经度索引<br/>V: 原始数据<br/>local_min_max_id: 极值标记 | 1D峰值数组[time] |

**核心算法**: 基于零交叉点和局部极值的峰值影响范围识别，支持并行加速

**应用**: 波动相位合成分析、传播速度计算、生命周期统计、事件检测

---

### 5. eof.py - 经验正交函数分析

#### 主要类

**`EOFAnalyzer`** - EOF分析主类

| 参数 | 说明 |
|------|------|
| `method` | 分解方法: 'svd'(numpy实现) 或 'xeofs'(xeofs库) |
| `apply_land_mask` | 是否应用陆地/海洋掩膜 |
| `ocean_only` | True保留海洋点，False保留陆地点 |
| `mask_resolution` | 掩膜分辨率: 'c'(粗~50km), 'l'(低~10km), 'i'(中~2km), 'h'(高~400m), 'f'(全~100m) |

| 方法 | 功能 | 参数说明 |
|------|------|----------|
| `fit(data, n_modes, dim_names)` | 执行EOF分解 | data: xr.DataArray<br/>n_modes: 模态数<br/>dim_names: 维度名称dict |
| `plot_vertical_profiles(n_modes, figsize, save_path)` | 绘制垂直剖面 | n_modes: 绘制模态数 |
| `save_results(filename)` | 保存分析结果 | filename: 输出文件路径 |
| `load_results(filename)` | 加载已保存结果 | filename: 输入文件路径 |

**支持特性**:
- 自动维度检测和转置
- 气候态去除（谐波滤波）
- 陆地/海洋掩膜功能
- 解释方差计算
- 结果序列化存储

**示例**:
```python
analyzer = EOFAnalyzer(method='svd', apply_land_mask=True, ocean_only=True)
results = analyzer.fit(data, n_modes=4)
fig = analyzer.plot_vertical_profiles(n_modes=4)
```

---

### 6. diagnostics.py - 诊断工具

#### 主要函数

| 函数 | 功能 | 参数 | 返回 |
|------|------|------|------|
| `calc_horizontal_GMS(ta, zg, plev, lon, lat, ua, va, hus)` | **水平总体湿稳定度** | ta: 温度(K)<br/>zg: 位势高度(m)<br/>plev: 压力层(Pa)<br/>ua,va: 风场(m/s)<br/>hus: 比湿(kg/kg) | GMS (无量纲) |
| `calc_vertical_GMS(ta, zg, plev, wa, hus, ua, va, lat, lon)` | **垂直总体湿稳定度** | wa: 垂直速度(Pa/s) | GMS (无量纲) |
| `gross_moist_stability(ta_path, zg_path, ua_path, va_path, wa_path, hus_path)` | **完整GMS计算** | 各变量NetCDF文件路径 | h_gms, v_gms |
| `calc_dse(ta, zg, plev)` | 计算干静力能 | ta: 温度<br/>zg: 位势高度 | DSE (J/kg) |
| `compute_dx_dy(lat, lon)` | 计算网格间距 | lat, lon: 纬度经度数组 | dx, dy (米) |
| `vertically_integrated_moist_flux_divergence(hus, ua, va, lat, lon)` | 垂直积分的湿通量散度 | hus: 比湿<br/>ua, va: 风场 | 能量通量散度 (W/m²) |

**物理常数**:
- Cp = 1004.0 J/(kg·K) - 定压比热
- g = 9.81 m/s² - 重力加速度
- L = 2.5×10⁶ J/kg - 蒸发潜热
- T_ref = 300.0 K - 参考温度

**计算公式**:
- DSE = Cp·T + g·Z
- 水平GMS = -T_ref · ∫(V·∇s)dp / ∫L·∇·(rV)dp
- 垂直GMS = -T_ref · ∫(ω·∂s/∂p)dp / ∫L·∇·(rV)dp

**物理意义**: 
- GMS > 0: 负反馈，抑制对流发展
- GMS < 0: 正反馈，促进对流组织化

**注意**: 诊断模块需要额外依赖 `metpy` 和 `geocat-comp`。如未安装，将显示警告信息。

---

### 7. plotting.py - 绘图功能

#### CCKW包络图

| 函数 | 功能 | 参数 |
|------|------|------|
| `get_cckw_envelope_curve(he, fmax)` | 计算CCKW包络曲线坐标 | he: 等效深度列表(m)<br/>fmax: 最大频率列表 |
| `plot_cckw_envelope(he, fmax, save_path, dpi)` | 绘制CCKW包络示意图 | save_path: 保存路径<br/>dpi: 分辨率(默认200) |

#### Wheeler-Kiladis频谱图

| 函数 | 功能 | 关键参数 |
|------|------|----------|
| `plot_wk_spectrum(power_symmetric, power_antisymmetric, background, wavenumber, frequency, ...)` | **绘制WK频谱** | max_wn: 最大波数<br/>max_freq: 最大频率<br/>add_matsuno_lines: 是否添加理论曲线<br/>he: 等效深度列表<br/>cpd_lines: 标注周期线<br/>cmap: 色标<br/>levels: 等值线水平 |
| `set_axis_for_wave(ax, text_size, freq_lines, depth, cpd_lines, max_wn_plot, max_freq_plot)` | 设置波动分析坐标轴 | freq_lines: 是否显示频率线<br/>depth: 是否显示等效深度线<br/>cpd_lines: 周期线列表(天) |

**默认设置**:
- 频谱等值线: [1, 1.2, 1.4, 1.6, 1.8, 2.0]
- 周期标注: [3, 6, 30]天
- 等效深度: [8, 25, 90]米

#### 地图绘图

| 函数 | 功能 | 参数 |
|------|------|------|
| `setup_map_axes(ax, title, box)` | 设置地图坐标轴 | box: [lon_min, lon_max, lat_min, lat_max] |
| `plot_spatial_field(data, ax, cmap, title, box, levels, add_coastlines)` | 绘制空间场 | data: xr.DataArray<br/>cmap: 色标<br/>add_coastlines: 是否添加海岸线 |

**依赖**: cartopy (可选，用于地图投影)

#### 泰勒图

**类**: `TaylorDiagram(refstd, fig, rect, srange, extend)`

| 方法 | 功能 | 参数 |
|------|------|------|
| `add_sample(stddev, corrcoef, *args, **kwargs)` | 添加模式样本点 | stddev: 标准差<br/>corrcoef: 相关系数 |
| `add_contours(levels, **kwargs)` | 添加RMS等值线 | levels: 等值线水平数 |

#### 通用工具

| 函数 | 功能 | 参数 |
|------|------|------|
| `save_figure(fig, filename, folder, fmt, dpi)` | 保存图形 | fmt: 'pdf'/'png'/'jpg'<br/>dpi: 分辨率(默认600) |

**色标支持**: 默认使用cmaps.amwg_blueyellowred，回退到'RdBu_r'

---

### 8. utils.py - 工具函数

#### 数据处理

| 函数 | 功能 | 参数 | 返回 |
|------|------|------|------|
| `load_data(path, var, lat_range)` | 加载NetCDF数据 | path: 文件路径<br/>var: 变量名<br/>lat_range: 纬度范围(默认-15,15) | data, lon, lat |
| `filter_series(series, min_wn, max_wn)` | 波数范围过滤 | series: 时间序列<br/>min_wn, max_wn: 波数边界 | 过滤后的序列 |
| `filter_paths_by_models(paths, model_names, loc, sep, case_sensitive, strict, missing_ok)` | **按模式名过滤文件路径** | paths: 文件路径列表<br/>model_names: 模式名列表<br/>loc: 分隔后的位置<br/>sep: 分隔符(默认'_')<br/>case_sensitive: 大小写敏感<br/>strict: 严格模式 | 匹配的路径列表 |
| `extract_model_name(path, loc, sep)` | 从路径提取模式名 | path: 文件路径<br/>loc: 分隔后的位置(默认1)<br/>sep: 分隔符(默认'_') | 模式名字符串 |

#### HEALPix网格转换

| 函数 | 功能 | 参数 | 返回 |
|------|------|------|------|
| `dataarray_healpix_to_equatorial_latlon(healpix_dataarray, nside, nest, minmax_lat)` | **HEALPix→等经纬度** | nside: HEALPix参数<br/>nest: 是否nested排序<br/>minmax_lat: 纬度范围 | 等经纬度DataArray |
| `get_region_healpix_(zoom, extent, nest)` | 获取区域网格索引 | extent: [lon_min, lon_max, lat_min, lat_max] | 网格索引数组 |

**优化**: 使用Numba @jit加速，速度提升10倍+

**依赖**: healpy (可选)

#### Radon变换

| 函数 | 功能 | 参数 | 返回 |
|------|------|------|------|
| `calc_radon_angle(field, theta_range)` | **计算传播角度** | field: 2D时空数组<br/>theta_range: 角度范围(默认0-180度) | theta, intensity, theta_max |
| `calc_c_from_theta(theta_deg, dx_deg, dt_sec, lat)` | 从角度计算相速度 | theta_deg: 角度(度)<br/>dx_deg: 经度间隔<br/>dt_sec: 时间间隔<br/>lat: 纬度 | 速度(m/s) |
| `plot_radon_energy_distribution(theta, energy, title, color, ax)` | 绘制Radon能量分布 | energy: Radon能量<br/>ax: matplotlib轴(可选) | theta_max, theta_ci (95%置信区间) |

**应用**: 
- 波动传播方向诊断
- 相速度估算
- 东传/西传波动识别

**物理原理**: Radon变换将时空场投影到不同角度，能量最大处对应主导传播方向

#### 色标和字体工具

| 函数 | 功能 | 参数 |
|------|------|------|
| `create_cmap_from_string(color_string)` | 从字符串创建色标 | color_string: 颜色列表(十六进制或RGB) | ListedColormap对象 |
| `set_matplotlib_font(font_dir, arial_font_path)` | 设置Matplotlib字体 | font_dir: 字体目录<br/>arial_font_path: Arial字体路径 | None |
| `save_figure(fig, filename, folder, fmt, dpi)` | 保存图形 | fig: matplotlib图形对象<br/>folder: 保存目录<br/>fmt: 格式<br/>dpi: 分辨率 | None |

#### 其他工具

| 函数 | 功能 |
|------|------|
| `get_curve(he, fmax)` | 获取Kelvin波包络曲线（与plotting.py中的get_cckw_envelope_curve相同） |

---

### 9. easyxp.py - 风场图例工具⭐

#### 主要函数

| 函数 | 功能 | 参数 |
|------|------|------|
| `simple_quiver_legend(ax, quiver, reference_value, unit, legend_location, ...)` | **创建简洁的风场图例** | ax: matplotlib轴对象<br/>quiver: Quiver对象<br/>reference_value: 参考风速值(默认10.0)<br/>unit: 单位字符串(默认'm/s')<br/>legend_location: 位置('lower right', 'lower left', 'upper right', 'upper left')<br/>box_width: 图例框宽度(默认0.11)<br/>box_height: 图例框高度(默认0.15)<br/>text_offset: 文本偏移量(默认0.02)<br/>font_size: 字体大小(默认7)<br/>label_separation: 标签间距(默认0.1)<br/>box_facecolor: 背景色(默认'white')<br/>box_edgecolor: 边框色(默认'k')<br/>box_linewidth: 边框宽度(默认0.8)<br/>zorder: 图层顺序(默认10) | None |

**特性**:
- ✅ 极简API：一行代码添加专业图例
- ✅ 灵活定位：支持四个角落位置
- ✅ 完全可定制：大小、字体、颜色、边框等
- ✅ 自动坐标转换：适配任何matplotlib投影
- ✅ 紧凑设计：不占用过多图像空间

**使用场景**:
- 气象风场图
- 海洋流场图
- 矢量场可视化
- 任何需要quiver图例的场景

---

## 📖 使用示例

### 完整工作流程

```python
import xarray as xr
from wave_tools.spectral import WKSpectralAnalysis, SpectralConfig
from wave_tools.filters import WaveFilter
from wave_tools.phase import optimize_peak_detection, remove_clm
from wave_tools.plotting import plot_wk_spectrum
from wave_tools.diagnostics import gross_moist_stability
from wave_tools.eof import EOFAnalyzer

# ===== 1. 频谱诊断 =====
data = xr.open_dataset('olr.nc')['olr'].sel(lat=slice(-15, 15))

config = SpectralConfig()
config.WINDOW_SIZE_DAYS = 96

analysis = WKSpectralAnalysis(config)
analysis.load_data(data=data)
analysis.preprocess()
analysis.compute_spectrum()
analysis.smooth_background()
analysis.save('spectrum.nc')

# 绘制频谱
plot_wk_spectrum(
    analysis.power_symmetric,
    analysis.power_antisymmetric,
    analysis.background,
    analysis.wavenumber.values,
    analysis.frequency.values,
    add_matsuno_lines=True,
    save_path='wk_spectrum.png'
)

# ===== 2. 波动提取 =====
wf = WaveFilter()
kelvin = wf.extract_wave_signal(data, wave_name='kelvin', use_parallel=True, n_harm=3)
er = wf.extract_wave_signal(data, wave_name='er', use_parallel=True)

# 保存结果
kelvin.to_netcdf('kelvin_filtered.nc')
er.to_netcdf('er_filtered.nc')

# ===== 3. 相位分析 =====
kelvin_lowpass = remove_clm(kelvin, cutoff=1/10, order=4)
kelvin_zonal = kelvin_lowpass.mean('lat')

V_peak, local_extrema = optimize_peak_detection(
    V=kelvin_zonal.values,
    kelvin_ref=kelvin_zonal,
    V_std=kelvin_zonal.std().values,
    Nstd=1.0,
    use_parallel=True
)

# ===== 4. EOF分析 =====
analyzer = EOFAnalyzer(method='svd', apply_land_mask=True, ocean_only=True)
results = analyzer.fit(data, n_modes=4)
fig = analyzer.plot_vertical_profiles(n_modes=4, save_path='eof_profiles.png')
analyzer.save_results('eof_results.pkl')

```

---

## 🔧 依赖库

### 核心依赖
```
numpy >= 1.19.0
xarray >= 0.16.0
scipy >= 1.5.0
matplotlib >= 3.3.0
pandas >= 1.1.0
joblib >= 1.0.0       # 并行计算
dask >= 2021.0.0      # 大规模数据并行处理 ⭐
psutil >= 5.8.0       # 内存监控 ⭐
```

### 可选依赖
```
cmaps                 # NCL气象色标
cartopy              # 地图投影
healpy               # HEALPix球面网格处理
numba                # JIT加速计算
metpy                # 气象计算工具
geocat-comp          # NCAR地球科学计算
xeofs                # EOF分析（可选方法）
global-land-mask     # 陆地/海洋掩膜
scikit-image         # Radon变换
h5netcdf             # HDF5/NetCDF4支持 ⭐
```

**安装全部依赖**:
```bash
# 核心依赖
pip install numpy xarray scipy matplotlib pandas joblib dask psutil

# 可选依赖
pip install cmaps cartopy healpy numba metpy geocat-comp xeofs global-land-mask scikit-image h5netcdf
```

**最小安装**（仅核心功能）:
```bash
pip install numpy xarray scipy matplotlib pandas
```

---

## 🆕 版本更新日志

### v1.0.0 (2026-02-13) - 当前版本

#### 新增功能 ⭐
- **CCKWFilter类**: 基于Dask的高性能波动滤波器
  - 支持Kelvin波和ER波提取
  - 自动应用浅水波色散关系
  - 完整的诊断信息输出
  - 一键式`process()`方法

- **交叉谱分析重构版** (`cross_spectrum_analysis.py`)
  - `MemoryMonitor`: 实时内存监控类
  - `load_netcdf_data`: 支持lazy loading的数据加载
  - `load_multiple_experiments`: 批量加载多实验数据
  - `analyze_cross_spectrum`: 一站式交叉谱分析函数
  - 支持海洋/陆地掩膜和变量缩放

- **easyxp模块**: 风场图例工具
  - `simple_quiver_legend`: 一行代码添加专业风场图例
  - 支持四个角落灵活定位
  - 完全可定制的样式

#### 性能优化 ⚡
- 所有波动滤波器支持Dask并行处理
- HEALPix转换使用Numba加速（10倍+速度提升）
- 内存优化的数据分块加载
- 自动垃圾回收和内存管理

#### 改进 ✨
- 完整的类型注解和文档字符串
- 详细的错误处理和警告信息
- 统一的函数命名规范
- 模块化的代码结构

---

## 📚 进阶使用

### 处理大数据集

```python
from wave_tools import CCKWFilter, MemoryMonitor
import xarray as xr

# 初始化内存监控
monitor = MemoryMonitor()
monitor.print_memory_status("开始前")

# 使用分块加载大数据集
large_data = xr.open_dataarray('large_file.nc', chunks={'time': 5000})

# 使用CCKWFilter处理
wave_filter = CCKWFilter(
    ds=large_data,
    sel_dict={'time': slice('1980', '2020'), 'lat': slice(-15, 15)},
    wave_name='kelvin',
    spd=1,
    n_workers=8,  # 增加并行进程数
    verbose=True
)

filtered_data = wave_filter.process()

# 检查内存使用
monitor.print_memory_status("处理完成后")

# 如果内存紧张，直接计算并保存统计量
std = filtered_data.std(dim='time').compute()
std.to_netcdf('kelvin_std.nc')
```

### 批量处理多个变量

```python
from wave_tools import CCKWFilter
import xarray as xr
from pathlib import Path

# 定义变量列表
variables = ['pr', 'rlut', 'ua', 'va', 'wa']
experiments = ['cntl', 'p4k', '4co2']
data_dir = Path('/path/to/data')
output_dir = Path('./filtered_data')
output_dir.mkdir(exist_ok=True)

# 批量处理
for exp in experiments:
    for var in variables:
        print(f"\n处理 {exp} - {var}")
        
        # 构建文件路径
        input_file = data_dir / f"{var}_{exp}.nc"
        output_file = output_dir / f"{var}_{exp}_kelvin.nc"
        
        if not input_file.exists():
            print(f"  跳过: 文件不存在")
            continue
        
        # 执行滤波
        try:
            wave_filter = CCKWFilter(
                ds=str(input_file),
                var=var,
                sel_dict={'time': slice('1980', '2020'), 'lat': slice(-15, 15)},
                wave_name='kelvin',
                spd=1,
                n_workers=4,
                verbose=False  # 批量处理时减少输出
            )
            
            filtered = wave_filter.process()
            filtered.to_netcdf(output_file)
            print(f"  ✅ 完成: {output_file.name}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue

print("\n所有处理完成！")
```

### 自定义波动参数

```python
from wave_tools.filters import WaveFilter

# 创建滤波器实例
wf = WaveFilter()

# 添加自定义波动类型
wf.add_wave_param(
    wave_name='my_custom_wave',
    freq_range=(5, 15),      # 周期5-15天
    wnum_range=(3, 10),      # 波数3-10
    equiv_depth=(20, 50)     # 等效深度20-50米
)

# 查看所有可用波动
print("可用波动:", wf.get_available_waves())

# 查看特定波动参数
params = wf.get_wave_params('my_custom_wave')
print("自定义波动参数:", params)

# 使用自定义波动滤波
filtered = wf.extract_wave_signal(
    data, 
    wave_name='my_custom_wave',
    use_parallel=True
)
```

---

## 💡 使用技巧

### 1. 性能优化建议

```python
# ✅ 推荐：使用Dask分块加载大文件
data = xr.open_dataarray('large_file.nc', chunks={'time': 5000})

# ❌ 避免：一次性加载所有数据到内存
# data = xr.open_dataarray('large_file.nc').load()

# ✅ 推荐：先选择再计算
filtered = wave_filter.process()
std = filtered.sel(lat=slice(-10, 10)).std(dim='time').compute()

# ❌ 避免：全部计算后再选择
# std = filtered.std(dim='time').compute()
# std_subset = std.sel(lat=slice(-10, 10))
```

### 2. 内存管理

```python
from wave_tools import MemoryMonitor
import gc

monitor = MemoryMonitor()

# 处理多个文件时定期清理
for i, file in enumerate(files):
    # 处理数据
    result = process_data(file)
    result.to_netcdf(f'output_{i}.nc')
    
    # 每5个文件检查一次内存
    if i % 5 == 0:
        monitor.print_memory_status(f"处理第{i}个文件后")
        gc.collect()  # 强制垃圾回收
```

### 3. 波动滤波参数选择

```python
# Kelvin波提取 - 不同研究目的的参数建议

# 🌊 对流耦合Kelvin波 (CCKW)
wave_filter = CCKWFilter(
    ds=data,
    wave_name='kelvin',
    sel_dict={'lat': slice(-15, 15)},  # 赤道附近
    spd=1  # 日数据
)

# 🌍 赤道Kelvin波 (全球尺度)
wave_filter = CCKWFilter(
    ds=data,
    wave_name='kelvin',
    sel_dict={'lat': slice(-20, 20)},  # 稍宽纬度带
    spd=4  # 6小时数据
)

# 🔬 研究特定周期的波动
wf = WaveFilter()
wf.add_wave_param(
    wave_name='kelvin_short',
    freq_range=(3, 10),    # 仅提取3-10天周期
    wnum_range=(5, 14),    # 较高波数
    equiv_depth=(25, 90)   # 较深等效深度
)
```

---

## ❓ 常见问题 (FAQ)

### Q1: 如何选择合适的滤波方法？

**A**: 
- **数据量 < 50GB，需要多种波动**: 使用 `WaveFilter`
- **数据量 > 50GB，仅需Kelvin/ER波**: 使用 `CCKWFilter`
- **需要最高精度的Kelvin波**: 使用 `CCKWFilter`（完整色散关系）

### Q2: 内存不足怎么办？

**A**:
```python
# 方法1: 增加数据分块大小
data = xr.open_dataarray('file.nc', chunks={'time': 1000})  # 减小分块

# 方法2: 分段处理
for year in range(1980, 2020):
    subset = data.sel(time=str(year))
    result = wave_filter.process()
    result.to_netcdf(f'output_{year}.nc')

# 方法3: 减少并行进程数
wave_filter = CCKWFilter(..., n_workers=2)  # 默认为4
```

### Q3: 如何验证滤波结果的正确性？

**A**:
```python
# 1. 检查功率谱
from wave_tools import calculate_wk_spectrum, plot_wk_spectrum

power_sym, power_asym, bg = calculate_wk_spectrum(filtered_data)
plot_wk_spectrum(power_sym, power_asym, bg, wavenumber, frequency,
                 add_matsuno_lines=True, save_path='filtered_spectrum.png')

# 2. 检查滤波后的方差比例
original_var = data.var(dim='time')
filtered_var = filtered_data.var(dim='time')
variance_ratio = (filtered_var / original_var).mean()
print(f"滤波后方差占比: {variance_ratio.values*100:.2f}%")
# Kelvin波通常占总方差的5-15%
```

### Q4: CCKWFilter支持哪些波动类型？

**A**: 目前支持：
- `'kelvin'`: Kelvin波（东传）
- `'er'`: 赤道Rossby波（西传）

如需其他波动类型（MRG、惯性重力波等），请使用 `WaveFilter`。

### Q5: 为什么交叉谱分析结果与NCL不完全一致？

**A**: 可能的原因：
1. **窗口函数不同**: 检查 `window_size` 参数
2. **平滑方法**: Python和NCL的平滑算法可能略有差异
3. **数据预处理**: 确保去趋势和年循环移除方法一致

### Q6: 如何处理非等经纬度网格数据？

**A**:
```python
# 使用HEALPix转换工具
from wave_tools.utils import dataarray_healpix_to_equatorial_latlon

regular_grid_data = dataarray_healpix_to_equatorial_latlon(
    healpix_data,
    nside=64,
    nest=True,
    minmax_lat=(-30, 30)
)
```

---

## 📝 引用

如果本工具包对您的研究有帮助，请引用：

```
Jianpu. (2026). Wave Tools: A Python package for tropical atmospheric wave analysis. 
Hohai University. Email: xianpuji@hhu.edu.cn
GitHub: https://github.com/[your-repo]/wave_tools
```

**相关文献**:
- Wheeler, M., & Kiladis, G. N. (1999). Convectively coupled equatorial waves: Analysis of clouds and temperature in the wavenumber–frequency domain. *Journal of the Atmospheric Sciences*, 56(3), 374-399.
- Kiladis, G. N., Wheeler, M. C., Haertel, P. T., Straub, K. H., & Roundy, P. E. (2009). Convectively coupled equatorial waves. *Reviews of Geophysics*, 47(2).
- Matsuno, T. (1966). Quasi-geostrophic motions in the equatorial area. *Journal of the Meteorological Society of Japan*, 44(1), 25-43.

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出功能请求！

### 报告问题
- 请在GitHub Issues中详细描述问题
- 提供最小可复现示例
- 说明您的环境（Python版本、操作系统等）

### 贡献代码
1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范
- 遵循PEP 8编码规范
- 添加完整的文档字符串
- 包含类型注解
- 添加测试用例

---

## 📞 联系方式

- **作者**: Jianpu
- **邮箱**: xianpuji@hhu.edu.cn
- **机构**: Hohai University (河海大学)
- **研究方向**: 热带大气波动、对流耦合过程、气候变化

---

## 📄 许可证

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

感谢以下项目和资源：
- [NCL (NCAR Command Language)](https://www.ncl.ucar.edu/) - 提供了波动滤波的参考实现
- [xarray](http://xarray.pydata.org/) - 强大的多维数组处理库
- [Dask](https://dask.org/) - 并行计算框架
- [cmaps](https://github.com/hhuangwx/cmaps) - NCL色标的Python实现
- 所有为本工具包提供反馈和建议的用户

---

## 📖 相关资源

### 教程和文档
- [Wheeler-Kiladis频谱分析教程](https://www.ncl.ucar.edu/Applications/Scripts/wheeler_kiladis_1.ncl)
- [Kelvin波动力学](https://glossary.ametsoc.org/wiki/Kelvin_wave)
- [EOF分析原理](https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/empirical-orthogonal-function-eof-analysis-and-rotated-eof-analysis)

### 相关工具
- [pywavelets](https://pywavelets.readthedocs.io/) - 小波变换
- [xeofs](https://github.com/nicrie/xeofs) - EOF分析专用库
- [metpy](https://unidata.github.io/MetPy/) - 气象数据处理

---

**最后更新**: 2026-02-13  
**版本**: v1.0.0

---

## 📖 附加文档

- **[CHANGELOG.md](CHANGELOG.md)** - 详细的版本更新日志
- **[QUICKREF.md](QUICKREF.md)** - 快速参考卡片（一页速查）
- **[wave_tools/requirements.txt](wave_tools/requirements.txt)** - 依赖列表

---

<div align="center">

**⭐ 如果这个工具包对您有帮助，请给个 Star！⭐**

Made with ❤️ by Jianpu @ Hohai University

</div>

