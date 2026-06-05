# Wave Tools - 快速参考卡片 (Quick Reference)

## 🚀 一分钟快速上手

### 1. 导入所有功能
```python
from wave_tools import *
import xarray as xr
```

### 2. 加载数据
```python
# 方法1: 直接加载
data = xr.open_dataarray('data.nc')

# 方法2: Lazy loading（推荐大文件）
data = xr.open_dataarray('data.nc', chunks={'time': 5000})

# 方法3: 使用工具包函数
data = load_netcdf_data('data.nc', chunks={'time': 5000}, verbose=True)
```

---

## 📊 常用功能速查

### A. Wheeler-Kiladis 频谱分析

```python
# 一行代码计算频谱
power_sym, power_asym, bg = calculate_wk_spectrum(
    data, 
    window_days=96, 
    skip_days=30
)

# 绘制频谱
plot_wk_spectrum(
    power_sym, power_asym, bg,
    wavenumber, frequency,
    add_matsuno_lines=True,
    save_path='spectrum.png'
)
```

### B. Kelvin 波提取

#### 方法1: CCKWFilter（推荐 ⭐）
```python
from wave_tools import CCKWFilter

wave_filter = CCKWFilter(
    ds='data.nc',                    # 或 xr.DataArray
    wave_name='kelvin',              # 'kelvin' 或 'er'
    sel_dict={'lat': slice(-15, 15), 'time': slice('1980', '2020')},
    spd=1,                           # 每天采样次数
    n_workers=4                      # 并行进程数
)

# 一键执行
kelvin = wave_filter.process()
```

#### 方法2: WaveFilter（多种波动）
```python
from wave_tools import WaveFilter

wf = WaveFilter()
kelvin = wf.extract_wave_signal(data, wave_name='kelvin', use_parallel=True)
er = wf.extract_wave_signal(data, wave_name='er')
mjo = wf.extract_wave_signal(data, wave_name='mjo')
```

### C. 交叉谱分析

```python
# 一站式分析（推荐 ⭐）
results, (fig, axes) = analyze_cross_spectrum(
    var1_name='pr',
    var2_name='rlut',
    experiments=['cntl', 'p4k'],
    data_dir='/data',
    mask=ocean_mask,
    output_dir='./figures'
)

# 查看结果
print(results['cntl']['prob_coh2'])  # 相干性阈值
```

### D. EOF 分析

```python
from wave_tools import EOFAnalyzer

analyzer = EOFAnalyzer(
    method='svd',              # 'svd' 或 'xeofs'
    apply_land_mask=True,
    ocean_only=True
)

results = analyzer.fit(data, n_modes=4)
fig = analyzer.plot_vertical_profiles(n_modes=4)
```

### E. 相位分析

```python
from wave_tools import optimize_peak_detection

# 检测波动峰值
peaks, extrema = optimize_peak_detection(
    V=data.values,
    kelvin_ref=data,
    V_std=data.std().values,
    Nstd=1.0,
    use_parallel=True
)
```

### F. 内存监控

```python
from wave_tools import MemoryMonitor

monitor = MemoryMonitor()
monitor.print_memory_status("检查点1")

# 获取内存信息
info = monitor.get_memory_info()
print(f"可用内存: {info['available_gb']:.2f} GB")
```

### G. 风场图例

```python
from wave_tools.easyxp import simple_quiver_legend

# 绘制风场
Q = ax.quiver(lon, lat, u, v)

# 添加图例（一行代码）
simple_quiver_legend(
    ax, Q, 
    reference_value=10.0, 
    unit='m/s',
    legend_location='lower right'
)
```

---

## 🎨 绘图速查

### 频谱图
```python
plot_wk_spectrum(power_sym, power_asym, bg, wnum, freq,
                 add_matsuno_lines=True, he=[12, 25, 50])
```

### 空间场
```python
plot_spatial_field(data, ax, cmap='RdBu_r', 
                   title='Title', box=[-180, 180, -30, 30])
```

### CCKW 包络
```python
plot_cckw_envelope(he=[12, 25, 50], fmax=[0.8, 0.8, 0.8])
```

### Taylor 图
```python
taylor = TaylorDiagram(refstd=1.0)
taylor.add_sample(stddev, corrcoef, label='Model1')
taylor.add_contours(levels=5)
```

---

## 🔧 实用工具速查

### 数据加载
```python
data, lon, lat = load_data('file.nc', var='pr', lat_range=(-15, 15))
```

### 模型文件筛选
```python
filtered = filter_paths_by_models(
    paths=file_list,
    model_names=['CESM', 'GFDL'],
    loc=1, sep='_'
)
```

### Radon 变换
```python
theta, intensity, theta_max = calc_radon_angle(field)
phase_speed = calc_c_from_theta(theta_max, dx_deg=2.5, dt_sec=86400, lat=0)
```

### HEALPix 转换
```python
regular_data = dataarray_healpix_to_equatorial_latlon(
    healpix_data, nside=64, nest=True
)
```

---

## 📦 波动参数速查表

| 波动 | 周期(天) | 波数 | 等效深度(m) | 传播方向 |
|------|---------|------|------------|---------|
| Kelvin | 3-20 | 2-14 | 8-90 | 东传 |
| ER | 9-72 | -10~-1 | 8-90 | 西传 |
| MRG | 3-10 | -10~-1 | 8-90 | 西传 |
| IG | 1-14 | 1-5 | 8-90 | 东传 |
| MJO | 20-100 | 1-5 | - | 东传 |
| TD | 2.5-5 | -20~-6 | - | 西传 |

---

## ⚙️ 性能优化速查

### 内存优化
```python
# 使用分块
data = xr.open_dataarray('file.nc', chunks={'time': 5000})

# 延迟计算
result = data.mean('time')  # 不立即计算
result.compute()            # 执行计算

# 垃圾回收
import gc
gc.collect()
```

### 并行处理
```python
# CCKWFilter
wave_filter = CCKWFilter(..., n_workers=8)

# WaveFilter
wf.extract_wave_signal(..., use_parallel=True, n_jobs=8)

# optimize_peak_detection
peaks, _ = optimize_peak_detection(..., use_parallel=True, n_jobs=8)
```

---

## 🆘 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| 内存不足 | 使用 `chunks={'time': 1000}` 减小分块 |
| 处理太慢 | 增加 `n_workers` 或 `n_jobs` |
| 结果不正确 | 检查滤波参数和数据预处理 |
| 导入失败 | `pip install [缺失的包]` |
| 精度问题 | 使用 `CCKWFilter`（完整色散关系） |

---

## 📞 获取帮助

```python
# 查看函数文档
help(CCKWFilter)
help(analyze_cross_spectrum)

# 打印工具包信息
print_info()

# 查看可用波动
list_available_waves()

# 查看版本
get_version()
```

---

**更新**: 2026-02-13 | **版本**: v1.0.0 | **作者**: Jianpu (xianpuji@hhu.edu.cn)
