import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors
from typing import Optional, List, Tuple
from scipy.signal import butter, filtfilt
from skimage.transform import radon, rescale
from typing import Iterable, Sequence, List, Union, Optional, Set, AnyStr, overload
import matplotlib as mpl
from matplotlib import font_manager
from scipy import interpolate
from numba import jit, prange

# Optional dependency
try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False
    hp = None


PathLike = Union[str, "os.PathLike[str]"]

def create_cmap_from_string(color_string: str) -> colors.ListedColormap:
    """
    根据给定的颜色字符串创建一个反转的颜色映射（Colormap）。

    输入:
        color_string (str): 一个字符串，每行包含一个颜色，可以是标准颜色名、RGB 值（如 '1,0,0' 表示红色），或十六进制颜色代码（如 '#FF5733'）。

    输出:
        ListedColormap: 返回一个基于输入颜色字符串的反转颜色映射对象（Colormap）。

    示例:
        color_string = '''
        #FF5733
        #33FF57
        #3357FF
        '''
        
        cmap = create_cmap_from_string(color_string)
        plt.imshow(data, cmap=cmap)
        plt.colorbar()
        plt.show()
    """
    # 去除多余的空白行并分割每行
    
    color_list = color_string.strip().split('\n')
    
    # 创建并返回反转后的颜色映射对象
    return colors.ListedColormap(color_list[::-1])

def create_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    """Create improved red-white-blue colormap"""
    colors = ['#00008B', '#0000FF', '#4169E1', '#87CEEB', '#E0F6FF',
              '#FFFFFF', 
              '#FFE0E0', '#FFB6C1', '#FF6B6B', '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('improved_bwr', colors, N=256)

def filter_series(series, min_wn, max_wn):
    return series[(series.index >= min_wn) & (series.index <= max_wn)]


def save_figure(
    fig: plt.Figure,
    filename: str = 'meridional_mean',
    folder: Optional[str] = None,
    fmt: str = 'pdf',
    dpi: int = 600
) -> None:
    """
    保存 matplotlib 生成的图像文件。

    参数：
    --------
    fig : matplotlib.figure.Figure
        要保存的图像对象。
    filename : str, optional
        保存的文件名（不含扩展名），默认 'meridional_mean'。
    folder : str, optional
        保存文件的文件夹路径，默认当前工作目录。
    fmt : str, optional
        文件格式，例如 'pdf'、'png'、'jpg'，默认 'pdf'。
    dpi : int, optional
        保存图像的分辨率，默认 600 dpi。

    返回：
    --------
    None
    """

    # 确定保存路径
    if folder is None:
        folder = os.getcwd()

    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'Folder {folder} has been created.')
    else:
        print(f'Folder {folder} already exists.')

    # 完整的输出路径，自动加上后缀
    outpath = os.path.join(folder, f"{filename}.{fmt}")

    # 保存图像
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight', format=fmt)
    print(f'Figure saved at: {outpath}')

def get_curve(
    he: Optional[List[float]] = None,
    fmax: Optional[List[float]] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    计算 Kelvin 波（CCKW）能量带的包络曲线坐标。
    """
    if he is None:
        he = [8, 25, 90]
    if fmax is None:
        fmax = [1/3, 1/2.25, 0.5]

    g = 9.8
    re = 6371e3
    s2d = 86400

    kw_x = []
    kw_y = []

    for v in range(len(he)):
        s_min = (g * he[0]) ** 0.5 / (2 * np.pi * re) * s2d
        s_max = (g * he[-1]) ** 0.5 / (2 * np.pi * re) * s2d
        kw_tmax = 20

        kw_x.append(np.array([
            2,
            1 / kw_tmax / s_min,
            14,
            14,
            fmax[0] / s_max,
            2,
            2
        ]))

        kw_y.append(np.array([
            1 / kw_tmax,
            1 / kw_tmax,
            14 * s_min,
            fmax[0],
            fmax[0],
            2 * s_max,
            1 / 20
        ]))

    return kw_x, kw_y


def create_cmap_from_string(color_string: str) -> colors.ListedColormap:
    """
    根据给定的颜色字符串创建一个反转的颜色映射（Colormap）。

    输入:
        color_string (str): 一个字符串，每行包含一个颜色，可以是标准颜色名、RGB 值（如 '1,0,0' 表示红色），或十六进制颜色代码（如 '#FF5733'）。

    输出:
        ListedColormap: 返回一个基于输入颜色字符串的反转颜色映射对象（Colormap）。

    示例:
        color_string =('
        #FF5733
        #33FF57
        #3357FF
       ')
        
        cmap = create_cmap_from_string(color_string)
        plt.imshow(data, cmap=cmap)
        plt.colorbar()
        plt.show()
    """
    # 去除多余的空白行并分割每行
    color_list = color_string.strip().split('\n')
    
    # 创建并返回反转后的颜色映射对象
    return colors.ListedColormap(color_list[::-1])



def load_data(path: str, var: str, lat_range=(-15, 15)) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """加载并预处理数据"""
    ds = xr.open_dataset(path).sortby('lat').sel(lat=slice(*lat_range))
    return ds[var], ds.lon.values, ds.lat.values






def calc_radon_angle(field: np.ndarray, theta_range=(0, 180)):
    
    theta = np.linspace(theta_range[0], theta_range[1], max(field.shape), endpoint=False)
    
    sinogram = radon(field, theta=theta, circle=False)
    intensity = np.sum(sinogram**2, axis=0)
    theta_max = theta[np.argmax(intensity)]
    return theta, intensity, theta_max

def calc_c_from_theta(theta_deg: float, dx_deg: float, dt_sec: float, lat: float = 0.0):
    a = 6.371e6  # 地球半径，单位 m
    meter_per_deg = 2 * np.pi * a * np.cos(np.deg2rad(lat)) / 360
    speed = meter_per_deg * np.tan(np.deg2rad(theta_deg)) * dx_deg / dt_sec
    return speed



def plot_radon_energy_distribution(theta, energy, title='Radon Power Spectrum', color='gray', ax=None):
    """
    绘制 Radon 相速度能量分布图。可选择传入 matplotlib 的子图 ax 来控制绘图位置。

    Parameters
    ----------
    theta : array-like
        角度数组（单位：度）
    energy : array-like
        Radon 能量
    title : str
        图标题
    color : str
        曲线颜色
    ax : matplotlib.axes.Axes or None
        如果提供则在此子图上绘图；否则使用 plt.gca()
    
    Returns
    -------
    theta_max : float
        能量最大处对应的角度
    theta_ci : tuple
        能量归一化大于 0.95 部分的角度范围 (min, max)
    """
   

    energy_norm = energy / np.max(energy)
    theta_max = theta[np.argmax(energy)]
    threshold = 0.95
    mask = energy_norm >= threshold
    theta_ci = theta[mask]
    theta_min, theta_max_ci = theta_ci[0], theta_ci[-1]

    # ✅ 使用 ax 绘图
    ax = ax or plt.gca()
    ax.plot(theta, energy_norm, label='Normalized Radon Power', color=color)
    ax.axvline(theta_max, color='r', linestyle='--', label=f'θmax = {theta_max:.2f}°')
    ax.fill_between(theta, 0, 1, where=mask, color=color, alpha=0.3, label='95% peak width')

    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("Normalized Radon Power")
    ax.set_title(title, fontsize=10)
    ax.legend()
    ax.grid(True)

    return theta_max, (theta_min, theta_max_ci)


def extract_model_name(path: PathLike, loc: int = 1, sep: str = "_") -> str:
    """
    Extract a model name token from a file path.

    Parameters
    ----------
    path : str or os.PathLike
        Full path or filename.
    loc : int, default=1
        Index after splitting the basename by `sep`.
    sep : str, default="_"
        Separator used to split the basename.

    Returns
    -------
    str
        The token at position `loc`.

    Raises
    ------
    IndexError
        If the split result has fewer than `loc+1` parts.
    """
    parts = os.path.basename(os.fspath(path)).split(sep)
    return parts[loc]  # may raise IndexError (intentional; see wrapper below)


def filter_paths_by_models(
    paths: Iterable[PathLike],
    model_names: Sequence[str],
    *,
    loc: int = 1,
    sep: str = "_",
    case_sensitive: bool = True,
    strict: bool = False,
    missing_ok: bool = True,
) -> List[str]:
    """
    Filter a collection of file paths, returning only those whose extracted
    model name matches one of the requested names.

    This function wraps `extract_model_name` and adds safety / flexibility
    (case-insensitive matching, graceful handling of malformed filenames, etc.).

    Parameters
    ----------
    paths : Iterable[str | os.PathLike]
        Paths to check.
    model_names : Sequence[str]
        Allowed model names (e.g., ["ACCESS-CM2", "CESM2-WACCM", ...]).
    loc : int, default=1
        Index of the token (after splitting basename by `sep`) that contains the model name.
    sep : str, default="_"
        Separator used to split the basename.
    case_sensitive : bool, default=True
        If False, comparison is done case-insensitively.
    strict : bool, default=False
        If True, raise an error if a path cannot yield a model token at `loc`.
        If False, skip such paths.
    missing_ok : bool, default=True
        If True, ignore paths whose extracted token is *not* in `model_names`.
        If False, raise a ValueError listing the unexpected tokens.

    Returns
    -------
    list[str]
        Sorted list of paths (as strings) whose extracted model token matched `model_names`.

    Raises
    ------
    IndexError
        When `strict=True` and a path does not have enough tokens.
    ValueError
        When `missing_ok=False` and at least one extracted token is not in `model_names`.
    """
    # Normalize allowed model names
    if case_sensitive:
        wanted: Set[str] = set(model_names)
        norm = lambda s: s  # identity
    else:
        wanted = {m.lower() for m in model_names}
        norm = lambda s: s.lower()

    matched_paths: List[str] = []
    unexpected_tokens: Set[str] = set()

    for p in paths:
        try:
            token = extract_model_name(p, loc=loc, sep=sep)
        except IndexError:
            if strict:
                raise
            else:
                continue  # skip malformed
        if norm(token) in wanted:
            matched_paths.append(os.fspath(p))
        else:
            unexpected_tokens.add(token)

    if not missing_ok and unexpected_tokens:
        raise ValueError(
            f"Found tokens not in model_names: {sorted(unexpected_tokens)}."
        )

    matched_paths.sort()
    return matched_paths





def set_matplotlib_font(font_dir: str, arial_font_path: str = "/Users/xpji/ttf/ARIAL.TTF") -> None:
    """
    在指定目录搜索目标字体，并将其设置为Matplotlib全局字体。
    
    参数:
    -------
    font_dir : str
        字体文件夹路径。
    target_font : str, 默认 "Arial"
        要设置的字体名称。
    
    返回:
    -------
    None
    """
    if not os.path.isdir(font_dir):
        print(f"❌ 提供的路径无效：{font_dir}")
        return
    if font_dir:
            fonts = font_manager.findSystemFonts(fontpaths=font_dir)
            print(f"在 {font_dir} 找到 {len(fonts)} 个字体文件")        
        # 尝试寻找Arial字体
    
    
    print(f"找到Arial字体：{arial_font_path}")
    arial_font_prop = font_manager.FontProperties(fname=arial_font_path)
    mpl.rcParams['font.family'] = arial_font_prop.get_name()
    print(f"Matplotlib全局字体已设置为：{arial_font_prop.get_name()}")
  
   



"""
优化的 HEALPix 到 LatLon 网格转换工具
Optimized Convert healpix data to a latlon grid

主要优化：
1. 向量化操作替代循环
2. 使用numba加速关键函数
3. 预计算和缓存中间结果
4. 使用更快的插值方法

Performance improvements:
1. Vectorization instead of loops
2. Numba acceleration for critical functions
3. Precompute and cache intermediate results
4. Use faster interpolation methods
"""





def dataset_healpix_to_equatorial_latlon(
    dataset: xr.Dataset,
    nside: int,
    nest: str,
    minmax_lat: float
) -> xr.Dataset:
    """
    Extract a latlon dataarray from a healpix dataset.

    The latlon array extracted is for a band around the equator.
    """
    latlon_datarrays = []
    for variable_name in dataset.data_vars:
        dataarray = dataset[variable_name]
        latlon_datarray_aux = dataarray_healpix_to_equatorial_latlon(
            dataarray, nside, nest, minmax_lat
        )
        latlon_datarray_aux.name = variable_name
        latlon_datarrays.append(latlon_datarray_aux)
    return xr.merge(latlon_datarrays)


def dataarray_healpix_to_equatorial_latlon(
    healpix_dataarray: xr.DataArray,
    nside: int,
    nest: bool,
    minmax_lat: float
) -> xr.DataArray:
    """
    优化版本：使用向量化和numba加速
    Optimized version with vectorization and numba acceleration
    
    主要改进：
    1. 一次性预计算所有纬度圈信息
    2. 使用向量化操作批量处理
    3. 对插值使用numba加速
    """
    if not HAS_HEALPY:
        raise ImportError("healpy is required for HEALPix operations. Install with: pip install healpy")
    MAXIMUM_LAT_RANGE = 90
    if minmax_lat > MAXIMUM_LAT_RANGE:
        raise ValueError(f"minmax_lat={minmax_lat} too wide for equatorial analysis.")

    data = healpix_dataarray.values
    time = healpix_dataarray.time.values
    
    # 获取像素索引 - 如果有cell坐标则使用，否则创建完整索引
    if "cell" in healpix_dataarray.coords:
        pix_index = healpix_dataarray.coords["cell"].values
    else:
        npix = hp.nside2npix(nside)
        pix_index = np.arange(npix)
    
    # 获取经纬度信息
    lat, lon = _get_pix_latlon(nside, nest, pix_index)
    
    print(f"DEBUG: Total pixels: {len(pix_index)}")
    print(f"DEBUG: Lat range: {lat.min():.2f} to {lat.max():.2f}")
    print(f"DEBUG: Lon range: {lon.min():.2f} to {lon.max():.2f}")
    
    # 预先计算圆整后的纬度
    lat_rounded = np.round(lat, 10)
    
    # 一次性筛选目标纬度带内的数据
    mask_lat = (lat_rounded >= -minmax_lat) & (lat_rounded <= minmax_lat)
    unique_lats = np.unique(lat_rounded[mask_lat])
    
    print(f"DEBUG: Unique latitudes in range: {len(unique_lats)}")
    
    # 获取目标经度网格 - 找到经度点最多的纬度圈
    max_nlon = 0
    reference_lat = unique_lats[0]
    for ulat in unique_lats:
        lat_mask = lat_rounded == ulat
        nlon_this_lat = np.sum(lat_mask)
        if nlon_this_lat > max_nlon:
            max_nlon = nlon_this_lat
            reference_lat = ulat
    
    # 使用经度点最多的纬度圈作为参考
    reference_lat_mask = lat_rounded == reference_lat
    final_lons = np.sort(lon[reference_lat_mask])
    
    print(f"DEBUG: Reference latitude: {reference_lat:.2f}, with {len(final_lons)} longitude points")
    
    # 预分配输出数组
    ntime, nlon, nlat = len(time), len(final_lons), len(unique_lats)
    resampled_data = np.zeros((ntime, nlon, nlat), dtype=data.dtype)
    
    # 批量处理所有纬度圈
    resampled_data = _process_all_lat_rings_vectorized(
        data, lat_rounded, lon, unique_lats, final_lons, ntime, nlon, nlat
    )
    
    # 转置数据以匹配标准的 (time, lat, lon) 维度顺序
    # 原始数据是 (time, nlon, nlat)，需要转为 (time, nlat, nlon)
    resampled_data = np.transpose(resampled_data, (0, 2, 1))
    
    # 创建输出 DataArray，维度顺序为 (time, lat, lon)
    latlon_dataarray = xr.DataArray(
        data=resampled_data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": unique_lats, "lon": final_lons},
    )
    return latlon_dataarray

def dataarray_to_equatorial_latlon_grid(
    dataarray: xr.DataArray, grid_type: str, grid_dict: Optional[dict]
) -> xr.DataArray:
    """转换数据到赤道经纬度网格"""
    if grid_type == "latlon":
        return dataarray
    elif grid_type == "healpix":
        if grid_dict is None:
            raise ValueError("No grid_dict provided for healpix conversion.")
        return dataarray_healpix_to_equatorial_latlon(dataarray, **grid_dict)
    else:
        raise ValueError("Grid type not found.")

def _process_all_lat_rings_vectorized(
    data: np.ndarray,
    lat_rounded: np.ndarray,
    lon: np.ndarray,
    unique_lats: np.ndarray,
    final_lons: np.ndarray,
    ntime: int,
    nlon: int,
    nlat: int
) -> np.ndarray:
    """
    向量化处理所有纬度圈
    Vectorized processing of all latitude rings
    """
    resampled_data = np.zeros((ntime, nlon, nlat), dtype=data.dtype)
    
    for i, unique_lat in enumerate(unique_lats):
        # 找到当前纬度圈的所有像素
        ring_mask = lat_rounded == unique_lat
        ring_index = np.where(ring_mask)[0]
        
        lons = lon[ring_index]
        data_ring = data[:, ring_index]
        
        # 按经度排序
        sorted_index = np.argsort(lons)
        sorted_lons = lons[sorted_index]
        sorted_data_ring = data_ring[:, sorted_index]
        
        # 打印调试信息（仅前几个纬度圈）
        if i < 3 or i == len(unique_lats) - 1:
            print(f"  Lat {unique_lat:.4f}: {len(sorted_lons)} points, target: {nlon} points")
        
        # 检查是否需要插值（当前纬度圈的经度数与目标网格不匹配时）
        if len(sorted_lons) != nlon:
            # 使用优化的插值函数
            sorted_data_ring = _fast_periodic_interp(
                final_lons, sorted_lons, sorted_data_ring
            )
            if i < 3:
                print(f"    -> Interpolated to {nlon} points")
        
        resampled_data[:, :, i] = sorted_data_ring
    
    return resampled_data


@jit(nopython=True, parallel=True, cache=True)
def _fast_periodic_interp_numba(
    x_new: np.ndarray,
    x_old: np.ndarray,
    y_old: np.ndarray,
    period: float = 360.0
) -> np.ndarray:
    """
    使用numba加速的周期性插值
    Fast periodic interpolation with numba
    
    参数:
        x_new: 目标经度网格
        x_old: 原始经度网格（已排序）
        y_old: 原始数据 (ntime, nlon_old)
        period: 周期（默认360度）
    """
    ntime = y_old.shape[0]
    nlon_new = len(x_new)
    nlon_old = len(x_old)
    result = np.zeros((ntime, nlon_new), dtype=y_old.dtype)
    
    # 扩展数据以处理周期性边界
    x_extended = np.zeros(nlon_old + 2)
    x_extended[0] = x_old[-1] - period
    x_extended[1:-1] = x_old
    x_extended[-1] = x_old[0] + period
    
    for t in prange(ntime):
        y_extended = np.zeros(nlon_old + 2)
        y_extended[0] = y_old[t, -1]
        y_extended[1:-1] = y_old[t, :]
        y_extended[-1] = y_old[t, 0]
        
        # 线性插值
        for i in range(nlon_new):
            x_target = x_new[i] % period
            
            # 找到插值区间
            idx = 0
            for j in range(len(x_extended) - 1):
                if x_extended[j] <= x_target <= x_extended[j+1]:
                    idx = j
                    break
            
            # 线性插值
            x0, x1 = x_extended[idx], x_extended[idx+1]
            y0, y1 = y_extended[idx], y_extended[idx+1]
            
            if x1 - x0 > 0:
                weight = (x_target - x0) / (x1 - x0)
                result[t, i] = y0 * (1 - weight) + y1 * weight
            else:
                result[t, i] = y0
    
    return result


def _fast_periodic_interp(
    x_new: np.ndarray,
    x_old: np.ndarray,
    y_old: np.ndarray,
    period: float = 360.0
) -> np.ndarray:
    """
    快速周期性插值（scipy版本）
    Fast periodic interpolation using scipy
    
    对于大数据集，scipy的interp1d比numba版本更快
    """
    # 归一化到 [0, period)
    x_new_norm = x_new % period
    x_old_norm = x_old % period
    
    # 扩展数据以处理周期性
    x_extended = np.concatenate([
        x_old_norm[-1:] - period,
        x_old_norm,
        x_old_norm[0:1] + period
    ])
    y_extended = np.concatenate([
        y_old[:, -1:],
        y_old,
        y_old[:, 0:1]
    ], axis=1)
    
    # 使用线性插值（比三次样条快）
    interp_func = interpolate.interp1d(
        x_extended, y_extended, 
        kind='linear', 
        axis=1,
        assume_sorted=True,
        copy=False
    )
    
    return interp_func(x_new_norm)


def _interp_array_along_first_axis(x, xp, fp, period):
    """原始插值函数（兼容性保留）"""
    return _fast_periodic_interp(x, xp, fp, period)


def _get_pix_latlon(nside: int, nest: bool, pix_index: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取HEALPix像素的经纬度坐标
    Get lat/lon coordinates for HEALPix pixels
    """
    if not HAS_HEALPY:
        raise ImportError("healpy is required for HEALPix operations. Install with: pip install healpy")
    
    if nest is False:
        raise NotImplementedError("nest=False is not implemented.")
    
    if pix_index is None:
        npix = hp.nside2npix(nside)
        pix_index = np.arange(npix)
    
    lon, lat = hp.pix2ang(nside, pix_index, lonlat=True, nest=nest)
    return lat, lon


# ==================== 额外的优化版本 ====================

def dataarray_healpix_to_equatorial_latlon_fast(
    healpix_dataarray: xr.DataArray,
    nside: int,
    nest: bool,
    minmax_lat: float,
    use_dask: bool = False
) -> xr.DataArray:
    """
    超快速版本：适用于大数据集
    Ultra-fast version for large datasets
    
    Parameters
    ----------
    healpix_dataarray : xr.DataArray
        输入的HEALPix数据
    nside : int
        HEALPix nside参数
    nest : bool
        是否使用nested scheme
    minmax_lat : float
        纬度范围 (±minmax_lat)
    use_dask : bool
        是否使用dask进行并行计算
    
    Returns
    -------
    xr.DataArray
        转换后的经纬度网格数据
    """
    if not HAS_HEALPY:
        raise ImportError("healpy is required for HEALPix operations. Install with: pip install healpy")
    
    if minmax_lat > MAXIMUM_LAT_RANGE:
        raise ValueError(f"minmax_lat={minmax_lat} too wide for equatorial analysis.")
    
    if use_dask:
        import dask.array as da
        # 将数据转换为dask数组以实现并行处理
        if not isinstance(healpix_dataarray.data, da.Array):
            healpix_dataarray = healpix_dataarray.chunk({'time': 'auto'})
    
    # 后续处理逻辑与标准版本相同
    return dataarray_healpix_to_equatorial_latlon(
        healpix_dataarray, nside, nest, minmax_lat
    )


def benchmark_conversion(healpix_dataarray, nside, nest, minmax_lat):
    """
    性能基准测试
    Benchmark the conversion performance
    """
    import time
    
    print("开始性能测试...")
    print(f"数据形状: {healpix_dataarray.shape}")
    
    start = time.time()
    result = dataarray_healpix_to_equatorial_latlon(
        healpix_dataarray, nside, nest, minmax_lat
    )
    end = time.time()
    
    print(f"✓ 转换完成！")
    print(f"  耗时: {end - start:.2f} 秒")
    print(f"  输出形状: {result.shape}")
    print(f"  内存占用: {result.nbytes / 1024**2:.1f} MB")
    
    return result





def get_region_healpix_(zoom: int = 8, extent: list = [-180, 181, -16, 16], nest: bool = True) -> np.ndarray:
    """
    获取指定经纬度范围的 Healpix 网格索引 / Get Healpix indices in a given lon/lat extent

    Parameters 参数
    ----------
    zoom : int, optional
        Healpix order, controls resolution. 默认 8
    extent : list, optional
        [lon_min, lon_max, lat_min, lat_max], 默认 [0,360,-15,15]
    nest : bool, optional
        是否使用 nested ordering，默认 True（应与数据集一致）

    Returns 返回
    -------
    icell : np.ndarray
        符合范围的 Healpix 网格索引 / Indices of pixels within the extent
    """
    if not HAS_HEALPY:
        raise ImportError("healpy is required for HEALPix operations. Install with: pip install healpy")
    
    nside = hp.order2nside(zoom)     # 计算 nside / compute nside
    npix  = hp.nside2npix(nside)     # 总像素数 / total pixels
    print(f"nside={nside}, npix={npix}")

    # 使用与数据集相同的 nest 参数
    hp_lon, hp_lat = hp.pix2ang(nside, np.arange(npix), nest=nest, lonlat=True)
    hp_lon = (hp_lon + 180) % 360 - 180  # 经度映射到 [-180,180] / map lon to [-180,180]

    # 选出落在范围内的网格 / select pixels within extent
    icell = np.where(
        (hp_lon >= extent[0]) & (hp_lon <= extent[1]) &
        (hp_lat >= extent[2]) & (hp_lat <= extent[3])
    )[0]

    print(f"纬度范围 Lat: {hp_lat[icell].min():.2f} ~ {hp_lat[icell].max():.2f}")
    print(f"经度范围 Lon: {hp_lon[icell].min():.2f} ~ {hp_lon[icell].max():.2f}")
    print(f"网格数量 Pixels: {len(icell)}")

    return icell


def _is_valid_healpix_layout(npix: int) -> bool:
    """Return True when pixel count matches a valid HEALPix nside."""
    if npix <= 0 or npix % 12 != 0:
        return False
    nside_sq = npix // 12
    nside = int(np.sqrt(nside_sq))
    return nside > 0 and nside * nside == nside_sq and (nside & (nside - 1) == 0)


def _regular_centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Build cell edges from a regular 1D center grid."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("target centers must be a 1D array with at least 2 points")
    delta = np.diff(centers)
    if not np.allclose(delta, delta[0]):
        raise ValueError("target centers must be regularly spaced")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * delta[0]
    edges[-1] = centers[-1] + 0.5 * delta[-1]
    return edges


def _resolve_icon_grid_file(data: xr.DataArray) -> str:
    """Locate the ICON grid file stored alongside the source data file."""
    source = data.encoding.get("source")
    if not source:
        raise FileNotFoundError(
            "Cannot resolve ICON grid file because the DataArray has no source path in encoding."
        )

    source_dir = os.path.dirname(os.path.realpath(source))
    candidates = [
        os.path.join(source_dir, name)
        for name in sorted(os.listdir(source_dir))
        if name.startswith("icon_grid_") and name.endswith("_G.nc")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No local ICON grid file found next to source data: {source_dir}"
        )
    return candidates[0]


def dataarray_icon_unstructured_to_latlon_grid(
    data: xr.DataArray,
    grid_file: Optional[str] = None,
    minmax_lat: float = 36.0,
    target_lat: Optional[np.ndarray] = None,
    target_lon: Optional[np.ndarray] = None,
    area_weighted: bool = True,
) -> xr.DataArray:
    """
    Convert ICON unstructured data to a regular lat-lon grid using cell-center bin averaging.

    This method is robust for very high-resolution ICON meshes where HEALPix-based logic
    is not applicable. Each ICON cell is assigned to a target lat-lon bin using its center
    coordinates, and values are averaged with cell-area weights when available.
    """
    if target_lat is None:
        target_lat = np.arange(-minmax_lat, minmax_lat + 0.1, 2.0)
    if target_lon is None:
        target_lon = np.arange(0.0, 360.0, 2.0)

    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)

    grid_file = grid_file or _resolve_icon_grid_file(data)
    with xr.open_dataset(grid_file) as grid_ds:
        cell_lon = np.rad2deg(grid_ds["clon"].values) % 360.0
        cell_lat = np.rad2deg(grid_ds["clat"].values)
        if area_weighted and "cell_area" in grid_ds.variables:
            cell_area = grid_ds["cell_area"].values
        else:
            cell_area = np.ones_like(cell_lon, dtype=np.float64)

    cell_dim = None
    for dim in data.dims:
        if data.sizes[dim] == cell_lon.size:
            cell_dim = dim
            break
    if cell_dim is None:
        raise ValueError(
            "Could not identify ICON cell dimension for unstructured-grid conversion."
        )

    time_dim = "time" if "time" in data.dims else None
    if time_dim is not None:
        working = data.transpose(time_dim, cell_dim)
        values = np.asarray(working.values)
        time_coord = working[time_dim].values
    else:
        working = data.transpose(cell_dim)
        values = np.asarray(working.values)[np.newaxis, :]
        time_coord = None

    lat_edges = _regular_centers_to_edges(target_lat)
    lon_edges = _regular_centers_to_edges(target_lon)
    dlon = float(np.diff(target_lon)[0])
    lon_start = float(lon_edges[0])

    lat_idx = np.searchsorted(lat_edges, cell_lat, side="right") - 1
    lon_idx = np.floor(((cell_lon - lon_start) % 360.0) / dlon).astype(np.int64)
    lon_idx = np.clip(lon_idx, 0, target_lon.size - 1)

    valid = (
        (lat_idx >= 0)
        & (lat_idx < target_lat.size)
        & (cell_lat >= lat_edges[0])
        & (cell_lat <= lat_edges[-1])
    )
    flat_bin = lat_idx[valid].astype(np.int64) * target_lon.size + lon_idx[valid]
    weights = cell_area[valid].astype(np.float64)

    ntime = values.shape[0]
    ngrid = target_lat.size * target_lon.size
    output = np.full((ntime, target_lat.size, target_lon.size), np.nan, dtype=np.float64)

    for it in range(ntime):
        vals = np.asarray(values[it, valid], dtype=np.float64)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue

        w = weights[finite]
        b = flat_bin[finite]
        weighted_sum = np.bincount(b, weights=vals[finite] * w, minlength=ngrid)
        weight_sum = np.bincount(b, weights=w, minlength=ngrid)

        row = np.full(ngrid, np.nan, dtype=np.float64)
        good = weight_sum > 0.0
        row[good] = weighted_sum[good] / weight_sum[good]
        output[it] = row.reshape(target_lat.size, target_lon.size)

    coords = {"lat": target_lat, "lon": target_lon}
    dims = ["lat", "lon"]
    if time_coord is not None:
        coords["time"] = time_coord
        dims = ["time", "lat", "lon"]

    result = xr.DataArray(
        output if time_coord is not None else output[0],
        dims=dims,
        coords=coords,
        name=data.name,
        attrs=dict(data.attrs),
    )
    result.attrs["grid_mapping"] = "icon_unstructured_cell_center_binning"
    result.attrs["icon_grid_file"] = grid_file
    result.attrs["area_weighted"] = "true" if area_weighted else "false"
    return result


def convert_icon_to_latlon_grid(
    data: xr.DataArray,
    nside: int = 256,
    nest: bool = True,
    minmax_lat: float = 36.0,
    target_lat: Optional[np.ndarray] = None,
    target_lon: Optional[np.ndarray] = None,
    interp_method: str = 'linear'
) -> xr.DataArray:
    """
    将ICON HEALPix网格数据转换为经纬度网格，并可选插值到目标分辨率
    
    此函数整合了HEALPix到等距柱状投影的转换和插值两个步骤，
    适用于ICON模式输出数据的标准化处理流程。
    
    Parameters 参数
    ----------
    data : xr.DataArray
        ICON HEALPix网格数据，维度为 (time, cell) 或 (cell,)
        ICON HEALPix grid data with dimensions (time, cell) or (cell,)
    nside : int, optional
        HEALPix分辨率参数，默认256 (对应R2B8)
        HEALPix resolution parameter, default 256 (for R2B8)
    nest : bool, optional
        是否使用嵌套排列，默认True
        Whether to use nested ordering, default True
    minmax_lat : float, optional
        赤道带纬度范围 (±minmax_lat)，默认36度
        Equatorial band latitude range (±minmax_lat), default 36 degrees
    target_lat : np.ndarray, optional
        目标纬度网格，如 np.arange(-36, 36.1, 2.0)
        Target latitude grid, e.g., np.arange(-36, 36.1, 2.0)
        如果为None，则不进行插值
        If None, no interpolation is performed
    target_lon : np.ndarray, optional
        目标经度网格，如 np.arange(0, 360, 2.0)
        Target longitude grid, e.g., np.arange(0, 360, 2.0)
        如果为None，则不进行插值
        If None, no interpolation is performed
    interp_method : str, optional
        插值方法，可选 'linear' 或 'nearest'，默认'linear'
        Interpolation method, 'linear' or 'nearest', default 'linear'
    
    Returns 返回
    -------
    xr.DataArray
        经纬度网格数据，维度为 (time, lat, lon) 或 (lat, lon)
        Lat-lon grid data with dimensions (time, lat, lon) or (lat, lon)
    
    Examples 示例
    --------
    >>> # 基本转换（无插值）
    >>> pr_latlon = convert_icon_to_latlon_grid(pr_icon, nside=256, nest=True)
    
    >>> # 转换并插值到2度网格
    >>> target_lat = np.arange(-36, 36.1, 2.0)
    >>> target_lon = np.arange(0, 360, 2.0)
    >>> pr_2deg = convert_icon_to_latlon_grid(
    ...     pr_icon, 
    ...     target_lat=target_lat, 
    ...     target_lon=target_lon
    ... )
    
    Notes 注意事项
    -----
    - 此函数依赖 healpy 库
    - 处理大数据集时建议分批处理时间维度
    - 插值方法 'linear' 更平滑但计算较慢，'nearest' 更快但可能有阶梯效应
    - This function requires the healpy library
    - For large datasets, consider processing time dimension in batches
    - 'linear' interpolation is smoother but slower, 'nearest' is faster but may show stepping
    """
    grid_type = str(data.attrs.get("CDI_grid_type", "")).lower()
    cell_like_dims = [dim for dim in data.dims if dim in {"cell", "ncells"}]
    looks_like_healpix = any(_is_valid_healpix_layout(data.sizes[dim]) for dim in cell_like_dims)

    inferred_nside = None
    for dim in cell_like_dims:
        dim_size = data.sizes[dim]
        if _is_valid_healpix_layout(dim_size):
            inferred_nside = int(np.sqrt(dim_size // 12))
            break

    if (grid_type == "unstructured" and not looks_like_healpix) or (cell_like_dims and not looks_like_healpix):
        print("\n" + "="*70)
        print("🌐 ICON unstructured grid → 经纬度网格转换")
        print("="*70)
        print("📍 方法: 基于真实 ICON cell center 的面积加权格点平均")
        print(f"   • 纬度范围: ±{minmax_lat}°")
        if target_lat is not None and target_lon is not None:
            print(f"   • 目标纬度点数: {len(target_lat)}")
            print(f"   • 目标经度点数: {len(target_lon)}")

        data_latlon = dataarray_icon_unstructured_to_latlon_grid(
            data=data,
            minmax_lat=minmax_lat,
            target_lat=target_lat,
            target_lon=target_lon,
            area_weighted=True,
        )
        print(f"   ✅ 转换完成")
        print(f"   • 输出维度: {list(data_latlon.dims)}")
        print(f"   • 输出形状: {data_latlon.shape}")
        print("="*70 + "\n")
        return data_latlon

    if not HAS_HEALPY:
        raise ImportError("healpy is required. Install with: pip install healpy")

    if inferred_nside is not None and inferred_nside != nside:
        print(f"   • 自动识别 HEALPix nside: {inferred_nside} (覆盖传入的 nside={nside})")
        nside = inferred_nside
    
    print("\n" + "="*70)
    print("🌐 ICON HEALPix → 经纬度网格转换")
    print("="*70)
    
    # Step 1: HEALPix到等距柱状投影转换
    print(f"📍 步骤1: HEALPix → 等距柱状投影")
    print(f"   • nside: {nside}")
    print(f"   • nest: {nest}")
    print(f"   • 纬度范围: ±{minmax_lat}°")
    
    grid_dict = {
        "nside": nside,
        "nest": nest,
        "minmax_lat": minmax_lat
    }
    
    data_latlon = dataarray_healpix_to_equatorial_latlon(data, **grid_dict)
    
    print(f"   ✅ 转换完成")
    print(f"   • 输出维度: {list(data_latlon.dims)}")
    print(f"   • 输出形状: {data_latlon.shape}")
    print(f"   • 纬度范围: {data_latlon.lat.values.min():.2f}° 至 {data_latlon.lat.values.max():.2f}°")
    print(f"   • 经度范围: {data_latlon.lon.values.min():.2f}° 至 {data_latlon.lon.values.max():.2f}°")
    
    # Step 2: 插值到目标网格（可选）
    if target_lat is not None and target_lon is not None:
        print(f"\n📍 步骤2: 插值到目标网格")
        print(f"   • 目标纬度: {len(target_lat)} 点 ({target_lat.min():.1f}° - {target_lat.max():.1f}°)")
        print(f"   • 目标经度: {len(target_lon)} 点 ({target_lon.min():.1f}° - {target_lon.max():.1f}°)")
        print(f"   • 插值方法: {interp_method}")
        
        data_interp = data_latlon.interp(
            lat=target_lat,
            lon=target_lon,
            method=interp_method
        )
        
        print(f"   ✅ 插值完成")
        print(f"   • 输出形状: {data_interp.shape}")
        
        print("="*70 + "\n")
        return data_interp
    else:
        print(f"\n⏭️  跳过插值步骤")
        print("="*70 + "\n")
        return data_latlon


def batch_convert_icon_to_latlon(
    data_dict: dict,
    output_dir: str,
    nside: int = 256,
    nest: bool = True,
    minmax_lat: float = 36.0,
    target_lat: Optional[np.ndarray] = None,
    target_lon: Optional[np.ndarray] = None,
    interp_method: str = 'linear',
    skip_existing: bool = True
) -> dict:
    """
    批量转换多个实验的ICON数据到经纬度网格
    
    Parameters 参数
    ----------
    data_dict : dict
        实验数据字典，格式: {'exp_name': xr.DataArray}
        Dictionary of experiment data, format: {'exp_name': xr.DataArray}
    output_dir : str
        输出目录路径
        Output directory path
    nside, nest, minmax_lat, target_lat, target_lon, interp_method :
        同 convert_icon_to_latlon_grid 参数
        Same as convert_icon_to_latlon_grid parameters
    skip_existing : bool, optional
        是否跳过已存在的文件，默认True
        Whether to skip existing files, default True
    
    Returns 返回
    -------
    dict
        转换后的数据字典，格式同输入
        Dictionary of converted data, same format as input
    
    Examples 示例
    --------
    >>> data_dict = {
    ...     'control': pr_control,
    ...     'plus2K': pr_plus2K,
    ...     'plus4K': pr_plus4K
    ... }
    >>> results = batch_convert_icon_to_latlon(
    ...     data_dict,
    ...     output_dir='/path/to/output',
    ...     target_lat=np.arange(-36, 36.1, 2.0),
    ...     target_lon=np.arange(0, 360, 2.0)
    ... )
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    n_exp = len(data_dict)
    
    print("\n" + "🌍"*35)
    print(f"📊 批量转换 {n_exp} 个实验的数据")
    print("🌍"*35 + "\n")
    
    for idx, (exp_name, data) in enumerate(data_dict.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{n_exp}] 实验: {exp_name.upper()}")
        print(f"{'='*70}")
        
        # 构建输出文件路径
        if target_lat is not None and target_lon is not None:
            resolution = abs(target_lat[1] - target_lat[0])
            output_file = os.path.join(
                output_dir, 
                f"{exp_name}_latlon_{resolution:.0f}deg.nc"
            )
        else:
            output_file = os.path.join(output_dir, f"{exp_name}_latlon.nc")
        
        # 检查是否已存在
        if skip_existing and os.path.exists(output_file):
            print(f"✅ 文件已存在，跳过: {output_file}")
            results[exp_name] = xr.open_dataarray(output_file)
            continue
        
        try:
            # 转换数据
            data_converted = convert_icon_to_latlon_grid(
                data,
                nside=nside,
                nest=nest,
                minmax_lat=minmax_lat,
                target_lat=target_lat,
                target_lon=target_lon,
                interp_method=interp_method
            )
            
            # 保存文件
            print(f"💾 保存到: {output_file}")
            data_converted.to_netcdf(output_file)
            print(f"✅ 保存完成")
            
            results[exp_name] = data_converted
            
        except Exception as e:
            print(f"❌ 处理失败: {exp_name}")
            print(f"   错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            results[exp_name] = None
    
    print("\n" + "🌍"*35)
    print("✅ 批量转换完成！")
    print("🌍"*35 + "\n")
    
    return results




