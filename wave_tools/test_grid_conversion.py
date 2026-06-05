"""
测试ICON网格转换函数

验证 convert_icon_to_latlon_grid 和 batch_convert_icon_to_latlon 是否正常工作
"""
import sys
sys.path.insert(0, '/work/mh1498/m301257')

import numpy as np
import xarray as xr
from wave_tools.utils import convert_icon_to_latlon_grid, batch_convert_icon_to_latlon

def test_single_conversion():
    """测试单个数据集转换"""
    print("\n" + "="*70)
    print("测试 1: convert_icon_to_latlon_grid")
    print("="*70)
    
    # 创建测试数据 (模拟ICON HEALPix数据)
    npix = 786432  # nside=256的像素数
    time = np.arange(10)
    
    # 创建模拟降水数据
    data = np.random.rand(len(time), npix) * 10  # 0-10 mm/day
    
    pr_icon = xr.DataArray(
        data,
        dims=['time', 'cell'],
        coords={'time': time, 'cell': np.arange(npix)},
        attrs={'units': 'mm/day', 'long_name': 'Precipitation Rate'}
    )
    
    print(f"✅ 创建测试数据: {pr_icon.shape}")
    
    # 执行转换（仅HEALPix转换，不插值）
    try:
        pr_latlon = convert_icon_to_latlon_grid(
            pr_icon,
            nside=256,
            nest=True,
            minmax_lat=36.0
        )
        print(f"✅ 转换成功！")
        print(f"   输出维度: {list(pr_latlon.dims)}")
        print(f"   输出形状: {pr_latlon.shape}")
        return True
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_single_conversion_with_interp():
    """测试带插值的转换"""
    print("\n" + "="*70)
    print("测试 2: convert_icon_to_latlon_grid (带插值)")
    print("="*70)
    
    # 创建测试数据
    npix = 786432
    time = np.arange(5)  # 减少时间步以加快测试
    
    data = np.random.rand(len(time), npix) * 10
    
    pr_icon = xr.DataArray(
        data,
        dims=['time', 'cell'],
        coords={'time': time, 'cell': np.arange(npix)},
        attrs={'units': 'mm/day'}
    )
    
    print(f"✅ 创建测试数据: {pr_icon.shape}")
    
    # 定义目标网格
    target_lat = np.arange(-36, 36.1, 2.0)
    target_lon = np.arange(0, 360, 2.0)
    
    # 执行转换和插值
    try:
        pr_2deg = convert_icon_to_latlon_grid(
            pr_icon,
            nside=256,
            nest=True,
            minmax_lat=36.0,
            target_lat=target_lat,
            target_lon=target_lon,
            interp_method='linear'
        )
        print(f"✅ 转换和插值成功！")
        print(f"   输出维度: {list(pr_2deg.dims)}")
        print(f"   输出形状: {pr_2deg.shape}")
        print(f"   目标网格: {len(target_lat)} lat × {len(target_lon)} lon")
        return True
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_conversion():
    """测试批量转换"""
    print("\n" + "="*70)
    print("测试 3: batch_convert_icon_to_latlon")
    print("="*70)
    
    # 创建多个实验的测试数据
    npix = 786432
    time = np.arange(3)  # 少量时间步
    
    data_dict = {}
    for exp_name in ['test_exp1', 'test_exp2']:
        data = np.random.rand(len(time), npix) * 10
        data_dict[exp_name] = xr.DataArray(
            data,
            dims=['time', 'cell'],
            coords={'time': time, 'cell': np.arange(npix)},
            attrs={'units': 'mm/day'}
        )
    
    print(f"✅ 创建 {len(data_dict)} 个测试数据集")
    
    # 定义目标网格
    target_lat = np.arange(-36, 36.1, 5.0)  # 粗网格以加快测试
    target_lon = np.arange(0, 360, 5.0)
    
    # 批量转换
    try:
        import tempfile
        output_dir = tempfile.mkdtemp()
        
        results = batch_convert_icon_to_latlon(
            data_dict=data_dict,
            output_dir=output_dir,
            nside=256,
            nest=True,
            minmax_lat=36.0,
            target_lat=target_lat,
            target_lon=target_lon,
            skip_existing=True
        )
        
        print(f"✅ 批量转换成功！")
        print(f"   处理实验数: {len(results)}")
        for exp_name, data in results.items():
            if data is not None:
                print(f"   - {exp_name}: {data.shape}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(output_dir)
        
        return True
    except Exception as e:
        print(f"❌ 批量转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🌍"*35)
    print("开始测试ICON网格转换函数")
    print("🌍"*35)
    
    results = []
    
    # 注意：这些测试需要healpy库
    try:
        import healpy
        print("✅ healpy 库已安装")
    except ImportError:
        print("❌ 缺少 healpy 库，请安装: pip install healpy")
        sys.exit(1)
    
    # 运行测试
    results.append(("单数据集转换", test_single_conversion()))
    results.append(("转换+插值", test_single_conversion_with_interp()))
    results.append(("批量转换", test_batch_conversion()))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
