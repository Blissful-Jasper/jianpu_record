# -*- coding: utf-8 -*-

import os
import glob
import re
import subprocess
from datetime import datetime, timedelta
import argparse
import pandas as pd
import xarray as xr


def get_time_difference(time1, time2):
    t1 = pd.to_datetime(time1)
    t2 = pd.to_datetime(time2)
    diff = (t2 - t1).total_seconds() / 60  # 计算分钟差异
    return diff


def convert_time(time):
    timestamp = pd.Timestamp(time)
    formatted_str = timestamp.strftime('%Y%m%d_%H%M')
    return str(formatted_str)

def convert_time2(self,time):
        time_str = str(time)
        timestamp = pd.Timestamp(time_str)
        formatted_str = timestamp.strftime('%Y%m%d')
        return str(formatted_str)    

def generate_new_time(time):
    ss = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in time]

    interval = pd.Timedelta(minutes=30)

    new_time = pd.date_range(start=ss[0] + 6 * interval, end=ss[-1] + 6 * interval, freq=interval)

    new_time_values = new_time.to_pydatetime()

    return new_time_values


def get_current_file_paths(processed_folders, start_index, interval, num_data):
    return processed_folders[start_index : start_index + interval * num_data][::interval]



# 指定输入和输出目录
# input_dir = "/Users/xpji/test_code/temp_nc"
# output_dir = "/Users/xpji/test_code/code"

parser = argparse.ArgumentParser()
parser.add_argument("start_time", help="起始时间（格式：'%Y%m%d %H%M'）")
parser.add_argument("end_time", help="结束时间（格式：'%Y%m%d %H%M'）")
parser.add_argument("--input_dir", help="输入目录路径", default="/home/gaby/test_run/temp_nc/")
parser.add_argument("--output_dir", help="输出目录路径", default="/home/gaby/test_run/model_input_nc/")
args = parser.parse_args()

# 获取参数值
start_time = args.start_time
end_time = args.end_time
input_dir = args.input_dir
output_dir = args.output_dir

# 通过命令行参数传入起始时间和结束时间

start_time = datetime.strptime(start_time, '%Y%m%d %H%M')
end_time = datetime.strptime(end_time, '%Y%m%d %H%M')

# 定义时间间隔
interval = timedelta(minutes=30)

# 初始化时间列表
time_list = []
folder_list = []
# 生成时间列表
current_time = start_time
while current_time < end_time:
    time_list.append(current_time.strftime('%Y%m%d_%H%M'))
    folder_list.append(current_time.strftime('%Y%m%d'))
    current_time += interval
    
file_paths=[]

for time_str in range(len(time_list)):
    file_name = 'NC_HS_H09_'+time_list[time_str]+'_FLDK_R100.nc'
    folder_path = os.path.join(input_dir,folder_list[time_str])
    file_path = os.path.join(folder_path,file_name)
    print(file_path)
    if os.path.exists(file_path):
        file_paths.append(file_path)
        
    else:
        print('file_path is not exist')


data_list = [xr.open_dataset(file_path) for file_path in sorted(file_paths)]
try:
    if len(data_list)==6:
        merged_data = xr.concat(data_list, dim="time")
        tbb_list = ['tbb_{:02d}'.format(band) for band in [8, 9, 10, 11, 13, 14, 16]]

        ds_band = merged_data[tbb_list].to_array().rename({'variable':'band'}).transpose('time','latitude','longitude','band')

        ds_all = xr.Dataset({'tbb': (['time','lat', 'lon', 'band'], ds_band.data)},
                            coords={'lat': ds_band.latitude.data,
                                    'lon': ds_band.longitude.data,
                                    'time': ds_band.time.data,
                                    'band':ds_band.band.data
                                    })
        output_date_dir = os.path.join(output_dir, time_list[0][0:8])
        if not os.path.exists(output_date_dir):
            os.makedirs(output_date_dir)
        output_nc_path = os.path.join(output_date_dir,
        'Model_input_'+time_list[0]+'-'+time_list[-1]+"_FLDK_R100.nc")

        if not os.path.exists(output_nc_path):
            ds_all.to_netcdf(output_nc_path)
            print('Saved data to{}'.format(output_nc_path))
        else:
            print('Data already exists at {}'.format(output_nc_path))
        
    else:
        print('the length of time is not enough')
except Exception as e:
    print(f"Error processing files in the current interval: {e}")



