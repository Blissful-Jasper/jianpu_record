#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:43:51 2023

@author: xpji
"""



import os
import xarray as xr
import tensorflow as tf
import numpy as np
import joblib
from scipy.ndimage import distance_transform_edt
# from skimage.util.shape import view_as_windows
import pandas as pd
from datetime import datetime
import metpy.calc as mpcalc
import argparse

class model_pred:
    def __init__(self,sat_path,  scaler_x_path, model_path, scaler_y_path, region_path,output_path):
        
        self.scaler_x_path = scaler_x_path
        self.model_path = model_path
        self.scaler_y_path = scaler_y_path
        self.region_path = region_path
        self.output_path = output_path
        self.sat_path = sat_path
        self.sat_data = xr.open_dataset(self.sat_path).sortby('lat')
        self.regions = np.load(self.region_path)
        
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.scaler_x = joblib.load(self.scaler_x_path)
        self.scaler_y = joblib.load(self.scaler_y_path)
        
    def fill_nans(self, data_matrix):
        indices = distance_transform_edt(
            np.isnan(data_matrix), return_distances=False, return_indices=True
        )
        return data_matrix[tuple(indices)]
    
    def assert_is_real_number(self, input_variable):
        return
    
    def assert_is_not_nan(self, input_variable):
        self.assert_is_real_number(input_variable)
        if np.isnan((input_variable)).any():
            raise ValueError('Input variable is NaN.')
    
    def assert_is_greater(self, input_variable, base_value, allow_nan=False):
        self.assert_is_not_nan(base_value)
        mask = np.logical_and(input_variable < base_value, ~np.isnan(input_variable))
        if np.logical_not(mask).all():
            return input_variable

        input_variable[mask] = np.nan
        filled_data = self.fill_nans(input_variable)
        if np.isnan(filled_data).any():
            raise ValueError("NaN values still exist after interpolation.")
        return filled_data
    
    def create_samples(self, x):
        trainX = []
        n_past = 6
        n_future = 6

        if len(x) >= n_past + n_future:
            for i in range(n_past, len(x) - n_future + 1):
                trainX.append(x[i - n_past:i])
            trainX = np.array(trainX)
            # print('trainX shape == {}.'.format(trainX.shape))
        else:
            print("Insufficient data length for creating samples.")
        return trainX
    
    def reshape_x(self, trainX):
        trainx = np.transpose(trainX, (0, 2, 3, 1, 4))
        x_data = []

        for i in range(trainx.shape[-2]):
            for j in range(trainx.shape[-1]):
           #     print(i, j)
                x_data.append(trainx[:,:,:,i,j])

        x_data = np.array(x_data)
        x_data = np.transpose(x_data,(1,2,3,0))
        return x_data
    
    def reshape_input_x(self, input_x):
        if len(input_x) < 6:
            raise ValueError("输入数据长度不够")
        elif len(input_x) == 6:
            trainX = np.expand_dims(input_x[:6], axis=0)
            trainx = np.transpose(trainX, (0, 2, 3, 1, 4))
            x_data = []

            for i in range(trainx.shape[-2]):
                for j in range(trainx.shape[-1]):
            #        print(i, j)
                    x_data.append(trainx[:, :, :, i, j])

            x_data = np.array(x_data)
            x_data = np.transpose(x_data,(1,2,3,0))
        else:
            past = 6
            data_split = []
            for k in range(0,len(input_x)-past+1):
                # print([k,k+past])
                data_split.append(input_x[k:k+past])
            data_split = np.array(data_split)

            trainx = np.transpose(data_split, (0, 2, 3, 1, 4))
            x_data = []

            for i in range(trainx.shape[-2]):
                for j in range(trainx.shape[-1]):
                    # print(i, j)
                    x_data.append(trainx[:, :, :, i, j])

            x_data = np.array(x_data)
            x_data = np.transpose(x_data,(1,2,3,0))
        return x_data
        
    def scaler(self,x):
        x_test = (self.scaler_x.transform(x.reshape(x.shape[0],-1))).reshape(x.shape)
        return x_test
        
    def scaler_inverse(self, y_pred):
        y_pred_inverse = self.scaler_y.inverse_transform(y_pred.reshape(y_pred.shape[0],-1)).reshape(y_pred.shape)
        y_pred_ori = np.expm1(y_pred_inverse)
        y_pred_ori[y_pred_ori<0] = 0
        return y_pred_ori

    def generate_new_time(self,time):
    
       time_str = [str(i) for i in pd.to_datetime(time)]
       ss = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in time_str]

       interval = pd.Timedelta(minutes=30)
       
       new_time = pd.date_range(start=ss[0] + 6 * interval, end=ss[-1] + 6 * interval, freq=interval)

       new_time_values = new_time.to_pydatetime()

       return new_time_values
    
    def convert_time(self,time):
        time_str = str(time)
        # 将字符串转换为 Pandas 的 Timestamp 对象
        timestamp = pd.Timestamp(time_str)
        
        # 使用 strftime() 方法自定义日期时间的格式
        formatted_str = timestamp.strftime('%Y%m%d_%H%M')
        
        # print('转换后的格式化字符串：', formatted_str)
        return str(formatted_str)
    def convert_time2(self,time):
        time_str = str(time)
        # 将字符串转换为 Pandas 的 Timestamp 对象
        timestamp = pd.Timestamp(time_str)
        
        # 使用 strftime() 方法自定义日期时间的格式
        formatted_str = timestamp.strftime('%Y%m%d')
        
        # print('转换后的格式化字符串：', formatted_str)
        return str(formatted_str)    
        
        
    def predict(self):
        

        nc_files = []
        for i in range(len(self.regions)):
            ds = self.sat_data.tbb.sel(lat=slice(self.regions[i][0],self.regions[i][1]),
                                   lon=slice(self.regions[i][2],self.regions[i][3]))
            time = ds.time.values
            new_time = self.generate_new_time(time)
            lat  = ds.lat 
            lon  = ds.lon
            # band = ds.band
		
            x =  self.assert_is_greater(ds, base_value=-1)
            x_res   =  self.reshape_input_x(x)
            x_scale =  self.scaler_x.transform(x_res.reshape(x_res.shape[0],-1)).reshape(x_res.shape)
            with tf.device('/gpu:0'):
                y_pred = self.scaler_inverse(self.model.predict(x_scale))

            data_vars = {'data': (['lat', 'lon', 'f_time'], y_pred[0])}
            coords = { 'lat': lat, 'lon': lon, 'f_time': new_time}
            ds_pred = xr.Dataset(data_vars, coords=coords)
            nc_files.append(ds_pred)

        ds_list = []
        for i in range(0, 16, 4):
            # print(i)
            ds_temp = xr.concat([nc_files[i], nc_files[i+1], nc_files[i+2], nc_files[i+3].sel(lat=slice(38.4,45))], dim='lat')

            ds_list.append(ds_temp)

        ds_com = xr.concat([ds_list[0],ds_list[1],ds_list[2],ds_list[3].sel(lon=slice(138.4,145))], dim='lon')

        ds_smooth = mpcalc.smooth_gaussian(ds_com.data, 8).values

        ds_all = xr.Dataset({'precip': (['latitude', 'longitude', 'time'], ds_smooth)},
                            coords={'latitude': ds_com.lat.data,
                                    'longitude': ds_com.lon.data,
                                    'time': new_time})
        output_date_dir = os.path.join(self.output_path, self.convert_time2(new_time[0])) 
        if not os.path.exists(output_date_dir):
            os.makedirs(output_date_dir)

        predict_output_path = os.path.join(output_date_dir, 
                    'NC_Pred_'+self.convert_time(new_time[0])+'-'+self.convert_time(new_time[-1])+"_R100_f30.nc")
        print(predict_output_path,'saved')
        if not os.path.exists(predict_output_path):
            
            ds_all.to_netcdf(predict_output_path)
            
        
        
        
  
         
        
    
def main(sat_path, scaler_x_path, model_path, scaler_y_path, region_path, output_path):
        
    predicter = model_pred(sat_path, scaler_x_path, model_path, scaler_y_path, region_path,output_path)
    predicter.predict()






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict Himawari-9 data to precip")
    parser.add_argument("--sat_path",  type=str,  help="model input Netcdf data")
    parser.add_argument("--scaler_x_path", type=str,  help="standard normolized for model input data")
    parser.add_argument("--model_path", type=str,     help="model save path")
    parser.add_argument("--scaler_y_path", type=str,  help="standard normolized inverse for model output data")
    parser.add_argument("--region_path", type=str,    help="slice region")
    parser.add_argument("--output_path", type=str,    help="prediction output path")
    args = parser.parse_args()
    
    main(args.sat_path, args.scaler_x_path, args.model_path, args.scaler_y_path, args.region_path,args.output_path)
