#!/bin/bash

#
#current_time="20240705 0600"
current_time="20230621 0600"
#current_time=$(date -d "$current_time + 10 min" +"%Y%m%d %H%M")
start_hour=3
end_min=10


# Base directory
base_dir="/DatadiskExt/xpji/jxp_deeplearn_model_code"
data_dir="/DatadiskExt/xpji/Hia8"
model_dir="/DatadiskExt/xpji/jxp_deeplearn_model_code/modelconfig/"

# Himawari-9 data path
ori_input_dir="$data_dir"

# python scipts
convert_script="$base_dir/code/01_convert.py"
model_input_script="$base_dir/code/02_create_model_input.py"
model_pred_script="$base_dir/code/03_model_pred.py"
time_up_script="$base_dir/code/04_time_up.py"
check_scipt="$base_dir/code/05_check_data.py"
plot_script="$base_dir/code/06_plot_predict.py"
# file path
temp_path="$base_dir/temp/"
temp_nc_path="$base_dir/temp_nc"
bin_path="$base_dir/bin/"
model_input_path="$base_dir/model_input_nc/"
model_predict_path="$base_dir/model_3hour_prediction/"

# model config path
scaler_x_path="$model_dir/scalerx.pkl"
model_path="$model_dir/model_unet.hdf5"
scaler_y_path="$model_dir/scalery.pkl"
region_path="$model_dir/regions.npy"



# log path
time_log="$base_dir/time_log"
time_log2="$base_dir/time_log2"
time_log3="$base_dir/time_log3"

# python path 
pypath="/Users/xpji/.conda/envs/tf-g/bin/python"


create_directory() {
  if [ ! -d "$1" ]; then
    mkdir -p "$1"
    echo "Created directory: $1"
  else
    echo "Directory already exists: $1"
  fi
}

# 检查并创建路径
create_directory "$temp_path"
create_directory "$temp_nc_path"
create_directory "$bin_path"
create_directory "$model_input_path"
create_directory "$model_predict_path"

