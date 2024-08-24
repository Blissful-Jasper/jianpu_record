#!/bin/bash

#source /home/gaby/.bashrc
#export LD_LIBRARY_PATH=/home/gaby/libraries/netcdf_4.2.1.1/lib:$LD_LIBRARY_PATH

source "/DatadiskExt/xpji/jxp_deeplearn_model_code/cf1.sh"





if [ ! -d "$time_log3" ]; then
  mkdir -p "$time_log3"
fi

if [ ! -d "$model_predict_path" ]; then
  mkdir -p "$model_predict_path"
fi

time_file3="$time_log3/current_time_03.txt"
python_output="$time_log3/python_output_03.log"

declare -A finish_predict_files


predict_files() {
  local model_input_path="$1"
  local start_time="$2"
  local python_output="$3"
  local model_pred_script="$model_pred_script"
  local scaler_x_path="$scaler_x_path"
  local model_path="$model_path"
  local scaler_y_path="$scaler_y_path"
  local model_predict_path="$model_predict_path"
  
  # Construct the expected file name based on the start_time
  start_time_formatted=$(date -d "$start_time" +"%Y%m%d_%H%M")
  end_time_formatted=$(date -d "$start_time + 150 minutes" +"%Y%m%d_%H%M")
  echo "start_time_formatted: $start_time_formatted"
  echo "end_time_formatted: $end_time_formatted"
  
  pre_start=$(date -d "$start_time + 180 minutes" +"%Y%m%d_%H%M")
  pre_end=$(date -d "$start_time + 330 minutes" +"%Y%m%d_%H%M")
  expected_file="${model_input_path}/Model_input_${start_time_formatted}-${end_time_formatted}_FLDK_R100.nc"
  output_file="${model_predict_path}${input_dir}/NC_Pred_${pre_start}-${pre_end}_R100_f30.nc"
  echo "$output_file"
  # Check if the expected file exists
  if [ -f "$expected_file" ]; then
    echo "The model input is prepared: $expected_file"
    $pypath "$model_pred_script" \
      --sat_path "$expected_file" \
      --scaler_x_path "$scaler_x_path" \
      --model_path "$model_path" \
      --scaler_y_path "$scaler_y_path" \
      --output_path "$model_predict_path" > "$python_output" 2>&1
    echo "$model_predict_path"
   
    if grep -q "saved" "$python_output"; then
      echo 'Predict finish'
      $pypath "$plot_script" --input_file "$output_file"
      return 0  # Success
      
    else
      echo 'Prediction failed. No "saved" keyword found in the output.'
      return 1  # Failure
    fi
  else
    echo "Error: File $expected_file not found."
    return 1  # Failure
  fi
}


if [ -f "$time_file3" ]; then
  current_time_rounded=$(cat "$time_file3")
  echo "Current Time: $current_time_rounded"

  start_time=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d %H%M")  
  echo "Model input Start Time: $start_time"
  end_time=$(date -d "$current_time_rounded - $end_min min" +"%Y%m%d %H%M")
  echo "Model input end Time: $(date -d "$current_time_rounded - 30 min" +"%Y%m%d %H%M")"
  input_dir=$(date -d "$start_time" +"%Y%m%d")
  echo "input_dir: $input_dir"

  python_output_03="$time_log3/python_output_03.log"
  
  if predict_files "$model_input_path/$input_dir" "$start_time" "$python_output_03"; then
    next_time=$(date -d "$end_time + 20 min" +"%Y%m%d %H%M")  
    echo "Next Start Time: $next_time"
    echo "$next_time" > "$time_file3"
  else
    echo "Predict is not finish. Retaining the current time."
  fi
  
else
  echo "Using Default Time: $current_time"
  current_minute=$(date -d "$current_time" +%M)
  current_minute_rounded=$(( (current_minute ) / 10 * 10 ))
  current_minute_formatted=$(printf "%02d" "$current_minute_rounded")
  current_time_rounded=$(date -d "$current_time" +"%Y%m%d %H${current_minute_formatted}")
  start_time=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d %H%M")
  echo "Model input Start Time: $start_time"
  end_time=$(date -d "$current_time_rounded - $end_min min" +"%Y%m%d %H%M")
  echo "Model input end Time: $(date -d "$current_time_rounded - 30 min" +"%Y%m%d %H%M")"
  input_dir=$(date -d "$start_time" +"%Y%m%d")
  echo "input_dir: $input_dir"
  python_output_03="$time_log3/python_output_03.log"
  
  if predict_files "$model_input_path/$input_dir" "$start_time" "$python_output_03"; then
    next_time=$(date -d "$end_time + 20 min" +"%Y%m%d %H%M")  
    echo "Next Start Time: $next_time"
    echo "$next_time" > "$time_file3"
  else
    echo "Predict is not finish. Retaining the current time."
  fi

 
fi


