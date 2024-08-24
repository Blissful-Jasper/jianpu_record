#!/bin/bash

# source /home/gaby/.bashrc
# export LD_LIBRARY_PATH=/home/gaby/libraries/netcdf_4.2.1.1/lib:$LD_LIBRARY_PATH

source "/DatadiskExt/xpji/jxp_deeplearn_model_code/cf1.sh"




if [ ! -d "$time_log2" ]; then
  mkdir -p "$time_log2"
fi
if [ ! -d "$temp_path" ]; then
  mkdir -p "$temp_path"
fi
if [ ! -d "$temp_nc_path" ]; then
  mkdir -p "$temp_nc_path"
fi

time_file2="$time_log2/current_time_02.txt"


process_data() {
  local input_dir="$1"
  local output_dir="$2"
  local start_time="$3"
  local end_time="$4"
  local model_input_script="$5"
  local python_output="$6"

  $pypath "$model_input_script" \
    "$start_time" \
    "$end_time" \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" > "$python_output" 2>&1

  if [ $? -ne 0 ]; then
    echo "Error processing files in $input_dir. Check $python_output for details."
    return 1
  fi

  # Check if the log contains "saved data to"
  if grep -q "Saved data to" "$python_output"; then
    return 0  # Success
  elif grep -q "Data already exists" "$python_output"; then
    return 0
  else
    return 1  # Failure
  fi
}


input_dir=$temp_nc_path
model_input_path=$model_input_path
model_input_script2=$model_input_scrip
error_log_file="$time_log2/error_log_file_02.txt"
error_count=0
max_errors=6
if [ -f "$time_file2" ]; then
  current_time_rounded=$(cat "$time_file2")
  echo "Current Time: $current_time_rounded"

  start_time=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d %H%M")  
  echo "Model input Start Time: $start_time"
  end_time=$(date -d "$current_time_rounded - $end_min min" +"%Y%m%d %H%M")
  echo "Model input end Time: $(date -d "$current_time_rounded - 30 min" +"%Y%m%d %H%M")"
  python_output_02="$time_log2/python_output_02.log"

  file_dir=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d")
  if process_data "$input_dir" "$model_input_path" "$start_time" "$end_time" "$model_input_script" "$python_output_02"; then
  next_time=$(date -d "$end_time + 20 min" +"%Y%m%d %H%M")
  echo "Next Start Time: $next_time"
  echo "$next_time" > "$time_file2"
  else

    echo "Error occurred, not writing to time_file"
    echo "Error occurred between $start_time and $end_time" >> "$error_log_file"
    ((error_count++))
    error_count2=$(grep "Error occurred between $start_time and $end_time" "$error_log_file" | wc -l)
    if [ $error_count2 -ge $max_errors ]; then
      # 
      echo "Error count reached maximum. Executing Python copy script..."
      
      # 
      process_data "$input_dir" "$model_input_path" "$start_time" "$end_time" "$model_input_script2" "$python_output_02"
      
      
      # 
      error_count=0
    fi
  fi
  
else
  echo "Using Default Time: $current_time"
  current_minute=$(date -d "$current_time" +%M)
  current_minute_rounded=$(( (current_minute ) / 10 * 10 ))
  current_minute_formatted=$(printf "%02d" "$current_minute_rounded")
  current_time_rounded=$(date -d "$current_time" +"%Y%m%d %H${current_minute_formatted}")
  start_time=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d %H%M")
  file_dir=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d")
  echo "Model input Start Time: $start_time"
  end_time=$(date -d "$current_time_rounded - $end_min min" +"%Y%m%d %H%M")
  echo "Model input end Time: $(date -d "$current_time_rounded - 30 min" +"%Y%m%d %H%M")"
  python_output_02="$time_log2/python_output_02.log"
  if process_data "$input_dir" "$model_input_path" "$start_time" "$end_time" "$model_input_script" "$python_output_02"; then
  next_time=$(date -d "$end_time + 20 min" +"%Y%m%d %H%M")
  echo "Next Start Time: $next_time"
  echo "$next_time" > "$time_file2"
  else
    echo "Error occurred, not writing to time_file"
  fi
fi




