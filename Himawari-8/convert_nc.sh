#!/bin/bash
# source /home/gaby/.bashrc
# export LD_LIBRARY_PATH=/home/gaby/libraries/netcdf_4.2.1.1/lib:$LD_LIBRARY_PATH

source "/DatadiskExt/xpji/jxp_deeplearn_model_code/cf1.sh"

if [ ! -d "$time_log" ]; then
  mkdir -p "$time_log"
fi
if [ ! -d "$temp_path" ]; then
  mkdir -p "$temp_path"
fi
if [ ! -d "$temp_nc_path" ]; then
  mkdir -p "$temp_nc_path"
fi
time_file="$time_log/current_time.txt"
python_output="$time_log/python_output.log"

#--------------------------------------------------------------------------------------------------------------------------------------------
# Function to check and delete files smaller than 2000KB
#--------------------------------------------------------------------------------------------------------------------------------------------
# Function to copy data from the previous time
copy_from_previous() {
    local prev_time_str="$1"
    local prev_output_file="$2"
    cp "$prev_output_file" "$output_file"
    $pypath $time_up_script "$output_file" "${date_str} ${time_str}"
    $pypath $check_scipt \
        --check_path "$output_file"
    
}


#--------------------------------------------------------------------------------------------------------------------------------------------
# Function to process a folder
process_folder() {
    local folder_path=$1
    local output_file="${temp_nc_path}/${date_str}/NC_HS_H09_${date_str}_${time_str}_FLDK_R100.nc"
    local error_counter=0   # Counter for consecutive 'Both conditions met, deleting the file.'
    local log_file="$time_log/error_log.txt" 
    if [[ -d $folder_path ]]; then
        # if folder path exist , run convert python code 
        echo "process $folder_path files..."
        echo "check file exist or not"
        if [  -f "$output_file" ]; then
            # if folder compare file exist, check the file
            echo "$output_file exist, checking data."
            file_size=$(du -k "$output_file" | cut -f1)
            if [ "$file_size" -lt 5000 ]; then
                echo " $file  smaller than 5000KB"
                error_counter=$((error_counter + 1))  # 
                echo "error_counter: $error_counter"
                echo "$date_str $time_str $current_date_time" >> "$log_file"
                rm -rf "$output_file"  # Delete problematic output file
            else
                echo "$output_file good"
            fi
        else
            # if folder exist, but compare file not exist, python running
            $pypath "$convert_script" \
                --root_path "$1" \
                --temp_path "$temp_path" \
                --temp_nc_path "$temp_nc_path" \
                --bin_path "$bin_path" \
                > "$python_output" 2>&1

            if grep -q "all files haved converted finish" "$python_output" && grep -q "Error" "$python_output"; then
                echo 'converting, but not finished competely.'
                error_counter=$((error_counter + 1))  # 
                
                echo "$date_str $time_str $current_date_time" >> "$log_file"
                rm -rf "$output_file"  # Delete problematic output file
            elif grep -q "all files haved converted finish" "$python_output"; then
                echo 'Convert finish,checking data'
                file_size=$(du -k "$output_file" | cut -f1)
                if [ "$file_size" -lt 5000 ]; then
                    echo " $file  smaller than 5000KB"
                    error_counter=$((error_counter + 1))  # 
                    echo "$date_str $time_str $current_date_time" >> "$log_file"
                    rm -rf "$output_file"  # Delete problematic output file
                else
                    echo "more than 5000kb"
                fi
                
            else
                echo 'Convert failed....'
                error_counter=$((error_counter + 1))  # 
                echo "$date_str $time_str $current_date_time" >> "$log_file"
                rm -rf "$output_file"  # Delete problematic output file
            fi
            
        fi    
    else
        if [  -f "$output_file" ]; then
            echo "folder $folder_path  not exist, but file has existed"
            file_size=$(du -k "$output_file" | cut -f1)
            if [ "$file_size" -lt 5000 ]; then
                echo " $file as it is smaller than 5000KB"
                error_counter=$((error_counter + 1))  # 
                echo "$date_str $time_str $current_date_time" >> "$log_file"
                rm -rf "$output_file"  # Delete problematic output file
            fi    
        else

            echo "folder $folder_path not exist, copying..."

            for ((i = 10; i <= 60; i += 10)); do
                prev_time_str=$(date -d "$time_str - $i minutes" "+%H%M")
                prev_date_str="$date_str"
                prev_date=$(date -d "$prev_date_str" "+%Y%m%d")  
                
                # 
                if [ "$(date -d "$time_str - $i minutes" "+%Y%m%d%H%M%S")" -ge "$(date -d "$date_str" "+%Y%m%d000000")" ]; then
                    prev_date_str="$date_str"  
                else
                    prev_date=$(date -d "$date_str - 1 day" "+%Y%m%d")  
                    prev_date_str=$(date -d "$prev_date" "+%Y%m%d")
                fi
                
                prev_output_file="${temp_nc_path}/${prev_date_str}/NC_HS_H09_${prev_date_str}_${prev_time_str}_FLDK_R100.nc"
                if [ -f "$prev_output_file" ]; then
                    echo "$prev_output_file found"
                    copy_from_previous "$prev_time_str" "$prev_output_file"
                    break
                else
                    echo "$prev_output_file not found."
                fi
            done
                
            
        fi    
    fi
    
    
    # Check if the same time has been recorded more than twice
    local count_same_time=$(grep -c "$date_str $time_str" "$log_file")
    if [ "$count_same_time" -gt 3 ]; then
        echo "The same time $date_str $time_str has been recorded more than twice."

        for ((i = 10; i <= 60; i += 10)); do
            prev_time_str=$(date -d "$time_str - $i minutes" "+%H%M")
            prev_date_str="$date_str"
            prev_date=$(date -d "$prev_date_str" "+%Y%m%d")  
            
            # 
            if [ "$(date -d "$time_str - $i minutes" "+%Y%m%d%H%M%S")" -ge "$(date -d "$date_str" "+%Y%m%d000000")" ]; then
                prev_date_str="$date_str"  
            else
                prev_date=$(date -d "$date_str - 1 day" "+%Y%m%d")  
                prev_date_str=$(date -d "$prev_date" "+%Y%m%d")
            fi
            now_output_file="${temp_nc_path}/${date_str}/NC_HS_H09_${date_str}_${time_str}_FLDK_R100.nc"
            prev_output_file="${temp_nc_path}/${prev_date_str}/NC_HS_H09_${prev_date_str}_${prev_time_str}_FLDK_R100.nc"
            if [ -f "$prev_output_file" ]; then
                echo "$prev_output_file found"
                if [ ! -f "$now_output_file" ]; then
                    copy_from_previous "$prev_time_str" "$prev_output_file"
                   
                else
                    echo "File already existing, skipping..."
                fi
                
                break
            else
                echo "$prev_output_file not found."
            fi
        done

        error_counter=0
    fi
}


if [ -f "$time_file" ]; then

    current_time_rounded=$(cat "$time_file")
    echo "Current Time: $current_time_rounded"

    start_time=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d %H%M")  
    echo "Model input Start Time: $start_time"
    end_time=$(date -d "$current_time_rounded - $end_min min" +"%Y%m%d %H%M")
    echo "Model input end Time: $(date -d "$current_time_rounded - 30 min" +"%Y%m%d %H%M")"
    end_time2=$(date -d "$end_time + $end_min min" +"%Y%m%d %H%M")
    
    while [[ "$start_time" < "$end_time2" ]]; do
        date_str=$(date -d "${start_time:0:8}" +"%Y%m%d")
        time_str="${start_time:9:12}"
        folder_name="${ori_input_dir}/${date_str}/${time_str}"
        
        if process_folder "$folder_name"; then
            echo "processing success"
        else
            echo "Error occurred"
        fi
        start_time=$(date -d "$start_time + 10 min" +"%Y%m%d %H%M")
       
    done
    if [ $? -eq 0 ]; then
        next_time=$(date -d "$current_time_rounded + 10 min" +"%Y%m%d %H%M")
        echo "$next_time" > "$time_file"
        echo "Next Start Time: $next_time"
    else
        echo "Error occurred, Next Start Time not updated"
    fi
    

else

    echo "Using Default Time: $current_time"
    current_minute=$(date -d "$current_time" +%M)
    current_minute_rounded=$(( (current_minute ) / 10 * 10 ))
    current_minute_formatted=$(printf "%02d" "$current_minute_rounded")
    current_time_rounded=$(date -d "$current_time" +"%Y%m%d %H${current_minute_formatted}")
    start_time=$(date -d "$current_time_rounded - $start_hour hour" +"%Y%m%d %H%M")
    end_time=$(date -d "$current_time_rounded - $end_min min" +"%Y%m%d %H%M")
    echo "Model input Start Time: $start_time"
    echo "Model input end Time: $(date -d "$current_time_rounded - 30 min" +"%Y%m%d %H%M")"
    echo "End Time: $end_time"
    end_time2=$(date -d "$end_time + $end_min min" +"%Y%m%d %H%M")

  
    while [[ "$start_time" < "$end_time2" ]]; do

        date_str=$(date -d "${start_time:0:8}" +"%Y%m%d")
        time_str="${start_time:9:12}"
        folder_name="${ori_input_dir}/${date_str}/${time_str}"
        
        if process_folder "$folder_name"; then
            echo "processing success"
        else
            echo "Error occurred"
        fi
        start_time=$(date -d "$start_time + 10 min" +"%Y%m%d %H%M")
    
    done

    next_time=$(date -d "$current_time_rounded + 10 min" +"%Y%m%d %H%M")
    echo "$next_time" > "$time_file"
    
    echo "Next Start Time: $next_time"
  
fi
