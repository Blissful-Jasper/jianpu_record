#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Jianpu)s

@email: xianpuji@hhu.edu.cn
"""


# import os
# import re
# import subprocess



# def decompress_file( source_path, unzip_file_path):
#         """
#         Decompresses a .bz2 file to a target path.

#         Parameters:
#             source_path (str): Path to the source .bz2 file.
#             unzip_file_path (str): Path to the target decompressed file.
#         """
#         try:
#             with open(unzip_file_path, 'wb') as output_file:
#                 subprocess.run(["/DatadiskExt/xpji/jxp_deeplearn_model_code/bin/bzip2", '-d', '-k', '-c', source_path],
#                                check=True, shell=False, stdout=output_file)
#         except subprocess.CalledProcessError as e:
#             print(f"Error decompressing {source_path}: {e}")

# def run_conversion_command( cmd, output_file_path):
#         """
#         Runs the conversion command using subprocess.

#         Parameters:
#             cmd (list): List containing the command and arguments.
#             output_file_path (str): Path to the output NetCDF file.
#         """
#         try:
#             subprocess.run(cmd, check=True)
#         except subprocess.CalledProcessError:
#             pass


# file_path   =  r"/DatadiskExt/xpji/hia9/20230621/0000/Z_SATE_C_RJTD_20230621001239_HS_H09_20230621_0000_B01_FLDK_R10_S0110.DAT.bz2"

# bin_path    = r"/DatadiskExt/xpji/jxp_deeplearn_model_code/bin/"

# unzip_file_path = r'/DatadiskExt/xpji/hia9/20230621/0000/Z_SATE_C_RJTD_20230621001239_HS_H09_20230621_0000_B01_FLDK_R10_S0110.DAT'


# output_file_path = r'/DatadiskExt/xpji/HS_H09_20230621_0000_B01_FLDK_R10_S0110.nc'

# decompress_file(file_path, unzip_file_path)

# file_basename = os.path.basename(unzip_file_path)

# output_file_name = re.search(r"HS_(.*)", file_basename).group(0).replace(".DAT", ".nc")

# band_number = re.search(r"B(\d{2})", output_file_name).group(0)

# bands = ["B08", "B09", "B10", "B11", "B13", "B14", "B16"]
# lat_start = 0  # Starting latitude
# lat_end = 45  # Ending latitude
# lon_start = 100  # Starting longitude
# lon_end = 145  # Ending longitude
# grid_scale = 0.1  # Grid scale


# cmd = ["/DatadiskExt/xpji/jxp_deeplearn_model_code/bin/hisd2/hisd2netcdf",
#                                 "-width", "451",
#                                 "-height", "451",
#                                 "-lat", str(lat_end),
#                                 "-lon", str(lon_start),
#                                 "-dlat", str(grid_scale),
#                                 "-dlon", str(grid_scale),
#                                 "-i", unzip_file_path,
#                                 "-o", output_file_path]
                      
# run_conversion_command(cmd, output_file_path)


import os
import re
import subprocess

def decompress_file(source_path: str, unzip_file_path: str) -> None:
    """
    Decompresses a .bz2 file to a target path.

    Parameters:
        source_path (str): Path to the source .bz2 file.
        unzip_file_path (str): Path to the target decompressed file.
    """
    try:
        # Use subprocess to call bzip2 for decompression
        subprocess.run(
            ["bzip2", "-d", "-k", "-c", source_path],
            check=True,
            stdout=open(unzip_file_path, 'wb')
        )
    except subprocess.CalledProcessError as e:
        print(f"Error decompressing {source_path}: {e}")

def run_conversion_command(cmd: list[str]) -> None:
    """
    Runs a conversion command using subprocess.

    Parameters:
        cmd (list): List containing the command and arguments.
    """
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Error running command: {' '.join(cmd)}")

def main():
    # Paths and parameters
    file_path = r"/DatadiskExt/xpji/hia9/20230621/0000/Z_SATE_C_RJTD_20230621001239_HS_H09_20230621_0000_B01_FLDK_R10_S0110.DAT.bz2"
    unzip_file_path = r"/DatadiskExt/xpji/hia9/20230621/0000/Z_SATE_C_RJTD_20230621001239_HS_H09_20230621_0000_B01_FLDK_R10_S0110.DAT"
    output_file_path = r"/DatadiskExt/xpji/HS_H09_20230621_0000_B01_FLDK_R10_S0110.nc"

    # Decompress the file
    decompress_file(file_path, unzip_file_path)

    # Determine output file name and extract band number
    file_basename = os.path.basename(unzip_file_path)
    output_file_name = re.search(r"HS_(.*)", file_basename).group(0).replace(".DAT", ".nc")
    band_number = re.search(r"B(\d{2})", output_file_name).group(0)

    # Define conversion parameters
    bands = ["B08", "B09", "B10", "B11", "B13", "B14", "B16"]
    lat_start = 0
    lat_end = 45
    lon_start = 100
    lon_end = 145
    grid_scale = 0.1

    # Conversion command
    cmd = [
        "/DatadiskExt/xpji/jxp_deeplearn_model_code/bin/hisd2/hisd2netcdf",
        "-width", "451",
        "-height", "451",
        "-lat", str(lat_end),
        "-lon", str(lon_start),
        "-dlat", str(grid_scale),
        "-dlon", str(grid_scale),
        "-i", unzip_file_path,
        "-o", output_file_path
    ]

    # Run conversion command
    run_conversion_command(cmd)

if __name__ == "__main__":
    main()








