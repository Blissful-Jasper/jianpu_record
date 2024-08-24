#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:31:47 2023

@author: xpji
"""
import argparse
import os
import re
import subprocess
import json
import xarray as xr
import numpy as np
import shutil
import time
from datetime import datetime

class DataConverter:
    def __init__(self, root_path, temp_path, temp_nc_path, bin_path):
        """
        Initialize the DataConverter object.

        Parameters:
            root_path (str): Path to the root directory of input data.
            temp_path (str): Path to the temporary directory for intermediate files.
            temp_nc_path (str): Path to the directory for storing final NetCDF files.
            bin_path (str): Path to the directory containing bzip2 and hisd2netcdf binaries.
        """
        self.root_path = root_path
        self.temp_path = temp_path
        self.temp_nc_path = temp_nc_path
        self.bin_path = bin_path
        self.last_root_path = None

    def clear_temp_dir(self):
        """
        Clear the temporary directory and recreate it if the root path has changed.
        """
        if self.root_path != self.last_root_path:
            shutil.rmtree(self.temp_path)
            os.makedirs(self.temp_path)
            self.last_root_path = self.root_path

    def decompress_file(self, source_path, unzip_file_path):
        """
        Decompresses a .bz2 file to a target path.

        Parameters:
            source_path (str): Path to the source .bz2 file.
            unzip_file_path (str): Path to the target decompressed file.
        """
        try:
            with open(unzip_file_path, 'wb') as output_file:
                subprocess.run([os.path.join(self.bin_path.strip(), 'bzip2'), '-d', '-k', '-c', source_path],
                               check=True, shell=False, stdout=output_file)
        except subprocess.CalledProcessError as e:
            print(f"Error decompressing {source_path}: {e}")

    def run_conversion_command(self, cmd, output_file_path):
        """
        Runs the conversion command using subprocess.

        Parameters:
            cmd (list): List containing the command and arguments.
            output_file_path (str): Path to the output NetCDF file.
        """
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            pass

    def convert_data(self, output_file_path, lat_end, lon_start, grid_scale, bands):
        """
        Converts source files to NetCDF format.

        Parameters:
            output_dir (str): Directory where the output NetCDF files will be saved.
            lat_end (float): Ending latitude for conversion.
            lon_start (float): Starting longitude for conversion.
            grid_scale (float): Grid scale for conversion.
            bands (list): List of bands to be included in the conversion.
        """
        if not os.path.exists(self.root_path):
            return

        for root, _, _ in os.walk(self.root_path):
            datas = sorted(os.listdir(root))

            if not datas:
                continue

            for data_name in datas:
                file_path = os.path.join(root, data_name)

                if file_path.endswith(".bz2") and "R20" in file_path:
                    unzip_file_path = os.path.join(self.temp_path, data_name.replace(".bz2", ""))

                    if os.path.exists(unzip_file_path):
                        try:
                            with open(unzip_file_path, 'rb'):
                                pass
                        except IOError:
                            os.remove(unzip_file_path)
                        else:
                            continue

                    self.decompress_file(file_path, unzip_file_path)

                    if unzip_file_path.endswith(".DAT") and not unzip_file_path.endswith(".nc"):
                        file_p = os.path.basename(unzip_file_path)
                        output_file_name = re.search(r"HS_(.*)", file_p).group(0).replace(".DAT", ".nc")

                        band_number = re.search(r"B(\d{2})", output_file_name).group(0)
                        output_file_path = os.path.join(self.temp_path, output_file_name)

                        if band_number in bands and not os.path.exists(output_file_path):
                            cmd = [
                                os.path.join(self.bin_path.strip(), "hisd2/hisd2netcdf"),
                                "-width", "451",
                                "-height", "451",
                                "-lat", str(lat_end),
                                "-lon", str(lon_start),
                                "-dlat", str(grid_scale),
                                "-dlon", str(grid_scale),
                                "-i", unzip_file_path,
                                "-o", output_file_path
                            ]
                            self.run_conversion_command(cmd, output_file_path)

    def check_data_quality(self, filename):
        """
        Check the quality of a NetCDF file and remove it if quality criteria are not met.

        Parameters:
            filename (str): Path to the NetCDF file to be checked.

        Returns:
            bool: True if the file meets quality criteria, False otherwise.
        """
        try:
            ds = xr.open_dataset(filename)
            target_vars = ['tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_13', 'tbb_14', 'tbb_16']
            if all(var in ds.data_vars for var in target_vars):
                ds.close()
                return True
            else:
                os.remove(filename)
                return False
        except:
            os.remove(filename)
            return False

    def combine_nc_files(self, bands):
        """
        Combine NetCDF files and save the combined data to target directory.

        Parameters:
            output_dir (str): Directory where the combined NetCDF files will be saved.
            bands (list): List of bands to be included in the combined files.
            temp_nc_path (str): Path to the directory for storing final NetCDF files.
        """
        for root, _, _ in os.walk(self.temp_path):
            datas = sorted(os.listdir(root))
            filtered_files = [os.path.join(root, data_name) for data_name in datas if
                              data_name.endswith('.nc') and not data_name.startswith('NC')]

            prefix_dict = {}
            for file_path in filtered_files:
                prefix = "_".join(os.path.splitext(os.path.basename(file_path))[0].split('_')[:-4])

                if prefix in prefix_dict:
                    prefix_dict[prefix].append(file_path)
                else:
                    prefix_dict[prefix] = [file_path]

            for key, prefix_files in prefix_dict.items():
                concatenated_data = {}

                for band in bands:
                    band_filtered_files = sorted(list(filter(lambda data_name: f"_{band}_" in data_name, prefix_files)))
                    if len(band_filtered_files) > 0:
                        region_data = None
                        for file_nc in band_filtered_files:
                            data = xr.open_dataset(file_nc).sortby('latitude')
                            if region_data is None:
                                region_data = xr.DataArray(data['tbb'], coords=data.coords)
                            else:
                                region_data = region_data.combine_first(data['tbb'])
                            data.close()

                        concatenated_data['tbb_' + f"{band}"[1:]] = region_data
                        time_str = re.search(r"2(.*)", key).group(0)
                        time = datetime.strptime(time_str, '%Y%m%d_%H%M')

                output_nc_path = os.path.join(self.temp_path, 'NC_' + "_".join(
                    os.path.basename(file_nc).split("_")[0:4]) + "_FLDK_R100.nc")

                if not os.path.exists(output_nc_path):
                    combined_dataset = xr.Dataset(concatenated_data, coords={'time': time})
                    combined_dataset.to_netcdf(output_nc_path)

                output_date_dir = os.path.join(self.temp_nc_path, os.path.basename(file_nc).split("_")[2]) 
                if not os.path.exists(output_date_dir):
                    os.makedirs(output_date_dir)

                target_path = os.path.join(output_date_dir, 'NC_' + "_".join(
                    os.path.basename(file_nc).split("_")[0:4]) + "_FLDK_R100.nc")

                if not os.path.exists(target_path):
                    shutil.copyfile(output_nc_path, target_path)
               

def main(root_path, temp_path, temp_nc_path, bin_path):
    """
    Main function to execute the data conversion process.

    Parameters:
        root_path (str): Path to the root directory of input data.
        temp_path (str): Path to the temporary directory for intermediate files.
        temp_nc_path (str): Path to the directory for storing final NetCDF files.
        bin_path (str): Path to the directory containing bzip2 and hisd2netcdf binaries.
    """

    converter = DataConverter(root_path, temp_path, temp_nc_path, bin_path)

    # Clear the temporary directory
    converter.clear_temp_dir()

    # Define the conversion parameters
    bands = ["B08", "B09", "B10", "B11", "B13", "B14", "B16"]
    lat_start = 0  # Starting latitude
    lat_end = 45  # Ending latitude
    lon_start = 100  # Starting longitude
    lon_end = 145  # Ending longitude
    grid_scale = 0.1  # Grid scale

    # Perform data conversion
    converter.convert_data(output_file_path=temp_path, lat_end=lat_end, lon_start=lon_start, grid_scale=grid_scale,
                           bands=bands)

    # Combine NetCDF files
    converter.combine_nc_files( bands=bands)
    print('all files haved converted finish')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, help="Path to the root directory of input data")
    parser.add_argument("--temp_path", type=str, help="Path to the temporary directory for intermediate files")
    parser.add_argument("--temp_nc_path", type=str, help="Path to the directory for storing final NetCDF files")
    parser.add_argument("--bin_path", type=str, help="Path to the directory containing bzip2 and hisd2netcdf binaries")
    args = parser.parse_args()

    root_path = args.root_path
    temp_path = args.temp_path
    temp_nc_path = args.temp_nc_path
    bin_path = args.bin_path

    main(root_path, temp_path, temp_nc_path, bin_path)




















