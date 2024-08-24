# -*- coding: utf-8 -*-

import netCDF4 as nc
import datetime
import argparse

def update_time_in_ncfile(input_file, new_time):
    try:
        # Open the NetCDF file in read-write mode
        with nc.Dataset(input_file, 'a') as dataset:
            # Assuming you have a 'time' variable in the NetCDF file
            if 'time' in dataset.variables:
                time_var = dataset.variables['time']
                # Convert the new time string to a datetime object
                new_time_obj = datetime.datetime.strptime(new_time, '%Y%m%d %H%M')
                # Convert the datetime object to the units used in the 'time' variable
                new_time_units = f"seconds since {new_time_obj.strftime('%Y-%m-%d %H:%M:%S')}"
                # Update the time variable's units and values
                time_var.units = new_time_units
                time_var[:] = [0.0]  # You may need to adjust this value depending on the units

            # Save the changes
            dataset.sync()
            print(f"Time in {input_file} updated to {new_time}")

    except Exception as e:
        print(f"Error updating time in {input_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Update time in a NetCDF file.")
    parser.add_argument("input_file", type=str, help="Path to the input NetCDF file")
    parser.add_argument("new_time", type=str, help="New time value in the format 'YYYYMMDD HHMM'")

    args = parser.parse_args()
    input_file = args.input_file
    new_time = args.new_time

    update_time_in_ncfile(input_file, new_time)

if __name__ == "__main__":
    main()
