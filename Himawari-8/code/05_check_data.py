import xarray as xr
import argparse
import os

def check_data_quality(filename):
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
            print('file is ok')
            return True
        else:
            print('bad file, file is removed')
            os.remove(filename)
            return False
    except:
        os.remove(filename)
        return False

parser = argparse.ArgumentParser()


parser.add_argument("--check_path", type=str, help="Path to the root directory of input data")

args = parser.parse_args()
check_path = args.check_path
check_data_quality(check_path)
