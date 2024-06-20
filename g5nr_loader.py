import xarray as xr
import numpy as np


def load_g5nr(file1, file2):
    """
    Load and process ECO-1280 data from two files.

    Parameters:
    - file1 (str):  U component
    - file2 (str):  V component

    Returns:
    - g5_u (numpy.ndarray): U-component of the vector field.
    - g5_v (numpy.ndarray): V-component of the vector field.
    - lat (numpy.ndarray): Latitude values.
    - lon (numpy.ndarray): Longitude values.
    """
    # Read UV data from files
    print('loading eco data')
    u_1 = xr.open_dataset(file1)
    v_1 = xr.open_dataset(file2)
    # Extract latitude and longitude values
    lat = u_1['lat'].values
    lon = u_1['lon'].values

    # Extract U and V components
    u_1 = u_1['U'][0, 5, :, :].values 
    v_1 = v_1['V'][0, 5, :, :].values

    return u_1, v_1, lat, lon