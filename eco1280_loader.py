import xarray as xr

def load_eco1280(file1, file2):
    """
    Load and process ECO-1280 data from two files.

    Parameters:
    - file1 (str): Path to the first ECO-1280 data file.
    - file2 (str): Path to the second ECO-1280 data file.

    Returns:
    - eco_u (numpy.ndarray): U-component of the vector field.
    - eco_v (numpy.ndarray): V-component of the vector field.
    - lat (numpy.ndarray): Latitude values.
    - lon (numpy.ndarray): Longitude values.
    """
    # Read UV data from files
    print('loading eco data')
    uv_1 = xr.open_dataset(file1)
    uv_2 = xr.open_dataset(file2)

    # Extract U and V components
    u1 = uv_1['ugrd_newP']
    v1 = uv_1['vgrd_newP']

    # Remove the third dimension
    eco_u = u1.values.reshape((1801, 3600))
    eco_v = v1.values.reshape((1801, 3600))

    # Extract latitude and longitude values
    lat = uv_1['lat_0'].values
    lon = uv_1['lon_0'].values

    return eco_u, eco_v, lat, lon