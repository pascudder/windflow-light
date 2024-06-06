from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from eco1280_loader import load_eco1280
import netCDF4 as nc

def calculate_rmse(x, y):
    return np.sqrt(np.nanmean((y - x) ** 2))

def main():

    # Load ECO1280 data
    file1 = './data/uv_2016-06-01_00:00:00_P500_out.nc' 
    file2 = './data/uv_2016-06-01_03:00:00_P500_out.nc'
    eco_u, eco_v, lat, lon = load_eco1280(file1, file2)

    # Load windflow data (Run 'run_windflow.py' first)
    print('loading windflow data')
    with nc.Dataset('data.nc', 'r') as f:
        w_lat = f.variables['lat'][:]
        w_lon = f.variables['lon'][:]
        gp_rad1 = f.variables['gp_rad1'][:]
        gp_rad2 = f.variables['gp_rad2'][:]
        w_u = f.variables['uwind'][:]
        w_v = f.variables['vwind'][:]
    assert np.all(lat == w_lat)

    expanded_lat = np.tile(lat, (3600, 1)).T
    mask = (expanded_lat <= 90) & (expanded_lat >= -90)  # Mask the region of interest
    lat_mask = np.radians(expanded_lat[mask])  # Convert latitude from degrees to radians

    # Select masked regions
    eu_mask = eco_u[mask]
    ev_mask = eco_v[mask]

    wu_mask = w_u[mask]
    wv_mask = w_v[mask]

    # Testing different multiplicative scaling values for wu_mask
    multiplicative_factors = np.arange(0.1, 2.1, 0.01)  # Smaller iterations for multiplicative factors
    best_multiplicative_rmse = float('inf')
    best_multiplicative_factor = None
    for scale in multiplicative_factors:
        scaled_wu_mask = (wu_mask * 0.1 * 111 * 1000 * np.cos(lat_mask)*scale) / 10800
        rmse_u = calculate_rmse(eu_mask, scaled_wu_mask)
        print(f'RMSE for multiplicative scaling factor {scale}: {rmse_u}')
        if rmse_u < best_multiplicative_rmse:
            best_multiplicative_rmse = rmse_u
            best_multiplicative_factor = scale
    print(f'Best multiplicative scaling factor: {best_multiplicative_factor} with RMSE: {best_multiplicative_rmse}')
   
   # Testing different additive scaling values for wu_mask
    additive_factors = np.arange(-1.0, 1.1, 0.05)  # Smaller iterations for additive factors
    best_additive_rmse = float('inf')
    best_additive_factor = None

    for add in additive_factors:
        scaled_wu_mask = (wu_mask * 111 * 1000*0.1* np.cos(lat_mask)) / 10800 + add
        rmse_u = calculate_rmse(eu_mask, scaled_wu_mask)
        print(f'RMSE for additive scaling factor {add}: {rmse_u}')
        if rmse_u < best_additive_rmse:
            best_additive_rmse = rmse_u
            best_additive_factor = add

    print(f'Best additive scaling factor: {best_additive_factor} with RMSE: {best_additive_rmse}')
    
    # Testing different multiplicative scaling values for wv_mask
    multiplicative_factors = np.arange(0.1, 2.1, 0.01)  # Smaller iterations for multiplicative factors
    best_multiplicative_rmse = float('inf')
    best_multiplicative_factor = None

    for scale in multiplicative_factors:
        scaled_wv_mask = (wv_mask * scale * 111 * 1000*0.1) / 10800
        rmse_v = calculate_rmse(ev_mask, scaled_wv_mask)
        print(f'RMSE for multiplicative scaling factor {scale}: {rmse_v}')
        if rmse_v < best_multiplicative_rmse:
            best_multiplicative_rmse = rmse_v
            best_multiplicative_factor = scale

    print(f'Best multiplicative scaling factor: {best_multiplicative_factor} with RMSE: {best_multiplicative_rmse}')

    # Testing different additive scaling values for wv_mask
    additive_factors = np.arange(-1.0, 1.1, 0.05)  # Smaller iterations for additive factors
    best_additive_rmse = float('inf')
    best_additive_factor = None

    for add in additive_factors:
        scaled_wv_mask = (wv_mask * 111 * 1000*0.1) / 10800 + add
        rmse_v = calculate_rmse(ev_mask, scaled_wv_mask)
        print(f'RMSE for additive scaling factor {add}: {rmse_v}')
        if rmse_v < best_additive_rmse:
            best_additive_rmse = rmse_v
            best_additive_factor = add

    print(f'Best additive scaling factor: {best_additive_factor} with RMSE: {best_additive_rmse}')


if __name__ == '__main__':
    main()
