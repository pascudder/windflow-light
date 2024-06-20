import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from eco1280_loader import load_eco1280

def load_windflow_data():
    print('loading windflow data')
    with nc.Dataset('data600.nc', 'r') as f:
        w_lat = f.variables['lat'][:]
        w_lon = f.variables['lon'][:]
        w_u = f.variables['uwind'][:]
        w_v = f.variables['vwind'][:]
    return w_lat, w_lon, w_u, w_v

def expand_lat_lon(lat, lon):
    expanded_lat = np.tile(lat, (lon.shape[0], 1)).T
    expanded_lon = np.tile(lon, (lat.shape[0], 1))
    return expanded_lat, expanded_lon

def filter_by_region(lat, lon, u, v, lat_range, lon_range):
    region_mask = (lat >= lat_range[0]) & (lat <= lat_range[1]) & (lon >= lon_range[0]) & (lon <= lon_range[1])
    return u[region_mask], v[region_mask]

def plot_histograms_by_region(eco_u, eco_v, w_u, w_v, lat, lon):
    regions = {
        'Region 1': {'lat_range': (-90, 0), 'lon_range': (0, 180)},
        'Region 2': {'lat_range': (0, 90), 'lon_range': (0, 180)},
        'Region 3': {'lat_range': (-90, 0), 'lon_range': (180, 360)},
        'Region 4': {'lat_range': (0, 90), 'lon_range': (180, 360)}
    }

    for region_name, ranges in regions.items():
        eco_u_region, eco_v_region = filter_by_region(lat, lon, eco_u, eco_v, ranges['lat_range'], ranges['lon_range'])
        w_u_region, w_v_region = filter_by_region(lat, lon, w_u, w_v, ranges['lat_range'], ranges['lon_range'])

        plt.figure(figsize=(12, 12))
        x_limits = (-60, 60)

        # Histogram for ECO1280 u component
        plt.subplot(2, 2, 1)
        plt.hist(eco_u_region.flatten(), bins=50, alpha=0.7, label='ECO1280 u Component')
        plt.xlabel('Wind u-component (m/s)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of ECO1280 u Component in {region_name}')
        plt.xlim(x_limits)
        plt.legend()

        # Histogram for ECO1280 v component
        plt.subplot(2, 2, 2)
        plt.hist(eco_v_region.flatten(), bins=50, alpha=0.7, color='green', label='ECO1280 v Component')
        plt.xlabel('Wind v-component (m/s)')
        plt.title(f'Histogram of ECO1280 v Component in {region_name}')
        plt.xlim(x_limits)
        plt.legend()

        # Histogram for Windflow u component
        plt.subplot(2, 2, 3)
        plt.hist(w_u_region.flatten(), bins=50, alpha=0.7, label='Windflow u Component')
        plt.xlabel('Wind u-component (m/s)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Windflow u Component in {region_name}')
        plt.xlim(x_limits)
        plt.legend()

        # Histogram for Windflow v component
        plt.subplot(2, 2, 4)
        plt.hist(w_v_region.flatten(), bins=50, alpha=0.7, color='red', label='Windflow v Component')
        plt.xlabel('Wind v-component (m/s)')
        plt.title(f'Histogram of Windflow v Component in {region_name}')
        plt.xlim(x_limits)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'histogram_components_{region_name}.png')
        plt.show()

def calculate_rmse(eco_u, w_u):
    return np.sqrt(np.mean((eco_u - w_u) ** 2))

def plot_rmse_by_longitude(eco_u, w_u, lon,title):
    rmses = []
    for i in range(len(lon)):
        rmse = calculate_rmse(eco_u[:, i], w_u[:, i])
        rmses.append(rmse)

    plt.figure(figsize=(12, 6))
    plt.plot(lon, rmses, label='RMSE')
    plt.xlabel('Longitude')
    plt.ylabel('RMSE (m/s)')
    plt.title('RMSE by Longitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'error_plots/{title}')

def plot_diff_by_longitude(eco, windflow, lon, title):
    plt.figure(figsize=(12, 6))
    
    # Plot eco_u
    eco_mean = np.nanmean(eco, axis=0)
    windflow_mean = np.nanmean(windflow, axis=0)
    plt.plot(lon, eco_mean, label='eco', color='blue')

    # Plot v_u
    plt.plot(lon, windflow_mean, label='windflow', color='red')
    
    plt.xlabel('Longitude')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed by Longitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'error_plots/{title}')

def plot_rmse_heatmap(x, y, lat, lon,title):
    rmse_matrix = np.sqrt((x - y) ** 2)
    
    plt.figure(figsize=(14, 8))
    plt.contourf(lon, lat, rmse_matrix, levels=100, cmap='viridis',vmax = 15)
    plt.colorbar(label='RMSE (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('RMSE Heatmap')
    plt.savefig(f'error_plots/{title}')

def main():
    # Load ECO1280 data
    file1 = './data/uv_2016-06-01_00:00:00_PX.nc' 
    file2 = './data/uv_2016-06-01_03:00:00_PX.nc'
    eco_u, eco_v, lat, lon = load_eco1280(file1, file2)

    # Load Windflow data
    w_lat, w_lon, w_u, w_v = load_windflow_data()

    # Check if latitudes and longitudes are the same (to ensure comparison is valid)
    assert np.all(lat == w_lat), "Latitude arrays do not match between datasets."

    # Expand lat and lon arrays
    expanded_lat, expanded_lon = expand_lat_lon(lat, lon)

        
    #RMSE chart
    plot_rmse_by_longitude(eco_u, w_u, lon,'u_lon_RMSE_600.png')
    plot_rmse_by_longitude(eco_v, w_v, lon,'v_lon_RMSE_600.png')
    plot_diff_by_longitude(eco_u, w_u, lon,'u_lon_diff_600.png')
    plot_diff_by_longitude(eco_v, w_v, lon,'v_lon_diff_600.png')

    plot_rmse_heatmap(eco_u, w_u, expanded_lat[:, 0], expanded_lon[0, :],'u_RMSE_600.png')
    plot_rmse_heatmap(eco_v, w_v, expanded_lat[:, 0], expanded_lon[0, :],'v_RMSE_600.png')
    eco_speed = (eco_u**2+eco_v**2)**0.5
    windflow_speed = (w_u**2+w_v**2)**0.5
    plot_rmse_heatmap(eco_speed, windflow_speed, expanded_lat[:, 0], expanded_lon[0, :],'speed_RMSE_600.png')




if __name__ == "__main__":
    main()
