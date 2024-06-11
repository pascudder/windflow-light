from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import netCDF4 as nc
from eco1280_loader import load_eco1280

# Load ECO1280 data
file1 = './data/uv_2016-06-01_00:00:00_P500_out.nc'
file2 = './data/uv_2016-06-01_03:00:00_P500_out.nc'
eco_u, eco_v, lat, _ = load_eco1280(file1, file2)

# Load windflow data (Run 'run_windflow.py' first)
print('loading windflow data')
with nc.Dataset('data.nc', 'r') as f:
    w_lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    gp_rad1 = f.variables['gp_rad1'][:]
    gp_rad2 = f.variables['gp_rad2'][:]
    w_u = f.variables['uwind'][:]
    w_v = f.variables['vwind'][:]
assert np.all(lat == w_lat)

# Calculate windspeed for ECO1280
ws_eco = np.sqrt(np.add(np.square(eco_u), np.square(eco_v)))
max_ws_eco = np.nanmax(ws_eco)
print("Eco maximum wind speed:", max_ws_eco)

# Calculate windspeed for Windflow
ws_windflow = np.sqrt(np.add(np.square(w_u), np.square(w_v)))
max_ws_windflow = np.nanmax(ws_windflow)
print("Windflow maximum wind speed:", max_ws_windflow)

# Plotting side by side
fig, axs = plt.subplots(2,1, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot ECO1280 data
c_inv = np.arange(0, 80, 1)
axs[0].coastlines(color='white')
contour_eco = axs[0].contourf(lon, lat, ws_eco, c_inv, transform=ccrs.PlateCarree(), cmap=plt.cm.jet)
cb_eco = fig.colorbar(contour_eco, ax=axs[0], orientation="vertical", pad=0.02, aspect=16, shrink=0.5)
cb_eco.set_label('m/s', size=10, rotation=0, labelpad=15)
cb_eco.ax.tick_params(labelsize=10)
qv_eco = axs[0].quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='k')
axs[0].set_title('ECO1280')

# Plot Windflow data
axs[1].coastlines(color='white')
contour_windflow = axs[1].contourf(lon, lat, ws_windflow, c_inv, transform=ccrs.PlateCarree(), cmap=plt.cm.jet)
cb_windflow = fig.colorbar(contour_windflow, ax=axs[1], orientation="vertical", pad=0.02, aspect=16, shrink=0.5)
cb_windflow.set_label('m/s', size=10, rotation=0, labelpad=15)
cb_windflow.ax.tick_params(labelsize=10)
qv_windflow = axs[1].quiver(lon[::60], lat[::40], w_u[::40, ::60], w_v[::40, ::60], color='k')
axs[1].set_title('Windflow displacement')

plt.tight_layout()
plt.savefig('combined_contour_quivers.png',bbox_inches='tight', dpi=300)
plt.show()
