'''plot humidity in plate carre projection with both windflow and ECO1280 data, overlayed with wind vectors / barbs'''

from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import netCDF4 as nc
from eco1280_loader import load_eco1280

#Load ECO1280 data
file1 = './data/uv_2016-06-01_00:00:00_P500_out.nc' 
file2 = './data/uv_2016-06-01_03:00:00_P500_out.nc'
eco_u, eco_v, lat, _ = load_eco1280(file1, file2)

#Load windflow data (Run 'run_windflow.py' first)
print('loading windflow data')
with nc.Dataset('data.nc', 'r') as f:
    w_lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    gp_rad1 = f.variables['gp_rad1'][:]
    gp_rad2 = f.variables['gp_rad2'][:]
    w_u = f.variables['uwind'][:]
    w_v = f.variables['vwind'][:]
assert np.all(lat == w_lat)

expanded_lat = np.tile(lat, (3600,1)).T
mask = (expanded_lat <=90) & (expanded_lat >= -90) # mask the region of interest
lat_mask = np.radians(expanded_lat)  # convert latitude from degrees to radians

#unit conversion to m/s
wu_mask = (w_u * 0.1 * 111 * 1000 * np.cos(lat_mask)) / 10800 #Could add a *1.12 scaling term to minimize RMSE.
wv_mask = (w_v * 0.1 * 111 * 1000 * 0.69) / 10800 #The 0.69 multiplicative scaling vector was added after empirical testing
                                                          #Minimizes RMSE but theoretically it shouldn't be needed.

#plot
projection = ccrs.PlateCarree()

print('plotting')
# ECO 1280 time 1
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap=plt.cm.jet)
qv = plt.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Humidity kg kg-1')
ax.coastlines(color='white')
plt.title('ECO1280 Humidity - Snapshot 1')
plt.tight_layout()
#plt.savefig('humidity_quivers_eco1.png',bbox_inches='tight', dpi=300)
plt.close()

# ECO1280 time 2
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap=plt.cm.jet)
qv = plt.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink = 0.7, label='Humidity kg kg-1')
plt.title('ECO1280 Humidity - Snapshot 2')
ax.coastlines(color='white')
plt.tight_layout()
plt.savefig('humidity_quivers_eco2.png',bbox_inches='tight', dpi=300)
plt.close()


# windflow time 1
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap=plt.cm.jet)
qv = plt.quiver(lon[::60], lat[::40], wu_mask[::40, ::60], wv_mask[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, shrink=0.7, ax=ax, label='Humidity kg kg-1')
plt.title('Windflow Humidity - Snapshot 1')
ax.coastlines(color='white')
plt.tight_layout()
#plt.savefig('humidity_quivers_windflow.png',bbox_inches='tight', dpi=300)
plt.close()

# windflow time 2
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap=plt.cm.jet)
qv = plt.quiver(lon[::60], lat[::40], wu_mask[::40, ::60], wv_mask[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink = 0.7, label='Humidity kg kg-1')
plt.title('Windflow Humidity - Snapshot 2')
ax.coastlines(color='white')
plt.tight_layout()
plt.savefig('humidity_quivers_windflow2.png', bbox_inches='tight', dpi=300)
plt.close()

