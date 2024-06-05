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

#convert from pixels to m/s
expanded_lat = np.radians(np.tile(lat, (3600, 1)).T)
w_u = (w_u * 0.1 * 111 * 1000 * np.cos(expanded_lat)) / 10800 #EXPANDED_LAT IS IN RADIANS
w_v = (w_v * 0.1 * 111 * 1000) / 10800

#plot
projection = ccrs.PlateCarree()

print('plotting')
# ECO 1280 time 1
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Humidity kg kg-1')
ax.coastlines(color='white')
plt.title('ECO1280 Humidity - Snapshot 1')
plt.tight_layout()
#plt.savefig('humidity_quivers_eco1.png', dpi=300)
plt.close()

# ECO1280 time 2
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink = 0.7, label='Humidity kg kg-1')
plt.title('ECO1280 Humidity - Snapshot 2')
ax.coastlines(color='white')
plt.tight_layout()
#plt.savefig('humidity_quivers_eco2.png', dpi=300)
plt.close()


# windflow time 1
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], w_u[::40, ::60], w_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, shrink=0.7, ax=ax, label='Humidity kg kg-1')
plt.title('Windflow Humidity - Snapshot 1')
ax.coastlines(color='white')
plt.tight_layout()
#plt.savefig('humidity_quivers_windflow.png', dpi=300)
plt.close()

# windflow time 2
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], w_u[::40, ::60], w_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink = 0.7, label='Humidity kg kg-1')
plt.title('Windflow Humidity - Snapshot 2')
ax.coastlines(color='white')
plt.tight_layout()
#plt.savefig('humidity_quivers_windflow2.png', dpi=300)
plt.close()


'''#barbs
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap='Blues_r')
bv = ax.barbs(lon[::100], lat[::80], W_u[::80, ::100], -W_v[::80, ::100], color='black', length=4, linewidth=0.8)
cbar = plt.colorbar(im, ax=ax, label='Humidity kg kg-1')
plt.title('Windflow Humidity')
ax.coastlines(color='white')
plt.tight_layout()
plt.savefig('humidity_barbs_windflow1_blues.png', dpi=300)
# plt.show()
plt.close()

fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap='Blues_r')
bv = ax.barbs(lon[::100], lat[::80], W_u[::80, ::100], -W_v[::80, ::100], color='black', length=4, linewidth=0.8)
cbar = plt.colorbar(im, ax=ax, label='Humidity kg kg-1')
plt.title('Windflow Humidity')
ax.coastlines(color='white')
plt.tight_layout()
plt.savefig('humidity_barbs_windflow2_blues.png', dpi=300)
# plt.show()
plt.close()
'''
