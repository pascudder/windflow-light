'''plot humidity in plate carre projection with both windflow and ECO1280 data, overlayed with wind vectors / barbs'''

from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import netCDF4 as nc

# read UV
print('reading UV')
uv_1 = xr.open_dataset('./data/uv_2016-06-01_00:00:00_P500_out.nc')
uv_2 = xr.open_dataset('./data/uv_2016-06-01_03:00:00_P500_out.nc')

u1 = uv_1['ugrd_newP']
v1 = uv_1['vgrd_newP']

Eco_u = u1.values.reshape((1801, 3600))
Eco_v = v1.values.reshape((1801, 3600))

lat = uv_1['lat_0'].values
lon = uv_1['lon_0'].values

with nc.Dataset('data.nc', 'r') as f:
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    gp_rad1 = f.variables['gp_rad1'][:]
    gp_rad2 = f.variables['gp_rad2'][:]
    W_u = f.variables['uwind'][:]
    W_v = f.variables['vwind'][:]

# plot stuff
projection = ccrs.PlateCarree()

print('plotting')
# ECO 1280 time 1
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], Eco_u[::40, ::60], Eco_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Humidity kg kg-1')
ax.coastlines(color='white')
plt.title('ECO1280 Humidity - Snapshot 1')
plt.tight_layout()
#plt.savefig('humidity_quivers_eco1_short.png', dpi=300)
plt.show()
plt.close()

# ECO1280 time 2
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], Eco_u[::40, ::60], Eco_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink = 0.7, label='Humidity kg kg-1')
plt.title('ECO1280 Humidity - Snapshot 2')
ax.coastlines(color='white')
plt.tight_layout()
#plt.savefig('humidity_quivers_eco2_short.png', dpi=300)
plt.show()
plt.close()


# windflow time 1
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], W_u[::40, ::60], -W_v[::40, ::60], color='black', width=0.0015)
print(gp_rad1)
cbar = plt.colorbar(im, shrink=0.7, ax=ax, label='Humidity kg kg-1')
plt.title('Windflow Humidity - Snapshot 1')
ax.coastlines(color='white')
plt.tight_layout()
#plt.savefig('humidity_quivers_windflow_Blues_short.png', dpi=200)
plt.show()
plt.close()

# windflow time 2
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7))
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap='Blues_r')
qv = plt.quiver(lon[::60], lat[::40], W_u[::40, ::60], -W_v[::40, ::60], color='black', width=0.0015)
cbar = plt.colorbar(im, ax=ax, shrink = 0.7, label='Humidity kg kg-1')
plt.title('Windflow Humidity - Snapshot 2')
ax.coastlines(color='white')
plt.tight_layout()
#plt.savefig('humidity_quivers_windflow2_Blues_short.png', dpi=200)
plt.show()
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
