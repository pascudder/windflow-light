import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from eco1280_loader import load_eco1280
import cartopy.crs as ccrs

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

expanded_lat = np.tile(lat, (3600, 1)).T
mask = (expanded_lat <= 90) & (expanded_lat >= -90) & ((gp_rad1 > 0) | (gp_rad2 > 0)) # Mask the region of interest, as well as the minimum humidity value.


lat_rad = np.radians(lat)
lon_rad = np.radians(lon)
a = np.cos(lat_rad)**2 * np.sin((lon_rad[1]-lon_rad[0])/2)**2
d = 2 * 6378.137 * np.arcsin(a**0.5)
size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1) # km
w_u = w_u * size_per_pixel * 1000/ 10800
w_v = w_v * size_per_pixel * 1000/ 10800

# Plot
projection = ccrs.PlateCarree()

print('plotting')
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': projection}, figsize=(20, 12))

# Combined plot time 1
ax = axes[0]
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap=plt.cm.jet,vmax=140)
qv_eco = ax.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015, label='ECO1280')
qv_wind = ax.quiver(lon[::60], lat[::40], w_u[::40, ::60].data, w_v[::40, ::60].data, color='red', width=0.0007, label='Windflow')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Humidity kg kg-1')
ax.coastlines(color='white')
ax.set_title('Combined Quivers - Snapshot 1')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

# Combined plot time 2
ax = axes[1]
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap=plt.cm.jet,vmax=140)
qv_eco = ax.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015, label='ECO1280')
qv_wind = ax.quiver(lon[::60], lat[::40], w_u[::40, ::60], w_v[::40, ::60], color='red', width=0.0007, label='Windflow')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Humidity kg kg-1')
ax.coastlines(color='white')
ax.set_title('Combined Quivers - Snapshot 2')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

plt.tight_layout()
plt.savefig('combined_humidity_quivers.png', bbox_inches='tight', dpi=300)
plt.show()
