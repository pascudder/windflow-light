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

# Plot
projection = ccrs.PlateCarree()

print('plotting')
fig, axes = plt.subplots(2,1, subplot_kw={'projection': projection}, figsize=(20, 12))

# Combined plot time 1
ax = axes[0]
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap=plt.cm.jet,vmax=140)
qv_eco = ax.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015, label='ECO1280')
qv_wind = ax.quiver(lon[::60], lat[::40], w_u[::40, ::60].data, w_v[::40, ::60].data, color='red', width=0.0012, label='Windflow')
cbar = plt.colorbar(im, ax=ax, shrink=0.5, label='Humidity kg kg-1')
ax.coastlines(color='white')
ax.set_title('Combined Quivers - Snapshot 1')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

# Combined plot time 2
ax = axes[1]
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap=plt.cm.jet,vmax=140)
qv_eco = ax.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015, label='ECO1280')
qv_wind = ax.quiver(lon[::60], lat[::40], w_u[::40, ::60], w_v[::40, ::60], color='red', width=0.0012, label='Windflow')
cbar = plt.colorbar(im, ax=ax, shrink=0.5, label='Humidity kg kg-1')
ax.coastlines(color='white')
ax.set_title('Combined Quivers - Snapshot 2')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

plt.tight_layout()
plt.savefig('combined_humidity_quivers.png', bbox_inches='tight', dpi=300)
plt.show()
