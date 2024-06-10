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

expanded_lat = np.tile(lat, (3600,1)).T
mask = (expanded_lat <=90) & (expanded_lat >= -90) # mask the region of interest
lat_mask = np.radians(expanded_lat)  # convert latitude from degrees to radians

#unit conversion to m/s
wu_mask = (w_u * 0.1 * 111 * 1000 * np.cos(lat_mask)) / 10800 #Could add a *1.12 scaling term to minimize RMSE.
wv_mask = (w_v * 0.1 * 111 * 1000) / 10800 #The 0.69 multiplicative scaling vector was added after empirical testing
                                                        #Minimizes RMSE but theoretically it shouldn't be needed.

# Plot
projection = ccrs.PlateCarree()

print('plotting')
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': projection}, figsize=(20, 12))

# Combined plot time 1
ax = axes[0]
im = ax.pcolormesh(lon, lat, gp_rad1, transform=projection, cmap=plt.cm.jet,vmax=140)
qv_eco = ax.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015, label='ECO1280')
qv_wind = ax.quiver(lon[::60], lat[::40], wu_mask[::40, ::60].data, wv_mask[::40, ::60].data, color='red', width=0.0007, label='Windflow')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Humidity kg kg-1')
ax.coastlines(color='white')
ax.set_title('Combined Quivers - Snapshot 1')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

# Combined plot time 2
ax = axes[1]
im = ax.pcolormesh(lon, lat, gp_rad2, transform=projection, cmap=plt.cm.jet,vmax=140)
qv_eco = ax.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='black', width=0.0015, label='ECO1280')
qv_wind = ax.quiver(lon[::60], lat[::40], wu_mask[::40, ::60], wv_mask[::40, ::60], color='red', width=0.0007, label='Windflow')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Humidity kg kg-1')
ax.coastlines(color='white')
ax.set_title('Combined Quivers - Snapshot 2')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

plt.tight_layout()
plt.savefig('combined_humidity_quivers.png', bbox_inches='tight', dpi=300)
plt.show()
