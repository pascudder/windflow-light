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

#calculate windspeed and plot
ws = (eco_u**2 + eco_v**2)**0.5
max_ws = np.nanmax(ws)
print("Eco maximum wind speed:", max_ws)
c_inv = np.arange(0, 80, 1)
fig = plt.figure(figsize=(13,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(color='white')
plt.contourf(lon, lat, ws, c_inv, transform=ccrs.PlateCarree(),cmap=plt.cm.jet) # takes some time
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('m/s',size=10,rotation=0,labelpad=15)
cb.ax.tick_params(labelsize=10)

# there are too many u and v, we must splice
# qv = plt.quiver(lon, lat, U2M_nans[0,:,:], V2M_nans[0,:,:], scale=350, color='k')

#splice array:
qv = plt.quiver(lon[::60], lat[::40], eco_u[::40, ::60], eco_v[::40, ::60], color='k')
plt.title('ECO1280')
plt.tight_layout()
plt.savefig('contour_quivers_eco.png',dpi = 300)


expanded_lat = np.tile(lat, (3600,1)).T
mask = (expanded_lat <=90) & (expanded_lat >= -90) # mask the region of interest
lat_mask = np.radians(expanded_lat[mask])  # convert latitude from degrees to radians

#select masked regions
eu_mask = eco_u[mask]
ev_mask = eco_v[mask]

wu_mask = w_u[mask]
wv_mask = w_v[mask]

#unit conversion to m/s
wu_mask = (wu_mask * 0.1 * 111 * 1000 * np.cos(lat_mask)) / 10800
wv_mask = (wv_mask * 0.1 * 111 * 1000) / 10800

#calculate windspeed and plot
ws = (w_u**2 + w_v**2)**0.5
max_ws = np.nanmax(ws)
print("Windflow maximum wind speed:", max_ws)
c_inv = np.arange(0, 80, 1)
fig = plt.figure(figsize=(13,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(color='white')
plt.contourf(lon, lat, ws, c_inv, transform=ccrs.PlateCarree(),cmap=plt.cm.jet) # takes some time
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('m/s',size=10,rotation=0,labelpad=15)
cb.ax.tick_params(labelsize=10)

# splice array:
qv = plt.quiver(lon[::60], lat[::40], w_u[::40, ::60], w_v[::40, ::60], color='k')
plt.tight_layout()
plt.title('Windflow displacement')
plt.savefig('contour_quivers_windflow.png',dpi = 300)
