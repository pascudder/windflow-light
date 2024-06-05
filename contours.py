from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import netCDF4 as nc

#Read Eco data
uv_1 = xr.open_dataset('./data/uv_2016-06-01_00:00:00_P500_out.nc')
uv_2 = xr.open_dataset('./data/uv_2016-06-01_03:00:00_P500_out.nc')

u1 = uv_1['ugrd_newP']
v1 = uv_1['vgrd_newP']

e_u = u1.values.reshape((1801, 3600))
e_v = v1.values.reshape((1801, 3600))

lat = uv_1['lat_0'].values
lon = uv_1['lon_0'].values

#calculate windspeed and plot
ws = (e_u**2 + e_v**2)**0.5
max_ws = np.nanmax(ws)
print("Maximum wind speed:", max_ws)
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
qv = plt.quiver(lon[::60], lat[::40], e_u[::40, ::60], e_v[::40, ::60], color='k')
plt.title('ECO1280')
plt.tight_layout()
plt.savefig('contour_quivers_eco.png',dpi = 300)

#read windflow data (Run 'run_windflow.py' first)
with nc.Dataset('data.nc', 'r') as f:
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    gp_rad1 = f.variables['gp_rad1'][:]
    gp_rad2 = f.variables['gp_rad2'][:]
    w_u = f.variables['uwind'][:]
    w_v = f.variables['vwind'][:]

#calculate windspeed and plot
ws = (w_u**2 + w_v**2)**0.5
max_ws = np.nanmax(ws)
print("Maximum wind speed:", max_ws)
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
