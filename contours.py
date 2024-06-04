from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs

uv_1 = xr.open_dataset('./data/uv_2016-06-01_00:00:00_P500_out.nc')
uv_2 = xr.open_dataset('./data/uv_2016-06-01_03:00:00_P500_out.nc')

u1 = uv_1['ugrd_newP']
v1 = uv_1['vgrd_newP']

u = u1.values.reshape((1801, 3600))
v = v1.values.reshape((1801, 3600))

lat = uv_1['lat_0'].values
lon = uv_1['lon_0'].values

ws = (u**2 + v**2)**0.5
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

# splice array:
qv = plt.quiver(lon[::60], lat[::40], u[::40, ::60], v[::40, ::60], color='k')
plt.title('ECO1280')
plt.tight_layout()
plt.savefig('contour_quivers_eco.png')


import torch
torch.device('cpu')
import numpy as np
from windflow import inference_flows
from windflow.datasets.daves_grids import Eco

checkpoint_file = 'model_weights/windflow.raft.pth.tar'
inference = inference_flows.FlowRunner('RAFT',
                                     overlap=128,
                                     tile_size=512,
                                     device=torch.device('cpu'),
                                     batch_size=1)
inference.load_checkpoint(checkpoint_file)

file1 = './data/gp_2016-06-01_00:00:00_P500_out.nc'
file2 = './data/gp_2016-06-01_03:00:00_P500_out.nc'

gp_1 = Eco(file1)
gp_ds1 = gp_1.open_dataset(scale_factor=25000, rescale=True)
gp_2 = Eco(file2)
gp_ds2 = gp_2.open_dataset(scale_factor=25000, rescale=True)

gp_rad1 = gp_ds1['gp_newP']
gp_rad2 = gp_ds2['gp_newP']

try:
    shape = np.shape(gp_rad1)
    gp_rad1 = gp_rad1.values.reshape((shape[0], shape[1]))
    gp_rad2 = gp_rad2.values.reshape((shape[0], shape[1]))
except ValueError as e:
    print(e)
    raise

_, flows = inference.forward(gp_rad1, gp_rad2)

u = flows[0]
v = flows[1]

ws = (u**2 + v**2)**0.5
max_ws = np.nanmax(ws)
print("Maximum wind speed:", max_ws)
c_inv = np.arange(0, 60, 1)
fig = plt.figure(figsize=(13,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(color='white')
plt.contourf(lon, lat, ws, c_inv, transform=ccrs.PlateCarree(),cmap=plt.cm.jet) # takes some time

cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('pixels',size=10,rotation=0,labelpad=15)
cb.ax.tick_params(labelsize=10)

# splice array:
qv = plt.quiver(lon[::60], lat[::40], u[::40, ::60], v[::40, ::60], color='k')
plt.tight_layout()
plt.title('Windflow pixel displacement')
plt.savefig('contour_quivers_windflow.png')
