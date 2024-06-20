import torch
torch.device('cpu')
import numpy as np
from windflow import inference_flows
from windflow.datasets.daves_grids import Eco
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr


file1 = '/data/pscudder/data/c1440_NR.inst30mn_3d_QV_Np.20060930_1700z.nc4'
file2 = '/data/pscudder/data/c1440_NR.inst30mn_3d_QV_Np.20060930_2000z.nc4'

gp_ds1 = xr.open_dataset(file1)
gp_ds2 = xr.open_dataset(file2)

lat = gp_ds1['lat'].values
lon = gp_ds1['lon'].values

gp_rad1 = gp_ds1['QV'][0, 5, :, :].values
gp_rad2 = gp_ds2['QV'][0, 5, :, :].values


projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': projection})
ax.pcolormesh(lon,lat, gp_rad1, transform=projection)
ax.coastlines()
plt.savefig('data.png')


checkpoint_file = 'model_weights/windflow.raft.pth.tar'
inference = inference_flows.FlowRunner('RAFT',
                                     overlap=512,
                                     tile_size=1024,
                                     device=torch.device('cpu'),
                                     batch_size=1,
                                     upsample_input=None,
                                     )
inference.load_checkpoint(checkpoint_file)

_, flows = inference.forward(gp_rad1, gp_rad2)
w_u = flows[0]
w_v = flows[1]


fig, axs = plt.subplots(1,2,figsize=(10,4))
speed = (flows[0]**2 + flows[1]**2)**0.5
axs = axs.flatten()
axs[0].imshow(gp_rad1,vmax = 125)
axs[0].set_title("Input frame 1")
axs[1].imshow(speed)
axs[1].set_title("Flow Intensity")
plt.tight_layout()
plt.savefig("Humidity.png", dpi=200)

#convert to m/s
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)
a = np.cos(lat_rad)**2 * np.sin((lon_rad[1]-lon_rad[0])/2)**2
d = 2 * 6378.137 * np.arcsin(a**0.5)
size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1) # km
w_u = w_u * size_per_pixel * 1000 / 10800
w_v = w_v * size_per_pixel * 1000 * 0.8 / 10800 #scale by 0.8 to remove bias

def save_data():
    with nc.Dataset('/data/pscudder/data/data_g5nr.nc', 'w') as f:
        lat_ = f.createDimension('lat', len(lat))
        lon_ = f.createDimension('lon', len(lon))
        lats = f.createVariable('lat', np.float32, ('lat',))
        lons = f.createVariable('lon', np.float32, ('lon',))
        gp_ra1 = f.createVariable('gp_rad1', np.float32, ('lat', 'lon',))
        gp_ra2 = f.createVariable('gp_rad2', np.float32, ('lat', 'lon',))
        flow1 = f.createVariable('uwind', np.float32, ('lat', 'lon'))
        flow2 = f.createVariable('vwind', np.float32, ('lat', 'lon'))
        
        flow1[:, :] = w_u
        flow2[:, :] = w_v 
        lats[:] = lat
        lons[:] = lon
        gp_ra1[:, :] = gp_rad1
        gp_ra2[:, :] = gp_rad2

save_data()