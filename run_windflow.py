print('reading windflow')
import torch
torch.device('cpu')
import numpy as np
from windflow import inference_flows
from windflow.datasets.daves_grids import Eco
import netCDF4 as nc


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

lat = gp_ds1['lat_0'].values
lon = gp_ds1['lon_0'].values

gp_rad1 = gp_ds1['gp_newP']
print(np.min(gp_rad1), np.max(gp_rad1))
gp_rad2 = gp_ds2['gp_newP']

try:
    shape = np.shape(gp_rad1)
    gp_rad1 = gp_rad1.values.reshape((shape[0], shape[1]))
    gp_rad2 = gp_rad2.values.reshape((shape[0], shape[1]))
except ValueError as e:
    print(e)
    raise


_, flows = inference.forward(gp_rad1, gp_rad2)

W_u = flows[0]
W_v = flows[1]


def save_data():
    with nc.Dataset('data.nc', 'w') as f:
        lat_ = f.createDimension('lat', len(lat))
        lon_ = f.createDimension('lon', len(lon))
        lats = f.createVariable('lat', np.float32, ('lat',))
        lons = f.createVariable('lon', np.float32, ('lon',))
        gp_ra1 = f.createVariable('gp_rad1', np.float32, ('lat', 'lon',))
        gp_ra2 = f.createVariable('gp_rad2', np.float32, ('lat', 'lon',))
        flow1 = f.createVariable('uwind', np.float32, ('lat', 'lon'))
        flow2 = f.createVariable('vwind', np.float32, ('lat', 'lon'))
        
        flow1[:, :] = W_u
        flow2[:, :] = W_v
        lats[:] = lat
        lons[:] = lon
        gp_ra1[:, :] = gp_rad1
        gp_ra2[:, :] = gp_rad2


save_data()