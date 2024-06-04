from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn


def main():
    
    #True wind data
    file1 = './data/uv_2016-06-01_00:00:00_P500_out.nc' 
    file2 = './data/uv_2016-06-01_03:00:00_P500_out.nc'

    Eco_u, Eco_v, lat, _ = load_eco1280(file1, file2)

    #Humidity grid to be passed into windflow - shape is (1801, 3600, 1)
    file1 = './data/gp_2016-06-01_00:00:00_P500_out.nc'
    file2 = './data/gp_2016-06-01_03:00:00_P500_out.nc'

    W_u, W_v, W_lat, _ = load_windflow(file1, file2)

    assert np.all(lat == W_lat)

    expanded_lat = np.tile(lat, (3600, 1)).T 
    print("Shape of expanded_lat:", expanded_lat.shape) #debugging
    
    mask = (expanded_lat <= 30) & (expanded_lat >= -30) # mask super northern and southern regions
                                                        #NEEDS CLARIFICATION - is this selecting the region between 30 degrees N and 30 degrees S? Seems too small.

    lat_mask = np.radians(expanded_lat[mask])  # convert latitude from degrees to radians

    #select masked regions
    Eu_mask = Eco_u[mask]
    Ev_mask = Eco_v[mask]

    Wu_mask = W_u[mask]
    Wv_mask = W_v[mask]

    # do calculations convert pixel to m/s
    Wu_mask = (Wu_mask * 0.1 * 111 * 1000 * np.cos(lat_mask)) / 10800
    Wv_mask = (Wv_mask * 0.1 * 111 * 1000) / 10800

    # calculate MSE of the u component
    x = Eu_mask
    y = Wu_mask
    
    print(f'MSE: u: {np.nanmean((y-x)**2)}')

    #U component plots
    ax = scatter(x, y, s=1, textbox=(-18, 55))
    ax.set_xlim(-20, 40)
    ax.set_ylim(-20, 40)
    ax.set_title("2016-06-01 00:00:00 P500 U component -30 to 30 lat")
    ax.set_xlabel('ECO1280 u-component m/s')
    ax.set_ylabel('Windflow u-component m/s')
    #plt.savefig("scatter.ucomp_500_90to90_pixel.png")

    ax = density_scatter(x, y, s=1, bins=150)
    ax.set_xlim(-20, 40)
    ax.set_ylim(-20, 40)
    ax.set_title("2016-06-01 00:00:00 P500 U component density -30 to 30 lat")
    ax.set_xlabel('ECO1280 u-component m/s')
    ax.set_ylabel('Windflow u-component m/s')
    #plt.savefig("scatter_density.ucomp_500_30to30_pixel.png")

    udiff = (y - x)
    ax = scatter(udiff, expanded_lat[mask], axline=False)
    ax.set_title("2016-06-01 00:00:00 P500 U diff (windflow - eco) vs lat")
    ax.set_xlabel('Udiff u-component (windflow - eco) m/s')
    ax.set_ylabel('Latitude')
    #plt.savefig("scatter_lat.ucomp_500_90to90_pixel.png")

    #calculate MSE of v component
    x = Ev_mask
    y = Wv_mask

    print(f'MSE: v: {np.nanmean((y-x)**2)}')

    #V component plots
    ax = scatter(x, y, s=1, textbox=(-18, 35))
    #ax.set_xlim(-40, 40)
    #ax.set_ylim(-40, 40)
    ax.set_title("2016-06-01 00:00:00 P500 V component 30 to 30 lat")
    ax.set_xlabel('ECO1280 v-component m/s')
    ax.set_ylabel('Windflow v-component m/s')
    #plt.savefig("scatter.vcomp_500_60to90_pixel.png")

    ax = density_scatter(x, y, s=1, bins=150)
    ax.set_xlim(-20, 40)
    ax.set_ylim(-20, 40)
    ax.set_title("2016-06-01 00:00:00 P500 V component density -30 to 30 lat")
    ax.set_xlabel('ECO1280 v-component m/s')
    ax.set_ylabel('Windflow v-component m/s')
    #plt.savefig("scatter_density.vcomp_500_30to30_pixel.png")

    vdiff = (y - x)
    ax = scatter(vdiff, expanded_lat[mask], axline=False)
    ax.set_title("2016-06-01 00:00:00 P500 V diff (windflow - eco) vs lat")
    ax.set_xlabel('Vdiff v-component (windflow - eco) m/s')
    ax.set_ylabel('Latitude')
    #plt.savefig("scatter_lat.vcomp_500_90to90_pixel.png")


def load_eco1280(file1, file2):
    # read UV
    print('reading UV')
    uv_1 = xr.open_dataset(file1)
    uv_2 = xr.open_dataset(file2)

    u1 = uv_1['ugrd_newP'] 
    v1 = uv_1['vgrd_newP']

    #remove 3rd dimension - value is 1 because input data should be a single grid slice
    Eco_u = u1.values.reshape((1801, 3600))
    Eco_v = v1.values.reshape((1801, 3600))

    lat = uv_1['lat_0'].values
    lon = uv_1['lon_0'].values
    
    return Eco_u, Eco_v, lat, lon


def load_windflow(file1, file2):

    # read windflow
    print('reading windflow')
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

    gp_1 = Eco(file1)
    gp_ds1 = gp_1.open_dataset(scale_factor=25000, rescale=True)
    gp_2 = Eco(file2)
    gp_ds2 = gp_2.open_dataset(scale_factor=25000, rescale=True)

    gp_rad1 = gp_ds1['gp_newP'] 
    gp_rad2 = gp_ds2['gp_newP']
    lat = gp_ds1['lat_0'].values
    lon = gp_ds1['lon_0'].values

    try:
        shape = np.shape(gp_rad1)
        gp_rad1 = gp_rad1.values.reshape((shape[0], shape[1]))
        gp_rad2 = gp_rad2.values.reshape((shape[0], shape[1]))
        print("Shape of reshaped gp_rad1:", gp_rad1.shape) #debugging
        print("Shape of reshaped gp_rad2:", gp_rad2.shape) #debugging
    except ValueError as e:
        print(e)
        raise

    # pass 2D humidity grid into windflow 
    _, flows = inference.forward(gp_rad1, gp_rad2)

    W_u = flows[0]
    W_v = flows[1]

    return W_u, -W_v, lat, lon


def scatter(x, y, s=1, textbox=(0, 0), axline=True):
    fig, ax = plt.subplots(figsize=(8,8))
    num_samples = len(x)
    bias = np.nanmean(y - x);
    rmse = np.sqrt(np.nanmean((y - x)**2))
    
    ax.scatter(x, y, s=s)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    props = dict(boxstyle='round', facecolor='aquamarine', alpha=0.5)
    textstr = f'Stats: \nSample Number: {num_samples} \nBias: {bias:.3f}\nRMSE: {rmse:.3f}'
    ax.text(xmin + 2, ymax - 2, textstr, fontsize=10, verticalalignment='top', bbox=props)
    
    if (axline):
        ax.axline((0, 0), slope=1, color='black', lw=1)
    
    return ax


def density_scatter(x , y, s=1, axline=True, sort=True, bins=20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram, pretty much directly copied from /home/sreiner/QOSAP/plotting_tools.py
    """
    fig, ax = plt.subplots(figsize=(8,8))

    data , x_e, y_e = np.histogram2d( x, y, bins=bins, density=True, range=[[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]])
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, s=s, **kwargs )
    if (axline):
        ax.axline((0, 0), slope=1, color='black', lw=1)

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    return ax


if __name__ == '__main__': 
    main()
