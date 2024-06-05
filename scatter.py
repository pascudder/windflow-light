from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from eco1280_loader import load_eco1280
import netCDF4 as nc

def main():

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

    expanded_lat = np.tile(lat, (3600, 1)).T #reshaped to (1801,3600)
    mask = (expanded_lat <= 30) & (expanded_lat >= -30) # mask super northern and southern regions
                                                        #This range seems small - consider working with higher values?
    lat_mask = np.radians(expanded_lat[mask])  # convert latitude from degrees to radians

    #select masked regions
    eu_mask = eco_u[mask]
    ev_mask = eco_v[mask]

    wu_mask = w_u[mask]
    wv_mask = w_v[mask]

    # do calculations convert pixel to m/s
    wu_mask = (wu_mask * 0.1 * 111 * 1000 * np.cos(lat_mask)) / 10800 #LAT_MASK IS IN RADIANS
    wv_mask = (wv_mask * 0.1 * 111 * 1000) / 10800

    # calculate MSE of the u component
    x = eu_mask
    y = wu_mask
    
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
    x = ev_mask
    y = wv_mask

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
