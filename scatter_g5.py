from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from g5nr_loader import load_g5nr
import netCDF4 as nc
from windflow.datasets import utils

def main():

    # Load ECO1280 data
    file1 = '/data/pscudder/data/c1440_NR.inst30mn_3d_U_Np.20060930_1700z.nc4' 
    file2 = '/data/pscudder/data/c1440_NR.inst30mn_3d_V_Np.20060930_1700z.nc4'
    eco_u, eco_v, lat, lon = load_g5nr(file1, file2)

    # Load windflow data
    print('loading windflow data')
    with nc.Dataset('/data/pscudder/data_g5nr.nc', 'r') as f:
        w_lat = f.variables['lat'][:]
        w_lon = f.variables['lon'][:]
        gp_rad1 = f.variables['gp_rad1'][:]
        gp_rad2 = f.variables['gp_rad2'][:]
        w_u = f.variables['uwind'][:]
        w_v = f.variables['vwind'][:]
    assert np.all(lat == w_lat)
   
    expanded_lat = np.tile(lat, (5760, 1)).T
    expanded_lon = np.tile(lon, (2881,1))
    mask = (expanded_lat <= 60) & (expanded_lat >= -60) # Mask the region of interest, as well as the minimum humidity value.

    # Select masked regions
    eco_u = eco_u[mask]
    eco_v = eco_v[mask]
    w_u = w_u[mask]
    w_v = w_v[mask]

    # Calculate RMSE of u component
    x = eco_u
    y = w_u

    print(f'RMSE: u: {np.sqrt(np.nanmean((y-x)**2))}')

    # U component plots
    ax = density_scatter(x, y, s=1, bins=150)
    ax.set_title("2016-06-01 00:00:00 P500 U component density -60 to 60 lat")
    ax.set_xlabel('G5NR u-component m/s')
    ax.set_ylabel('Windflow u-component m/s')
    plt.savefig("scatterplots/scatter_density_ucomp_g5.png", bbox_inches='tight')

    udiff = (y - x)
    ax = scatter(udiff, expanded_lat[mask], axline=False)
    ax.set_title("2016-06-01 00:00:00 P500 U diff (windflow - eco) vs lat")
    ax.set_xlabel('Udiff u-component (windflow - eco) m/s')
    ax.set_ylabel('Latitude')
    plt.savefig("scatterplots/scatter_lat_ucomp_g5.png", bbox_inches='tight')

    udiff = (y - x)
    ax = scatter(udiff, expanded_lon[mask], axline=False)
    ax.set_title("2016-06-01 00:00:00 P500 U diff (windflow - eco) vs lon")
    ax.set_xlabel('Udiff u-component (windflow - eco) m/s')
    ax.set_ylabel('Longitude')
    plt.savefig("scatterplots/scatter_lon_ucomp_g5.png", bbox_inches='tight')

    # Calculate RMSE of v component
    x = eco_v
    y = w_v

    print(f'RMSE: v: {np.sqrt(np.nanmean((y-x)**2))}')

    # V component plots

    ax = density_scatter(x, y, s=1, bins=150)
    ax.set_title("2016-06-01 00:00:00 P500 V component density -60 to 60 lat")
    ax.set_xlabel('G5NR v-component m/s')
    ax.set_ylabel('Windflow v-component m/s')
    plt.savefig("scatterplots/scatter_density_vcomp_g5.png", bbox_inches='tight')

    vdiff = (y - x)
    ax = scatter(vdiff, expanded_lat[mask], axline=False)
    ax.set_title("2016-06-01 00:00:00 P500 V diff (windflow - eco) vs lat")
    ax.set_xlabel('Vdiff v-component (windflow - eco) m/s')
    ax.set_ylabel('Latitude')
    plt.savefig("scatterplots/scatter_lat.vcomp_g5.png", bbox_inches='tight')

    vdiff = (y - x)
    ax = scatter(vdiff, expanded_lon[mask], axline=False)
    ax.set_title("2016-06-01 00:00:00 P500 V diff (windflow - eco) vs lon")
    ax.set_xlabel('Vdiff v-component (windflow - eco) m/s')
    ax.set_ylabel('Longitude')
    plt.savefig("scatterplots/scatter_lon.vcomp_g5.png", bbox_inches='tight')

def scatter(x, y, s=1, textbox=(0, 0), axline=True):
    fig, ax = plt.subplots(figsize=(8, 8))
   
    num_samples = len(x)
    bias = np.nanmean(x)
    rmse = np.sqrt(np.nanmean((x**2)))
    
    ax.scatter(x, y, s=s)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    textstr = f'Num Samples: {num_samples}\nBias: {bias:.3f}\nRMSE: {rmse:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    
    if axline:
        ax.axline((0, 0), slope=1, color='black', lw=1)
    
    return ax


def density_scatter(x , y, s=1, axline=True, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True, range=[[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]])
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, s=s, **kwargs)
    if axline:
        ax.axline((0, 0), slope=1, color='black', lw=1)

    num_samples = len(x)
    bias = np.nanmean(y - x)
    rmse = np.sqrt(np.nanmean((y - x)**2))

    textstr = f'Num Samples: {num_samples}\nBias: {bias:.3f}\nRMSE: {rmse:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    return ax


if __name__ == '__main__':
    main()
