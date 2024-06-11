from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from eco1280_loader import load_eco1280
import netCDF4 as nc
import torch
import torch.nn
from mpl_toolkits.basemap import Basemap
from metpy.plots import colortables

def downsample(x, pool=25):
    x = torch.from_numpy(x[np.newaxis])
    xlr = torch.nn.AvgPool2d(pool, stride=pool)(x)
    return xlr.detach().numpy()[0]

def flow_quiver_plot(u, v, u2, v2, x=None, y=None, ax=None, 
                     down=25, vmin=None, vmax=None, latlon=False,
                     size=10, cmap='jet', colorbar=False):
    
    intensity = (u**2 + v**2) ** 0.5
    intensity2 = (u2**2 + v2**2) ** 0.5

    u_l = downsample(u, down)
    v_l = downsample(v, down)
    u2_l = downsample(u2, down)
    v2_l = downsample(v2, down)

    intensity_l = (u_l ** 2 + v_l**2) ** 0.5
    intensity2_l = (u2_l ** 2 + v2_l**2) ** 0.5

    if (x is None) or (y is None):
        x = np.arange(0, u_l.shape[1]) * down + down/2.
        y = np.arange(0, v_l.shape[0]) * down + down/2.
        X, Y = np.meshgrid(x, y)
    else:
        x = downsample(x, down)
        y = downsample(y, down)
        X, Y = x, y

    if not ax:
        ratio = 1. * u.shape[0] / u.shape[1]
        hi = int(ratio * size)
        wi = int(size) * 2 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(wi, hi), frameon=False)
        for ax in [ax1, ax2]:
            ax.axis('off')
    else:
        fig = plt.gcf()
        ax1, ax2 = ax
    
    if vmax is None:
        vmax = max(np.nanmax(intensity), np.nanmax(intensity2))
    if vmin is None:
        vmin = min(np.nanmin(intensity), np.nanmin(intensity2))
    
    im1 = ax1.imshow(intensity, origin='upper', cmap=cmap, vmax=vmax, vmin=vmin)
    im2 = ax2.imshow(intensity2, origin='upper', cmap=cmap, vmax=vmax, vmin=vmin)
    
    if colorbar:
        fig.colorbar(im1, ax=ax1, label='Wind Speed (m/s)', aspect=50, pad=0.01, shrink=0.3)
        fig.colorbar(im2, ax=ax2, label='Wind Speed (m/s)', aspect=50, pad=0.01, shrink=0.3)
        
    scale = 150
    try:
        ax1.quiver(X, Y, v_l, u_l, latlon=latlon, scale_units='inches', scale=scale)
        ax2.quiver(X, Y, v2_l, u2_l, latlon=latlon, scale_units='inches', scale=scale)
    except: 
        ax1.quiver(X, Y, v_l, u_l, scale_units='inches', scale=scale)
        ax2.quiver(X, Y, v2_l, u2_l, scale_units='inches', scale=scale)
        
    for ax in [ax1, ax2]:
        if hasattr(ax, 'xaxis'):
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_aspect('equal')
    
    return ax1, ax2


def main():

    # Load ECO1280 data
    file1 = './data/uv_2016-06-01_00:00:00_P500_out.nc' 
    file2 = './data/uv_2016-06-01_03:00:00_P500_out.nc'
    eco_u, eco_v, lat, lon = load_eco1280(file1, file2)

    # Load windflow data
    print('loading windflow data')
    with nc.Dataset('data.nc', 'r') as f:
        w_lat = f.variables['lat'][:]
        w_lon = f.variables['lon'][:]
        gp_rad1 = f.variables['gp_rad1'][:]
        gp_rad2 = f.variables['gp_rad2'][:]
        w_u = f.variables['uwind'][:]
        w_v = f.variables['vwind'][:]
    assert np.all(lat == w_lat)

    expanded_lat = np.tile(lat, (3600, 1)).T
    mask = (expanded_lat <= 90) & (expanded_lat >= -90) & ((gp_rad1 > 0) | (gp_rad2 > 0)) # Mask the region of interest, as well as the minimum humidity value.


    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    a = np.cos(lat_rad)**2 * np.sin((lon_rad[1]-lon_rad[0])/2)**2
    d = 2 * 6378.137 * np.arcsin(a**0.5)
    size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1) # km
    w_u = w_u * size_per_pixel * 1000/ 10800
    w_v = w_v * size_per_pixel * 1000/ 10800



    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(40, 20))
    ax1, ax2 = flow_quiver_plot(w_u, w_v, eco_u, eco_v, ax=(ax1, ax2),colorbar=True)
    ax1.set_title('Wind Flow Quiver')
    ax2.set_title('Eco Quiver')
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
    plt.savefig('combined_quiver.png',bbox_inches='tight',dpi=300)

if __name__ == '__main__':
    main()
    
