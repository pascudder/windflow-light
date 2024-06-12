# windflow-light 

Perform optical flow inference on geostationary satellite images from a pretrained RAFT model. 

![windflow example](./humidity_plots/Humidity.png)

See  `python predict.py` for a basic example.

## Install

`conda env create -f environment_cpu.yml`

## Usage

Usage for applying pretrained model to ECO1280 data

### contour.py

![Contours](./contour_plots/combined_quiver.png)

Plot and compare wind vectors derived from the ECO1280 data with those inferred from the windflow model.

`gp_2016-06-01_00:00:00_P500_out.nc`

of the form `gp_yyyy-mm-dd_HH:MM:SS_P[pressurelvl]_out.nc`

fields:

| Field | long name |
| --- | --- |
| gp_newP | specific humidity |
| lev_p | pressure level |
| lat_0 | latitude |
| lon_0 | longitude |

UV Components:

`uv_2016-06-01_00:00:00_P500_out.nc`

of the form `uv_yyyy-mm-dd_HH:MM:SS_P[pressure level]_out.nc`

fields:

| Field | long name |
| --- | --- |
| ugrd_newP | U component of wind |
| vgrd_newP | V component of wind |
| lat_0 | latitude |
| lon_0 | longitude |
| lev_p | pressure level |

latitude, longitude, and pressure level are the same in both of these files. 

### eco1280_loader.py

1. Read in UV Component file; extract and return lat, lon, u and v comp. 

### run_windflow.py

1. Read in Humidity file; extract humidity data, scale  or perform preprocessing if necessary ( This particular example scales by 25000 since the ECO1280 data was scaled by that amount. We want our humidity values to be in the 0 to 255 range)
2. Load model checkpoint. Set the tile size and overlap: Larger tile size should result in increased accuracy(and computation cost). Overlap is currently at 25% of tile size.
3. Perform inference on the model with the humidity data. The preprocessed humidity data is fed into the model to compute flows between the two time steps. This produces predictions for the u and v in units of **pixels displacement**
4. Convert pixel displacement to speed in units of m/s. We are using the haversine formula to calculate the size of each pixel, as we need to convert from a rectangular image to a sphere.

Calculate Intermediate Variable \(a\)

$$
a = \cos(\text{lat})^2 \cdot \sin\left(\frac{\text{lon}[1] - \text{lon}[0]}{2}\right)^2
$$

Calculate Distance \(d\)

$$
d = 2 \cdot (\text{equatorial radius of earth} + \text{height of atmosphere}) \cdot \arcsin(\sqrt{a}) \quad \text{(in km)}
$$

Calculate Size Per Pixel

$$
\text{size per pixel} = \text{np.repeat(np.expanddims(d, -1), len(lonrad), axis=1)} \quad \text{(in km)}
$$

Convert to Meters

$$
\text{size per pixel} = \text{size per pixel} \times 1000 \quad \text{(in meters)}
$$

Convert to Meters Per Second

Since these images are 3 hours apart, divide by \(3 \times 60 \times 60 = 10800\) to get units of seconds.

$$
\text{sizeperpixelmps} = \frac{\text{size per pixel}}{10800} \quad \text{(in m/s)}
$$

### scatter.py

![scatter u](./scatterplots/scatter_density_ucomp_500.png)
![scatter v](./scatterplots/scatter_density_vcomp.png)


1. Read in UV file and Humidity file
2. Create a mask for regions of interest (-30 to 30, or -60 to 60, -90 to 90)

There is density scatter, normal scatter, and plots with difference between real u/v and computed u/v. 

### humidity.py

![eco1280 example](./humidity_plots/combined_humidity_quivers.gif)

Compare the inferred wind vectors (red) with the truth data (black) over a 3 hour time window.
 
## Citation

Vandal, T., Duffy, K., McCarty, W., Sewnath, A., & Nemani, R. (2022). Dense feature tracking of atmospheric winds with deep optical flow, Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

## Acknowledgements

External packages and flownet code was used from: https://github.com/celynw/flownet2-pytorch/ <br>
Funded by NASA ROSES Earth Science Research from Geostationary Satellite Program (2020-2023)
