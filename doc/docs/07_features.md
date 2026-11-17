# Features

`TopoPyScale` (TPS) has a number of features to customize your downscaling. This pages is an attempt to describe all features currently implemented.

## Spatial vs. Point downscaling

TPS can be used to downscaled in two modes. The simplest case is to use the library to downscale weather timeseries at a list of points. This list of points id specified in a `.csv` file using coordinate system of the same projection (*i.e.* EPSG code). At minimum each line of the `.csv` file should contain:

```text
name,x,y
```

Using the setting spatial in `toposub` will lead you using the clustering method decribed in more details here below. 

## Fetching ERA5 data from CDS

As of now, TPS includes a class to implementing all the functionalities in dealing with the ERA5 data independent from `Topoclass`. This makes the process, now very long, of fetching data more flexible and robuts. The class eith inherits the config from `Topoclass`, or can be directly pointed towards the `config.yml` file.

```python
from TopoPyScale import fetch_era5 as fe 
from TopoPyScale import topoclass as tc 

config_file = 'config.yml'
mp = tc.Topoclass(config_file)
era = fe.FetchERA5(config=mp.config)
era.go_fetch()
```

The class integrate a number of options to make the downloading more flexible. For instance only download Surface Level data, with a given set of variables.

```python
from TopoPyScale import fetch_era5 as fe 
era = fe.FetchERA5(config='config.yml')

varoi = ['geopotential_at_surface',
			'surface_thermal_radiation_downwards',
            'surface_solar_radiation_downwards',
            'surface_pressure',
            'total_precipitation', 
            '2m_temperature'
        ]

era.varoi_surf = varoi
era.go_fetch()
```

This way it is possible to use routines for TPS project but not only.


## Convert ERA5 to Zarr

As explained in the (Datasets)[04_datasetSources.md], TPS relies on ERA5 data. In the future more datasets could be included. The minimum data required are at pressure levels `t,p,q,u,v`, and the surface level:
```
- 'geopotential_at_surface'
- '2m_dewpoint_temperature'
- 'surface_thermal_radiation_downwards',
- 'surface_solar_radiation_downwards','surface_pressure',
- 'total_precipitation'
- '2m_temperature'
- 'toa_incident_solar_radiation',
- 'friction_velocity'
- 'instantaneous_moisture_flux'
- 'instantaneous_surface_sensible_heat_flux'
```

TPS requests daily files of the ERA5 pressure levels and surface level, and concatenates them with `cdo` into yearly files (`inputs/climate/yearly/`). Downscaling then reads ERA5 from a zarr store, `inputs/climate/ERA5.zarr`, built from the yearly files the first time `mp.downscale_climate()` runs (and rebuilt if it does not cover the project period). The conversion opens one pair of PLEV/SURF files at a time and appends it to the store. It can also be run explicitly:

```python
from TopoPyScale import fetch_era5 as fe

fe.netcdf_to_zarr(path_to_netcdfs='inputs/climate/yearly',
                  zarrout='inputs/climate/ERA5.zarr',
                  plev_name='PLEV_*.nc',
                  surf_name='SURF_*.nc',
                  time_chunk=720)
```

The store holds `z, t, q, u, v, r` on pressure levels and `z_surf, sp, t2m, d2m, tp, ssrd, strd` at the surface, as float32 compressed with Blosc. The function also harmonizes files downloaded from the new CDS (`valid_time`, `pressure_level`, `expver`).

## Parallelization

TPS uses parallelization for a number of steps: threads to manage download requests to the CDS server, multiple cores for clustering, and threads over blocks of points for the downscaling (`project.parallelization.setting.multicore.CPU_cores`).

## Solar geometry

`solar_geom.get_solar_geom()` computes, for every point and time step:

- the solar zenith, azimuth and elevation angles at the centre of the radiation accumulation window (`t − 30 min` for ERA5), with the NREL Solar Position Algorithm of pvlib (`nrel_numpy`, about 0.0003° accuracy). The time-dependent part of the algorithm is computed once for all points; only the location-dependent part is evaluated per point, as arrays. For 2400 points and one year of hourly data this takes about 12 s, instead of about 65 s point by point;
- the top-of-atmosphere irradiance on a horizontal surface `SWtoa`, averaged over the accumulation window with `toposcale.solar_substeps` sub-steps (default 6). It uses the solar constant `toposcale.solar_constant` (default 1361 W m⁻²) scaled by the Earth–Sun distance, consistently with the accumulated ERA5 fluxes it is compared to (clearness index). Averaging over the window matters at sunrise and sunset, where the value at the window centre differs from the hourly mean by up to about 35 W m⁻².

`ds_solar.nc` is stored as float32. A file computed by an older version prints a warning when loaded; delete it to recompute it.


## Horizons

`mp.compute_horizon()` computes the horizon angle of each point in the directions `dem.horizon_increments` degrees apart (the centres of the bins: -175, -165, ..., 175 for 10°; 0 = south, 90 = east, ±180 = north). The downscaling sets the direct beam to zero when the sun is below the horizon of the nearest direction. The horizons are saved to `outputs/da_horizon.nc`, with dimensions (`azimuth`, `point_name`), and as the columns `hori_azi_<azimuth>` of `df_centroids`.

From each point, `topo_param.horizon_rays()` follows a straight ray in each direction, one cell at a time along the major grid axis, interpolating the DEM bilinearly. The observer is on the DEM surface at the exact position of the point. The horizon is the steepest elevation angle to the terrain along the ray, or 0 if no terrain is higher than the point. A ray stops at the edge of the DEM, or as soon as no terrain further away could be steeper than the horizon already found: the highest elevation of the DEM seen from that distance is below it. This shortcut does not change the result. No-data cells never form a horizon. Earth curvature is ignored, as in topocalc (about 0.1° for terrain 20 km away).

Earlier versions computed the horizons of every DEM pixel with topocalc and took the pixel nearest to each point. topocalc follows lines of sight shared by many pixels: for a direction off the grid axes, each line is a staircase, and all lines step at the same columns (or rows). The pixels just before a step read their nearest terrain from the adjacent row. This leaves streaks in the horizon maps for directions a few degrees off the grid axes: strongest 5° off (azimuths ±5°, ±85°, ±95°, ±175° with a 10° increment, a step every 11 cells), still visible 15° off, and absent at 45°. On the Davos example (SRTM 30 m), at azimuth 95°, these pixels differ from a straight ray by 3.6° on average, and by more than 17° for 5% of them. Along the grid axes, where topocalc's lines are exact rows and columns, the rays give the same horizons as topocalc.

Computing at the points only is also faster and lighter. For the 50 centroids of the Davos example (747 × 500 cells), it takes 0.15 s and writes 24 kB, instead of 7.5 s and 108 MB for the full grid. Each point and direction costs about 10 µs on that DEM. With `sampling.method: grid`, every pixel is a point: about 2 minutes for its 373,500 pixels on one core. Blocks of points run in `project.parallelization.setting.multicore.CPU_cores` threads.

An existing `da_horizon.nc` is loaded if it holds the same points (names and positions), and recomputed otherwise. A file computed on the full DEM grid by an earlier version is still used, at the nearest pixel, with a warning: delete it to recompute the horizons at the points. For maps of horizons, `topo_param.compute_horizon()` still computes the full grid with topocalc.


## Clustering

Clustering offer a number of possibilities. First, any DEM can by segmented by clustering (k-means) using a list of features. These features can be topographic or anything which can be labeled to the pixel level (e.g. watershed, glaciated land, etc.). The clustering will return a list of groups of pixels that fall within the same "cluster". The method implemented is K-means from [Scikit-learn](https://sklearn.org/stable/getting_started.html). To speed up the processing, TPS will use the routine `minibatch_kmean()` to spread the computational load on multiple core. 

### Clustering features

In the config file, the line `clustering_features: {'x':1, 'y':1, 'elevation':1, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1}` will define which variables are used to perform the clustering. A variable must be defined in `ds_param` to be used. Each feature are associated to a weight (default 1), but can be modulated higher or lower than one to diminish or enhance the effect of the selected variable. 
The value in `clustering_features` is a multiplier applied during the feature scaling prior to the K-means clustering.
This option allows to apply feature importance relative to the others. **This multiplier must be $>0$.** 
It may be useful for instance to emphasize the role of elevation in case of sharp terrain or any other variable.  


With the example below, the elevation as a factor of 4 relative to other features. 
```python
clustering_features: {'x':1, 'y':1, 'elevation':4, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1}
```

If you add new features specific to your project, be aware that all features are standardize using the `scikit-learn` function `StandardScaler()`. Other types of scaler not yet available.

It is recommanded to keep the bare minimum variable: `x,y,elevation`.

### Searching optimum number of clusters

There are no clear cut method to decide the ideal number of clusters. However there exists a number of technics to find a compromise between a sufficiently fine segmentation and the lowest number of clusters (less computation). These methods are implemented `Topo_sub`. It includes a method to search for the optimal number of clusters: `search_optimum_number_of_clusters()`. This may be used in the following manner:

```python
from TopoPyScale import topoclass as tc
import numpy as np

config_file = './config.yml'
mp = tc.Topoclass(config_file)
mp.compute_dem_param()
df = mp.search_optimum_number_of_clusters(cluster_range=np.arange(100,1000,50),plot=False)
```

This will compute the scores Within-Cluster-Sum-of-Squares (WCSS) Davies-Bouldin, Calinski-Harabasz, and the RMSE between the original DEM elevation and the clustered DEM elevation. Any of these may be used to decide of an appropriate number of clusters  

### Masking

While the input DEM is a raster file and have rectangular boundings, TPS allows to mask part of the DEM. The mask should be a GeoTiff of the same grid (same boundings) as the mains DEM. Pixels categorized as `1` will be considered for downscaling, the one as `0` will be ignored. This should be specified into the config file with the following: `clustering_mask: clustering/catchment_mask.tif` within the section `toposub`.

### Groups

The clustering can be defined in subgroups. This may be usefull for instance when one wants to split a watershed in different section (i.g. glacier, alpine, fan, lake, *etc.*). This can be done by creating a raster file of the same size and bounds as the DEM with an arbitrary set of integers. Each integers will define a group. The path to the raster file must be indicated in the config file as part of the `toposub` section as follow: `clustering_groups: clustering/groups.tif`.

Now one can set relative number of clusters per group. Default, it maintain cluster density per area homogenous across groups.

To trigger the option.
1 in config.yml:
add clustering_group_weights: 'inputs/dem/gw.csv' in toposub
2. create a csv file, here called gw.csv
```
group,weight
8,0.8
4,0.2
```
This will use 20% of clusters to group 4 and 80% for group 8.

## FSM integration

As part of TPS, we integrated a number of functions to interact with the snow model **FSM**. 

## Realtime

**@Joel can you write smthg?**
