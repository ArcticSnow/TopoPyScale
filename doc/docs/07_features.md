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

Currently, TPS will request daily files from the ERA5 Pressure levels and Surface level. These files are then concatenated using `cdo` into yearly files. As of now it is possible to convert these yearly netcdf to a Zarr store archive. This has the advantage to speed up the downscaling step as IO processes are lowered. For this you need to specify in the configuration file a name for a zarr store in the section `climate/era5` with the following: `zarr_store: ERA5.zarr`. The Zarr store will be created locally in the `inputs/climate/` folder. It is also possible to convert a stack of netcdf to a Zarr using the command:

```python
import xarray as xr 
from TopoPyScale import fetch_era5 as fe 


ds_plev = xr.open_mfdataset('inputs/climate/PLE*.nc')
ds_surf = xr.open_mfdataset('inputs/climate/SUR*.nc')

fe.convert_to_zarr(ds_plev, 
					'inputs/climate/ERA5.zarr', 
					chuncks={'latitude':3, 'longitude':3, 'time':1000, 'level':7},
					compressor=None, 
					mode='w')

fe.convert_to_zarr(ds_surf, 
					'inputs/climate/ERA5.zarr', 
					chuncks={'latitude':3, 'longitude':3, 'time':1000},
					compressor=None, 
					mode='a')
```

This will first create the Zarr store filled with the data from `ds_plev` (notice the `mode='w'`), and append the `ds_surf` data to the same zarr store (notice `mode='a'`). The default compression is using Blosc (when compressor is set to `None`. It is possible to use any other compression options supported by the Zarr library. 

While this step improves significantly computation afterwards, converting the stack of netcdf to zarr can be long.

## Parallelization

TPS uses parallelization for a number of steps. For most, the Python library multiprocessing is used to either handle multithreads (management of downloading request from cds server), or multicore (clustering, solar geometry, downscaling). An optional method using [Dask](https://docs.dask.org/en/stable/) is available to perform the dowscaling step. 

Settings for parallelization should be adapted and considered according to your machine (laptop, server, HPC, ...).

## Clustering

Clustering offer a number of possibilities. First, any DEM can by segmented by clustering (k-means) using a list of features. These features can be topographic or anything which can be labeled to the pixel level (e.g. watershed, glaciated land, etc.). The clustering will return a list of groups of pixels that fall within the same "cluster". The method implemented is K-means from [Scikit-learn](https://sklearn.org/stable/getting_started.html). To speed up the processing, TPS will use the routine `minibatch_kmean()` to spread the computational load on multiple core. 

### Clustering features

In the config file, the line `clustering_features: {'x':1, 'y':1, 'elevation':1, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1}` will define which variables are used to perform the clustering. A variable must be defined in `ds_param` to be used. Each feature are associated to a weight (default 1), but can be modulated higher or lower than one to diminish or enhance the effect of the selected variable. The value in `clustering_features` is a multiplier to scale the feature importance relative to the other. This may be useful for instance to emphasize the role of elevation in case of sharp terrain. Here the elevation as a factor of 4 relative to other features. 

```python
clustering_features: {'x':1, 'y':1, 'elevation':4, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1}
```

If you add new features specific to your project, be aware that all features are standardize using the `scikit-learn` function `StandardScaler()`. Other types of scaler not yet available.

It is recommanded to keep at the bare minimum the variable: `x,y,elevation`.

### Searching optimum number of clusters

There are not clear cut method to decide the ideal number of clusters. However there exists a number of technics to find a compromise between a sufficiently fine segmentation and the lowest number of clusters (less computation). These methods are implemented `Topo_sub`. It includes a method to search for the optimal number of clusters: `search_optimum_number_of_clusters()`. This may be used in the following manner:

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