## Project Organisation

To run a TopoPyScale project, you need to have the following file structure:

```
my_project/
    ├── inputs/
        ├── dem/
            ├── my_dem.tif
            ├── my_dem_mask.tif     (OPTIONAL: to mask part of the DEM)
            ├── my_dem_groups.tif   (OPTIONAL: to delineate groups of clusters)
            └── pts_list.csv        (OPTIONAL: to downscale to specific points)
        └── climate/                (climate.era5.path)
            └── ERA5.zarr           (written by mp.get_era5(), see Zarr below)
    ├── outputs/                    (outputs.directory)
            ├── tmp/
            ├── df_centroids.pck, ds_param.nc, ds_solar.nc, da_horizon.nc
            ├── physics_checks.csv  (if toposcale.physics_checks is set)
            └── downscaled/
                └── down.zarr       (downscaled timeseries of all points)
    ├── pipeline.py (OPTIONAL: script for the downscaling instructions)
    └── config.yml

```

`TopoPyScale` automatically creates the `inputs/` and `outputs/` folder structure, in which climatic and topographic forcings are stored. Relative paths in `config.yml` are relative to `project.directory`, which defaults to the directory where Python runs.

## File `config.yml`

The configuration file contains all parameters needed to run a downscaling job. It includes general information about the job, as well as specific routine and values. Examples of `config.yml` file can be found in the repository [TopoPyScale_examples](https://github.com/ArcticSnow/TopoPyScale_examples).

The configuration consists of a YAML file, which is a common standard for storing configurations. Further help on YAML syntax can be found [here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html), and the Python packages [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation) and [Munch](https://pypi.org/project/munch/) allows to interact with such kind of file. **Be aware that YAML is indent sensitive.**

The example below lists every key TopoPyScale reads. Keys marked `(optional)` can be left out: their default is given in the tables further down. A key left empty (e.g. `extent:`) is the same as a key left out.

```yaml
project:
  name: Name of the project               # metadata
  description: Describe the project       # metadata
  authors:                                # written in the metadata of exported files
    - Author 1 (can add contact and affiliation here)
    - Author 2
  date: Date at which the project is run  # metadata
  directory: /path/to/project/            # (optional) default: current working directory

  # start and end date of the period of interest. The end day is included
  start: 2018-01-01
  end: 2018-01-31

  # (optional) [latN, latS, lonW, lonE] in degrees. Box of the ERA5 download. Default: extent of the DEM
  extent: [60.8, 60.4, 7.1, 7.7]


  # indicate which climate data to use. Currently only era5 available (see climate section below)
  climate: era5

  parallelization:
    setting:
        multicore:
            CPU_cores: 6                  # sky view factor, horizon, clustering and downscaling threads, FSM runs

#.....................................................................................................
climate:
  precip_lapse_rate: False         # (optional) apply the precipitation lapse-rate correction (MicroMet monthly factors, Northern Hemisphere)
  precip_lapse_rate_max_factor: 5  # (optional) bound of the precipitation elevation factor, kept within [1/max, max]. Default 5

  # Settings for the ERA5 atmospheric forcings
  era5:
    path: inputs/climate/             # relative to the project directory, or absolute (e.g. a shared ERA5 archive)
    product: reanalysis               # ensemble not available yet
    timestep: 1h                      # 1h, 3h or 6h
    accumulation_window: 1h           # (optional) window of the accumulated radiation and precipitation, ending at the time stamp. ERA5: 1h whatever the timestep

    # Choose pressure levels relevant to your project and available in ERA5 Pressure Levels
    plevels: [ 700,750,775,800,825,850,875,900,925,950,975,1000 ]
    download_threads: 8               # parallel CDS requests (one month each, at most 10), or GDEX processes
    realtime: False                   # (optional) CDS only. Set project.end to the last full day available and clean outputs/downscaled/
    data_repository: cds              # (optional) cds (Copernicus Climate Data Store, default) or gdex (NCAR GDEX, see Dataset Sources)
    gdex_root:                        # (optional) gdex only: path or URL of GDEX d633000. Default /gdex/data/d633000 on NCAR systems, else OSDF over HTTPS
    cds_download_format: unarchived   # (optional) unarchived or zip, both are read
    output_format: netcdf             # (optional) file format of get_era5_snowmapper() only
    zarr_store: ERA5.zarr             # (optional) name of the ERA5 zarr store in the climate folder. Default ERA5.zarr

#.....................................................................................................
dem:
  path: inputs/dem/                       # (optional) folder of the DEM, relative to the project directory or absolute. Default inputs/dem/
  file: myDEM.tif                         # name of the DEM file. Must be a raster
  epsg: 32632                             # EPSG code of the DEM projection (projected, in metres)
  horizon_increments: 10                  # azimuth step of the horizons computed at the points, in degrees
  solar_position_method: nrel_numpy       # (optional) pvlib solar position method. Default nrel_numpy

#.....................................................................................................
sampling:

  # toposub (DEM clustering), points (list of points) or grid (every DEM pixel)
  method: toposub

  # In case method == 'points'
  points:
    csv_file: pts_list.csv              # in inputs/dem/, or absolute path. Columns x,y in the DEM projection
    name_column: pt_name                # (optional) column containing point_name. If not provided, point_name will be automatically assigned

  # In case method == 'toposub'
  toposub:
    clustering_method: minibatchkmean   # clustering method available: kmean, minibatchkmean
    n_clusters: 50                      # number of cluster to segment the DEM
    random_seed: 2                      # random seed for the K-mean clustering
    clustering_features: { 'x': 1, 'y': 1, 'elevation': 1, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1 }  # features used in the clustering with their relative importance (multiplier after scaling)
    clustering_mask: inputs/dem/catchment_mask.tif      # (optional) .tif mask (0/1) on the DEM grid, relative to the project directory
    clustering_groups: inputs/dem/groups.tif            # (optional) .tif of cluster groups (integers, e.g. land cover) on the DEM grid
    clustering_group_weights: inputs/dem/gr_wgths.csv   # (optional) .csv with columns group,weight: fraction of the clusters per group. Default: proportional to group area

#.....................................................................................................
toposcale:
  interpolation_method: idw               # horizontal interpolation of ERA5: idw or linear
  LW_terrain_contribution: True           # (optional) terrain contribution to longwave. Default False
  physics_checks: True                    # (optional) run physics-consistency checks after downscaling: True (report only), 'warning' or 'error' (raise)
  point_block: 100                        # (optional) number of points downscaled together. Default 100
  time_block: 8760                        # (optional) number of time steps downscaled together. Default one year
  solar_constant: 1361                    # (optional) total solar irradiance at 1 AU [W m-2]. Default 1361
  solar_substeps: 6                       # (optional) sub-steps of the accumulation window: average of the top-of-atmosphere irradiance and sunlit fraction of the direct beam. Default 6
  surface_effect:                         # (optional) surface-effect correction of air temperature and dew point (inversions, cold-air pools). Default off
    method: none                          # none, redcapp or dreamit
    coefficients:                         # (optional) coefficients file from surface_calib.save_coefficients. Default: published values
    context_dem:                          # (optional) DEM covering the points plus the window (30 km wide for redcapp, 50 km radius for dreamit). Default: project DEM
    dewpoint: True                        # (optional) correct the humidity too. Default True
    lw_consistent: True                   # (optional) longwave from the corrected temperature and humidity. Default True
    allow_partial_window: False           # (optional) accept windows leaving the context DEM. Default False
    diagnostics: False                    # (optional) store t_pl, dT_surf and dTd_surf (also with method none, for the calibration). Default False

#.....................................................................................................
outputs:
  directory: outputs                    # (optional) relative to the project directory, or absolute. Default outputs
  variables: all                        # (optional) list of variables to downscale and store, e.g. ['t', 'tp', 'SW']. Default all
  file:
    clean_outputs: False                # (optional) remove the entire outputs/ directory when Topoclass is created
    clean_FSM: False                    # (optional) remove the fsm_sims/ and ensemble/ directories when Topoclass is created
    df_centroids: df_centroids.pck      # (pickle)  dataframe containing the points of interest with their topographic features
    ds_param: ds_param.nc               # (netcdf)  topographic parameters (slope, aspect, etc.)
    ds_solar: ds_solar.nc               # (netcdf)  solar geometry
    da_horizon: da_horizon.nc           # (netcdf)  horizon angles of the points
    zarr_store: down.zarr               # (optional) store of the downscaled timeseries in outputs/downscaled/. Default down.zarr

clean_up:
  rm_tmp_dirs: True                     # (optional) remove the tmp directories after downscaling
```

The file `config.yml` is parsed by TopoPyScale at the time the class `topoclass('config.yml')` is created.

## Possible values of configurations

### Project

| Field       | Example Value         | Required | Possible Values | Description |
|-------------|-----------------------|----------|-----------------|-------------|
| name        | Finse                 | no       | string          | metadata: name of the project. Not used by the code |
| description | Downscaling for Finse | no       | string          | metadata: description of the project. Not used by the code |
| authors     | [S. Filhol]           | no       | list of strings | written as creator in the metadata of the files exported by `to_cryogrid()`, `to_crocus()`, `to_surfex()` and `to_musa()` |
| date        | Nov 2021              | no       | string          | metadata: creation date of the downscaling. Not used by the code |
| directory   | ./path_to_my_project/ | no       | string          | path where the downscaling project is located. Default: Python current working directory |
| start       | 2018-10-01            | yes      | %Y-%m-%d        | start date of the downscaling. ERA5 must cover it |
| end         | 2018-12-31            | yes      | %Y-%m-%d        | end date of the downscaling, included: time steps run up to the last one of that day. ERA5 must cover it |
| extent      | [60.8, 60.4, 7.1, 7.7] | no      | [latN, latS, lonW, lonE] | lat/lon box in degrees. The ERA5 download adds a 0.4° margin. If the DEM file does not exist, an interactive prompt offers to download one for this box. Default: extent of the DEM |
| climate     | era5                  | yes      | era5            | source of climate data, the name of the section in `climate`. ERA5 is the only supported dataset at the moment |
| ***parallelization*** |             |          |                 | |
| **setting**   |                     |          |                 | |
| *multicore*   |                     |          |                 | |
| CPU_cores   | 4                     | yes      | integer         | number of cores for the sky view factor and horizon computation, minibatch k-means clustering, `to_musa()`, and the number of threads downscaling blocks of points |

The former keys `parallelization.downscaling_method` and `parallelization.setting.dask` are no longer read.

### Climate

| Field             | Example Value      | Required | Possible Values | Description |
|-------------------|--------------------|----------|-----------------|-------------|
| precip_lapse_rate | True               | no       | True, False     | apply the precipitation lapse-rate correction, monthly factors of MicroMet [@listonMicroMet2006] (Northern Hemisphere). Default False |
| precip_lapse_rate_max_factor | 5       | no       | float > 1       | bound of the precipitation elevation factor (1 + c dz)/(1 - c dz), which otherwise diverges for dz ~ 2.9 km in winter. Default 5 |
| **era5**          |                    |          |                 | as of now TopoPyScale only supports ERA5 data |
| path              | inputs/climate/    | yes      | string          | folder of the climate data, relative to the project directory or absolute. With an absolute path (e.g. a shared archive), no `tmp/` folder is created in it |
| product           | reanalysis         | yes      | reanalysis      | no other product available at the moment |
| timestep          | 1h                 | yes      | 1h, 3h, 6h      | time step of the download and of the downscaling. Lower case (pandas no longer accepts `1H`; it is converted). With 3h or 6h, radiation and precipitation are still the 1-h ERA5 accumulation ending at the time stamp: `tp` is that hour times the time step, sampled rather than summed |
| accumulation_window | 1h               | no       | 1h, 3h, 0h      | window over which radiation is accumulated, ending at the time stamp (ERA5: 1h whatever the timestep). The solar geometry is computed at its centre, see [Solar geometry](07_features.md#solar-geometry). Default 1h |
| plevels           | [700,800,900,1000] | yes      | array           | ERA5 pressure levels to use, in hPa. The lowest pressure level must lie above the highest elevation of the DEM |
| download_threads  | 8                  | no       | integer         | number of parallel CDS requests, one month of one stream each. At most 10, the number of requests of a user the CDS runs at a time (more only wait in its queue, and lower the priority of the user). With GDEX: number of processes reading days in parallel, default 4 |
| data_repository   | cds                | no       | cds, gdex       | where ERA5 is fetched from: `cds` (Copernicus Climate Data Store, default) or `gdex` (NCAR GDEX, best on NCAR systems). Both give the same store. See [Dataset Sources](04_datasetSources.md#from-ncar-gdex) |
| gdex_root         | /gdex/data/d633000 | no       | path or URL     | `gdex` only: root of GDEX dataset d633000. Default `/gdex/data/d633000` if it exists (NCAR systems), else `https://osdf-director.osg-htc.org/ncar/gdex/d633000` |
| cds_download_format | unarchived       | no       | unarchived, zip | download format delivered by the CDS. Default unarchived. Zip files (the CDS sends one for surface fields whatever this setting) are read as well |
| realtime          | False              | no       | True, False     | `cds` only: set `project.end` to the last full day available and empty `outputs/downscaled/`, for daily updates with the latest month (ERA5T). Each run fetches the current month again |
| output_format     | netcdf             | no       | netcdf, grib    | file format of `get_era5_snowmapper()` only |
| zarr_store        | ERA5.zarr          | no       | string          | name of the ERA5 zarr store in the climate folder, written by `mp.get_era5()`. Default ERA5.zarr |

### dem

| Field              | Example Value   | Required | Possible Values          | Description |
|--------------------|-----------------|----------|--------------------------|-------------|
| path               | C:/GIS/DEMs     | no       | string                   | folder of the DEM file, relative to the project directory or absolute. Default `inputs/dem/` |
| file               | ASTER_Finse.tif | yes      | `*.tif`                  | filename of the DEM |
| epsg               | 32632           | yes      | EPSG CRS projection code | EPSG code of the DEM projection. Must be projected (metres). Also the projection of the `x,y` of a list of points |
| horizon_increments | 10              | yes      | 1-90                     | azimuth step of the horizons, computed at the points along straight rays (see [Horizons](07_features.md#horizons)). The azimuths are the centres of the bins: -175, -165, ..., 175 for 10. Unit: `degree` |
| solar_position_method | nrel_numpy   | no       | nrel_numpy, or a method of [pvlib](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html) | sun position algorithm. `nrel_numpy` (default) is vectorized over all points; other pvlib methods are run point by point and may require a specific [installation](01_install.md) |

### Sampling

| Field               | Example Value                                                                     | Required                                    | Possible Values                         | Description |
|---------------------|-----------------------------------------------------------------------------------|---------------------------------------------|-----------------------------------------|-------------|
| method              | toposub                                                                           | yes                                         | toposub, points, grid                   | `toposub`: cluster centroids of the DEM (also `cluster`); `points`: list of points in a `.csv` file (also `point`); `grid`: every pixel of the DEM |
| **points**          |                                                                                   |                                             |                                         | |
| csv_file            | station_list.csv                                                                  | only if method is `points`                  | `*.csv`, `*.txt`                        | `.csv` file of the points, in `inputs/dem/` or an absolute path. Must contain the columns `x,y` in the DEM projection (`dem.epsg`) |
| name_column         | pt_name                                                                           | no                                          | column names of the csv file            | column with a unique ID of the points (number or string), used to name the downscaled points. Default: row number, zero-padded |
| **toposub**         |                                                                                   |                                             |                                         | |
| clustering_method   | minibatchkmean                                                                    | only if method is `toposub`                 | kmean, minibatchkmean                   | clustering method. minibatchkmean is parallelized and a lot faster. See scikit-learn documentation |
| n_clusters          | 10                                                                                | only if method is `toposub`                 | integer                                 | number of clusters k-means segments the DEM into |
| random_seed         | 2                                                                                 | only if method is `toposub`                 | integer                                 | random seed to use in k-means |
| clustering_features | {'x':1, 'y':1, 'elevation':4, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1} | only if method is `toposub`                 | python dict: {feature_name: importance} | features of `ds_param` used in the clustering. Importance is a multiplier applied to a feature after scaling, in case one feature is more important for segmenting the DEM. Default should be 1 |
| clustering_mask     | inputs/dem/catchment_mask.tif                                                     | no (only used if method is `toposub`)       | `*.tif`                                 | `.tif` mask (pixels at 1 are used, 0 ignored), relative to the project directory or absolute. Same grid and resolution as the DEM |
| clustering_groups   | inputs/dem/VEG_CODE.tif                                                           | no (only used if method is `toposub`)       | `*.tif`                                 | `.tif` of integer groups to split the clustering into (e.g. vegetation codes 1-9), relative to the project directory or absolute. Same grid and resolution as the DEM |
| clustering_group_weights | inputs/dem/gr_weights.csv                                                    | no (only used with `clustering_groups`)     | `*.csv`                                 | `.csv` with the columns `group,weight` (or `weights`): fraction of the clusters given to each group, summing to 1. Each group gets ceil(n_clusters × weight) clusters. Default: proportional to the group area. See [Groups](07_features.md#groups) |

### Toposcale

| Field                   | Example Value | Required | Possible Values | Description |
|-------------------------|---------------|----------|-----------------|-------------|
| interpolation_method    | idw           | yes      | idw, linear     | horizontal interpolation of ERA5 to the points: inverse squared distance weight, or bilinear in latitude/longitude |
| LW_terrain_contribution | True          | no       | True, False     | add the longwave emitted by the surrounding terrain. Default False |
| point_block             | 100           | no       | int             | number of points downscaled together (also the shard size of the output store). Default 100 |
| time_block              | 8760          | no       | int             | number of time steps downscaled together (also the chunk size of the output store). Default: one year |
| solar_constant          | 1361          | no       | float           | total solar irradiance at 1 AU [W m-2], scaled by the Earth-Sun distance to compute the top-of-atmosphere irradiance. Default 1361 |
| solar_substeps          | 6             | no       | int             | number of sub-steps of the radiation accumulation window, used to average the top-of-atmosphere irradiance and to scale the direct beam by the part of the window the point is not shaded by terrain. 1: one shading test at the centre of the window. Default 6 |
| physics_checks          | True          | no       | True, False, 'warning', 'error' | run physics-consistency checks on the downscaled outputs and write `outputs/physics_checks.csv`. 'warning'/'error' raise when a check fails at that level. Default False. See [Physics Checks](09_physics_checks.md) |
| surface_effect          | {method: redcapp} | no   | dict            | surface-effect correction of air temperature and dew point: `method` none, redcapp or dreamit, plus `coefficients`, `context_dem`, `dewpoint`, `lw_consistent`, `allow_partial_window`, `diagnostics` (see the example above). Default off. See [Features](07_features.md#surface-effect-correction-of-air-temperature) |

### Outputs

| Field         | Example Value      | Required | Possible Values     | Description |
|---------------|--------------------|----------|---------------------|-------------|
| directory     | C:/ERA5/Downscaled | no       | string              | folder of the outputs, relative to the project directory or absolute. Default `outputs` |
| variables     | all                | no       | all, or a list      | variables to downscale and store in `down.zarr`, and default selection of `to_netcdf()`. Anything but a non-empty list means all: t, u, v, q, p, tp, precip_lapse_rate, wd, ws, w, vp, cse, LW, SW, SW_direct, SW_diffuse, cos_illumination. The physics checks and the exporters need most of them |
| **file**      |                    |          |                     | |
| clean_outputs | True               | no       | True, False         | remove the whole `outputs/` folder when `Topoclass` is created. With `False`, existing DEM parameters, clusters, solar geometry and horizons are loaded instead of recomputed. Default False |
| clean_FSM     | True               | no       | True, False         | remove the `fsm_sims/` and `ensemble/` folders when `Topoclass` is created. Default False |
| df_centroids  | df_centroids.pck   | yes      | `*.pck`             | filename of the dataframe of the points/centroids where downscaling takes place. Saved in `outputs/` |
| ds_param      | ds_param.nc        | yes      | `*.nc`              | filename of the dataset of DEM morphometrics and cluster labels. Saved in `outputs/` |
| ds_solar      | ds_solar.nc        | yes      | `*.nc`              | filename of the solar geometry of the points. Saved in `outputs/` |
| da_horizon    | da_horizon.nc      | yes      | `*.nc`              | filename of the horizon angles of the points (`azimuth`, `point_name`). Saved in `outputs/`, recomputed when the points change |
| zarr_store    | down.zarr          | no       | string              | name of the zarr store of the downscaled timeseries, in `outputs/downscaled/`. Default down.zarr |

The cluster map is written by `mp.toposub.write_landform()` to `outputs/landform.tif`; the former key `outputs.file.landform` is not read.

### clean_up

| Field       | Example Value | Required | Possible Values | Description |
|-------------|---------------|----------|-----------------|-------------|
| rm_tmp_dirs | True          | no       | True, False     | remove `outputs/tmp/` and the climate `tmp/` folder after downscaling. Default False |

## File `csv` format for a list of points

The list of points is a comma-separated value file (`.csv`) which must contain at least the fields `x,y`, in the projection of the DEM (`dem.epsg`). The file is looked for in `inputs/dem/`, unless `sampling.points.csv_file` is an absolute path. All other columns are loaded into the dataframe and can be used for further analysis (but are not required by TopoPyScale).

An example of a list of points, for a DEM in UTM 32N (`epsg: 32632`):

```csv
Name,stn_number,latitude,longitude,x,y
Finsevatne,SN25830,60.5938,7.527,419320.867306002,6718447.86246835
Fet-I-Eidfjord,SN49800,60.4085,7.2798,405243.856317655,6698143.36494597
Skurdevikåi,SN29900,60.3778,7.5693,421114.679132306,6694343.36865902
Midtstova,SN53530,60.6563,7.2755,405730.30171528,6725742.26010349
FV50-Vestredalen,SN53990,60.7418,7.5748,422296.164018722,6734871.61164008
Klevavatnet,SN53480,60.7192,7.2085,402259.379226592,6732844.21093029
```

This file is loaded in the `mp.toposub.df_centroids` dataframe. With `name_column: Name`, the points are named after the column `Name`.

## Updating an older `config.yml`

Keys renamed or no longer read:

| Old setting | Now |
|-------------|-----|
| `climate.era5.timestep: 1H` | `1h` (pandas no longer accepts `H`; upper case is converted) |
| `project.parallelization.downscaling_method`, `project.parallelization.setting.dask` | not read: downscaling runs `CPU_cores` threads over blocks of points |
| `project.split` (`IO`, `time`, `space`) | removed: long periods are processed by blocks of time steps, `toposcale.time_block` |
| `outputs.file.downscaled_pt` | removed with the netcdf downscaling: the points are written to `outputs.file.zarr_store` |
| `outputs.file.landform` | not read: `outputs/landform.tif` |
| `sampling.points.epsg` | not read: `x,y` must be in the DEM projection |
| `sampling.points.ID_col` | `sampling.points.name_column` |
| `data_repository: google_cloud_storage` | not supported: `cds` or `gdex` |
| `clean_up.delete_tmp_dirs` | `clean_up.rm_tmp_dirs` |
| `climate.era5.rm_daily` | not read: there are no daily files (a message is printed) |
| `climate.era5.cds_output_format` | not read: netcdf is downloaded (a message is printed for `grib`) |

## Parallelization

TopoPyScale uses parallelization for a number of steps: threads sending the download requests to the CDS (`download_threads`, at most 10), or processes fetching days from GDEX, and `project.parallelization.setting.multicore.CPU_cores` cores for the sky view factor, the horizons and the clustering. Solar geometry is vectorized over points. Downscaling processes blocks of points in parallel threads, `CPU_cores` of them.

Settings for parallelization should be adapted and considered according to your machine (laptop, server, HPC, ...).


## Zarr

Downscaling reads ERA5 from a zarr store, `inputs/climate/ERA5.zarr`, and writes all downscaled points to one zarr store, `outputs/downscaled/down.zarr`. Zarr stores data in compressed chunks, so that only the chunks needed are read.

- `mp.get_era5()` writes ERA5 to the store as the downloads arrive, see [Features](07_features.md#era5-download). A store written this way records which time steps it holds: `mp.downscale_climate()` refuses it while the project period has gaps, and running `mp.get_era5()` again fills them.
- netcdf files placed in the climate folder by hand (`PLEV_*.nc` and `SURF_*.nc`, in `inputs/climate/` or `inputs/climate/yearly/`) are converted to a store the first time `mp.downscale_climate()` runs, and again when the store does not cover the project period. The files are converted one pair at a time, without `xr.open_mfdataset`. It can also be done explicitly with `mp.era5_obj.to_zarr()` or `fetch_era5.netcdf_to_zarr()`.
- Downscaling processes blocks of points (`toposcale.point_block`) and of time steps (`toposcale.time_block`) at once. The ERA5 nodes around the points are read once per time block, and all points of a block are downscaled together as arrays.
- The output store has dimensions `(point_name, time)`. See [Output Formats](05_outputs.md).

