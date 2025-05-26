# Project Configuration

## Project Organisation

To run a TopoPyScale project, you need to have the following file structure:

```
my_project/
    ├── inputs/
        ├── dem/ 
            ├── my_dem.tif
            └── pts_list.csv  (OPTIONAL: to downscale to specific points)
        └── climate/
            ├── PLEV*.nc
            └── SURF*.nc
    ├── outputs/
            ├── tmp/
    ├── pipeline.py (OPTIONAL: script for the downscaling instructions)
    └── config.yml

```

`TopoPyScale` will automatically generates the `inputs/` and `outputs/` folder structure in which climatic and topographic forcings will be stored. Then, `TopoPyScale` is implemented with the assumption that the Python console will be open from the root path of `my_project/`

## File `config.yml`

The configuration file contains all parameters needed to run a downscaling job. It includes general information about the job, as well as specific routine and values. Examples of `config.yml` file can be found in the repository[TopoPyScale_examples](https://github.com/ArcticSnow/TopoPyScale_examples).

The configuration consists of a YAML file, which is a common standard for storing configurations. Further help on YAML syntax can be found [here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html), and the Python packages [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation) and [Munch](https://pypi.org/project/munch/) allows to interact with such kind of file. **Be aware that YAML is indent sensitive.**

For TopoPyScale, the configuration file must contain at least the following:

```yaml
project:
  name: Name of the project
  description: Describe the project
  authors:
    - Author 1 (can add contact and affiliation here)
    - Author 2
    - Author 3
  date: Date at which the project is run. This is metadata
  directory: /path/to/project/

  # start and end date of the timeperiod of interest
  start: 2018-01-01
  end: 2018-01-31
  split:
    IO: False       # Flag to split downscaling in time or not
    time: 2         # number of years to split timeline in
    space: None     # NOT IMPLEMENTED

  # This is for the option of fetching DEM with API (NOT YET SUPPORTED)
  extent:

  # Indicate the number of core to use
  CPU_cores: 4

  # indicate which climate data to use. Currently only era5 available (see climate section below)
  climate: era5

#.....................................................................................................
climate:
  # For now TopoPyScale only supports ERA5-reanalysis input climate data
  era5:
    path: inputs/climate/   # Can either be a absolute path or relative to the project directory
    product: reanalysis     # ensemble not available yet
    timestep: 1H            # 1H, 3H 6H or else.

    # Choose pressure levels relevant to your project and evailable in ERA5 Pressure Levels
    plevels: [ 700,750,775,800,825,850,875,900,925,950,975,1000 ]
    download_threads: 1               # Number of threads to request downloads with cdsapi
    realtime: False                   # (Optional) Forces redownload of latest month of ERA5 data upon each run of code (allows daily updates for realtime applications)
    data_repository: cds              # repository from where to download data: cds (copernicus official ERA5), google_cloud_storage (Google archive of ERA5)
    cds_output_format: netcdf         # netcdt or grib. Grib is not supported by topoclass
    cds_download_format: unarchived   # unarchived or zip
    rm_daily: False                   # remove 

  precip_lapse_rate: False     # Apply precipitation lapse-rate correction (currently valid for Northern Hemisphere only)

#.....................................................................................................
dem:
  path: C:/GIS/DEMs                       # (optional) Absolute path where the DEM file is stored
  file: myDEM.tif                         # Name of the dem file. Must be a raster.
  epsg: 32632                             # projection EPSG code
  horizon_increments: 10                  # horizon increment angle in degrees
  solar_position_method: nrel_numpy       # (optional) method to compute solar_geom with pvlib libraries.  

#.....................................................................................................
sampling:

  # choose downscaling using dem segmentation 'toposub' or a list of points 'points'. Possible values: toposub, points
  method: toposub

  # In case method == 'points', indicate a file with a list of points and the point coordinate projection EPSG code
  points:
    csv_file: pt_list.csv               # filename of list of points
    epsg: 4326                          # EPSG code of the points (x,y) coordinates in file
    name_column: pt_name                # (optional) column containing point_name. If not provided, point_name will be automatically assigned

  # In case method == 'toposub'
  toposub:
    clustering_method: minibatchkmean   # clustering method available: kmean, minibatchkmean
    n_clusters: 50                      # number of cluster to segment the DEM
    random_seed: 2                      # random seed for the K-mean clustering 
    clustering_features: { 'x': 1, 'y': 1, 'elevation': 1, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1 }  # dictionnary of the features of choice to use in clustering with their relative importance. Relative importance is a multiplier after scaling
    clustering_mask: clustering/catchment_mask.tif # optional path to tif containing a mask (0/1)
    clustering_groups: clustering/groups.tif # optional path to a tif containing cluster groups (int values), e.g. land cover

#.....................................................................................................
toposcale:
  interpolation_method: idw               # interpolation methods available: linear or idw
  LW_terrain_contribution: True           # (bool)    Turn ON/OFF terrain contribution to longwave

#.....................................................................................................
outputs:
  directory: outputs                    # (optional) absolute path where to store the final downscaled products.
  variables: all                        # list of variables to export in netcdf. ['t','p','SW']. Default None or all
  file:
    clean_outputs: False                # (bool)    remove the entire outputs/ directory prior to downscaling
    clean_FSM: True                     # (bool)    remove the entire sim/ directory
    df_centroids: df_centroids.pck      # (pickle)  dataframe containing the points of interest with their topographic features
    ds_param: ds_param.nc               # (netcdf)  topographic parameters (slope, aspect, etc.)
    ds_solar: ds_solar.nc               # (netcdf)  solar geometry
    da_horizon: da_horizon.nc           # (netcdf)  horizon angles
    landform: landform.tif              # (geotiff) rasters of of cluster labels, [TopoSub]
    downscaled_pt: down_pt_*.nc         # (netcdf)  Downscales

clean_up:
  rm_tmp_dirs: True                   # (optional: bool) remove the created tmp directories after downscaling?
```

The file `config.yml` is parsed by TopoPyScale at the time the class `topoclass('config.yml')` is created.

## Possible values of configurations

### Project

| Field       | Example Value         | Required                     | Possible Values | Description                                                                                                                  |
|-------------|-----------------------|------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------|
| name        | Finse                 | yes                          | string          | metadata: Name of the project                                                                                                |
| description | Downscaling for Finse | yes                          | string          | metadata: Description of the project                                                                                         |
| authors     |                       | yes                          | list of strings | metadata: name of downscaling authors                                                                                        |
| date        | Nov 2021              | yes                          | string          | metadata: creation date of the downscaling                                                                                   |
| directory   | ./path_to_my_project/ | no                           | string          | path where the downscaling project is located. If empty, default will be python current working directory                    |
| start       | 2018-10-01            | yes                          | %Y-%m-%d        | start date of the downscaling. Currently must date available in ERA5 dataset                                                 |
| end         | 2018-12-31            | yes                          | %Y-%m-%d        | end date of the downscaling. Currently must date available in ERA5 dataset. FYI, `TopoPyScale` downsloads ERA5 data monthly. |
| **split**   |                       |                              |                 |                                                                                                                              |
| IO          | False                 | yes                          | True, False     | Use True to split in time the downscaling project in case of long timeseries. This is decrease memory usage.                 |
| time        | 1                     | only if `split.IO` is `True` |                 | Number of years to chunck climate data timeseries                                                                            |
| space       | None                  | no                           |                 | not yet implemented                                                                                                          |
| extent      | None                  | no                           |                 | not yet implemented                                                                                                          |
| CPU_cores   | 4                     | yes                          | integer         | Number of cores to use                                                                                                       |
| climate     | era5                  | yes                          | era5            | source of climate data. ERA5 is the only supported dataset at the moment                                                     |

### Climate

| Field             | Example Value      | Required | Possible Values | Description                                                                                                        |
|-------------------|--------------------|----------|-----------------|--------------------------------------------------------------------------------------------------------------------|
| **era5**          |                    |          |                 | As of now TopoPyScale only supports ERA5 data                                                                      |
| path              | inputs/climate/    | y        | string          | path to store climate data (either relative to the project directory or absolute path possible)                    |
| product           | reanalysis         | y        | reanalysis      | no other product available at the moment.                                                                          |
| timestep          | 1H                 | y        | 1H              | timestep to run TopoPyScale. Currently only 1H available                                                           |
| plevels           | [700,800,900,1000] | y        | array           | Indicate ERA5 pressure level to use. The lower pressure level must be higher than the highest elevation of the DEM |
| download_threads  | 12                 | y        | integer         | Number of downloading threads to use                                                                    |
| data_repository  | google_cloud_storage | y        | string         | Indicate which data repositoryt to download data from: 'cds or google_cloud_storage                  |
| cds_output_format  | netcdf | n       | string         | indicate file format CDS will deliver. netcdf or grib|
| cds_download_format  | unarchived | n       | string         | indicate download format CDS will deliver. unarchived or zip|  
| rm_daily  | False | n       | string         | remove or not daily downloads. To save storage after download|
| realtime          | False              | n        | True, False     | Upon each new run of code redownloads latest month (ERA5T) to obtain daily updates of partial months.              |
| precip_lapse_rate | True               | y        | True, False     | Apply precipitation lapse rate                                                                                     |

### dem

| Field              | Example Value   | Required | Possible Values          | Description                                            |
|--------------------|-----------------|----------|--------------------------|--------------------------------------------------------|
| path               | C:/GIS/DEMs     | no       | str (absolute path)      | vAbsolute path where the DEM file is stored            |
| file               | ASTER_Finse.tif | yes      | `*.tif`                  | filename of the DEM                                    |
| epsg               | 32632           | yes      | EPSG CRS projection code | EPSG CRS projection code of the DEM geoTiff            |
| horizon_increments | 10              | yes      | 1-90                     | sector angle to compute horizon angle. Unit: `degree`. |
| solar_position_method | nrel_c              | no      | See methods of [pvlib](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html)                     | pvlib can use different method to compute sun position. May require specific [installation](01_install.md) |

### Sampling

| Field               | Example Value                                                                     | Required                                    | Possible Values                         | Description                                                                                                                                                                                                                          |
|---------------------|-----------------------------------------------------------------------------------|---------------------------------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| method              | toposub                                                                           | yes                                         | toposub, points                         | choice to run dowscaling for a list of `points` in a `.csv` file, or for a spatial job using `toposub` clustering method                                                                                                             |
| **points**          |                                                                                   |                                             |                                         |                                                                                                                                                                                                                                      |
| csv_file            | station_list.csv                                                                  | only if method is `points`                  | `*.csv`, `*.txt`                        | name of the `.csv` file containing the list of points. must contain at least the fields `x,y`                                                                                                                                        |
| epsg                | 4326                                                                              | only if method is `points`                  | EPSG CRS projection code                | EPSG CRS projection code of the coordinate `x,y` provided in the `.csv` file                                                                                                                                                         |
| name_column         | pt_name                                                                              | no                                          | All column names of the csv file        | Name of the column containing a ID of the points (unique!). This Id can be number or string. It will be used to name the downscaled data at the points.                                                                              |
| **toposub**         |                                                                                   |                                             |                                         |                                                                                                                                                                                                                                      |
| clustering_method   | minibatchkmean                                                                    | only if method is `toposub`                 | kmean, minibatchkmean                   | clustering method. minibatchkmean is parallelized and lot faster. See scikit-learn documentation.                                                                                                                                    |
| n_clusters          | 10                                                                                | only if method is `toposub`                 | integer                                 | number of cluster k-mean will segement DEM by                                                                                                                                                                                        |
| random_seed         | 2                                                                                 | only if method is `toposub`                 | integer                                 | random seed to use in k-mean                                                                                                                                                                                                         |
| clustering_features | {'x':1, 'y':1, 'elevation':4, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1} | only if method is `toposub`                 | python dict: {feature_name: importance} | Python dictionary that list which features the clustering must be done with. Importance value is a scaling factor applied to specific feature in case one feature may be more important for segmenting the DEM. Default should be 1. |
| clustering_mask     | clustering/catchment_mask.tif                                                     | optional (only used if method is `toposub`) | '**/*.tif'                              | Path (or relative path) to a .tif file containing the mask (0/1 values) which pixels of the DEM are used. Needs to have the same grid/resolution as the input DEM.                                                                   |
| clustering_groups   | clustering/VEG_CODE.tif                                                           | optional (only used if method is `toposub`) | '**/*.tif'                              | Path (or relative path) to a .tif file containing integer values of groups to split the clustering into (e.g. Vegetation codes 1-9). Needs to have the same grid/resolution as the input DEM.                                        |

### Toposcale

| Field                   | Example Value | Required | Possible Values | Description                                                           |
|-------------------------|---------------|----------|-----------------|-----------------------------------------------------------------------|
| interpolation_method    | idw           | y        | idw, linear     | interpolation method: inverse distance weight or linear interpolation |
| LW_terrain_contribution | True          | y        | True, False     | Use longwave terrain contribution correction or not                   |

### Outputs

| Field         | Example Value      | Required | Possible Values     | Description                                                                                                                                                                                                                                                   |
|---------------|--------------------|----------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| directory     | C:/ERA5/Downscaled | no       | str (absolute path) | The absolute path where to store the final downscaled product                                                                                                                                                                                                 
| variables     | all                | yes      | all, or             | Variable to export when using `to_netcdf()` function                                                                                                                                                                                                          |
| **file**      |                    |          |                     |                                                                                                                                                                                                                                                               |
| clean_outputs | True               | yes      | True, False         | remove all files from `/outputs/` folder prior to downscaling. If `False` all files in `outputs/` are kept, but `outputs/tmp/` is removed. `False` can be used to speed up job if computing DEM morphometrics, solar geometries, and horizons have been done. |
| clean_FSM     | True               | yes      | True, False         | remove all files from `/fsm_sims/` folder prior to downscaling and running FSM. If false, existing files will not be deleted.                                                                                                                                 |
| df_centroids  | df_centroids.pck   | yes      | `*.pck`             | filename to store dataframe of the points/centroids downscaling takes place . File is saved in `outputs/`                                                                                                                                                     |
| ds_param      | ds_param.nc        | yes      | `*.nc`              | filename to store dataset of the DEM morphometric and cluster/centroid labels map. File is saved in `outputs/`                                                                                                                                                |
| ds_solar      | ds_solar.nc        | yes      | `*.nc`              | filename to store dataset of the solar geometry of DEM. File is saved in `outputs/`                                                                                                                                                                           |
| da_horizon    | da_horizon.nc      | yes      | `*.nc`              | filename to store DataArray of the DEM horizons. File is saved in `outputs/`                                                                                                                                                                                  |
| landform      | landform.tif       | no       | `*.tif`             | filename to store raster of the points/centroids downscaling map. File is saved in `outputs/`                                                                                                                                                                 |
| downscaled_pt | down_pt_*.nc       | yes      | `*.nc`              | filename to store dataset of the dowsncaled points/centroids. File is saved in `outputs/`                                                                                                                                                                     |

### clean_up

| Field           | Example Value | Required | Possible Values | Description                                                                         |
|-----------------|---------------|----------|-----------------|-------------------------------------------------------------------------------------|
| delete_tmp_dirs | True          | no       | True, False     | If True, the created tmp directories will get deleted after downscaling the climate |

## File `csv` format for a list of points

The list of points is a comma-separated value file (`.csv`) which must contain at least the fields `x,y`. All other columns will
be loaded into a dataframe and can be used for further analysis (but won't be required by TopoPyScale).

An example of a list of points:

```csv
Name,stn_number,latitude,longitude,x,y
Finsevatne,SN25830,60.5938,7.527,419320.867306002,6718447.86246835
Fet-I-Eidfjord,SN49800,60.4085,7.2798,405243.856317655,6698143.36494597
Skurdevikåi,SN29900,60.3778,7.5693,421114.679132306,6694343.36865902
Midtstova,SN53530,60.6563,7.2755,405730.30171528,6725742.26010349
FV50-Vestredalen,SN53990,60.7418,7.5748,422296.164018722,6734871.61164008
Klevavatnet,SN53480,60.7192,7.2085,402259.379226592,6732844.21093029
```

This file will loaded in `mp.toposub.df_centroids` dataframe.

