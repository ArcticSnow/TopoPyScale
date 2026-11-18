<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topoclass`
Toposcale class definition 

S. Filhol, September 2021 

project/  config.ini 
    -> input/ 
        -> dem/ 
        -> climate/ 
    -> output/ 



---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Topoclass`
A python class to bring the typical use-case of toposcale in a user friendly object 

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.__init__`

```python
__init__(config_file)
```








---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L678"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.check_physics`

```python
check_physics(
    strict=False,
    snow_partition_method=None,
    fname='physics_checks.csv',
    points=None
)
```

Run physics-consistency checks on the downscaled outputs (see topo_checks.py) and save the report to csv. 



**Args:**
 
 - <b>`strict`</b> (bool or str):  False to only report; True or 'error' to raise on errors; 'warning' to raise on  warnings too 
 - <b>`snow_partition_method`</b> (str or list):  partition method(s) to probe, e.g. 'continuous' 
 - <b>`fname`</b> (str):  report file name, written to the outputs directory 
 - <b>`points`</b> (list):  optional subset of point_name to check 



**Returns:**
 
 - <b>`dataframe`</b>:  report with one row per check and variable 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.compute_dem_param`

```python
compute_dem_param()
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L568"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.compute_horizon`

```python
compute_horizon()
```

Horizon angles of the points (df_centroids), every dem.horizon_increments degrees, along straight rays from each point (topo_param.compute_horizon_points). Saved to outputs/da_horizon.nc and as the columns hori_azi_<azimuth> of df_centroids. An existing horizon file is loaded if it holds these points (names and positions), and recomputed otherwise. 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.compute_solar_geometry`

```python
compute_solar_geometry()
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L602"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.downscale_climate`

```python
downscale_climate()
```

Downscale the climate forcing to the points, into the zarr store outputs/downscaled/<zarr_store> (topo_scale_zarr.ClimateDownscaler). 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.extract_pts_param`

```python
extract_pts_param(method='nearest', **kwargs)
```

Function to use a list point as input rather than cluster centroids from DEM segmentation (topo_sub.py/self.clustering_dem()). 



**Args:**
  df (dataFrame): 
 - <b>`method`</b> (str):  method of sampling 
 - <b>`**kwargs`</b>:  pd.read_csv() parameters 

**Returns:**
 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L295"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.extract_topo_cluster_param`

```python
extract_topo_cluster_param()
```

Function to segment a DEM in clusters and retain only the centroids of each cluster. 

:return: 



**TODO:**
 
- try to migrate most code of this function to topo_sub.py as a function itself segment_topo() 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L498"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.extract_topo_param`

```python
extract_topo_param()
```

Function to select which 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L765"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.get_WMO_observations`

```python
get_WMO_observations()
```

Function to download and parse in-situ data from WMO database 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L708"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.get_era5`

```python
get_era5()
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.get_era5_snowmapper`

```python
get_era5_snowmapper(surf_plev, lastday)
```

Funtion to call fetching of ERA5 data 

**TODO:**
 
- merge monthly data into one file (cdo?)- this creates massive slow down! 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L761"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.get_ifs_forecast`

```python
get_ifs_forecast()
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.get_lastday`

```python
get_lastday()
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.load_project`

```python
load_project()
```

Function to load pre-existing TopoPyScale project saved in files 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L758"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.remap_netcdf`

```python
remap_netcdf(filepath, file_type)
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.search_optimum_number_of_clusters`

```python
search_optimum_number_of_clusters(
    cluster_range=array([100, 300, 500, 700, 900]),
    scaler_type=StandardScaler(),
    plot=True
)
```

Function to test what would be an appropriate number of clusters 

**Args:**
 
 - <b>`cluster_range`</b> (int array):  numpy array or list of number of clusters to compute scores. 



**Returns:**
 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L820"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_crocus`

```python
to_crocus(fname_format='CROCUS_pt_*.nc', scale_precip=1)
```

function to export toposcale output to crocus format .nc. This functions saves one file per point_name 



**Args:**
 
 - <b>`fout_format`</b> (str):  filename format. point_name is inserted where * is 
 - <b>`scale_precip`</b> (float):  scaling factor to apply on precipitation. Default is 1 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L789"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_cryogrid`

```python
to_cryogrid(fname_format='Cryogrid_pt_*.nc', precip_partition='continuous')
```

wrapper function to export toposcale output to cryosgrid format from TopoClass 



**Args:**
 
 - <b>`fname_format`</b> (str):  filename format. point_name is inserted where * is 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L814"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_fsm`

```python
to_fsm(fname_format='FSM_pt_*.txt')
```

function to export toposcale output to FSM format 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L897"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_geotop`

```python
to_geotop(fname_format='meteo_*.txt')
```

function to export toposcale output to FSM format 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L903"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_musa`

```python
to_musa(
    layout='grid',
    path=None,
    bbox=None,
    stride=1,
    cell_mask=None,
    start=None,
    end=None
)
```

function to export TopoPyScale output as forcing for MuSA MuSA: https://github.com/ealonsogzl/MuSA 

Writes daily forcing files, mask.nc, and DEM.nc ('grid') or points_meta.csv ('points') to path. See topo_export.to_musa() for the layouts, files, units and the matching MuSA config.py settings. 



**Args:**
 
 - <b>`layout`</b> (str):  'grid' (default): one MuSA cell per DEM cell, var(time, northing, easting) on the DEM grid,  each cell with the forcing of its TopoSUB cluster. Needs the toposub sampling method. 
 - <b>`'points'`</b>:  one MuSA cell per point or cluster, var(time, y, x) with y = point id and x = [0]. 
 - <b>`path`</b> (str):  output folder. Default: outputs/musa_<layout> 
 - <b>`bbox`</b> (list):  [xmin, xmax, ymin, ymax] in DEM coordinates, to crop the DEM for layout 'grid' 
 - <b>`stride`</b> (int):  keep every stride-th DEM cell for layout 'grid' 
 - <b>`cell_mask`</b> (array):  boolean (y, x) array of the DEM cells to simulate for layout 'grid' 
 - <b>`start`</b> (str):  first timestep to export. Default: first downscaled timestep 
 - <b>`end`</b> (str):  last timestep to export. Default: last downscaled timestep 



**Returns:**
 
 - <b>`dict`</b>:  grid shape, number of cells to simulate, and date_ini, date_end and dt for MuSA's config.py 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L872"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_netcdf`

```python
to_netcdf(file_out='output.nc', variables=None)
```

function to export toposcale output to one single generic netcdf format, compressed 



**Args:**
 
 - <b>`file_out`</b> (str):  name of export file 
 - <b>`variables`</b> (list str):  list of variable to export. Default exports all variables 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L860"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_snowmodel`

```python
to_snowmodel(fname_format='Snowmodel_stn_*.csv')
```

function to export toposcale output to snowmodel format .ascii, for single station standard 

 fout_format: str, filename format. point_name is inserted where * is 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L891"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_snowpack`

```python
to_snowpack(fname_format='smet_pt_*.smet')
```

function to export toposcale output to FSM format 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L835"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.to_surfex`

```python
to_surfex(
    fname_format='FORCING*.nc',
    scale_precip=1,
    year_start=None,
    year_end=None
)
```

function to export toposcale output to surfex format .nc. This functions saves one file per point_name 



**Args:**
 
 - <b>`fout_format`</b> (str):  filename format. point_name is inserted where * is 
 - <b>`scale_precip`</b> (float):  scaling factor to apply on precipitation. Default is 1 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L755"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Topoclass.unzip_file`

```python
unzip_file(filepath)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L951"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Plotting`
Sub-Class with plotting functions for topoclass object 




---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L1002"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Plotting.horizon`

```python
horizon()
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L984"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Plotting.map_center_clusters`

```python
map_center_clusters(
    background_var='elevation',
    cmap=<matplotlib.colors.ListedColormap object at 0x7f056d90f1d0>,
    hillshade=True,
    **kwargs
)
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L991"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Plotting.map_clusters`

```python
map_clusters(**kwargs)
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L977"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Plotting.map_terrain`

```python
map_terrain(var='elevation', hillshade=True, **kwargs)
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L960"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Plotting.map_variable`

```python
map_variable(
    var='t',
    time_step=1,
    time=None,
    cmap=<matplotlib.colors.LinearSegmentedColormap object at 0x7f056d31be90>,
    hillshade=True,
    **kwargs
)
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L999"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Plotting.solar_geom`

```python
solar_geom()
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topoclass.py#L996"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Plotting.timeseries`

```python
timeseries()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
