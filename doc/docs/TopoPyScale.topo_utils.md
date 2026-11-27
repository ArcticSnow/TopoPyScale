<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_utils`





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `as_str`

```python
as_str(values)
```

Fixed-width unicode array (zarr v3 returns numpy StringDType, which cannot be cast with astype(str)). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `horizon_matches_points`

```python
horizon_matches_points(da_horizon, df_points, tol=0.001)
```

True if da_horizon holds the horizons of exactly these points (names and positions within tol m). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `horizon_at_points`

```python
horizon_at_points(da_horizon, df_points)
```

Horizon profile of each point of df_points, in the order of its rows. 



**Args:**
 
 - <b>`da_horizon`</b> (dataarray):  horizon angles [rad] of the points (azimuth, point_name), from  topo_param.compute_horizon_points 
 - <b>`df_points`</b> (dataframe):  points with columns point_name, x, y 



**Returns:**
 
 - <b>`dataarray`</b>:  horizon angles [rad], (azimuth, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `open_downscaled_store`

```python
open_downscaled_store(store)
```

Open the zarr store of downscaled points (dimensions point_name, time), lazily with Dask. 



**Args:**
 
 - <b>`store`</b> (str):  path to the zarr store, e.g. outputs/downscaled/down.zarr 



**Returns:**
 
 - <b>`dataset`</b>:  point_name as plain strings 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `open_netcdf_stack`

```python
open_netcdf_stack(files, concat_dim)
```

Open netcdf files one by one (lazily, with Dask) and concatenate them along concat_dim. 



**Args:**
 
 - <b>`files`</b> (list):  netcdf files, in the order of concatenation 
 - <b>`concat_dim`</b> (str):  dimension along which to concatenate 



**Returns:**
 dataset 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_downscaled`

```python
read_downscaled(path='outputs/downscaled/down.zarr')
```

Read downscaled points as one dataset, lazily with Dask. 



**Args:**
 
 - <b>`path`</b> (str):  zarr store of downscaled points (*.zarr) 



**Returns:**
 
 - <b>`dataset`</b>:  dimensions point_name, time 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ds_to_indexed_dataframe`

```python
ds_to_indexed_dataframe(ds, dim_order=None)
```

Function to convert an Xarray dataset with multi-dimensions to indexed dataframe (and not a multilevel indexed dataframe). WARNING: this only works if the variable of the dataset have all the same dimensions! 

By default the ds.to_dataframe() returns a multi-index dataframe. Here the coordinates are transfered as columns in the dataframe 



**Args:**
 
 - <b>`ds`</b> (dataset):  xarray dataset with all variable of same number of dimensions 
 - <b>`dim_order`</b> (list):  order of the dimensions in the rows (e.g. ['y', 'x']: row-major for a map). Default: the  order of ds.dims, which depends on how the dataset was built (e.g. (x, y) from a GeoTIFF, (y, x) from a  netcdf file) 



**Returns:**
 pandas dataframe: 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `multicore_pooling`

```python
multicore_pooling(fun, fun_param, n_cores)
```

Function to perform multiprocessing on n_cores 

**Args:**
 
 - <b>`fun`</b> (obj):  function to distribute 
 - <b>`fun_param zip(list)`</b>:  zip list of function arguments 
 - <b>`n_core`</b> (int):  number of cores 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `multithread_pooling`

```python
multithread_pooling(fun, fun_param, n_threads)
```

Function to perform multiprocessing on n_threads 

**Args:**
 
 - <b>`fun`</b> (obj):  function to distribute 
 - <b>`fun_param zip(list)`</b>:  zip list of functino arguments 
 - <b>`n_core`</b> (int):  number of threads 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_versionning`

```python
get_versionning()
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `FsmMetParser`

```python
FsmMetParser(file, freq='1h', resample=False)
```

Parses FSM forcing files from toposcale sims 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `FsmSnowParser`

```python
FsmSnowParser(file, freq='1h', resample=False)
```

parses FSM output fuiles 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `FsmPlot_ensemble`

```python
FsmPlot_ensemble(wdir, simvar, sampleN)
```

Function to plot ensemble results for a given variable and point. 



**Args:**
 
 - <b>`wdir`</b>:  full path to sim directory 
 - <b>`simvar`</b>:  variable to plot (str) 
 - <b>`sampleN`</b>:  sample number to plot 

**Returns:**
 
 - <b>`sampleN`</b>:  sample number 



**Example:**
 FsmPlot_ensemble('/home/joel/sim/topoPyscale_davos/', "HS", 0) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `SmetParser`

```python
SmetParser(file, doresample=True, freq='1h', NA_val=-999)
```

val_file = full path to a smet resample = "TRUE" or "FALSE" freq = '1D' '1h' etc 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `getCoordinatePixel`

```python
getCoordinatePixel(map_path, x_coord, y_coord, epsg_in, epsg_out)
```

Function to find which sample a point exists in. Can accept any combination of projections of the map and points. 



**Args:**
 
 - <b>`map_path`</b>:  full path to map 
 - <b>`x_coord`</b>:  x coord of point 
 - <b>`y_coord`</b>:  y coord of point 
 - <b>`epsg_in`</b>:  input proj epsg code (lonlat: 4326) 
 - <b>`epsg_out`</b>:  ouput proj epsg code 



**Returns:**
 
 - <b>`sampleN`</b>:  sample number 



**Example:**
 # get epsg of map from TopoPyScale import topo_da as da epsg_out, bbxox = da.projFromLandform(map_path) 

getCoordinatePixel("/home/joel/sim/topoPyscale_davos/landform.tif", 9.80933, 46.82945, 4326, 32632) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L404"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `time_vecs`

```python
time_vecs(start_date, end_date)
```

Function to build timestamp vectors of year starts and year ends for arbitrary time range. 



**Args:**
 
 - <b>`start_date`</b> (str):  _description_ 
 - <b>`end_date`</b> (str):  _description_ 



**Returns:**
 
 - <b>`start_year`</b>:  vector of start of years (datetime64) 
 - <b>`end_year`</b>:  vector of end of year (datetime64) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
