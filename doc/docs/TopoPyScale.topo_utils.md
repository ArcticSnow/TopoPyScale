<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_utils`





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ds_to_indexed_dataframe`

```python
ds_to_indexed_dataframe(ds)
```

Function to convert an Xarray dataset with multi-dimensions to indexed dataframe (and not a multilevel indexed dataframe). WARNING: this only works if the variable of the dataset have all the same dimensions! 

By default the ds.to_dataframe() returns a multi-index dataframe. Here the coordinates are transfered as columns in the dataframe 



**Args:**
 
 - <b>`ds`</b> (dataset):  xarray dataset with all variable of same number of dimensions 



**Returns:**
 pandas dataframe: 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_versionning`

```python
get_versionning()
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `FsmMetParser`

```python
FsmMetParser(file, freq='1h', resample=False)
```

Parses FSM forcing files from toposcale sims 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `FsmSnowParser`

```python
FsmSnowParser(file, freq='1H', resample=False)
```

parses FSM output fuiles 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `SmetParser`

```python
SmetParser(file, doresample=True, freq='1H', NA_val=-999)
```

val_file = full path to a smet resample = "TRUE" or "FALSE" freq = '1D' '1H' etc 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `val_plots`

```python
val_plots()
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_utils.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
