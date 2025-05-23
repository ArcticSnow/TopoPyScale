<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_param`
Set of functions to work with DEMs S. Filhol, Oct 2021 



**TODO:**
 
- an improvement could be to first copmute horizons, and then SVF to avoid computing horizon twice 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_epsg_pts`

```python
convert_epsg_pts(xs, ys, epsg_src=4326, epsg_tgt=3844)
```

Simple function to convert a list fo poitn from one projection to another oen using PyProj 



**Args:**
 
 - <b>`xs`</b> (array):  1D array with X-coordinate expressed in the source EPSG 
 - <b>`ys`</b> (array):  1D array with Y-coordinate expressed in the source EPSG 
 - <b>`epsg_src`</b> (int):  source projection EPSG code 
 - <b>`epsg_tgt`</b> (int):  target projection EPSG code 



**Returns:**
 
 - <b>`array`</b>:  Xs 1D arrays of the point coordinates expressed in the target projection 
 - <b>`array`</b>:  Ys 1D arrays of the point coordinates expressed in the target projection 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_extent_latlon`

```python
get_extent_latlon(dem_file, epsg_src)
```

Function to extract DEM extent in Lat/Lon 



**Args:**
 
 - <b>`dem_file`</b> (str):  path to DEM file (GeoTiFF) 
 - <b>`epsg_src`</b> (int):  EPSG projection code 



**Returns:**
 
 - <b>`dict`</b>:  extent in lat/lon, {latN, latS, lonW, lonE} 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_pts_param`

```python
extract_pts_param(df_pts, ds_param, method='nearest')
```

Function to sample DEM parameters for a list point. This is used as an alternative the the TopoSub method, to perform downscaling at selected locations (x,y) WARNING: the projection and coordiante system of the EDM and point coordinates MUST be the same! 



**Args:**
 
 - <b>`df_pts`</b> (dataframe):  list of points coordinates with coordiantes in (x,y). 
 - <b>`ds_param`</b> (dataset):  dem parameters 
 - <b>`method`</b> (str):  sampling method. Supported 'nearest', 'linear' interpolation, 'idw' interpolation (inverse-distance weighted) 



**Returns:**
 
 - <b>`dataframe`</b>:  df_pts updated with new columns ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf'] 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_dem_param`

```python
compute_dem_param(
    dem_file,
    fname='ds_param.nc',
    project_directory=PosixPath('.'),
    output_folder='outputs'
)
```

Function to compute and derive DEM parameters: slope, aspect, sky view factor 



**Args:**
 
 - <b>`dem_file`</b> (str):  path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM) 



**Returns:**
 
 - <b>`dataset`</b>:  x, y, elev, slope, aspect, svf 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_horizon`

```python
compute_horizon(
    dem_file,
    azimuth_inc=30,
    num_threads=None,
    fname='da_horizon.nc',
    output_directory=PosixPath('outputs')
)
```

Function to compute horizon angles for 



**Args:**
 
 - <b>`dem_file`</b> (str):  path and filename of the dem 
 - <b>`azimuth_inc`</b> (int):  angle increment to compute horizons at, in Degrees [0-359] 
 - <b>`num_threads`</b> (int):  number of threads to parallize on 



**Returns:**
 
 - <b>`dataarray`</b>:  all horizon angles for x,y,azimuth coordinates  






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
