<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_param`
Set of functions to work with DEMs S. Filhol, Oct 2021 



**TODO:**
 
- an improvement could be to first copmute horizons, and then SVF to avoid computing horizon twice 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

## <kbd>function</kbd> `compute_dem_complexity`

```python
compute_dem_complexity(
    elev,
    slope,
    weights={'slope': 0.5, 'elev_norm': 0.3, 'curvature': 0.2}
)
```

Function to compute topographic complexity index. This index is an arbitray index considering  



**Args:**
 
 - <b>`elev`</b> (array):  ndarray of elevations  
 - <b>`slope`</b> (array):  ndarray of slopes  
 - <b>`weights`</b> (dict):  dictionnary of the relative weights of slope vs normalized elevation vs curvature. Sum must be equal to 1 



**Returns:**
 
 - <b>`array`</b>:  complexity index  


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `open_dem`

```python
open_dem(dem_file)
```

Function to quickly open a DEM raster using xarray and removing the band dimension, combersome when dealing with DEM 



**Args:**
 
 - <b>`dem_file`</b> (str):  path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM) 



**Returns:**
 
 - <b>`dataset`</b>:  x, y, elevation 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Horizon angles of every pixel of the DEM, with topocalc (e.g. for maps). 

topocalc follows lines of sight that are staircases stepping at the same columns (or rows) for all pixels. For azimuths a few degrees off the grid axes, the pixels just before a step read their nearest terrain from the adjacent row, which leaves streaks in the horizon maps. Topoclass uses compute_horizon_points() instead, which traces a straight ray from each point. 



**Args:**
 
 - <b>`dem_file`</b> (str):  path and filename of the dem 
 - <b>`azimuth_inc`</b> (int):  angle increment to compute horizons at, in Degrees [0-359] 
 - <b>`num_threads`</b> (int):  number of threads to parallize on 



**Returns:**
 
 - <b>`dataarray`</b>:  all horizon angles for x,y,azimuth coordinates  




---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L357"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `horizon_rays`

```python
horizon_rays(
    elevation,
    x,
    y,
    px,
    py,
    azimuths,
    max_rays=2000000,
    n_threads=1,
    prune=True
)
```

Horizon angles of points along straight rays, for a list of azimuths. 

From each point, a ray is followed in each direction one cell at a time along the major grid axis, the DEM being interpolated bilinearly at each sample. The horizon is the steepest elevation angle to a sample, or 0 if no sample is higher than the point (as topocalc). The observer is on the DEM surface, interpolated at the point. Rays stop at the edge of the DEM, or as soon as no sample further away can be steeper than the horizon found ((max DEM elevation - point elevation) / distance), which does not change the result. No-data cells (NaN) never form a horizon. 



**Args:**
 
 - <b>`elevation`</b> (2D array):  DEM (y, x) 
 - <b>`x, y (1D arrays)`</b>:  coordinates of the DEM columns and rows (regular grid, either orientation) 
 - <b>`px, py (1D arrays)`</b>:  coordinates of the points, in the projection of the DEM 
 - <b>`azimuths`</b> (1D array):  directions [deg]: 0 = south, 90 = east, -90 = west, +-180 = north 
 - <b>`max_rays`</b> (int):  number of rays (azimuths x points) in memory at once 
 - <b>`n_threads`</b> (int):  number of threads tracing blocks of points. The result does not depend on it 
 - <b>`prune`</b> (bool):  stop the rays when no further sample can be steeper. Only for testing 



**Returns:**
 
 - <b>`array`</b>:  horizon elevation angles [rad], (azimuth, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_horizon_points`

```python
compute_horizon_points(
    dem_file,
    df_points,
    azimuth_inc=10,
    fname='da_horizon.nc',
    output_directory=PosixPath('outputs'),
    n_threads=1
)
```

Horizon angles of a list of points, along straight rays from each point (see horizon_rays). 



**Args:**
 
 - <b>`dem_file`</b> (str):  path and filename of the dem 
 - <b>`df_points`</b> (dataframe):  points with columns x, y (projection of the DEM) and point_name 
 - <b>`azimuth_inc`</b> (int):  angle increment of the azimuths [deg], centred in their bins as compute_horizon() 
 - <b>`fname`</b> (str):  filename of the netcdf file written in output_directory. None to skip writing 
 - <b>`output_directory`</b> (Path):  output directory 
 - <b>`n_threads`</b> (int):  number of threads 



**Returns:**
 
 - <b>`dataarray`</b>:  horizon angles [rad], (azimuth, point_name), with the point coordinates x, y 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
