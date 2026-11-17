<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_param`
Set of functions to work with DEMs: slope, aspect, sky view factor and horizons. S. Filhol, Oct 2021 

Horizons at points are traced along straight rays (horizon_rays). Horizons of every cell, used for the sky view factor (sky_view_factor) and maps (compute_horizon), are computed with a line sweep (horizon_sweep). Both are described in the documentation (Features > Slope, aspect and sky view factor; Features > Horizons). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_dem_param`

```python
compute_dem_param(
    dem_file,
    fname='ds_param.nc',
    project_directory=PosixPath('.'),
    output_folder='outputs',
    n_processes=1
)
```

Function to compute and derive DEM parameters: slope, aspect (gradient_d8), sky view factor (sky_view_factor). An existing file is loaded as is; it is rewritten only if parameters are missing from it. 



**Args:**
 
 - <b>`dem_file`</b> (str):  path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM) 
 - <b>`fname`</b> (str):  netcdf file name 
 - <b>`project_directory`</b> (Path):  project directory 
 - <b>`output_folder`</b> (str):  folder of fname, in project_directory 
 - <b>`n_processes`</b> (int):  number of processes computing the sky view factor 



**Returns:**
 
 - <b>`dataset`</b>:  x, y, elev, slope, aspect, svf 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_horizon`

```python
compute_horizon(
    dem_file,
    azimuth_inc=30,
    n_processes=1,
    fname='da_horizon_grid.nc',
    output_directory=PosixPath('outputs'),
    near=32
)
```

Horizon angles of every cell of the DEM (e.g. for maps), with the line sweep of horizon_sweep(). Topoclass computes the horizons of the points with compute_horizon_points() instead. 



**Args:**
 
 - <b>`dem_file`</b> (str):  path and filename of the dem 
 - <b>`azimuth_inc`</b> (int):  angle increment to compute horizons at, in Degrees [0-359] 
 - <b>`n_processes`</b> (int):  number of processes, one azimuth each 
 - <b>`fname`</b> (str):  netcdf file written in output_directory. None to skip writing 
 - <b>`output_directory`</b> (Path):  output directory 
 - <b>`near`</b> (int):  cells traced exactly along each ray before the sweep takes over 



**Returns:**
 
 - <b>`dataarray`</b>:  horizon angles [rad], (azimuth, y, x) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gradient_d8`

```python
gradient_d8(dem, dx, dy, aspect_rad=False)
```

Slope and aspect from the 3x3 neighbourhood (Horn's finite differences), port of topocalc.gradient.gradient_d8 (USDA ARS NWRC, public domain). Edges are extrapolated linearly. 

For a cell e and its neighbours  | a | b | c |  | d | e | f |  | g | h | i | dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 dx) and dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 dy). 



**Args:**
 
 - <b>`dem`</b> (array):  elevation (y, x) 
 - <b>`dx, dy (float)`</b>:  cell size along x and y (dy < 0 for a north-up raster) 
 - <b>`aspect_rad`</b> (bool):  return the aspect in IPW radians (0 = south, positive east) instead of degrees 



**Returns:**
 slope [rad], aspect [deg clockwise from north, or IPW rad] 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L476"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `horizon_sweep`

```python
horizon_sweep(elevation, dx, dy, azimuth, near=32)
```

Horizon elevation angle [rad] of every cell for one azimuth, by a line sweep (see _sweep). 



**Args:**
 
 - <b>`elevation`</b> (2D array):  DEM (y, x), NaN for no data 
 - <b>`dx, dy (float)`</b>:  cell size along x and y, signed as the coordinates (dy < 0 for a north-up raster) 
 - <b>`azimuth`</b> (float):  direction [deg]: 0 = south, 90 = east, -90 = west, +-180 = north 
 - <b>`near`</b> (int):  cells traced exactly along each ray before the sweep takes over 



**Returns:**
 
 - <b>`array`</b>:  horizon angles [rad] (>= 0), NaN on no-data cells 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sky_view_factor`

```python
sky_view_factor(elevation, dx, dy, n_angles=72, near=32, n_processes=1)
```

Sky view factor of every cell, from horizons computed by a line sweep (horizon_sweep). 

The sky view factor integrates the sky visible above the local horizon, weighted for a tilted surface (Dozier and Frew, 1990, eq. 7b):  svf = 1/(2 pi) * integral over azimuth phi of [cos(S) sin^2(H) + sin(S) cos(phi - A) (H - sin(H) cos(H))] with S the slope, A the aspect and H the horizon angle from zenith in direction phi. It is evaluated at n_angles azimuths, keeping the positive part of the integrand, as topocalc did. Azimuths are added in order whatever the number of processes, so the result does not depend on it. 



**Args:**
 
 - <b>`elevation`</b> (2D array):  DEM (y, x) 
 - <b>`dx, dy (float)`</b>:  cell size along x and y, signed as the coordinates (dy < 0 for a north-up raster) 
 - <b>`n_angles`</b> (int):  number of azimuths, from -180 deg every 360 / n_angles 
 - <b>`near`</b> (int):  cells traced exactly along each ray before the sweep takes over 
 - <b>`n_processes`</b> (int):  number of processes, one azimuth each 



**Returns:**
 
 - <b>`array`</b>:  sky view factor (0 to 1), NaN on no-data cells and their neighbours (slope undefined, as topocalc) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L581"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_param.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
