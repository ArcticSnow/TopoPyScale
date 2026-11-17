<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_scale_zarr`
S. Filhol, 2025 

TopoSCALE downscaling from an ERA5 zarr store to a zarr store of downscaled points. 

The ERA5 store (pressure levels z, t, q, u, v and surface fields z_surf, sp, t2m, d2m, tp, ssrd, strd) is read in blocks of time steps and points. For each block, only the ERA5 nodes surrounding the points of the block are read, and all points are downscaled at once with the array functions of toposcale_core.py. Results are written to one zarr store with dimensions (point_name, time), chunked by point and by time block. 

Create the ERA5 store from yearly netcdf files with fetch_era5.netcdf_to_zarr(). 

**Global Variables**
---------------
- **PRECIP_LAPSE_MAX_FACTOR**
- **PLEV_VARS**
- **SURF_VARS**
- **SOLAR_VARS**
- **varout_default**
- **VAR_ATTRS**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tp_attrs`

```python
tp_attrs(tstep_h)
```






---

## <kbd>class</kbd> `ClimateDownscaler`
Downscale an ERA5 zarr store to a list of points. 



**Args:**
 
 - <b>`era5_zarr_path`</b> (str):  path to the ERA5 zarr store (see fetch_era5.netcdf_to_zarr) 
 - <b>`output_path`</b> (str):  directory of the output store 
 - <b>`df_centroids`</b> (dataframe):  points with point_name, x, y, lat, lon, elevation, slope, aspect, svf 
 - <b>`da_horizon`</b> (dataarray):  horizon angles [rad] of the points (azimuth, point_name), from  topo_param.compute_horizon_points, or of the DEM grid (azimuth, y, x), taken at the nearest pixel 
 - <b>`ds_solar`</b> (dataset):  solar geometry (point_name, time), from solar_geom.get_solar_geom 
 - <b>`target_EPSG`</b> (int):  EPSG code of x, y 
 - <b>`start_date, end_date (str)`</b>:  period to downscale (end_date included) 
 - <b>`tstep`</b> (str):  '1h', '3h' or '6h' 
 - <b>`varout`</b> (list):  variables to store, default varout_default 
 - <b>`interp_method`</b> (str):  'idw' or 'linear' 
 - <b>`lw_terrain_flag`</b> (bool):  add the longwave emitted by the surrounding terrain 
 - <b>`precip_lapse_rate_flag`</b> (bool):  apply the precipitation elevation factor 
 - <b>`store_name`</b> (str):  name of the output zarr store, default 'down.zarr' 
 - <b>`precip_lapse_rate_max_factor`</b> (float):  bound of the precipitation elevation factor 
 - <b>`point_block`</b> (int):  number of points processed together, also the shard size of the output store along points 
 - <b>`time_block`</b> (int):  number of time steps processed together, also the chunk size of the output store along time. 
 - <b>`Default`</b>:  one year 
 - <b>`max_box_mb`</b> (float):  the ERA5 nodes around all points are read once per time block if they fit in this  memory (MB), otherwise once per point block 

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    era5_zarr_path,
    output_path,
    df_centroids,
    da_horizon,
    ds_solar,
    target_EPSG,
    start_date,
    end_date,
    tstep,
    varout=None,
    interp_method='idw',
    lw_terrain_flag=True,
    precip_lapse_rate_flag=False,
    store_name='down.zarr',
    precip_lapse_rate_max_factor=5.0,
    point_block=100,
    time_block=None,
    max_box_mb=2000
)
```








---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `downscale`

```python
downscale(n_core=1)
```

Downscale all points and write them to the output store. 



**Args:**
 
 - <b>`n_core`</b> (int):  number of threads processing point blocks in parallel. Blocks write disjoint shards. 



**Returns:**
 
 - <b>`dataset`</b>:  the output store opened with xarray 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `process_block`

```python
process_block(p0, p1, t0, t1, era=None)
```

Downscale points p0:p1 (positions in df_centroids) for time steps t0:t1 and write them to the store. 



**Args:**
 
 - <b>`era`</b> (tuple):  optional (arrays, box) already read by _read_era() on a box containing the points 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `setup_output_store`

```python
setup_output_store()
```

Create the output store: coordinates, attributes and empty variables chunked (1 point, time_block). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
