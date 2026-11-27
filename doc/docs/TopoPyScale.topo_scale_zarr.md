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
- **ACC_VARS**
- **DIRECT_DIFFUSE**
- **varout_default**
- **VAR_ATTRS**
- **SURFACE_DIAGNOSTICS**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `accumulation`

```python
accumulation(era_times, era_attrs, tstep_h)
```

How the accumulated fields of an ERA5 store cover the time steps of the downscaling. 

A store holds at each time stamp the accumulation over the accumulation_hours hours ending at the stamp (store attribute; 1 h for ERA5 files and for stores written before it was recorded). When the store is finer than the time step and its accumulations cover its own steps (an hourly store downscaled at 3 h), the k values of each window are summed. 



**Args:**
 
 - <b>`era_times`</b> (DatetimeIndex):  time axis of the store 
 - <b>`era_attrs`</b> (dict):  attributes of the store 
 - <b>`tstep_h`</b> (int):  time step of the downscaling [h] 



**Returns:**
 
 - <b>`k`</b> (int):  number of stored values summed per time step, acc_hours (int): hours accumulated per time step 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tp_attrs`

```python
tp_attrs(tstep_h)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClimateDownscaler`
Downscale an ERA5 zarr store to a list of points. 



**Args:**
 
 - <b>`era5_zarr_path`</b> (str):  path to the ERA5 zarr store (see era5_store and fetch_era5.netcdf_to_zarr). z_surf may  be static (latitude, longitude) or given at every time step 
 - <b>`output_path`</b> (str):  directory of the output store 
 - <b>`df_centroids`</b> (dataframe):  points with point_name, x, y, lat, lon, elevation, slope, aspect, svf 
 - <b>`da_horizon`</b> (dataarray):  horizon angles [rad] of the points (azimuth, point_name), from  topo_param.compute_horizon_points, or of the DEM grid (azimuth, y, x), taken at the nearest pixel 
 - <b>`ds_solar`</b> (dataset):  precomputed solar geometry (point_name, time), from solar_geom.get_solar_geom. None 
 - <b>`(recommended)`</b>:  the solar geometry of each block of points and time steps is computed when the block is downscaled, from the coordinates of the points, without holding it for the whole period in memory 
 - <b>`target_EPSG`</b> (int):  EPSG code of x, y 
 - <b>`start_date, end_date (str)`</b>:  period to downscale (end_date included) 
 - <b>`tstep`</b> (str):  '1h', '3h' or '6h' 
 - <b>`varout`</b> (list):  variables to store, default varout_default 
 - <b>`interp_method`</b> (str):  'idw' or 'linear' 
 - <b>`lw_terrain_flag`</b> (bool):  add the longwave emitted by the surrounding terrain 
 - <b>`precip_lapse_rate_flag`</b> (bool):  apply the precipitation elevation factor 
 - <b>`surface_effect`</b> (dict):  surface-effect correction, from surface_effect.setup() (None: off). Its diagnostics  t_pl, dT_surf and dTd_surf are added to the outputs when 'diagnostics' is true 
 - <b>`store_name`</b> (str):  name of the output zarr store, default 'down.zarr' 
 - <b>`precip_lapse_rate_max_factor`</b> (float):  bound of the precipitation elevation factor 
 - <b>`point_block`</b> (int):  number of points processed together, also the shard size of the output store along points 
 - <b>`time_block`</b> (int):  number of time steps processed together, also the chunk size of the output store along time. 
 - <b>`Default`</b>:  one year 
 - <b>`max_box_mb`</b> (float):  the ERA5 nodes around all points are read once per time block if they fit in this  memory (MB), otherwise once per point block 
 - <b>`shading_substeps`</b> (int):  sub-steps of the radiation accumulation window at which terrain shading is tested; 
 - <b>`the direct beam is scaled by the sunlit fraction of the window (1`</b>:  one test at the centre, all or 
 - <b>`nothing). Default`</b>:  solar_substeps (or the solar_substeps of ds_solar) 
 - <b>`accumulation_window`</b> (str):  window over which the radiation fluxes are accumulated, ending at the time stamp. 
 - <b>`Default`</b>:  the hours the accumulated fields cover per time step (see accumulation(): the time step for stores of window sums or hourly stores, 1 h for stores sampling the hourly ERA5 accumulations). Used when ds_solar is None 
 - <b>`solar_substeps`</b> (int):  sub-steps of the window averaged for the top-of-atmosphere irradiance. Default 6 
 - <b>`solar_method`</b> (str):  pvlib solar position method, default 'nrel_numpy' (vectorized) 
 - <b>`solar_constant`</b> (float):  total solar irradiance at 1 AU [W m-2] 
 - <b>`direct_diffuse`</b> (str):  direct/diffuse split of the ERA5 shortwave: 'empirical' (default, from the clearness  index, Ruiz-Arias et al. 2010) or 'era5' (the direct radiation fdir of the store) 
 - <b>`lw_terrain_temperature`</b>:  temperature of the terrain emitting longwave to the point (lw_terrain_flag): 'air'  (default, the air temperature at the point), 'fixed' (273.15 K), a value [K], or a DataArray  (point_name, time) [K], e.g. surface temperatures of a snow model 
 - <b>`diffuse_model`</b> (str):  'isotropic' (default) or 'hay_davies' (circumsolar part from the sun direction) 
 - <b>`terrain_albedo`</b>:  albedo of the surrounding terrain reflecting shortwave to the point: 0 (default, off), a  value, or a DataArray (point_name, time), e.g. snow albedo of a snow model. Adds the output SW_terrain 
 - <b>`wind`</b> (dict):  wind options from wind.setup() (None: pressure-level wind). Blending needs u10 and v10 in the  ERA5 store 

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ClimateDownscaler.__init__`

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
    max_box_mb=2000,
    surface_effect=None,
    shading_substeps=None,
    accumulation_window=None,
    solar_substeps=6,
    solar_method='nrel_numpy',
    solar_constant=1361.0,
    direct_diffuse='empirical',
    lw_terrain_temperature='air',
    diffuse_model='isotropic',
    terrain_albedo=0.0,
    wind=None
)
```








---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L527"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ClimateDownscaler.downscale`

```python
downscale(n_core=1)
```

Downscale all points and write them to the output store. 



**Args:**
 
 - <b>`n_core`</b> (int):  number of threads processing point blocks in parallel. Blocks write disjoint shards. 



**Returns:**
 
 - <b>`dataset`</b>:  the output store opened with xarray 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L496"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ClimateDownscaler.process_block`

```python
process_block(p0, p1, t0, t1, era=None)
```

Downscale points p0:p1 (positions in df_centroids) for time steps t0:t1 and write them to the store. 



**Args:**
 
 - <b>`era`</b> (tuple):  optional (arrays, box) already read by _read_era() on a box containing the points 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale_zarr.py#L363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ClimateDownscaler.setup_output_store`

```python
setup_output_store()
```

Create the output store: coordinates, attributes and empty variables chunked (1 point, time_block). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
