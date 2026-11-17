<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_scale`
Toposcale functionalities S. Filhol, Oct 2021 

======= Organization of input data to Toposcale ====== dem_description (nb_points)  X  Y,  elev,  slope,  aspect,  svf,  x_ERA,  y_ERA,  elev_ERA, 

horizon_angles (nb_points * bins) solar_geom (nb_points * time * 2) surface_climate_data (spatial + time + var) plevels_climate_data (spatial + time + var + plevel) 

======= Dataset naming convention:  ========= ds_surf => dataset direct from ERA5 single level surface ds_plev => dataset direct from ERA5 pressure levels 

ds_surf_pt => 3*3 grid ERA5 single surface level around a given point (lat,lon) ds_plev_pt => 3*3 grid ERA5 pressure level around a given point (lat,lon) 

plev_interp => horizontal interpolation of the 3*3 grid at the point (lat,lon) for the pressure levels surf_interp => horizontal interpolation of the 3*3 grid at the point (lat,lon) for the single surface level 

top => closest pressure level above the point (lat,lon,elev) for each timesep bot => closest pressure level below the point (lat,lon,elev) for each timesep 

down_pt => downscaled data time series (t, u, v, q, LW, SW, tp) 

**Global Variables**
---------------
- **g**
- **R**
- **PRECIP_LAPSE_COEF**
- **PRECIP_LAPSE_MAX_FACTOR**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clear_files`

```python
clear_files(path: Union[Path, str])
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pt_downscale_interp`

```python
pt_downscale_interp(row, ds_plev_pt, ds_surf_pt, meta)
```

Function to interpolate atmospheric variables horizontally nd vertically to point 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pt_downscale_radiations`

```python
pt_downscale_radiations(row, ds_solar, horizon_da, meta, output_dir)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_scale.py#L359"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `downscale_climate`

```python
downscale_climate(
    project_directory,
    climate_directory,
    output_directory,
    df_centroids,
    horizon_da,
    ds_solar,
    target_EPSG,
    start_date,
    end_date,
    tstep,
    interp_method='idw',
    lw_terrain_flag=True,
    precip_lapse_rate_flag=False,
    file_pattern='down_pt*.nc',
    n_core=4,
    precip_lapse_rate_max_factor=5.0
)
```

DEPRECATED: use topo_scale_zarr.ClimateDownscaler (Topoclass.downscale_climate does). 

Function to perform downscaling of climate variables (t,q,u,v,tp,SW,LW) based on Toposcale logic 



**Args:**
 
 - <b>`project_directory`</b> (str):  path to project root directory climate_directory (str): output_directory (path): 
 - <b>`df_centroids`</b> (dataframe):  containing a list of point for which to downscale (includes all terrain data) 
 - <b>`horizon_da`</b> (dataarray):  horizon angles [rad] of the points (azimuth, point_name) or of the DEM grid (azimuth, y, x) 
 - <b>`target_EPSG`</b> (int):  EPSG code of the DEM start_date: end_date: 
 - <b>`interp_method`</b> (str):  interpolation method for horizontal interp. 'idw' or 'linear' 
 - <b>`lw_terrain_flag`</b> (bool):  flag to compute contribution of surrounding terrain to LW or ignore 
 - <b>`tstep`</b> (str):  timestep of the input data, default = 1h 
 - <b>`file_pattern`</b> (str):  filename pattern for storing downscaled points, default = 'down_pt_*.nc' 
 - <b>`n_core`</b> (int):  number of core on which to distribute processing 
 - <b>`precip_lapse_rate_max_factor`</b> (float):  bound of the precipitation elevation factor, see precip_lapse_factor() 



**Returns:**
 
 - <b>`dataset`</b>:  downscaled data organized with time, point_name, lat, long 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
