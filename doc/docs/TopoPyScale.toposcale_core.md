<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.toposcale_core`
TopoSCALE physics on numpy arrays, for many points at once. 

All functions are pure: they take numpy arrays and return numpy arrays, with no I/O. Arrays are time-major: (time, point) for point fields, (time, level, point) after horizontal interpolation of pressure levels. 

Formulas of TopoSCALE (Fiddes and Gruber, 2014), used by topo_scale_zarr.py. 

**Global Variables**
---------------
- **PRECIP_LAPSE_COEF**
- **PRECIP_LAPSE_MAX_FACTOR**
- **G**
- **R**
- **EPS**
- **SIGMA**
- **LW_X1**
- **LW_X2**
- **T_TERRAIN**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `horizontal_weights`

```python
horizontal_weights(row, lats, lons, Xs, Ys, method='idw')
```

Weights of the ERA5 nodes surrounding a point for the horizontal interpolation. 



**Args:**
 
 - <b>`row`</b>:  point with attributes x, y (DEM projection) and lat, lon (WGS84) 
 - <b>`lats`</b> (array):  latitudes of the nodes (1D) 
 - <b>`lons`</b> (array):  longitudes of the nodes (1D) 
 - <b>`Xs`</b> (array):  x coordinates of the nodes in the DEM projection (2D, latitude x longitude) 
 - <b>`Ys`</b> (array):  y coordinates of the nodes in the DEM projection (2D, latitude x longitude) 
 - <b>`method`</b> (str):  'idw' (inverse squared distance) or 'linear' (bilinear in latitude/longitude) 



**Returns:**
 
 - <b>`array`</b>:  2D weights (latitude x longitude), non-negative and summing to 1 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `precip_lapse_factor`

```python
precip_lapse_factor(elev_diff, months, max_factor=5.0)
```

Precipitation elevation adjustment factor (1 + c dz) / (1 - c dz) of Liston and Elder (2006). 

The factor diverges for c dz -> 1 (dz ~ 2.9 km with c = 0.35 km-1) and becomes negative beyond. c dz is therefore bounded so that the factor stays within [1/max_factor, max_factor]. 



**Args:**
 
 - <b>`elev_diff`</b> (array):  elevation of the point minus elevation of the ERA5 surface [m] 
 - <b>`months`</b> (array):  month of each time step (1-12) 
 - <b>`max_factor`</b> (float):  upper bound of the factor (> 1) 



**Returns:**
 
 - <b>`array`</b>:  adjustment factor 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `neighbours`

```python
neighbours(points, lats, lons, transformer, method='idw')
```

Indices of the 3x3 ERA5 nodes around each point and their interpolation weights. 



**Args:**
 
 - <b>`points`</b> (dataframe):  with columns x, y (DEM projection), lat, lon, point_name 
 - <b>`lats, lons (array)`</b>:  ERA5 latitude and longitude coordinates (1D) 
 - <b>`transformer`</b> (pyproj.Transformer):  WGS84 -> DEM projection 
 - <b>`method`</b> (str):  'idw' or 'linear', see horizontal_weights() 



**Returns:**
 ilat (int array, npts x 3), ilon (int array, npts x 3), weights (float array, npts x 3 x 3) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `weight_matrix`

```python
weight_matrix(ilat, ilon, weights, nlat, nlon)
```

Interpolation weights as a matrix (node x point) on a box of nlat x nlon ERA5 nodes. 



**Args:**
 
 - <b>`ilat, ilon (int arrays)`</b>:  npts x 3, indices of the neighbours within the box 
 - <b>`weights`</b> (array):  npts x 3 x 3 
 - <b>`nlat, nlon (int)`</b>:  size of the box 



**Returns:**
 
 - <b>`array or scipy sparse matrix`</b>:  (nlat * nlon, npts). Dense for small boxes, sparse otherwise. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_weights`

```python
apply_weights(field, W)
```

Horizontal interpolation of field (..., nlat, nlon) with a weight matrix from weight_matrix(): (..., npts). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `interpolate_horizontal`

```python
interpolate_horizontal(field, ilat, ilon, weights)
```

Weighted sum over the 3x3 neighbours of each point. 



**Args:**
 
 - <b>`field`</b> (array):  (..., nlat, nlon), the last two dimensions indexed by ilat and ilon 
 - <b>`ilat, ilon (int arrays)`</b>:  npts x 3, indices in the last two dimensions of field 
 - <b>`weights`</b> (array):  npts x 3 x 3 



**Returns:**
 
 - <b>`array`</b>:  (..., npts) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `interpolate_vertical`

```python
interpolate_vertical(plev, elevation, level_hpa)
```

Linear interpolation in height between the two pressure levels bracketing each point, and pressure from the hypsometric equation. When a point is below the lowest pressure level, the lowest level is used (per time step). 



**Args:**
 
 - <b>`plev`</b> (dict):  'z' (geopotential height, m), 't', 'q', 'u', 'v', each (time, level, point).  Levels sorted by increasing pressure (the last level is the lowest in altitude) 
 - <b>`elevation`</b> (array):  point elevation [m], (point,) 
 - <b>`level_hpa`</b> (array):  pressure of the levels [hPa], (level,) 



**Returns:**
 
 - <b>`dict`</b>:  't', 'q', 'u', 'v', 'p' (Pa), each (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `q_from_dewpoint`

```python
q_from_dewpoint(d2m, sp)
```

Specific humidity [kg kg-1] from dew point [K] and pressure [Pa] (Magnus, as meteo_util.dewT_2_q_magnus). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vapor_pressure_from_q`

```python
vapor_pressure_from_q(q, p)
```

Vapour pressure [Pa] from specific humidity and pressure (exact inverse of q_from_dewpoint). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dewpoint_from_vp`

```python
dewpoint_from_vp(vp)
```

Dew point [K] from vapour pressure [Pa] (inverse of the Magnus formula of q_from_dewpoint). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `surface_signal`

```python
surface_signal(plev, surf, level_hpa)
```

ERA5 surface effect at the grid surface: 2 m temperature and dew point minus their pressure-level values at the elevation of the ERA5 surface (inversions, cold-air pools, diurnal cycle missing from free-atmosphere air). 



**Args:**
 
 - <b>`plev`</b> (dict):  pressure-level fields as in interpolate_vertical, geopotential height 'z' in m 
 - <b>`surf`</b> (dict):  'z_surf' (m2 s-2), 't2m', 'd2m', each (time, point) 



**Returns:**
 dT, dTd [K], both (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `surface_correction`

```python
surface_correction(t, q, p, dT, dTd, weight, beta=0.0, dewpoint=True)
```

Add the terrain-weighted surface effect to pressure-level temperature and humidity (REDCAPP, DReaMIT): T = T_pl + w dT + beta and Td = Td_pl + w dTd, with Td capped at T. w = 0 and beta = 0 return t and q. 



**Args:**
 
 - <b>`t, q, p`</b>:  pressure-level temperature [K], specific humidity [kg kg-1] and pressure [Pa] at the points 
 - <b>`dT, dTd`</b>:  surface_signal() at the points, (time, point) 
 - <b>`weight`</b> (array):  surface-coupling weight w, (point,) 
 - <b>`beta`</b> (array or float):  additive offset [K], (time, 1) or scalar 
 - <b>`dewpoint`</b> (bool):  correct the humidity too 



**Returns:**
 corrected t, q 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mixing_ratio`

```python
mixing_ratio(q)
```

As meteo_util.mixing_ratio. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vapor_pressure`

```python
vapor_pressure(p, w)
```

As meteo_util.vapor_pressure [Pa]. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `wind_speed_direction`

```python
wind_speed_direction(u, v)
```

Wind speed [m s-1] and direction the wind blows from, clockwise from north [rad, 0-2pi]. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `precipitation`

```python
precipitation(tp_m, elev_diff, months, tstep_h, lapse_rate_flag, max_factor=5.0)
```



**Args:**
 
 - <b>`tp_m`</b> (array):  interpolated ERA5 precipitation [m] over the hour ending at the time stamp, (time, point) 
 - <b>`elev_diff`</b> (array):  point elevation - ERA5 surface elevation [m], (time, point) 
 - <b>`months`</b> (array):  month of each time step, (time,) 
 - <b>`tstep_h`</b> (int):  time step in hours 

**Returns:**
 tp [mm per time step], precip_lapse_rate [-], both (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `longwave`

```python
longwave(t, vp, t2m, vp_surf, strd, svf, slope, terrain_flag)
```

Downwelling longwave [W m-2]: clear-sky emissivity at the point plus the cloud emissivity of ERA5. 



**Args:**
 
 - <b>`t, vp`</b>:  air temperature [K] and vapour pressure [Pa] at the points, (time, point) 
 - <b>`t2m, vp_surf, strd`</b>:  ERA5 surface temperature [K], vapour pressure [Pa], and strd [J m-2 over 1 h], (time, point) 
 - <b>`svf, slope`</b>:  sky view factor [-] and slope [rad], (point,) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `horizon_at_sun`

```python
horizon_at_sun(profiles, horizon_azimuth_deg, sun_azimuth)
```

Horizon angle in the direction of the sun (nearest azimuth bin, as xarray .sel(method='nearest')). 



**Args:**
 
 - <b>`profiles`</b> (array):  horizon angles [rad], (point, azimuth) 
 - <b>`horizon_azimuth_deg`</b> (array):  azimuth of the bins [deg], sorted ascending 
 - <b>`sun_azimuth`</b> (array):  solar azimuth [rad], (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `shortwave`

```python
shortwave(ssrd, sp, p, solar, slope, aspect, svf, shade)
```

Shortwave partition and terrain corrections. 



**Args:**
 
 - <b>`ssrd`</b> (array):  ERA5 ssrd [J m-2 over 1 h], (time, point) 
 - <b>`sp`</b> (array):  ERA5 surface pressure [Pa], (time, point) 
 - <b>`p`</b> (array):  pressure at the point [Pa], (time, point) 
 - <b>`solar`</b> (dict):  'mu0', 'SWtoa', 'zenith', 'azimuth' (rad), 'sunset' (bool), each (time, point) 
 - <b>`slope, aspect, svf (array)`</b>:  (point,), angles in rad, aspect 0 = south 
 - <b>`shade`</b> (bool array):  sun behind the local horizon, (time, point) 



**Returns:**
 
 - <b>`dict`</b>:  'SW', 'SW_direct', 'SW_diffuse', 'cos_illumination', each (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L381"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `downscale_block`

```python
downscale_block(
    plev,
    surf,
    level_hpa,
    months,
    points,
    solar,
    shade,
    tstep_h,
    lw_terrain_flag,
    precip_lapse_rate_flag,
    precip_lapse_rate_max_factor=5.0,
    surface=None
)
```

Downscale one block of time steps for a set of points. 



**Args:**
 
 - <b>`plev`</b> (dict):  horizontally interpolated pressure-level fields 'z' (m2 s-2), 't', 'q', 'u', 'v',  each (time, level, point) 
 - <b>`surf`</b> (dict):  horizontally interpolated surface fields 'z_surf' (m2 s-2), 'sp', 't2m', 'd2m', 'tp', 'ssrd',  'strd', each (time, point) 
 - <b>`level_hpa`</b> (array):  pressure levels [hPa] 
 - <b>`months`</b> (array):  month of each time step 
 - <b>`points`</b> (dict):  'elevation', 'slope', 'aspect', 'svf', each (point,) 
 - <b>`solar`</b> (dict):  see shortwave(), each (time, point) 
 - <b>`shade`</b> (bool array):  (time, point) 
 - <b>`tstep_h`</b> (int):  time step of the data in hours 
 - <b>`surface`</b> (dict):  optional surface-effect correction (surface_effect.py): 'weight' (point,) or None for the  diagnostics only, 'beta' (time,) or scalar, 'dewpoint' (bool), 'lw_consistent' (bool, longwave from the 
 - <b>`corrected temperature and humidity). None`</b>:  no correction, outputs unchanged 



**Returns:**
 
 - <b>`dict of (time, point) arrays`</b>:  t, q, u, v, p, ws, wd, tp, precip_lapse_rate, w, vp, cse, LW, SW, SW_direct, SW_diffuse, cos_illumination; with surface, also t_pl, dT_surf, dTd_surf 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
