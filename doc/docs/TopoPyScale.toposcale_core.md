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
- **MU_MIN**
- **DIFFUSE_MODELS**

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

## <kbd>function</kbd> `seasonal_month`

```python
seasonal_month(months, latitude=None)
```

Month of the northern-hemisphere season of each time step and point: the month itself north of the equator, the month shifted by six south of it (January in the south is July in the north), for the monthly coefficients derived in the northern hemisphere. 



**Args:**
 
 - <b>`months`</b> (array):  month of each time step (1-12), (time,) or (time, point) 
 - <b>`latitude`</b> (array):  latitude of the points [deg], (point,). None: months unchanged 



**Returns:**
 
 - <b>`array`</b>:  months (1-12), (time, point) if latitude is given 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `precip_lapse_factor`

```python
precip_lapse_factor(elev_diff, months, max_factor=5.0)
```

Precipitation elevation adjustment factor (1 + c dz) / (1 - c dz) of Liston and Elder (2006). 

The factor diverges for c dz -> 1 (dz ~ 2.9 km with c = 0.35 km-1) and becomes negative beyond. c dz is therefore bounded so that the factor stays within [1/max_factor, max_factor]. 



**Args:**
 
 - <b>`elev_diff`</b> (array):  elevation of the point minus elevation of the ERA5 surface [m] 
 - <b>`months`</b> (array):  month of each time step (1-12), in the northern-hemisphere season (seasonal_month) 
 - <b>`max_factor`</b> (float):  upper bound of the factor (> 1) 



**Returns:**
 
 - <b>`array`</b>:  adjustment factor 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_weights`

```python
apply_weights(field, W)
```

Horizontal interpolation of field (..., nlat, nlon) with a weight matrix from weight_matrix(): (..., npts). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `q_from_dewpoint`

```python
q_from_dewpoint(d2m, sp)
```

Specific humidity [kg kg-1] from dew point [K] and pressure [Pa] (Magnus, as meteo_util.dewT_2_q_magnus). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vapor_pressure_from_q`

```python
vapor_pressure_from_q(q, p)
```

Vapour pressure [Pa] from specific humidity and pressure (exact inverse of q_from_dewpoint). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dewpoint_from_vp`

```python
dewpoint_from_vp(vp)
```

Dew point [K] from vapour pressure [Pa] (inverse of the Magnus formula of q_from_dewpoint). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mixing_ratio`

```python
mixing_ratio(q)
```

As meteo_util.mixing_ratio. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vapor_pressure`

```python
vapor_pressure(p, w)
```

As meteo_util.vapor_pressure [Pa]. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `wind_speed_direction`

```python
wind_speed_direction(u, v)
```

Wind speed [m s-1] and direction the wind blows from, clockwise from north [rad, 0-2pi]. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `wind_components`

```python
wind_components(ws, wd)
```

u, v [m s-1] of a wind of speed ws blowing from the direction wd (clockwise from north, rad). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `wind_terrain`

```python
wind_terrain(
    u,
    v,
    slope,
    aspect,
    curvature,
    slope_scale,
    slope_weight=0.58,
    curvature_weight=0.42
)
```

Wind weighting and diverting factors of MicroMet (Liston and Elder, 2006, eqs. 15-19). 



**Args:**
 
 - <b>`u, v (array)`</b>:  wind [m s-1], (time, point) 
 - <b>`slope, aspect (array)`</b>:  [rad], (point,), aspect 0 = south, positive towards east (TopoPyScale convention) 
 - <b>`curvature`</b> (array):  terrain curvature scaled to [-0.5, 0.5] over the DEM, (point,) 
 - <b>`slope_scale`</b> (float):  twice the steepest slope of the DEM [rad], scaling the slope in the wind direction 
 - <b>`slope_weight, curvature_weight (float)`</b>:  gs and gc 



**Returns:**
 
 - <b>`u, v (array)`</b>:  wind after the terrain modification, (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `precipitation`

```python
precipitation(
    tp_m,
    elev_diff,
    months,
    tstep_h,
    lapse_rate_flag,
    max_factor=5.0,
    acc_hours=1,
    latitude=None
)
```



**Args:**
 
 - <b>`tp_m`</b> (array):  interpolated ERA5 precipitation [m] accumulated over the acc_hours hours ending at the time stamp,  (time, point) 
 - <b>`elev_diff`</b> (array):  point elevation - ERA5 surface elevation [m], (time, point) 
 - <b>`months`</b> (array):  month of each time step, (time,) 
 - <b>`tstep_h`</b> (int):  time step in hours. tp_m is scaled by tstep_h / acc_hours: 1 when the accumulation covers the  time step 
 - <b>`acc_hours`</b> (int):  hours accumulated in tp_m 
 - <b>`latitude`</b> (array):  latitude of the points [deg], (point,): the monthly coefficients follow the season of each 
 - <b>`hemisphere. None`</b>:  northern-hemisphere months 

**Returns:**
 tp [mm per time step], precip_lapse_rate [-], both (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `longwave`

```python
longwave(
    t,
    vp,
    t2m,
    vp_surf,
    strd,
    svf,
    slope,
    terrain_flag,
    acc_seconds=3600.0,
    t_terrain=273.15
)
```

Downwelling longwave [W m-2]: clear-sky emissivity at the point plus the cloud emissivity of ERA5. 



**Args:**
 
 - <b>`t, vp`</b>:  air temperature [K] and vapour pressure [Pa] at the points, (time, point) 
 - <b>`t2m, vp_surf, strd`</b>:  ERA5 surface temperature [K], vapour pressure [Pa], and strd [J m-2 accumulated over  acc_seconds], (time, point) 
 - <b>`svf, slope`</b>:  sky view factor [-] and slope [rad], (point,) 
 - <b>`acc_seconds`</b> (float):  accumulation period of strd [s] 
 - <b>`t_terrain`</b> (float or array):  temperature of the surrounding terrain [K], scalar or (time, point), for the  longwave it emits (emissivity 0.99) when terrain_flag 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L389"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L406"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `shortwave`

```python
shortwave(
    ssrd,
    sp,
    p,
    solar,
    slope,
    aspect,
    svf,
    shade,
    acc_seconds=3600.0,
    fdir=None,
    diffuse_model='isotropic',
    albedo=0.0
)
```

Shortwave partition and terrain corrections. 



**Args:**
 
 - <b>`ssrd`</b> (array):  ERA5 ssrd [J m-2 accumulated over acc_seconds], (time, point) 
 - <b>`sp`</b> (array):  ERA5 surface pressure [Pa], (time, point) 
 - <b>`p`</b> (array):  pressure at the point [Pa], (time, point) 
 - <b>`solar`</b> (dict):  'mu0', 'SWtoa', 'zenith', 'azimuth' (rad), 'sunset' (bool), each (time, point) 
 - <b>`slope, aspect, svf (array)`</b>:  (point,), angles in rad, aspect 0 = south 
 - <b>`shade`</b> (bool array):  sun behind the local horizon, (time, point) 
 - <b>`acc_seconds`</b> (float):  accumulation period of ssrd [s], the window of solar['SWtoa'] 
 - <b>`fdir`</b> (array):  ERA5 direct solar radiation on a horizontal surface [J m-2 accumulated over acc_seconds], 
 - <b>`(time, point). None`</b>:  the direct/diffuse split of the grid comes from the clearness index (Ruiz-Arias et al., 2010) 
 - <b>`diffuse_model`</b> (str):  'isotropic': sky diffuse scaled by the sky view factor; 'hay_davies': the circumsolar  part, a share of the diffuse equal to the anisotropy index (direct / TOA, Hay and Davies 1980), comes  from the direction of the sun (projected on the slope and shaded as the beam), the rest is isotropic 
 - <b>`albedo`</b> (float or array):  albedo of the surrounding terrain, scalar or (time, point). The terrain seen by  the point (1 - svf) reflects the global radiation of a level surface at the point, added to the diffuse. 
 - <b>`0`</b>:  no terrain reflection 



**Returns:**
 
 - <b>`dict`</b>:  'SW', 'SW_direct', 'SW_diffuse' (sky and terrain), 'cos_illumination', 'SW_terrain' (reflected by the terrain, part of SW_diffuse), each (time, point) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/toposcale_core.py#L469"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    surface=None,
    acc_hours=1,
    radiation=None,
    wind=None
)
```

Downscale one block of time steps for a set of points. 



**Args:**
 
 - <b>`plev`</b> (dict):  horizontally interpolated pressure-level fields 'z' (m2 s-2), 't', 'q', 'u', 'v',  each (time, level, point) 
 - <b>`surf`</b> (dict):  horizontally interpolated surface fields 'z_surf' (m2 s-2), 'sp', 't2m', 'd2m', 'tp', 'ssrd',  'strd', and optionally 'fdir' (direct/diffuse split of ERA5), each (time, point) 
 - <b>`level_hpa`</b> (array):  pressure levels [hPa] 
 - <b>`months`</b> (array):  month of each time step 
 - <b>`points`</b> (dict):  'elevation', 'slope', 'aspect', 'svf', each (point,), and optionally 'lat' [deg] for the  season of the precipitation coefficients 
 - <b>`solar`</b> (dict):  see shortwave(), each (time, point) 
 - <b>`shade`</b> (bool array):  (time, point) 
 - <b>`tstep_h`</b> (int):  time step of the data in hours 
 - <b>`surface`</b> (dict):  optional surface-effect correction (surface_effect.py): 'weight' (point,) or None for the  diagnostics only, 'beta' (time,) or scalar, 'dewpoint' (bool), 'lw_consistent' (bool, longwave from the 
 - <b>`corrected temperature and humidity). None`</b>:  no correction, outputs unchanged 
 - <b>`acc_hours`</b> (int):  hours accumulated in surf 'tp', 'ssrd' and 'strd' (ending at the time stamp) 
 - <b>`radiation`</b> (dict):  options of the radiation: 'lw_terrain_temperature' ('air': the air temperature at the 
 - <b>`point, 'fixed'`</b>:  273.15 K, a value [K], or an array (time, point)), 'diffuse_model' ('isotropic' or 
 - <b>`'hay_davies'), 'albedo' (of the surrounding terrain, scalar or (time, point); 0`</b>:  no reflection). 
 - <b>`Default`</b>:  fixed terrain temperature, isotropic diffuse, no reflection 
 - <b>`wind`</b> (dict):  wind options (wind.setup): 'blend' (point,) weight of the ERA5 10 m wind surf['u10'],  surf['v10']; 'terrain' 'liston_elder' with 'curvature' (point,), 'slope_scale', 'slope_weight', 
 - <b>`'curvature_weight'. None`</b>:  pressure-level wind 



**Returns:**
 
 - <b>`dict of (time, point) arrays`</b>:  t, q, u, v, p, ws, wd, tp, precip_lapse_rate, w, vp, cse, LW, SW, SW_direct, SW_diffuse, cos_illumination; with surface, also t_pl, dT_surf, dTd_surf 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
