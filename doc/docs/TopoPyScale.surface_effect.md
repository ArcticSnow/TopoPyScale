<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.surface_effect`
Surface-effect correction of the downscaled air temperature and dew point: REDCAPP and DReaMIT. 

TopoSCALE takes air temperature from pressure levels, i.e. free-atmosphere air, without the inversions, cold-air pools and diurnal cycle that ERA5 resolves at its own surface. Both methods add back a terrain-weighted share of the ERA5 surface signal dT = t2m - T_pl(z_grid) (toposcale_core.surface_signal): 

 T = T_pl(z_point) + w dT + beta(t),     Td = Td_pl(z_point) + w dTd  (capped at T) 


- REDCAPP (Cao et al. 2017, GMD 10, 2905): w = alpha [H (1 - S) + S] + beta V (1 - S), S = exp(-R / gamma),  with H the hypsometric position and R the elevation range within a 30 km wide circle, and V = MRVBF / 8  (Gallant and Dowling 2003). Published coefficients: Swiss Alps. 
- DReaMIT (Pozsgay et al. 2026, GMD 19, 7197): w = alpha_intercept + exp(alpha_slope h) - 1, with h the hypsometric  position within 50 km, and beta(t) = beta_amp cos(2 pi (t - t_star)) + beta_bias (t: fraction of the year).  Published coefficients: ERA5, Yukon and Northwest Territories (GlobSim). DReaMIT defines dT against a linear lapse  rate fitted above the inversion; here T_pl is the TopoSCALE pressure-level interpolation, as in REDCAPP. 

Hypsometric position = fraction of the DEM cells within the window that are higher than the point: 0 on the highest summit, 1 at the bottom of the lowest valley (both papers). 

Config (toposcale.surface_effect), off by default:  method: none | redcapp | dreamit  coefficients: file written by surface_calib.save_coefficients (default: published values)  context_dem: DEM for the terrain indices, covering the points plus the window (default: project DEM)  dewpoint: true               # correct the humidity too  lw_consistent: true          # longwave from the corrected temperature and humidity  allow_partial_window: false  # accept windows that leave the context DEM (flagged in df_centroids)  diagnostics: false           # store t_pl, dT_surf and dTd_surf (also with method: none) 

**Global Variables**
---------------
- **METHODS**
- **DEFAULTS**
- **INDEX_COLUMNS**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `weight_dreamit`

```python
weight_dreamit(hypso_pos, alpha_intercept, alpha_slope, **_)
```

DReaMIT surface-coupling weight alpha(h) (Pozsgay et al. 2026, eq. 5). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `weight_redcapp`

```python
weight_redcapp(hypso_pos, relief, mrvbf, alpha, beta, gamma, **_)
```

REDCAPP land-surface correction factor LSCF (Cao et al. 2017). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `weight`

```python
weight(method, df, coef)
```

Weight of each point (rows of df with the INDEX_COLUMNS) for a method and its coefficients. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `beta_t`

```python
beta_t(times, beta_amp=0.0, t_star=0.0, beta_bias=0.0, **_)
```

DReaMIT seasonal offset beta(t) [K] (Pozsgay et al. 2026, eq. 6), t the fraction of the year. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_coefficients`

```python
load_coefficients(method, path=None)
```

Published coefficients of a method, updated with a coefficients file (surface_calib.save_coefficients). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mrvbf`

```python
mrvbf(z, res, slope_threshold=16.0, max_steps=8, pctl_radius=6)
```

Multiresolution index of valley bottom flatness (Gallant and Dowling 2003, Water Resour. Res. 39, 1347). 

Steps 1 and 2 at the DEM resolution, then the DEM is smoothed and coarsened 3 times per step and the slope threshold halved. Steps stop when the coarse DEM is smaller than the lowness window. 



**Args:**
 
 - <b>`z`</b> (2d array):  elevation [m], nan for no data 
 - <b>`res`</b> (float):  cell size [m] 
 - <b>`slope_threshold`</b> (float):  slope threshold of the first step [%]; 16 for a 25 m DEM (Gallant and Dowling),  REDCAPP uses 50 for a smoother index 
 - <b>`max_steps`</b> (int):  number of steps (8: index up to about 8) 
 - <b>`pctl_radius`</b> (int):  radius of the elevation-percentile window [cells] from step 2 (3 at step 1) 



**Returns:**
 mrvbf (2d array), number of steps computed 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `terrain_indices`

```python
terrain_indices(
    df,
    dem_file,
    radius_km,
    epsg=None,
    with_mrvbf=False,
    mrvbf_slope_threshold=50.0
)
```

Hypsometric position, elevation range and MRVBF of points. 



**Args:**
 
 - <b>`df`</b> (dataframe):  points with x, y (projected, m) and elevation 
 - <b>`dem_file`</b> (str):  context DEM, covering the points plus the radius; reprojected to epsg if needed 
 - <b>`radius_km`</b> (float):  window radius [km] 
 - <b>`with_mrvbf`</b> (bool):  compute MRVBF (REDCAPP), on the context DEM 



**Returns:**
 dataframe with hypso_pos, relief [m], mrvbf, window_coverage (share of the circle inside the DEM) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_effect.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup`

```python
setup(df, cfg, dem_file, epsg)
```

Terrain indices and weights of the points, and the arguments of toposcale_core.downscale_block. 



**Args:**
 
 - <b>`df`</b> (dataframe):  df_centroids (x, y, elevation); index columns and w_surf are added in place 
 - <b>`cfg`</b> (dict):  config toposcale.surface_effect 
 - <b>`dem_file`</b> (str):  project DEM, used when cfg has no context_dem 
 - <b>`epsg`</b> (int):  projection of x, y 



**Returns:**
 dict for downscale_block(surface=...), or None when the correction and its diagnostics are off 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
