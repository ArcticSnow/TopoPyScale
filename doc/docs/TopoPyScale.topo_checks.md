<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_checks`
Physics-consistency diagnostics for TopoPyScale downscaled outputs. 

The downscaling chain (horizontal interpolation -> vertical interpolation -> radiation partitioning and terrain corrections -> export) can silently break basic physical constraints: humidity above saturation, direct beam in terrain shadow, precipitation created by an unbounded elevation factor, time misalignment between accumulated ERA5 fluxes and the solar geometry, unit mix-ups between the netcdf attributes and the values. This module checks the outputs against such constraints and returns a tidy report. 

Every check returns rows with the columns:  check, variable, status, n_fail, n_total, frac_fail, worst, note where status is one of 'pass', 'info', 'warning', 'error'. 

Typical usage after a downscaling run: 

 from TopoPyScale import topo_checks as tc  report = tc.run_all_checks(mp.downscaled_pts,  ds_solar=mp.ds_solar,  df_centroids=mp.toposub.df_centroids,  ds_param=mp.toposub.ds_param)  print(tc.summarize(report))  tc.raise_on_failure(report, level='error') 

or simply `mp.check_physics()` from a Topoclass object. 

**Global Variables**
---------------
- **G**
- **R_D**
- **EPS**
- **SIGMA**
- **S0**
- **S0_MAX**
- **T_TERRAIN**
- **STATUS_ORDER**
- **REPORT_COLUMNS**
- **DEFAULT_BOUNDS**
- **MAX_PRECIP_RATE**
- **UNIT_SIGNATURES**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `esat_water`

```python
esat_water(T)
```

Saturation vapour pressure over water [Pa], same formula as meteo_util.q_2_rh (Bolton 1980). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `relative_humidity`

```python
relative_humidity(T, p, q)
```

Unclipped relative humidity w.r.t. water [0-1] (meteo_util.q_2_rh clips to 1 and hides supersaturation). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_time_axis`

```python
check_time_axis(ds)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_finite`

```python
check_finite(ds, variables=None)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_ranges`

```python
check_ranges(ds, bounds=None)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_unit_attributes`

```python
check_unit_attributes(ds)
```

Compare the declared `units` attribute with the magnitude of the values. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_humidity`

```python
check_humidity(ds, rh_tol=0.02)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_wind`

```python
check_wind(ds)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L259"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_pressure_hypsometric`

```python
check_pressure_hypsometric(ds, terrain, tol=0.01)
```

Pressure differences between points must follow the hypsometric equation with the downscaled temperature. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_lapse_rate`

```python
check_lapse_rate(ds, terrain, min_range=300.0)
```

Effective temperature lapse rate across points should not be super-adiabatic. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_shortwave`

```python
check_shortwave(ds, sol=None, terrain=None, tol=1.0)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L349"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_shortwave_timing`

```python
check_shortwave_timing(
    ds,
    sol,
    terrain,
    max_slope_deg=10.0,
    tol_minutes=15.0,
    min_days=5
)
```

Compare the daily centroid (in local solar time) of downscaled SW with the one of the TOA irradiance computed from ds_solar, for nearly flat points. ERA5 ssrd at time t is the accumulation over [t - 1h, t]: if the solar geometry is evaluated at t instead of t - 30 min, SW lags the sun by about +30 min, which biases kt (low in the morning, clipped in the afternoon), the diffuse/direct split and the illumination on slopes. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L384"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_longwave`

```python
check_longwave(ds, terrain=None, tol=0.02, min_emissivity=0.4)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_precipitation`

```python
check_precipitation(ds, max_factor=5.0)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L415"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_domain_precip_budget`

```python
check_domain_precip_budget(ds, ds_param=None, weights=None, bounds=(0.67, 1.5))
```

Area-weighted domain precipitation relative to the horizontally interpolated ERA5 precipitation. Without precip lapse rate the ratio is 1. With it, the ratio tells how much water the elevation factor adds to the domain. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L442"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_centroid_aspect`

```python
check_centroid_aspect(df_centroids, min_slope_deg=5.0, tol_deg=10.0)
```

The centroid `aspect` used for the illumination angle must agree with the circular mean carried by the clustering features (aspect_cos, aspect_sin). Aspect is in [0, 2 pi) with 0 = south, so an arithmetic mean of a south-facing cluster (pixels near 0 and near 2 pi) gives ~pi, i.e. a north-facing slope. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_dem_orientation`

```python
check_dem_orientation(ds_param)
```

slope/aspect (gradient_d8 with dy < 0) assume a north-up raster. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_snow_partition`

```python
check_snow_partition(method='continuous', n_draws=2000)
```

Probe meteo_util.partition_snow: closure, bounds, monotonic in T, and RH unit convention used by exporters. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L500"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_all_checks`

```python
run_all_checks(
    ds_down,
    ds_solar=None,
    df_centroids=None,
    ds_param=None,
    snow_partition_method=None,
    points=None,
    bounds=None
)
```

Run all physics-consistency checks on downscaled outputs. 



**Args:**
 
 - <b>`ds_down`</b> (dataset):  downscaled points (e.g. Topoclass.downscaled_pts), dims (point_name, time) 
 - <b>`ds_solar`</b> (dataset):  solar geometry (Topoclass.ds_solar) 
 - <b>`df_centroids`</b> (dataframe):  Topoclass.toposub.df_centroids (elevation, slope, aspect, svf, lon, hori_azi_*) 
 - <b>`ds_param`</b> (dataset):  Topoclass.toposub.ds_param, used for area weights of the precipitation budget 
 - <b>`snow_partition_method`</b> (str or list):  partition method(s) to probe, e.g. 'continuous' 
 - <b>`points`</b> (list):  optional subset of point_name to check 
 - <b>`bounds`</b> (dict):  optional override of DEFAULT_BOUNDS 



**Returns:**
 
 - <b>`dataframe`</b>:  report, worst status first 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L549"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `summarize`

```python
summarize(report, min_status='info')
```

Human-readable summary of the checks at or above `min_status`. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_checks.py#L560"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `raise_on_failure`

```python
raise_on_failure(report, level='error')
```






---

## <kbd>class</kbd> `PhysicsConsistencyError`










---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
