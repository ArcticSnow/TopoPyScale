<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.surface_calib`
Calibration of the surface-effect correction (surface_effect.py) from station records. 

1. Run TopoPyScale on the stations as a points project with toposcale.surface_effect: {method: none,  diagnostics: true}: the store then holds t_pl, dT_surf and dTd_surf at each station. 2. Read the observations (read_meteoswiss, read_csv) and pair them with the store (pair). 3. calibrate(): per-station weights (diagnostic), coefficients of the method by pooled least squares,  cross-validation (leave one station out, or contiguous time blocks with fewer than 3 stations) and  moving-block bootstrap intervals. 4. save_coefficients(): YAML file for toposcale.surface_effect.coefficients. 

Model fitted, per station s and hour t:  T_obs - T_pl = w(theta; indices_s) dT + beta(t) [+ c_s] 

**Global Variables**
---------------
- **DREAMIT_SLOPES**
- **REDCAPP_GAMMAS**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dewpoint_from_rh`

```python
dewpoint_from_rh(t, rh)
```

Dew point [K] from temperature [K] and relative humidity [%] (Magnus, as toposcale_core). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_meteoswiss`

```python
read_meteoswiss(path)
```

MeteoSwiss open-data hourly file (ogd-smn_<sta>_h_*.csv): t [K] and td [K], indexed by the END of the hour (UTC). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_csv`

```python
read_csv(path, time, t, rh=None, td=None, celsius=True, **kwargs)
```

Generic station file: column names of time, temperature and relative humidity [%] or dew point. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pair`

```python
pair(ds_down, obs, hour_ending=True)
```

Pair station observations with a downscaled store run with surface-effect diagnostics. 



**Args:**
 
 - <b>`ds_down`</b> (dataset):  downscaled points (point_name, time) with t_pl, dT_surf, dTd_surf, q, p 
 - <b>`obs`</b> (dict):  point_name -> dataframe of t and td [K] (read_meteoswiss, read_csv) 
 - <b>`hour_ending`</b> (bool):  observations are means over the hour ending at the time stamp; the model is then  averaged over the two time steps bracketing that hour 



**Returns:**
 dataframe with point_name, time, y (t obs - t_pl), dT, yd (td obs - td_pl), dTd 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fit_stations`

```python
fit_stations(data, min_abs_dT=1.0)
```

Per-station weight: slope of T_obs - T_pl on dT (with intercept), over hours with |dT| > min_abs_dT. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fit`

```python
fit(method, data, idx, fit_beta='none', station_offsets=False)
```

Coefficients of a method by pooled least squares over all stations and hours: profile over the nonlinear parameter (DReaMIT alpha_slope, REDCAPP gamma), linear least squares for the others. 



**Args:**
 
 - <b>`method`</b> (str):  'dreamit' or 'redcapp' 
 - <b>`data`</b> (dataframe):  pair() output 
 - <b>`idx`</b> (dataframe):  terrain indices of the stations (surface_effect.terrain_indices), indexed by point_name 
 - <b>`fit_beta`</b> (str):  'none', 'bias' (constant offset) or 'seasonal' (DReaMIT beta(t)); else beta = 0 
 - <b>`station_offsets`</b> (bool):  one free offset per station (fits w on the variation of dT only) 



**Returns:**
 dict of coefficients (and 'offsets' per station if station_offsets) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `predict`

```python
predict(method, coef, data, idx)
```

Corrected temperature minus T_pl for the rows of data (offsets per station if present in coef). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `scores`

```python
scores(sim, obs)
```

Bias, RMSE and correlation. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cross_validate`

```python
cross_validate(method, data, idx, by='station', n_blocks=2, **kw)
```

Out-of-sample predictions of T_obs - T_pl: leave one station out (by='station'), or leave one of n_blocks contiguous time blocks out (by='time'). 



**Returns:**
  series of predictions aligned with data 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bootstrap`

```python
bootstrap(method, data, idx, n=200, block='5D', seed=0, **kw)
```

Moving-block bootstrap over time (blocks shared by all stations): 2.5 and 97.5 % percentiles. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calibrate`

```python
calibrate(
    method,
    data,
    indices,
    fit_beta='none',
    station_offsets=False,
    n_boot=200,
    cv_by=None
)
```

Full calibration: per-station weights, pooled fit, cross-validation and bootstrap, for each window radius. 



**Args:**
 
 - <b>`method`</b> (str):  'dreamit' or 'redcapp' 
 - <b>`data`</b> (dataframe):  pair() output 
 - <b>`indices`</b> (dict):  radius_km -> terrain indices of the stations (indexed by point_name); the radius with the  lowest cross-validated RMSE is kept 
 - <b>`cv_by`</b> (str):  'station' or 'time'; default 'station' with 3 stations or more, else 'time' 



**Returns:**
 
 - <b>`dict`</b>:  method, radius_km, coefficients, intervals_95, cv (scores per station, out of sample), fit (scores in sample), stations (per-station weights), cv_by 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/surface_calib.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_coefficients`

```python
save_coefficients(result, path)
```

Write a calibrate() result as a coefficients file for toposcale.surface_effect.coefficients. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
