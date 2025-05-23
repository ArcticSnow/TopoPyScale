<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.meteo_util`




**Global Variables**
---------------
- **var_era_plevel**
- **var_era_surf**
- **g**
- **R**
- **eps0**
- **S0**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `steep_snow_reduce_all`

```python
steep_snow_reduce_all(snow, slope)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `partition_snow`

```python
partition_snow(
    precip,
    temp,
    rh=None,
    sp=None,
    method='continuous',
    tair_low_thresh=272.15,
    tair_high_thresh=274.15
)
```

Function to partition precipitation in between rain vs snow based on temperature threshold and mixing around freezing. The methods to partition snow/rain precipitation are: 
- continuous: method implemented into Cryogrid. 
- jennings2018 methods are from the publication Jennings et al (2018). DOI: https://doi.org/10.1038/s41467-018-03629-7 



**Args:**
 
 - <b>`precip`</b> (array):  1D array, precipitation in mm/hr 
 - <b>`temp`</b> (arrray):  1D array, air temperature in K 
 - <b>`rh`</b> (array):  1D array, relative humidity in % 
 - <b>`sp`</b> (array):  1D array, surface pressure in Pa 
 - <b>`method`</b> (str):  'continuous', 'Jennings2018_bivariate', 'Jennings2018_trivariate'. 
 - <b>`tair_low_thresh`</b> (float):  lower temperature threshold under which all precip is snow. degree K 
 - <b>`tair_high_thresh`</b> (float):  higher temperature threshold under which all precip is rain. degree K 



**Returns:**
 
 - <b>`array`</b>:  1D array rain 
 - <b>`array`</b>:  1D array snow 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `q_2_rh`

```python
q_2_rh(temp, pressure, qair)
```

Function to convert specific humidity (q) to relative humidity (RH) following Bolton (1980) 



**Args:**
 
 - <b>`temp`</b> (array):  temperature in degree K 
 - <b>`pressure`</b> (array):  aire pressure in Pa 
 - <b>`qair`</b> (array):  specific humidity in kg/kg 

**Returns:**
 
 - <b>`array`</b>:  relative humidity in the range of [0-1] 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mixing_ratio`

```python
mixing_ratio(ds, var)
```

Function to compute mixing ratio 



**Args:**
 
 - <b>`ds`</b>:  dataset (following ERA5 variable convention) q-specific humidity in kg/kg 
 - <b>`var`</b>:  dictionnary of variable name (for ERA surf / ERA plevel) Returns: dataset with new variable 'w' 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `t_rh_2_dewT`

```python
t_rh_2_dewT(ds, var)
```

Function to compute dea temperature from air temperature and relative humidity 



**Args:**
 
 - <b>`ds`</b>:  dataset with temperature ('t') and relative humidity ('r') variables 
 - <b>`var`</b>:  dictionnary of variable name (for ERA surf / ERA plevel) 

**Returns:**
 
 - <b>`dataset`</b>:  with added variable 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dewT_2_q_magnus`

```python
dewT_2_q_magnus(ds, var)
```

A version of the Magnus formula with the AERK parameters. AERK from Alduchov and Eskridge (1996) Note, e=Magnus(tdc) and es=Magnus(tc) 



**Args:**
 
 - <b>`ds`</b>:  dataset with 
 - <b>`var`</b>:  dictionnary of variable name (for ERA surf / ERA plevel) 

**Returns:**
 
 - <b>`array`</b>:  specific humidity in kg/kg 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/meteo_util.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vapor_pressure`

```python
vapor_pressure(ds, var)
```

Function to compute Vapor pressure from mixing ratio based on 2nd order Taylor series expansion 



**Args:**
 
 - <b>`ds`</b> (dataset):  dataset with pressure and mix_ratio variables 
 - <b>`var`</b> (dict):  variable name correspondance  

**Returns:**
 
 - <b>`dataset`</b>:  input dataset with new `vp` columns 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
