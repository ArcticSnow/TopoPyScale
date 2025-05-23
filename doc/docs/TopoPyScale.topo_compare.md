<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_compare.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_compare`





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_compare.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `correct_trend`

```python
correct_trend(df, reference_col='obs', target_col='dow', apply_correction=True)
```

Function to estimate linear trend correction. 

**Args:**
  df (dataframe): 
 - <b>`reference_col`</b> (str):  name of the reference column, i.e. observation timeseries 
 - <b>`target_col`</b> (str):  name of the target column, i.e. downscale timeseries 
 - <b>`apply_correction`</b> (bool):  applying correction to target data 



**Returns:**
 
 - <b>`dict`</b>:  metrics of the linear trend estimate 
 - <b>`dataframe`</b> (optional):  corrected values 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_compare.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `correct_seasonal`

```python
correct_seasonal(
    df,
    reference_col='obs',
    target_col='dow',
    plot=True,
    apply_correction=True
)
```

Function to correct for seasonal signal.  1. it groups all days by day_of_year and extract the median difference.  2. it computes the 31 day rolling mean (padded on both sides) 



**Args:**
 
 - <b>`df`</b> (dataframe):  index must be a datetime, with daily timestep. other columns are reference and target. 
 - <b>`reference_col`</b> (str):  name of the reference column, i.e. observation timeseries 
 - <b>`target_col`</b> (str):  name of the target column, i.e. downscale timeseries plot (bool): 
 - <b>`apply_correction`</b> (bool):  apply correction and return corrected data 



**Returns:**
 dataframe - correction for each day of year, starting on January 1. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_compare.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `obs_vs_downscaled`

```python
obs_vs_downscaled(
    df,
    reference_col='obs',
    target_col='dow',
    trend_correction=True,
    seasonal_correction=True,
    param={'xlab': 'Downscaled [unit]', 'ylab': 'Observation [unit]', 'xlim': (-20, 20), 'ylim': (-20, 20), 'title': None},
    plot='heatmap'
)
```

Function to compare Observation to Downscaled for one given variable. 



**Args:**
 
 - <b>`df`</b> (dataframe):  pandas dataframe containing corresponding Observation and Downscaled values 
 - <b>`reference_col`</b> (str):  name of the reference column. Observation 
 - <b>`target_col`</b> (str):  name of the target column. Downscaled timeseries 
 - <b>`trend_correction`</b> (bool):  remove trend by applying a linear regression 
 - <b>`seasonal_correction`</b> (bool):  remove seasonal signal by deriving median per day of the year and computing the rolling mean over 31d on the median 
 - <b>`param`</b> (dict):  parameter for comparison and plotting 
 - <b>`plot`</b> (str):  plot type: heatmap or timeseries 



**Returns:**
 
 - <b>`dict`</b>:  metrics of regression and comparison 
 - <b>`dataframe`</b>:  dataframe containing the seasonal corrections to applied 
 - <b>`dataframe`</b>:  corrected values 

Inspired by this study: https://reader.elsevier.com/reader/sd/pii/S0048969722015522?token=106483481240DE6206D83D9C7EC9D4990C2C3DE6F669EDE39DCB54FF7495A31CC57BDCF3370A6CA39D09BE293EACDDBB&originRegion=eu-west-1&originCreation=20220316094240 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
