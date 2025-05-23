<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/solar_geom.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.solar_geom`
Function to compute Solar angles S. Filhol, Oct 2021 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/solar_geom.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_solar_geom`

```python
get_solar_geom(
    df_position,
    start_date,
    end_date,
    tstep,
    sr_epsg='4326',
    num_threads=None,
    fname='ds_solar.nc',
    project_ouput=PosixPath('.'),
    method_solar='nrel_numpy'
)
```

Function to compute solar position for each location given in the dataframe azimuth is define with 0 towards South, negative in W-dir, and posiive towards E-dir 



**Args:**
 
 - <b>`df_position`</b> (dataframe):  point_name as index, latitude, longitude 
 - <b>`start_date`</b> (str):  start date  "2014-05-10" 
 - <b>`end_date (str`</b>:  end date   "2015-05-10" 
 - <b>`tstep`</b> (str):  time step, ex: '6H' 
 - <b>`sr_epsg`</b> (str):  source EPSG code for the input coordinate 
 - <b>`num_threads`</b> (int):  number of threads to parallelize computation on. default is number of core -2 
 - <b>`fname`</b> (str):  name of netcdf file to store solar geometry 
 - <b>`project_ouput`</b> (str):  path to project root directory 



**Returns:**
 
 - <b>`dataset`</b>:  solar angles in degrees 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
