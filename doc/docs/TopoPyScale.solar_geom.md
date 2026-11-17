<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/solar_geom.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.solar_geom`
Function to compute Solar angles S. Filhol, Oct 2021 

The solar position follows the NREL Solar Position Algorithm (Reda and Andreas, 2004) as implemented in pvlib (method 'nrel_numpy'). Its time-dependent part (heliocentric position of the Earth, nutation, sidereal time, sun right ascension and declination) does not depend on the location: it is computed once for all time steps, and only the location-dependent part is evaluated for each point, as arrays (time x point). 

**Global Variables**
---------------
- **S0_DEFAULT**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/solar_geom.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `spa_time_terms`

```python
spa_time_terms(times)
```

Location-independent part of the NREL SPA, for a DatetimeIndex in UTC. 



**Returns:**
 
 - <b>`dict of 1D arrays (time)`</b>:  apparent sidereal time v, sun right ascension alpha and declination delta [deg], equatorial horizontal parallax xi [deg], Earth-Sun distance R [AU] 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/solar_geom.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `spa_positions`

```python
spa_positions(terms, latitude, longitude, elevation)
```

Location-dependent part of the NREL SPA, broadcast over time and points. 



**Args:**
 
 - <b>`terms`</b> (dict):  from spa_time_terms() 
 - <b>`latitude, longitude (array)`</b>:  degrees, (point,) 
 - <b>`elevation`</b> (array):  m, (point,) 



**Returns:**
 
 - <b>`zenith, azimuth (clockwise from north), elevation angle`</b>:  degrees, (time, point), without refraction (the 'zenith', 'azimuth' and 'elevation' columns of pvlib.solarposition.get_solarposition) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/solar_geom.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cos_zenith_geocentric`

```python
cos_zenith_geocentric(terms, latitude, longitude)
```

Cosine of the geocentric solar zenith angle, (time, point): cheap, used to average the top-of-atmosphere irradiance over sub-steps. It neglects the parallax (< 0.003 deg) and the atmospheric refraction. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/solar_geom.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_solar_geom`

```python
get_solar_geom(
    df_position,
    start_date,
    end_date,
    tstep,
    sr_epsg='4326',
    fname='ds_solar.nc',
    project_ouput=PosixPath('.'),
    method_solar='nrel_numpy',
    accumulation_window='1h',
    solar_constant=1361.0,
    n_substeps=6
)
```

Compute the solar geometry and top-of-atmosphere irradiance for each point. Azimuth is defined with 0 towards South, negative towards West and positive towards East. 



**Args:**
 
 - <b>`df_position`</b> (dataframe):  point_name, elevation, and either longitude/latitude (or lon/lat) or x/y in sr_epsg 
 - <b>`start_date`</b> (str):  start date  "2014-05-10" 
 - <b>`end_date`</b> (str):  end date   "2015-05-10" 
 - <b>`tstep`</b> (str):  time step, ex: '6h' 
 - <b>`sr_epsg`</b> (str):  EPSG code of x, y, used when longitude/latitude are not given 
 - <b>`fname`</b> (str):  name of netcdf file to store solar geometry 
 - <b>`project_ouput`</b> (str):  path to project root directory 
 - <b>`method_solar`</b> (str):  pvlib solar position method. 'nrel_numpy' (default) is vectorized over points 
 - <b>`accumulation_window`</b> (str):  length of the window over which the radiation fluxes are accumulated, ending at 
 - <b>`the time stamp (ERA5`</b>:  '1h' whatever the sampling time step). Angles are evaluated at the centre of this window, and SWtoa is the mean over the window. Use '0h' for instantaneous fluxes. 
 - <b>`solar_constant`</b> (float):  total solar irradiance at 1 AU [W m-2]. Default 1361 
 - <b>`n_substeps`</b> (int):  number of sub-steps used to average SWtoa over the accumulation window 



**Returns:**
 
 - <b>`dataset`</b>:  zenith, azimuth, elevation [rad], mu0, SWtoa [W m-2], sunset (point_name, time) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
