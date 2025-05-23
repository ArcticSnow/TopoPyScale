<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_obs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_obs`
Tools to download and compare Downsclaed timeseries to observation All observation in folder inputs/obs/ S. Filhol December 2021 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_obs.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_metno_obs`

```python
get_metno_obs(
    sources,
    voi,
    start_date,
    end_date,
    client_id='97a0e2bc-a262-48fe-9dea-5e9c894e9328'
)
```

Function to download observation data from MetNo FROST API (Norwegian Meteorological institute) 

Args  sources (list): station code, e.g. 'SN25830'  voi (list): variables to download  start_date (str): starting date  end_date (str): ending date  client_id (str): FROST_API_CLIENTID 

Return:   dataframe: all data combined together 

List of variable: https://frost.met.no/element table Find out about stations: https://seklima.met.no/  WARNING: Download max one year at the time to not reach max limit of data to download. 



**TODO:**
 
- convert df to xarray dataset with stn_id, time as coordinates 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_obs.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `combine_metno_obs_to_xarray`

```python
combine_metno_obs_to_xarray(fnames='metno*.pckl', path='inputs/obs/')
```

Function to convert metno format to usable dataset 



**Args:**
 
 - <b>`fnames`</b> (str pattern):  pattern of the file to load 
 - <b>`path`</b> (str):  path of the file to load 



**Returns:**
 
 - <b>`dataset`</b>:  dataset will all data organized in timeseries and station number 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_obs.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fetch_WMO_insitu_observations`

```python
fetch_WMO_insitu_observations(
    years,
    months,
    bbox,
    target_path='./inputs/observations',
    product='sub_daily',
    var=['air_pressure', 'air_pressure_at_sea_level', 'air_temperature', 'dew_point_temperature', 'wind_from_direction', 'wind_speed']
)
```

Function to download WMO in-situ data from land surface in-situ observations from Copernicus. https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-surface-land?tab=overview 



**Args:**
 
 - <b>`year`</b> (str or list):  year(s) to download 
 - <b>`month`</b> (str or list):  month(s) to download 
 - <b>`bbox`</b> (list):  bonding box in lat-lon [lat_south, lon_west, lat_north, lon_east] 
 - <b>`target`</b> (str):  filename 
 - <b>`var`</b> (list):  list of variable to download. available: 



**Returns:**
 Store to disk the dataset as zip file 



**TODO:**
 
    - [x] test function 
    - [ ] check if can download large extent at once or need multiple requests? 
    - [x] save data in individual files for each stations. store either as csv or netcdf (better) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_obs.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_WMO_insitu_observations`

```python
parse_WMO_insitu_observations(
    fname=None,
    file_pattern='inputs/observations/surf*subset_csv*.csv',
    path='./inputs/observations'
)
```

Function to parse WMO files formated from CDS database. parse in single station file 



**Args:**
  fname:  file_pattern:  path: 



**Returns:**
 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
