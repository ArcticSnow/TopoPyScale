<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_export`
Functions to export topo_scale output to formats compatible with existing models (e.g. CROCUS, Cryogrid, Snowmodel, ...) 

S. Filhol, December 2021 

TODO; 
- SPHY forcing (grids) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_scaling_and_offset`

```python
compute_scaling_and_offset(da, n=16)
```

Compute offset and scale factor for int conversion 



**Args:**
 
 - <b>`da`</b> (dataarray):  of a given variable 
 - <b>`n`</b> (int):  number of digits to account for 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_netcdf`

```python
to_netcdf(ds, fname='output.nc', variables=None, complevel=9)
```

Generic function to save a datatset to one single compressed netcdf file 



**Args:**
 
 - <b>`fname`</b> (str):  name of export file 
 - <b>`variables`</b> (list str):  list of variable to export. Default exports all variables 
 - <b>`complevel`</b> (int):  Compression level. 1-9 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_musa`

```python
to_musa(
    ds,
    df_pts,
    da_label,
    fname_met='musa_met.nc',
    fname_labels='musa_labels.nc',
    path='outputs/',
    climate_dataset_name='ERA5',
    project_authors='S. Filhol'
)
```

Function to export to MuSa standard. 

**Args:**
  ds:  df_pts:  da_labels:  fname:  path: 



**Returns:**
  Save downscaled timeseries and toposub cluster mapping 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_cryogrid`

```python
to_cryogrid(
    ds,
    df_pts,
    fname_format='Cryogrid_pt_*.nc',
    path='outputs/',
    label_map=False,
    da_label=None,
    snow_partition_method='continuous',
    climate_dataset_name='ERA5',
    project_author='S. Filhol'
)
```

Function to export TopoPyScale downscaled dataset in a netcdf format compatible for Cryogrid-community model. 



**Args:**
 
 - <b>`ds`</b> (dataset):  downscaled values from topo_scale 
 - <b>`df_pts`</b> (dataframe):  with metadata of all point of downscaling 
 - <b>`fname_format`</b> (str):  file name for output 
 - <b>`path`</b> (str):  path where to export files 
 - <b>`label_map`</b> (bool):  export cluster_label map 
 - <b>`da_label`</b> (dataarray): (optional) dataarray containing label value for each pixel of the DEM 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_fsm2`

```python
to_fsm2(
    ds_down,
    fsm_param,
    simulation_path='fsm_sim',
    fname_format='fsm_',
    namelist_options=None,
    n_digits=None,
    snow_partition_method='continuous',
    cluster_method=True,
    epsg_ds_param=2056,
    temperature_correction=0,
    forest_param_scaler={'hcan': 100, 'lai': 100}
)
```

Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd). FSM2oshd includes canopy structures processes one simulation consists of 2 driving file: 
   - met.txt with variables:  year, month, day, hour, SWdir, SWdif, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt 
   - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10 



**Args:**
 
 - <b>`ds_down`</b>:   Downscaled weather variable dataset 
 - <b>`fsm_param`</b>:   terrain and canopy parameter dataset 
 - <b>`df_centroids`</b>:   cluster centroids statistics (terrain + canopy) 
 - <b>`ds_tvt`</b> (dataset, int, float, or str):   transmisivity. Can be a dataset, a constant or 'svf_for' 
 - <b>`simulation_path`</b> (str):  'fsm_sim' 
 - <b>`fname_format`</b> (str): 'fsm_' 
 - <b>`namelist_options`</b> (dict):  {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]} 
 - <b>`n_digits`</b> (int):  Number of digits (for filename system) 
 - <b>`snow_partition_method`</b> (str):  method for snow partitioning. Default: 'continuous' 
 - <b>`cluster_method`</b> (bool):  boolean to be True is using cluster appraoch 
 - <b>`epsg_ds_param`</b> (int):  epsg code of ds_parma: example: 2056 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L651"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_fsm2oshd`

```python
to_fsm2oshd(
    ds_down,
    fsm_param,
    ds_tvt,
    simulation_path='fsm_sim',
    fname_format='fsm_',
    namelist_options=None,
    n_digits=None,
    snow_partition_method='continuous',
    cluster_method=True,
    epsg_ds_param=2056,
    temperature_correction=0,
    forest_param_scaler={'vfhp': 100, 'fveg': 100, 'fves': 100, 'hcan': 100, 'lai5': 100}
)
```

Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd). FSM2oshd includes canopy structures processes one simulation consists of 2 driving file: 
   - met.txt with variables:  year, month, day, hour, SWdir, SWdif, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt 
   - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10 



**Args:**
 
 - <b>`ds_down`</b>:   Downscaled weather variable dataset 
 - <b>`fsm_param`</b>:   terrain and canopy parameter dataset 
 - <b>`df_centroids`</b>:   cluster centroids statistics (terrain + canopy) 
 - <b>`ds_tvt`</b> (dataset, int, float, or str):   transmisivity. Can be a dataset, a constant or 'svf_for' 
 - <b>`simulation_path`</b> (str):  'fsm_sim' 
 - <b>`fname_format`</b> (str): 'fsm_' 
 - <b>`namelist_options`</b> (dict):  {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]} 
 - <b>`n_digits`</b> (int):  Number of digits (for filename system) 
 - <b>`snow_partition_method`</b> (str):  method for snow partitioning. Default: 'continuous' 
 - <b>`cluster_method`</b> (bool):  boolean to be True is using cluster appraoch 
 - <b>`epsg_ds_param`</b> (int):  epsg code of ds_parma: example: 2056 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L964"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_fsm`

```python
to_fsm(
    ds,
    fname_format='FSM_pt_*.tx',
    snow_partition_method='continuous',
    n_digits=None
)
```

Function to export data for FSM. 



**Args:**
 
 - <b>`ds`</b> (dataset):  downscaled_pts, 
 - <b>`df_pts`</b> (dataframe):  toposub.df_centroids, 
 - <b>`fname_format`</b> (str pattern):  output format of filename 
 - <b>`snow_partition_method`</b> (str):  snow/rain partitioning method: default 'jennings2018_trivariate' 

format is a text file with the following columns year month  day   hour  SW      LW      Sf         Rf     Ta  RH   Ua    Ps (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa) 

See README.md file from FSM source code for further details 



**TODO:**
 
- Check unit DONE jf 
- Check format is compatible with compiled model DONE jf 
- ensure ds.point_name.values always 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L1012"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_TC`

```python
to_TC(ds, fname_format='pt_*.tx')
```

Function to export data for the T&C model. https://hyd.ifu.ethz.ch/research-data-models/t-c.html 



**Args:**
 
 - <b>`ds`</b> (dataset):  downscaled_pts, 
 - <b>`df_pts`</b> (dataframe):  toposub.df_centroids, 
 - <b>`fname_format`</b> (str pattern):  output format of filename 

format is a text file with the following columns datetime    TA  P       PRESS RH  SWIN    LWIN    WS   WDIR (datetime) (°C) (mm/h)  (Pa)  (%) (W/m2)  (W/m2)  (m/s) (°) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L1055"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_micromet_single_station`

```python
to_micromet_single_station(
    ds,
    df_pts,
    fname_format='Snowmodel_pt_*.csv',
    na_values=-9999,
    headers=False
)
```

Function to export TopoScale output in the format for Listn's Snowmodel (using Micromet). One CSV file per point_name 



**Args:**
 
 - <b>`ds`</b> (dataset):  TopoPyScale xarray dataset, downscaled product 
 - <b>`df_pts`</b> (dataframe):  with point list info (x,y,elevation,slope,aspect,svf,...) 
 - <b>`fname_format`</b> (str):  filename format. point_name is inserted where * is 
 - <b>`na_values`</b> (int):  na_value default 
 - <b>`headers`</b> (bool):  add headers to file 



Example of the compatible format for micromet (headers should be removed to run the model: 

year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip (yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt) 

2002   10    1   12.00    101   426340.0  4411238.0  3598.0     0.92    57.77     4.80   238.29 -9999.00 2002   10    2   12.00    101   426340.0  4411238.0  3598.0    -3.02    78.77 -9999.00 -9999.00 -9999.00 2002   10    8   12.00    101   426340.0  4411238.0  3598.0    -5.02    88.77 -9999.00 -9999.00 -9999.00 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L1111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_crocus`

```python
to_crocus(
    ds,
    df_pts,
    fname_format='CROCUS_pt_*.nc',
    scale_precip=1,
    climate_dataset_name='ERA5',
    project_author='S. Filhol',
    snow_partition_method='continuous'
)
```

Functiont to export toposcale output to CROCUS netcdf format. Generates one file per point_name 



**Args:**
 
 - <b>`ds`</b> (dataset):  Toposcale downscaled dataset. 
 - <b>`df_pts`</b> (dataframe):  with point list info (x,y,elevation,slope,aspect,svf,...) 
 - <b>`fname_format`</b> (str):  filename format. point_name is inserted where * is 
 - <b>`scale_precip`</b> (float):  scaling factor to apply on precipitation. Default is 1 
 - <b>`climate_dataset_name`</b> (str):  name of original climate dataset. Default 'ERA5', 
 - <b>`project_author`</b> (str):  name of project author(s) 
 - <b>`snow_partition_method`</b> (str):  snow/rain partitioning method: default 'jennings2018_trivariate' 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L1254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_surfex`

```python
to_surfex(
    ds,
    df_pts,
    fname_format='FORCING*.nc',
    scale_precip=1,
    climate_dataset_name='ERA5',
    project_author='S. Filhol',
    snow_partition_method='continuous',
    year_start='2023-08-01 00:00:00',
    year_end='2024-06-30 23:00:00'
)
```

Function to export toposcale output to Surfex forcing netcdf format. Generates one file per year 



**Args:**
 
 - <b>`ds`</b> (dataset):  Toposcale downscaled dataset. 
 - <b>`df_pts`</b> (dataframe):  with point list info (x,y,elevation,slope,aspect,svf,...) 
 - <b>`fname_format`</b> (str):  filename format. point_name is inserted where * is 
 - <b>`scale_precip`</b> (float):  scaling factor to apply on precipitation. Default is 1 
 - <b>`climate_dataset_name`</b> (str):  name of original climate dataset. Default 'ERA5', 
 - <b>`project_author`</b> (str):  name of project author(s) 
 - <b>`snow_partition_method`</b> (str):  snow/rain partitioning method: default 'jennings2018_trivariate' 
 - <b>`year_start`</b> (str):  start of year for yearly Surfex forcing 



**ToDo:**
 
- [x] split to yearly files, 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L1416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_snowpack`

```python
to_snowpack(ds, fname_format='smet_pt_*.tx')
```

Function to export data for snowpack model as smet. https://models.slf.ch/docserver/meteoio/SMET_specifications.pdf 



**Args:**
 
 - <b>`ds`</b> (dataset):  downscaled_pts, 
 - <b>`df_pts`</b> (dataframe):  toposub.df_centroids, 
 - <b>`fname_format`</b> (str pattern):  output format of filename 

format is a text file with the following columns (coulumns can vary) 

SMET 1.1 ASCII [HEADER] station_id       = meteoc1 station_name     = WFJ2 latitude         = 46.829650 longitude        = 9.809328 altitude         = 2539.0 easting          = 780851.861845 northing         = 189232.420554 epsg             = 21781 nodata           = -999 tz               = 0 plot_unit        = time K - m/s ° W/m2 W/m2 kg/m2 - plot_description = time air_temperature relative_humidity wind_velocity wind_direction incoming_short_wave_radiation incoming_long_wave_radiation water_equivalent_precipitation_sum - plot_color       = 0x000000 0x8324A4 0x50CBDB 0x297E24 0x64DD78 0xF9CA25 0xD99521 0x2431A4 0xA0A0A0 plot_min         = -999 253.15 0 0 0 0 150 0 -999 plot_max         = -999 283.15 1 30 360 1400 400 20 -999 fields           = timestamp TA RH VW DW ISWR ILWR PSUM PINT [DATA] 2014-09-01T00:00:00   271.44   1.000    5.9   342     67    304  0.000    0.495 2014-09-01T01:00:00   271.50   0.973    6.7   343    128    300  0.166    0.663 2014-09-01T02:00:00   271.46   0.968    7.4   343    244    286  0.197    0.788 2014-09-01T03:00:00   271.41   0.975    7.6   345    432    273  0.156    0.626 2014-09-01T04:00:00   271.57   0.959    7.8   347    639    249  0.115    0.462 2014-09-01T05:00:00   271.52   0.965    8.2   350    632    261  0.081    0.323 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_export.py#L1483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_geotop`

```python
to_geotop(ds, fname_format='geotop_pt_*.txt')
```

Function to export data for snowpack model as smet. https://models.slf.ch/docserver/meteoio/SMET_specifications.pdf 



**Args:**
 
 - <b>`ds`</b> (dataset):  downscaled_pts, 
 - <b>`df_pts`</b> (dataframe):  toposub.df_centroids, 
 - <b>`fname_format`</b> (str pattern):  output format of filename 



Date format = DD/MM/YYYY hh:mm Air temperature = Degree Celsius Relative Humidity = % Radiations components such SWin, SWout, LWin = W/m2 Precipitation = mm/hr Wind speed = m/s Wind direction = degree Air pressure = mbar 

The format of the file is in *.txt format. 





format is a text file with the following columns (coulumns can vary) 

Date,AirT,WindS,WindDr,RelHum,Swglob,Swout,Lwin,Lwout,Iprec,AirPressure 01/09/2015 00:00,1.5,0.7,165,59,0,0,223.9,315.5,0,581.02897 01/09/2015 01:00,1.2,0.6,270,59,0,0,261.8,319,0,581.02897 01/09/2015 02:00,0.8,0.5,152,61,0,0,229.9,312.2,0,581.02897 01/09/2015 03:00,0.8,0.5,270,60,0,0,226.1,310.8,0,581.02897 01/09/2015 04:00,0.3,0.8,68,60,0,0,215.3,309.6,0,581.02897 01/09/2015 05:00,0.2,0.6,270,62,0,0,230.2,309.1,0,581.02897 01/09/2015 06:00,0.3,0.6,35,62,0,0,222.8,306.7,0,581.02897 01/09/2015 07:00,-0.2,0.3,270,65,0,0,210,305.5,0,581.02897 01/09/2015 08:00,0,0.6,52,59,114,23,218.3,312.3,0,581.02897 01/09/2015 09:00,1.9,1.5,176,57,173,35,220.2,322.8,0,581.02897 01/09/2015 10:00,3.4,1.9,183,47,331,67,245.8,372.6,0,581.02897 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
