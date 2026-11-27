<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.sim_fsm`
Methods to generate required simulation files and run simulations of various models using tscale forcing J. Fiddes, February 2022 



**TODO:**
 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `infer_met_timestep`

```python
infer_met_timestep(metfile, default=3600)
```

Time step [s] of an FSM forcing file, from its first two rows (year month day hour ...). 



**Args:**
 
 - <b>`metfile`</b> (str):  path to the FSM forcing file 
 - <b>`default`</b> (float):  value returned, with a warning, if the time step cannot be inferred 



**Returns:**
 
 - <b>`float`</b>:  time step in seconds 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fsm_nlst`

```python
fsm_nlst(nconfig, metfile, nave, dt_seconds=3600)
```

Function to generate namelist parameter file that is required to run the FSM model. https://github.com/RichardEssery/FSM 



**Args:**
 
 - <b>`nconfig`</b> (int):  which FSm configuration to run (integer 1-31) 
 - <b>`metfile`</b> (str):  path to input tscale file (relative as Fortran fails with long strings (max 21 chars?)) 
 - <b>`nave`</b> (int):  number of forcing steps to average output over eg if forcing is hourly and output required is daily then nave = 24  (4 with 6-hourly forcing) 
 - <b>`dt_seconds`</b> (int):  time step of the forcing file [s] (FSM namelist dt). Default 3600 (hourly) 



**Returns:**
 NULL (writes out namelist text file which configures a single FSM run) 



**Notes:**

> constraint is that Fortran fails with long strings (max?) definition: https://github.com/RichardEssery/FSM/blob/master/nlst_CdP_0506.txt 
>
>
>Example nlst: &config / &drive met_file = 'data/met_CdP_0506.txt' zT = 1.5 zvar = .FALSE. / &params / &initial Tsoil = 282.98 284.17 284.70 284.70 / &outputs out_file = 'out_CdP_0506.txt' / 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `txt2ds`

```python
txt2ds(fname)
```

Function to read a single FSM text file output as a xarray dataset 

**Args:**
 
 - <b>`fname`</b> (str):  filename 



**Returns:**
 xarray dataset of dimension (time, point_id) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_netcdf`

```python
to_netcdf(fname_fsm_sim, complevel=9)
```

Function to convert a single FSM simulation output file (.txt) to a compressed netcdf file (.nc) 



**Args:**
 
 - <b>`fname_fsm_sim`</b> (str):  filename to convert from txt to nc 
 - <b>`complevel`</b> (int):  Compression level. 1-9 



**Returns:**
 NULL (FSM simulation file written to disk) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_netcdf_parallel`

```python
to_netcdf_parallel(
    fsm_sims='./fsm_sims/sim_FSM*.txt',
    n_core=6,
    complevel=9,
    delete_txt_files=False
)
```

Function to convert FSM simulation output (.txt files) to compressed netcdf. This function parallelize jobs on n_core 



**Args:**
 
 - <b>`fsm_sims`</b> (str or list):  file pattern or list of file output of FSM. Note that must be in short relative path!. Default = 'fsm_sims/sim_FSM_pt*.txt' 
 - <b>`n_core`</b> (int):  number of cores. Default = 6: 
 - <b>`complevel`</b> (int):  Compression level. 1-9 
 - <b>`delete_txt_files`</b> (bool):  delete simulation txt files or not. Default=True 



**Returns:**
 NULL (FSM simulation file written to disk) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_dataset`

```python
to_dataset(fname_pattern='sim_FSM_pt*.txt', fsm_path='./fsm_sims/')
```

Function to read FSM outputs of one simulation into a single dataset. 

**Args:**
  fname_pattern:  fsm_path: 



**Returns:**
  dataset (xarray) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_pt_fsm`

```python
read_pt_fsm(fname)
```

Function to load FSM simulation output into a pandas dataframe 

**Args:**
 
 - <b>`fname`</b> (str):  path to simulation file to open 



**Returns:**
 pandas dataframe 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fsm_sim_parallel`

```python
fsm_sim_parallel(
    fsm_input='outputs/FSM_pt*.txt',
    fsm_nconfig=31,
    fsm_nave=24,
    fsm_exec='./FSM',
    n_core=6,
    n_thread=100,
    delete_nlst_files=True,
    fsm_dt=None
)
```

Function to run parallelised simulations of FSM 



**Args:**
 
 - <b>`fsm_input`</b> (str or list):  file pattern or list of file input to FSM. Note that must be in short relative path!. Default = 'outputs/FSM_pt*.txt' 
 - <b>`fsm_nconfig`</b> (int):   FSM configuration number. See FSM README.md: https://github.com/RichardEssery/FSM. Default = 31 
 - <b>`fsm_nave`</b> (int):  number of timestep to average for outputs. e.g. If input is hourly and fsm_nave=24, outputs will be daily. Default = 24 
 - <b>`fsm_exec`</b> (str):  path to FSM executable. Default = './FSM' 
 - <b>`n_core`</b> (int):  number of cores. Default = 6 
 - <b>`n_thread`</b> (int):  number of threads when creating simulation configuration files. Default=100 
 - <b>`delete_nlst_files`</b> (bool):  delete simulation configuration files or not. Default=True 
 - <b>`fsm_dt`</b> (int):  time step of the forcing [s] (FSM namelist dt). Default: inferred from the first forcing file 



**Returns:**
 NULL (FSM simulation file written to disk) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L332"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fsm_sim`

```python
fsm_sim(nlstfile, fsm_exec, delete_nlst_files=True)
```

Function to simulate the FSM model https://github.com/RichardEssery/FSM 



**Args:**
 
 - <b>`nlstfile`</b> (int):  which FSm configuration to run (integer 1-31) 
 - <b>`fsm_exec`</b> (str):  path to input tscale file (relative as Fortran fails with long strings (max 21 chars?)) 



**Returns:**
 NULL (FSM simulation file written to disk) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L355"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `agg_by_var_fsm`

```python
agg_by_var_fsm(var='snd', fsm_path='./outputs/fsm_sims', n_workers=8)
```

Function to make single variable multi cluster files as preprocessing step before spatialisation. This is much more efficient than looping over individual simulation files per cluster. For V variables , C clusters and T timesteps this turns C individual files of dimensions V x T into V individual files of dimensions C x T. 

Currently written for FSM files but could be generalised to other models. 



**Args:**
 
 - <b>`var`</b> (str):  column name of variable to extract, one of: alb, rof, hs, swe, gst, gt50  alb - albedo  rof - runoff  snd - snow height (m)  swe - snow water equivalent (mm)  gst - ground surface temp (10cm depth) degC  gt50 - ground temperature (50cm depth) degC 


 - <b>`fsm_path`</b> (str):  location of simulation files 
 - <b>`n_workers`</b> (int):  number of parallel threads for file reading (default: 8) 

**Returns:**
 dataframe 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `agg_by_var_fsm_ensemble`

```python
agg_by_var_fsm_ensemble(var='snd', W=1)
```

Function to make single variable multi cluster files as preprocessing step before spatialisation. This is much more efficient than looping over individual simulation files per cluster. For V variables , C clusters and T timesteps this turns C individual files of dimensions V x T into V individual files of dimensions C x T. 

Currently written for FSM files but could be generalised to other models. 



**Args:**
 
 - <b>`var`</b> (str):  column name of variable to extract, one of: alb, rof, hs, swe, gst, gt50  alb - albedo  rof - runoff  hs - snow height (m)  swe - snow water equivalent (mm)  gst - ground surface temp (10cm depth) degC  gt50 - ground temperature (50cm depth) degC 

W - weight vector from PBS 

**Returns:**
 dataframe 

ncol: 4 = rof 5 = hs 6 = swe 7 = gst 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `timeseries_means_period`

```python
timeseries_means_period(df, start_date, end_date)
```

Function to extract results vectors from simulation results. This can be entire time period some subset or sing day. 



**Args:**
 
 - <b>`df`</b> (dataframe):  results df 
 - <b>`start_date`</b> (str):  start date of average (can be used to extract single day) '2020-01-01' 
 - <b>`end_date`</b> (str):  end date of average (can be used to extract single day) '2020-01-03' 



**Returns:**
 
 - <b>`dataframe`</b>:  averaged dataframe 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L560"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `topo_map`

```python
topo_map(df_mean, mydtype, outname='outputmap.tif')
```

Function to map results to toposub clusters generating map results. 



**Args:**
 
 - <b>`df_mean`</b> (dataframe):  an array of values to map to dem same length as number of toposub clusters 
 - <b>`mydtype`</b>:  HS (maybeGST) needs "float32" other vars can use "int16" to save space 

Here 's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron' s answer: https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `topo_map_headless`

```python
topo_map_headless(df_mean, mydtype, outname='outputmap.tif')
```

Headless server version of Function to map results to toposub clusters generating map results. 



**Args:**
 
 - <b>`df_mean`</b> (dataframe):  an array of values to map to dem same length as number of toposub clusters 
 - <b>`mydtype`</b>:  HS (maybeGST) needs "float32" other vars can use "int16" to save space 

Here 's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron' s answer: https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L661"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `topo_map_forcing`

```python
topo_map_forcing(ds_var, n_decimals=2, dtype='float32', new_res=None)
```

Function to map forcing to toposub clusters generating gridded forcings 



**Args:**
 
 - <b>`ds_var`</b>:  single variable of ds eg. mp.downscaled_pts.t 
 - <b>`n_decimals`</b> (int):  number of decimal to round vairable. default 2 
 - <b>`dtype`</b> (str):  dtype to export raster. default 'float32' 
 - <b>`new_res`</b> (float):  optional parameter to resample output to (in units of projection 

Return: 
 - <b>`grid_stack`</b>:  stack of grids with dimension Time x Y x X 

Here 's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron' s answer: https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L740"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `topo_map_sim`

```python
topo_map_sim(ds_var, n_decimals=2, dtype='float32', new_res=None)
```

Function to map sim results to toposub clusters generating gridded results 



**Args:**
 
 - <b>`ds_var`</b>:  single variable of ds eg. mp.downscaled_pts.t 
 - <b>`n_decimals`</b> (int):  number of decimal to round vairable. default 2 
 - <b>`dtype`</b> (str):  dtype to export raster. default 'float32' 
 - <b>`new_res`</b> (float):  optional parameter to resample output to (in units of projection 

Return: 
 - <b>`grid_stack`</b>:  stack of grids with dimension Time x Y x X 

Here 's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron' s answer: https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L805"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `topo_map_sim_memsafe`

```python
topo_map_sim_memsafe(ds_var, n_decimals=2, dtype='float32', new_res=None)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L862"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_ncdf`

```python
write_ncdf(
    wdir,
    grid_stack,
    var,
    units,
    epsg,
    res,
    mytime,
    lats,
    lons,
    mydtype,
    newfile,
    outname=None
)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L903"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `agg_stats`

```python
agg_stats(df)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L928"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `climatology`

```python
climatology(HSdf, fsm_path)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L942"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `climatology_plot`

```python
climatology_plot(
    var,
    mytitle,
    HSdf_daily_median,
    HSdf_daily_quantiles,
    HSdf_realtime=None,
    plot_show=False
)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L1001"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `climatology_plot2`

```python
climatology_plot2(
    mytitle,
    HSdf_daily_median,
    HSdf_daily_quantiles,
    HSdf_realtime=None,
    plot_show=False
)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L1079"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `concat_fsm`

```python
concat_fsm(mydir)
```

A small routine to concatinate fsm results from separate years in a single file.  This is mainly needed when a big job is split into years for parallel processing on eg a cluster 

rsync -avz --include="<file_pattern>" --include="*/" --exclude="*" <username>@<remote_server_address>:/path/to/remote/directory/ <local_destination_directory> 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L1119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `netcdf_day_to_geotiff`

```python
netcdf_day_to_geotiff(netcdf_file, target_date, output_tiff, variable=None)
```

Extract a specific day from a NetCDF file and write it as a GeoTIFF. 



**Args:**
 
 - <b>`netcdf_file`</b> (str):  Path to the NetCDF file (e.g., '202409_HS.nc') 
 - <b>`target_date`</b> (str):  Date to extract in 'YYYY-MM-DD' format (e.g., '2024-09-01') 
 - <b>`output_tiff`</b> (str):  Output GeoTIFF filename (e.g., 'snow_depth_2024-09-01.tif') 
 - <b>`variable`</b> (str):  Variable name to extract. If None, uses the first data variable 



**Returns:**
 
 - <b>`None`</b>:  Writes GeoTIFF file to disk 

# Extract September 1st, 2024 from your snow depth file sim.netcdf_day_to_geotiff( '202409_HS.nc',  '2024-09-01',  'snow_depth_2024-09-01.tif' ) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm.py#L1245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `netcdf_lonlat_to_timeseries`

```python
netcdf_lonlat_to_timeseries(
    netcdf_file,
    target_lon,
    target_lat,
    variable=None,
    output_csv=None
)
```

Extract a time series for a specific longitude/latitude location from a NetCDF file. 



**Args:**
 
 - <b>`netcdf_file`</b> (str):  Path to the NetCDF file (e.g., '202409_HS.nc') 
 - <b>`target_lon`</b> (float):  Target longitude 
 - <b>`target_lat`</b> (float):  Target latitude   
 - <b>`variable`</b> (str):  Variable name to extract. If None, uses the first data variable 
 - <b>`output_csv`</b> (str):  Optional CSV filename to save the time series 



**Returns:**
 
 - <b>`pandas.DataFrame`</b>:  Time series data with datetime index and variable values 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
