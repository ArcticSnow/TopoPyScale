<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.fetch_era5`
Retrieve ecmwf data with cdsapi. 


- J. Fiddes, Origin implementation 
- S. Filhol adapted in 2021 

**Global Variables**
---------------
- **var_surf_name_google**
- **var_plev_name_google**
- **GDEX_POSIX_ROOT**
- **GDEX_OSDF_ROOT**
- **GDEX_PLEV**
- **GDEX_SURF_AN**
- **GDEX_SURF_ACCUMU**
- **GDEX_INVARIANT_Z**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `append_nc_to_zarr_region`

```python
append_nc_to_zarr_region(ncfile, zarrstore, surf=False)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L294"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `netcdf_to_zarr`

```python
netcdf_to_zarr(
    path_to_netcdfs='inputs/climate/yearly',
    zarrout='inputs/climate/ERA5.zarr',
    plev_name='PLEV_*.nc',
    surf_name='SURF_*.nc',
    time_chunk=720,
    chunks=None,
    max_chunk_mb=16,
    compressor=None
)
```

Convert pairs of PLEV and SURF netcdf files (e.g. yearly files) into one zarr store, file by file. 

Each pair is opened with xr.open_dataset, merged (the surface geopotential is renamed z_surf), cast to float32 and appended along time. The store holds z, t, q, u, v, r (time, level, latitude, longitude) and z_surf, sp, t2m, d2m, tp, ssrd, strd (time, latitude, longitude), the layout read by topo_scale_zarr.ClimateDownscaler. 



**Args:**
 
 - <b>`path_to_netcdfs`</b> (str):  folder of the netcdf files 
 - <b>`zarrout`</b> (str):  output zarr store (overwritten) 
 - <b>`plev_name`</b> (str):  pattern of the pressure level files, * standing for the period (e.g. the year) 
 - <b>`surf_name`</b> (str):  pattern of the surface files 
 - <b>`time_chunk`</b> (int):  number of time steps per chunk 
 - <b>`chunks`</b> (dict):  optional chunk sizes per dimension, overriding the automatic choice 
 - <b>`max_chunk_mb`</b> (float):  target size of a pressure-level chunk, used to choose the spatial chunk size 
 - <b>`compressor`</b> (obj):  zarr codec. Default: BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle') 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_zarr`

```python
to_zarr(
    ds,
    fout,
    chuncks={'x': 3, 'y': 3, 'time': 8760, 'level': 7},
    compressor=None,
    mode='w'
)
```

Function to convert a dataset to a zarr store. 



**Args:**
 
 - <b>`ds`</b> (dataset):  dataset to convert to zarr 
 - <b>`fout`</b> (str):  name of the zarr archive 
 - <b>`chuncks`</b> (dict):  {'x':3, 'y':3, 'time':8760, 'level':7} 
 - <b>`compressor`</b> (obj):  default is None, otherwise use an encoder compatible with zarr3. Blosc is default 
 - <b>`mode`</b> (str):  mode to write to zarr store. See the xarray to_zarr() documentation 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fetch_era5_google_from_zarr`

```python
fetch_era5_google_from_zarr(
    eraDir,
    startDate,
    endDate,
    lonW,
    latS,
    lonE,
    latN,
    plevels,
    bucket='gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
)
```

Function to download data from Zarr repository on Google Cloud Storage (https://github.com/google-research/arco-era5/tree/main) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gdex_root`

```python
gdex_root(root=None)
```

Root of GDEX dataset d633000: root if given, else the local mount on NCAR systems, else OSDF over HTTPS. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L569"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `retrieve_era5_gdex`

```python
retrieve_era5_gdex(
    startDate,
    endDate,
    eraDir,
    latN,
    latS,
    lonE,
    lonW,
    step='1h',
    surf_plev='all',
    plevels=None,
    num_processes=4,
    root=None,
    rm_daily=False
)
```

Fetch ERA5 from NCAR GDEX (dataset d633000) instead of the CDS, into the same daily and yearly files. 

Writes eraDir/daily/dSURF_YYYYMMDD.nc and dPLEV_YYYYMMDD.nc, then eraDir/yearly/SURF_YYYY.nc and PLEV_YYYY.nc, with the variables, units and layout of the CDS files: z, t, u, v, q on pressure levels (ascending) and z, t2m, d2m, sp, ssrd, strd, tp at the surface; latitude descending, longitude in -180..180. Accumulated fluxes are over the hour ending at the time stamp, as on the CDS. Days already in eraDir/daily are skipped. 

On NCAR systems the files are read from /gdex/data. Elsewhere they are read over HTTPS (OSDF), which needs aiohttp. Pressure-level files are chunked as one global field per hour and per variable: whatever the size of the box, about 1 GB per variable and day is transferred. 



**Args:**
 
 - <b>`startDate, endDate`</b>:  first and last day 
 - <b>`eraDir`</b> (Path):  climate folder 
 - <b>`latN, latS, lonE, lonW (float)`</b>:  box [deg] 
 - <b>`step`</b> (str):  '1h', '3h' or '6h' 
 - <b>`surf_plev`</b> (str):  'surf', 'plev' or 'all' 
 - <b>`plevels`</b> (list):  pressure levels [hPa] 
 - <b>`num_processes`</b> (int):  days fetched in parallel (processes, as HDF5 reads hold a global lock) 
 - <b>`root`</b> (str):  GDEX root (path or URL). Default: /gdex/data/d633000 if it exists, else OSDF over HTTPS 
 - <b>`rm_daily`</b> (bool):  remove the daily folder once merged into yearly files 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L650"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `retrieve_era5`

```python
retrieve_era5(
    product,
    startDate,
    endDate,
    eraDir,
    latN,
    latS,
    lonE,
    lonW,
    step,
    num_threads=10,
    surf_plev='surf',
    plevels=None,
    realtime=False,
    output_format='netcdf',
    download_format='unarchived',
    new_CDS_API=True,
    rm_daily=False,
    surf_varoi=None,
    plev_varoi=None,
    verify_flag=True
)
```

Sets up era5 surface retrieval. * Creates list of year/month pairs to iterate through. * MARS retrievals are most efficient when subset by time. * Identifies preexisting downloads if restarted. * Calls api using parallel function. 



**Args:**
 
 - <b>`product`</b>:  "reanalysis" (HRES) or "ensemble_members" (EDA) startDate: endDate: 
 - <b>`eraDir`</b>:  directory to write output 
 - <b>`latN`</b>:  north latitude of bbox 
 - <b>`latS`</b>:  south latitude of bbox 
 - <b>`lonE`</b>:  easterly lon of bbox 
 - <b>`lonW`</b>:  westerly lon of bbox 
 - <b>`step`</b>:  timestep to use: 1, 3, 6 
 - <b>`num_threads`</b>:  number of threads to use for downloading data 
 - <b>`surf_plev`</b>:  download surface single level or pressure level product: 'surf' or 'plev' 
 - <b>`output_format`</b> (str):  default is "netcdf", can be "grib". 
 - <b>`download_format`</b> (str):  default "unarchived". Can be "zip" 
 - <b>`new_CDS_API`</b>:  flag to handle new formating of SURF files with the new CDS API (2024). 
 - <b>`rm_daily`</b>:  remove folder containing all daily ERA5 file. Option to clear space of data converted to yearly files. 
 - <b>`surf_varoi`</b>:  list of surface variable to download. Default is None, automatically assigning minimum required for the downscaling 
 - <b>`plev_varoi`</b>:  list of pressure level variables to download. . Default is None, automatically assigning minimum required for the downscaling 
 - <b>`verify_flag`</b>:  bypass ssl certifcate flag. Default must be True 



**Returns:**
 Monthly era surface files stored in disk. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L877"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_request_surf`

```python
era5_request_surf(
    dataset,
    year,
    month,
    day,
    bbox,
    target,
    product,
    time,
    verify_flag=True,
    output_format='netcdf',
    download_format='unarchived',
    varoi=None
)
```

CDS surface api call 



**Args:**
 
 - <b>`dataset`</b> (str):  copernicus dataset (era5) 
 - <b>`year`</b> (str or list):  year of interest 
 - <b>`month`</b> (str or list):  month of interest 
 - <b>`bbox`</b> (list):  bonding box in lat-lon 
 - <b>`target`</b> (str):  filename 
 - <b>`product`</b> (str):  type of model run. defaul: reanalysis 
 - <b>`time`</b> (str or list):  hours for which to download data 
 - <b>`output_format`</b> (str):  default is "netcdf", can be "grib". 
 - <b>`download_format`</b> (str):  default "unarchived". Can be "zip" 
 - <b>`varoi`</b> (list):  list of variable of interest to download. Default to None which fallsback to minimum needed for TPS 



**Returns:**
 Store to disk dataset as indicated 



**TODO:**
 
- [ ] change SURF request to the ZARR repository for timeseries retrieval. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L926"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_request_plev`

```python
era5_request_plev(
    dataset,
    year,
    month,
    day,
    bbox,
    target,
    product,
    time,
    plevels,
    verify_flag=True,
    output_format='netcdf',
    download_format='unarchived',
    varoi=None
)
```

CDS plevel api call 



**Args:**
 
 - <b>`dataset`</b> (str):  copernicus dataset (era5) 
 - <b>`year`</b> (str or list):  year of interest 
 - <b>`month`</b> (str or list):  month of interest 
 - <b>`bbox`</b> (list):  bonding box in lat-lon 
 - <b>`target`</b> (str):  filename 
 - <b>`product`</b> (str):  type of model run. defaul: reanalysis 
 - <b>`time`</b> (str or list):  hours to query 
 - <b>`plevels`</b> (str or list):  pressure levels to query 
 - <b>`output_format`</b> (str):  default is "netcdf", can be "grib". 
 - <b>`download_format`</b> (str):  default "unarchived". Can be "zip" 
 - <b>`varoi`</b> (list):  list of variable of interest to download. Default to None which fallsback to minimum needed for TPS 



**Returns:**
 Store to disk dataset as indicated 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L977"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_realtime_surf`

```python
era5_realtime_surf(eraDir, dataset, bbox, product)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L1000"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_realtime_plev`

```python
era5_realtime_plev(eraDir, dataset, bbox, product, plevels)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L1025"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `return_last_fullday`

```python
return_last_fullday()
```

TODO: NEED docstring and explanation  


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L1059"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `grib2netcdf`

```python
grib2netcdf(gribname, outname=None)
```

Function to convert grib file to netcdf 



**Args:**
 
 - <b>`gribname`</b>:  filename fo grib file to convert 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L1073"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `unzip_file`

```python
unzip_file(file_path)
```

Function to unpack and repack as NETCDF data sent by the new CDS API. CDS can send either weird zip file shipped as netcdf file for Surface data.  The option of zip format can be used to download the data.  



**Args:**
 
 - <b>`file_path`</b>:  path of era5 data. Typically in TopoPyScale project it will be at: ./inputs/climate/daily 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L1129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remap_CDSbeta`

```python
remap_CDSbeta(file_pattern, file_type='SURF')
```

Remapping of variable names from CDS beta to CDS legacy standard. 



**Args:**
 
 - <b>`file_pattern`</b> (str):  path and pattern of the file to be remapped. example: 'inputs/climate/daily/dSURF_*.nc' 
 - <b>`file_type`</b> (str):  type of file: SURF or PLEV.   


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L1192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_request_surf_snowmapper`

```python
era5_request_surf_snowmapper(
    today,
    latN,
    latS,
    lonE,
    lonW,
    eraDir,
    output_format='netcdf'
)
```

CDS surface api call 



**Args:**
 
 - <b>`dataset`</b> (str):  copernicus dataset (era5) 
 - <b>`today`</b> (datetime object):  day to download 
 - <b>`target`</b> (str):  filename 
 - <b>`product`</b> (str):  type of model run. defaul: reanalysis 
 - <b>`time`</b> (str or list):  hours for which to download data 
 - <b>`format`</b> (str):  "grib" or "netcdf" 



**Returns:**
 Store to disk dataset as indicated 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L1242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_request_plev_snowmapper`

```python
era5_request_plev_snowmapper(
    today,
    latN,
    latS,
    lonE,
    lonW,
    eraDir,
    plevels,
    output_format='netcdf'
)
```

CDS plevel api call 



**Args:**
 
 - <b>`dataset`</b> (str):  copernicus dataset (era5) 
 - <b>`year`</b> (str or list):  year of interest 
 - <b>`month`</b> (str or list):  month of interest 
 - <b>`bbox`</b> (list):  bonding box in lat-lon 
 - <b>`target`</b> (str):  filename 
 - <b>`product`</b> (str):  type of model run. defaul: reanalysis 
 - <b>`time`</b> (str or list):  hours to query 
 - <b>`plevels`</b> (str or list):  pressure levels to query 



**Returns:**
 Store to disk dataset as indicated 


---

## <kbd>class</kbd> `FetchERA5`
Class to fetch data from ERA5 based on config of loaded by the main class Topoclass.config or config loaded directly from yml file. 



**Args:**
 
 - <b>`config`</b> (str or DefaultMunch):  point to a config object (inherited from Topoclass, or filepath to yml) 
 - <b>`varoi_surf`</b> (list):  list of surface variables to download. Default is None which will fetch only necessary for TPS  
 - <b>`varoi_plev`</b> (list):  list of pressure level variables to download. Default is None which will fetch only necessary for TPS 
 - <b>`verify_ssl`</b> (bool):  boolean flag to stop ssl certificate. DO NOT USE!!! unless needed.   

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(config, varoi_surf=None, varoi_plev=None, verify_ssl=True)
```








---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_era5_snowmapper`

```python
get_era5_snowmapper(surf_plev, lastday)
```

Funtion to call fetching of ERA5 data 

**TODO:**
 
- merge monthly data into one file (cdo?)- this creates massive slow down! 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `go_fetch`

```python
go_fetch(surf_plev='all')
```

Funtion to call fetching of ERA5 data 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `load_config`

```python
load_config(config_file)
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_zarr`

```python
to_zarr(zarr_store=None, time_chunk=720)
```

Convert the yearly PLEV/SURF netcdf files of the climate folder to the zarr store used for downscaling. 



**Args:**
 
 - <b>`zarr_store`</b> (str):  name of the store in the climate folder. Default: climate.<source>.zarr_store or ERA5.zarr 
 - <b>`time_chunk`</b> (int):  number of time steps per chunk 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
