<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.fetch_era5`
Retrieve ERA5 from the CDS (cdsapi) or NCAR GDEX into the zarr store read by the downscaling (see era5_store.py). 


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
- **GDEX_FDIR**
- **GDEX_WIND10**
- **GDEX_INVARIANT_Z**
- **CDS_DATASETS**
- **CDS_MAX_PARALLEL**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `climate_netcdf_dir`

```python
climate_netcdf_dir(climate_path, plev_name='PLEV_*.nc')
```

Folder of the PLEV/SURF netcdf files to convert to zarr: climate_path/yearly (where get_era5() merges the daily files) if it holds some, else climate_path itself (files placed there by hand, as in the examples). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `append_nc_to_zarr_region`

```python
append_nc_to_zarr_region(ncfile, zarrstore, surf=False)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L370"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gdex_root`

```python
gdex_root(root=None)
```

Root of GDEX dataset d633000: root if given, else the local mount on NCAR systems, else OSDF over HTTPS. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L370"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gdex_root`

```python
gdex_root(root=None)
```

Root of GDEX dataset d633000: root if given, else the local mount on NCAR systems, else OSDF over HTTPS. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cds_request`

```python
cds_request(
    task,
    variables,
    area,
    step,
    plevels=None,
    product='reanalysis',
    download_format='unarchived',
    hours=None
)
```

CDS API request of a task: its days (one month at most), all time steps (or the given hours, e.g. [0, 1, ...]), the box on the 0.25 deg grid. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L656"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fetch_era5_to_zarr`

```python
fetch_era5_to_zarr(
    start,
    end,
    eraDir,
    latN,
    latS,
    lonE,
    lonW,
    step='1h',
    plevels=None,
    streams=('plev', 'surf'),
    source='cds',
    product='reanalysis',
    zarr_store='ERA5.zarr',
    n_workers=None,
    plev_variables=None,
    surf_variables=None,
    gdex_root=None,
    download_format='unarchived',
    verify=True,
    retries=3,
    retry_wait=60.0,
    time_chunk=720
)
```

Fetch ERA5 for a period and a box into the zarr store eraDir/zarr_store, read by the downscaling. 

From the CDS, one request per stream and month (as ECMWF advises), at most 10 at a time (the CDS per-user limit). From NCAR GDEX, one day per stream and task, read in spawned processes. Workers only download; the main thread reads each file as it arrives, checks it and writes it to the store (see era5_store.py). The surface geopotential is fetched once. Time steps already in the store are skipped, so an interrupted or failed run is completed by running again, and a longer period only fetches what is new. Data of the earlier layout (yearly/PLEV_*.nc, daily/dPLEV_*.nc...) are read instead of downloaded again. 



**Args:**
 
 - <b>`start, end`</b>:  first and last day 
 - <b>`eraDir`</b> (Path):  climate folder 
 - <b>`latN, latS, lonE, lonW (float)`</b>:  box [deg], reduced to the nodes of the 0.25 deg grid inside it 
 - <b>`step`</b> (str):  '1h', '3h' or '6h'. At 3h and 6h the accumulated surface fields are fetched hourly and stored as  their sums over the time step ending at the stamp 
 - <b>`plevels`</b> (list):  pressure levels [hPa] 
 - <b>`streams`</b> (list):  'plev' and/or 'surf' 
 - <b>`source`</b> (str):  'cds' or 'gdex' 
 - <b>`product`</b> (str):  CDS product type, 'reanalysis' 
 - <b>`zarr_store`</b> (str):  store name in eraDir 
 - <b>`n_workers`</b> (int):  parallel downloads. Default 8 for the CDS (at most 10), 4 processes for GDEX 
 - <b>`plev_variables, surf_variables (list)`</b>:  CDS variable names. Default: what the downscaling reads 
 - <b>`gdex_root`</b> (str):  GDEX root (path or URL). Default: /gdex/data/d633000 if it exists, else OSDF over HTTPS 
 - <b>`download_format`</b> (str):  CDS download format, 'unarchived' or 'zip' (both are read) 
 - <b>`verify`</b> (bool):  verify the SSL certificate of the CDS 
 - <b>`retries`</b> (int):  new attempts of a failed download, after retry_wait, 2 retry_wait, 4 retry_wait... seconds 
 - <b>`retry_wait`</b> (float):  seconds before the first new attempt 
 - <b>`time_chunk`</b> (int):  time steps per chunk of the store 



**Returns:**
 
 - <b>`Path`</b>:  the store 

Raises RuntimeError listing the downloads that failed, once all others are in the store. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L768"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `return_last_fullday`

```python
return_last_fullday()
```

TODO: NEED docstring and explanation  


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L802"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `unzip_file`

```python
unzip_file(file_path)
```

Repack as one netCDF file, in place, a CDS download that is a zip of netCDF files (the CDS sends one for surface fields of different kinds, instantaneous and accumulated). The result is standardized (see remap_CDSbeta) and named like the input with the .nc extension. A netCDF file is left unchanged. 



**Args:**
 
 - <b>`file_path`</b> (str):  downloaded file 



**Returns:**
 
 - <b>`str`</b>:  path of the netCDF file 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L828"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remap_CDSbeta`

```python
remap_CDSbeta(file_pattern, file_type='SURF')
```

Rewrite netCDF files of the CDS with the names of the earlier CDS (time, level), ascending levels, and without the number and expver coordinates (see era5_store.standardize). Files already remapped are left as they are. 



**Args:**
 
 - <b>`file_pattern`</b> (str):  path and pattern of the files, e.g. 'inputs/climate/daily/dSURF_*.nc' 
 - <b>`file_type`</b> (str):  SURF or PLEV (kept for compatibility, both are handled the same way) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L849"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L899"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FetchERA5`
Class to fetch data from ERA5 based on config of loaded by the main class Topoclass.config or config loaded directly from yml file. 



**Args:**
 
 - <b>`config`</b> (str or DefaultMunch):  point to a config object (inherited from Topoclass, or filepath to yml) 
 - <b>`varoi_surf`</b> (list):  list of surface variables to download. Default is None which will fetch only necessary for TPS  
 - <b>`varoi_plev`</b> (list):  list of pressure level variables to download. Default is None which will fetch only necessary for TPS 
 - <b>`verify_ssl`</b> (bool):  boolean flag to stop ssl certificate. DO NOT USE!!! unless needed.   

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FetchERA5.__init__`

```python
__init__(config, varoi_surf=None, varoi_plev=None, verify_ssl=True)
```








---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FetchERA5.get_era5_snowmapper`

```python
get_era5_snowmapper(surf_plev, lastday)
```

Funtion to call fetching of ERA5 data 

**TODO:**
 
- merge monthly data into one file (cdo?)- this creates massive slow down! 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FetchERA5.go_fetch`

```python
go_fetch(surf_plev='all')
```

Fetch ERA5 into the zarr store of the climate folder (climate.<source>.zarr_store, default ERA5.zarr), from the CDS or NCAR GDEX (climate.<source>.data_repository). See fetch_era5_to_zarr(). 



**Args:**
 
 - <b>`surf_plev`</b> (str):  'all', 'surf' or 'plev' 



**Returns:**
 
 - <b>`Path`</b>:  the store 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FetchERA5.load_config`

```python
load_config(config_file)
```





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FetchERA5.to_zarr`

```python
to_zarr(zarr_store=None, time_chunk=720)
```

Convert the yearly PLEV/SURF netcdf files of the climate folder to the zarr store used for downscaling. 



**Args:**
 
 - <b>`zarr_store`</b> (str):  name of the store in the climate folder. Default: climate.<source>.zarr_store or ERA5.zarr 
 - <b>`time_chunk`</b> (int):  number of time steps per chunk 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
