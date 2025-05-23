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

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fetch_era5_google`

```python
fetch_era5_google(
    eraDir,
    startDate,
    endDate,
    lonW,
    latS,
    lonE,
    latN,
    plevels,
    step='3H',
    num_threads=1
)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    new_CDS_API=True
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



**Returns:**
 Monthly era surface files stored in disk. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    output_format='netcdf',
    download_format='unarchived'
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



**Returns:**
 Store to disk dataset as indicated 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    output_format='netcdf',
    download_format='unarchived'
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



**Returns:**
 Store to disk dataset as indicated 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L373"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_realtime_surf`

```python
era5_realtime_surf(eraDir, dataset, bbox, product)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_realtime_plev`

```python
era5_realtime_plev(eraDir, dataset, bbox, product, plevels)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L462"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `return_last_fullday`

```python
return_last_fullday()
```

TODO: NEED docstring and explanation  


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L499"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `grib2netcdf`

```python
grib2netcdf(gribname, outname=None)
```

Function to convert grib file to netcdf 



**Args:**
 
 - <b>`gribname`</b>:  filename fo grib file to convert 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L515"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `process_SURF_file`

```python
process_SURF_file(wdir)
```

Function to unpack and repack as NETCDF data sent by the new CDS API, which sends a ZIP file as NETCDF file. 



**Args:**
 
 - <b>`wdir`</b>:  path of era5 data. Typically in TopoPyScale project it will be at: ./inputs/climate 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L576"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remap_CDSbeta`

```python
remap_CDSbeta(wdir)
```

Remapping of variable names from CDS beta to CDS legacy standard. 



**Args:**
 
 - <b>`wdir`</b>:  path of era5 data. Typically in TopoPyScale project it will be at: ./inputs/climate 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L650"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5.py#L700"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
