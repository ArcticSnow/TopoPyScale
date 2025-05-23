<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.fetch_era5`
Retrieve ecmwf data with cdsapi. 


- J. Fiddes, Origin implementation 
- S. Filhol adapted in 2021 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/retrieve_era5#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    output_format='netcdf'
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



**Returns:**
 Monthly era surface files stored in disk. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/era5_request_surf#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_request_surf`

```python
era5_request_surf(
    dataset,
    year,
    month,
    bbox,
    target,
    product,
    time,
    output_format='netcdf'
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
 - <b>`format`</b> (str):  "grib" or "netcdf" 



**Returns:**
 Store to disk dataset as indicated 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/era5_request_plev#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_request_plev`

```python
era5_request_plev(
    dataset,
    year,
    month,
    bbox,
    target,
    product,
    time,
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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/era5_realtime_surf#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_realtime_surf`

```python
era5_realtime_surf(eraDir, dataset, bbox, product)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/era5_realtime_plev#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `era5_realtime_plev`

```python
era5_realtime_plev(eraDir, dataset, bbox, product, plevels)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/return_last_fullday#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `return_last_fullday`

```python
return_last_fullday()
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/grib2netcdf#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `grib2netcdf`

```python
grib2netcdf(gribname)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/process_SURF_file#L375"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `process_SURF_file`

```python
process_SURF_file(wdir)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/remap_CDSbeta#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remap_CDSbeta`

```python
remap_CDSbeta(wdir)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/era5_request_surf_snowmapper#L503"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5/era5_request_plev_snowmapper#L553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
