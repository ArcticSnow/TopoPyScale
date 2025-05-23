<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5_zarr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.fetch_era5_zarr`





---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5_zarr.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_valid_netcdf`

```python
is_valid_netcdf(file_path)
```

Check if a NetCDF file is valid. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5_zarr.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hypsometric_pressure`

```python
hypsometric_pressure(surface_pressure, geopotential_height, temperature)
```

Calculate the pressure at a given geopotential height using the hypsometric equation. 



**Args:**
 
 - <b>`surface_pressure`</b> (array):  Surface pressure in Pascals. 
 - <b>`geopotential_height`</b> (array):  Geopotential height in meters. 
 - <b>`temperature`</b> (array):  Temperature in Kelvin. 



**Returns:**
 
 - <b>`array`</b>:  Pressure at the given geopotential height in Pascals. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5_zarr.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `q_2_rh`

```python
q_2_rh(temp, surface_pressure, geopotential_height, qair)
```

Convert specific humidity (q) to relative humidity (RH). 



**Args:**
 
 - <b>`temp`</b> (array):  Temperature in Kelvin. 
 - <b>`surface_pressure`</b> (array):  Surface pressure in Pascals. 
 - <b>`geopotential_height`</b> (array):  Geopotential height in meters. 
 - <b>`qair`</b> (array):  Specific humidity in kg/kg. 



**Returns:**
 
 - <b>`array`</b>:  Relative humidity in percentage (0-100%). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5_zarr.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `process_month`

```python
process_month(ds, start, end, output_dir, surface_vars, pressure_vars)
```

Process and save data for a given month. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/fetch_era5_zarr.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `main`

```python
main(start_date, end_date, output_dir, dataset_path)
```

Main function to process ERA5 data. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
