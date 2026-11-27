<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.era5_store`
ERA5 zarr store filled as the downloads arrive, whatever the source (CDS or NCAR GDEX). 

The store (inputs/climate/ERA5.zarr by default) is created by the first data written, with the time axis of the whole period allocated. Chunks not yet written read as NaN. 


- z, t, q, u, v (time, level, latitude, longitude): pressure levels, float32 
- t2m, d2m, sp, tp, ssrd, strd (time, latitude, longitude): surface fields, float32 
- z_surf (latitude, longitude): surface geopotential, static 
- filled_plev, filled_surf (time): 1 where the time step was written, after validation 

At 3 h or 6 h time steps, the accumulated surface fields (tp, ssrd, strd...) are fetched hourly and stored as the sum over the time step ending at the stamp (store attribute accumulation_hours), not as the hour ending at the stamp: the hours of the first window of a request on the previous day are fetched with it. 

Workers only transfer files. Every netCDF file is read by the main thread, one at a time, as netCDF-C and HDF5 are not thread safe. Each file is standardized (CDS names, ERA5/ERA5T expver, level order), checked (time steps, variables, levels, grid, missing values), written to its time slice of the store, then deleted. 

The filled flags make the store its own record of what was fetched: a rerun only fetches the missing time steps, a longer period extends the store, and the downscaling refuses a store with gaps. 

S. Filhol, J. Fiddes and contributors 

**Global Variables**
---------------
- **GRID**
- **STREAMS**
- **STEP_HOURS**
- **SHORT_NAMES**
- **PLEV_VARIABLES**
- **SURF_VARIABLES**
- **FDIR**
- **REQUIRED**
- **DIMS**
- **REQUEST_KEYS**
- **ACCUMULATED**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `accumulated_names`

```python
accumulated_names(variables)
```

CDS variable names of accumulated fields. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `window_hours`

```python
window_hours(stamps, acc_hours)
```

Hours of the windows of acc_hours hours ending at each stamp, (stamp, hour). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `time_axis`

```python
time_axis(start, end, step)
```

Time steps from the first hour of start to the last time step of end. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `snap_area`

```python
snap_area(latN, latS, lonW, lonE, res=0.25)
```

Box [N, W, S, E] on the nodes of the ERA5 grid inside it: the nodes the CDS and GDEX return for the box, so that every source and the store agree on the grid. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `grid_of`

```python
grid_of(area, res=0.25)
```

Latitudes (descending) and longitudes (ascending) of the nodes of a box from snap_area(). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `month_tasks`

```python
month_tasks(start, end, streams=('plev', 'surf'))
```

One task per stream and calendar month (or part of it within the period), the request size ECMWF advises. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `day_tasks`

```python
day_tasks(start, end, streams=('plev', 'surf'))
```

One task per stream and day. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `standardize`

```python
standardize(ds)
```

Harmonize an ERA5 netCDF file: CDS names (valid_time, pressure_level), ERA5 completed by ERA5T where expver is a dimension, extra coordinates dropped (number, expver...), levels ascending, latitude descending, longitude ascending. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `open_raw`

```python
open_raw(paths)
```

Read downloaded files into memory, standardized: netCDF files, or zip files of netCDF files (the CDS sends one for surface fields of different kinds, instantaneous and accumulated). Several files (a request split in parts) are combined by time and variable. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `retry`

```python
retry(fun, retries=3, wait=60.0, fatal=None, label='')
```

Call fun(); on failure call it again after wait, 2 wait, 4 wait... seconds, unless fatal(error) is True. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L530"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `store_gaps`

```python
store_gaps(store, start, end)
```

What a store lacks to cover the period [start, end]: a list of messages, empty if complete. None for a store without filled flags (built from netCDF files by fetch_era5.netcdf_to_zarr), which is not checked. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L634"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `import_legacy`

```python
import_legacy(store, tasks, legacy, area, plev_variables, surf_variables)
```

Write the tasks found in earlier netCDF files to the store. Returns the tasks still to download. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L659"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run`

```python
run(store, tasks, fetch, n_workers=1, processes=False)
```

Fetch the tasks in parallel workers, threads (or spawned processes for sources read with HDF5), and write each result to the store from this thread as soon as it arrives. 



**Args:**
 
 - <b>`store`</b> (Era5Store):  store 
 - <b>`tasks`</b> (list):  tasks to fetch 
 - <b>`fetch`</b> (callable):  fetch(task) -> list of downloaded files (picklable when processes is True) 
 - <b>`n_workers`</b> (int):  number of workers 
 - <b>`processes`</b> (bool):  workers are processes instead of threads 

Raises RuntimeError listing the tasks that failed, after all the others are written. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Task`
One download: a stream over whole days. The stream is 'plev', 'surf' or 'zsurf' (the static surface geopotential, fetched once). 

<a href="https://github.com/ArcticSnow/TopoPyScale/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Task.__init__`

```python
__init__(key: str, stream: str, days: tuple) → None
```








---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Task.times`

```python
times(step)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Era5Store`
ERA5 zarr store written by time slices (see the module docstring). 



**Args:**
 
 - <b>`path`</b> (str):  store path 
 - <b>`request`</b> (dict):  what the store holds: product, area [N, W, S, E] on the grid, levels (hPa), timestep,  plev_variables and surf_variables (CDS names). An existing store must hold the same. 
 - <b>`times`</b> (DatetimeIndex):  time steps of the period. An existing store is extended to cover them. 
 - <b>`time_chunk`</b> (int):  time steps per chunk 
 - <b>`max_chunk_mb`</b> (float):  target size of a pressure-level chunk, used to choose the spatial chunk size 
 - <b>`compressor`</b> (obj):  zarr codec. Default: BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle') 

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Era5Store.__init__`

```python
__init__(path, request, times, time_chunk=720, max_chunk_mb=16, compressor=None)
```






---

#### <kbd>property</kbd> Era5Store.exists







---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Era5Store.aggregate`

```python
aggregate(task, ds)
```

Hourly surface data of a task to its time stamps: instantaneous fields at the stamps, accumulated fields summed over the acc_hours hours ending at each stamp. 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Era5Store.check`

```python
check(task, ds)
```

Check a standardized dataset against the task and the store; returned as float32, dimensions ordered. 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Era5Store.missing`

```python
missing(tasks)
```

Tasks with time steps not yet written (and the surface geopotential task if it is missing). 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Era5Store.task_times`

```python
task_times(task)
```

Time steps the downloaded data of a task hold: its time stamps, or for surface fields stored as window sums all the hours of its windows, from the previous day. 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L489"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Era5Store.write`

```python
write(task, ds)
```

Check ds (a standardized dataset of the task) and write it to its time slice, then flag the time steps. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LegacyFiles`
netCDF files of the earlier layout, read instead of downloading again: climate/yearly/PLEV_*.nc and SURF_*.nc, the same in the climate folder, and climate/daily/dPLEV_*.nc and dSURF_*.nc. 

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L565"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LegacyFiles.__init__`

```python
__init__(folder)
```








---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L570"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LegacyFiles.index`

```python
index(stream)
```

(file, first time, last time) of the readable files of a stream. 

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/era5_store.py#L585"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LegacyFiles.read`

```python
read(task, step, area, times=None)
```

Data of the task from the files, standardized; None if the files do not hold all of it on the grid. Reads the variables of self.names ({'plev': [...], 'surf': [...]}, short names) at times (default: the time stamps of the task). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
