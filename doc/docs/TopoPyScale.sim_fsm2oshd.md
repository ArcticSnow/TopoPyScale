<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.sim_fsm2oshd`
Methods to perform FSM2oshd simulations S. Filhol, September 2023 



**TODO:**
 


- open and forest namelist 
- run open/forest simulation 
- combine results based on weights of open vs forest 
- recover information of forest cover fraction from namelist file. Python namelist parser package? 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_fsm_ds_with_mask`

```python
read_fsm_ds_with_mask(path)
```

function to read fsm as ds, adding a dummy cluster for masked area 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fsm2oshd_sim_parallel`

```python
fsm2oshd_sim_parallel(
    simulation_path='./fsm_sim,',
    fsm_forest_nam='fsm__forest*.nam',
    fsm_open_nam='fsm__open*.nam',
    fsm_exec='./FSM_OSHD',
    n_cores=6,
    delete_nlst_files=False
)
```

Function to run parallelised simulations of FSM 



**Args:**
 
 - <b>`fsm_forest_nam`</b> (str):  file pattern of namelist file for forest setup. Default = 'fsm_sim/fsm__forest*.nam' 
 - <b>`fsm_open_nam`</b> (str):  file pattern of namelist file for open setup. Default = 'fsm_sim/fsm__open*.nam' 
 - <b>`fsm_exec`</b> (str):  path to FSM executable. Default = './FSM' 
 - <b>`n_cores`</b> (int):  number of cores. Default = 6 
 - <b>`delete_nlst_files`</b> (bool):  delete simulation configuration files or not. Default=True 



**Returns:**
 NULL (FSM simulation file written to disk) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `txt2ds`

```python
txt2ds(fname)
```

Function to read a single FSM text file output as a xarray dataset 

**Args:**
 
 - <b>`fname`</b> (str):  filename 



**Returns:**
 xarray dataset of dimension (time, point_name) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `output_to_ds`

```python
output_to_ds(fname_pattern, complevel=9, remove_file=False, fout='fsm_outputs')
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `aggregate_all_open_forest`

```python
aggregate_all_open_forest(
    fname_df_forest,
    fname_forest_outputs='fsm_sim/fsm_outputs_fores_*.nc',
    fname_open_outputs='fsm_sim/fsm_outputs_open_*.nc',
    fout='fsm_sim/fsm_outputs',
    n_cores=6,
    n_digits=3,
    remove_tmp_outputs=True
)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_pt_fsm2oshd`

```python
read_pt_fsm2oshd(fname)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_pt_met_fsm2oshd`

```python
read_pt_met_fsm2oshd(fname)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
