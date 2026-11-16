<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.sim_fsm2oshd`
Methods to perform FSM2oshd simulations S. Filhol, September 2023 

Workflow: 1. topo_export.to_fsm2oshd() writes, for each point, a met file and namelists for the 'forest' and 'open' modes:  <simulation_path>/<fname_format>_forest_<ind>.nam, ..._open_<ind>.nam, outputs ..._outputs_<mode>_<ind>.txt 2. fsm2oshd_sim_parallel() runs FSM2oshd on all namelists and converts each text output to netcdf 3. aggregate_all_open_forest() combines the forest and open outputs of each point, weighted by the forest proportion 4. read_fsm_ds_with_mask() reads the combined outputs as one dataset 

**Global Variables**
---------------
- **N_BITS**
- **FSM_OUTPUT_COLUMNS**
- **FSM_MET_COLUMNS**
- **OUTPUT_ATTRS**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `namelist_output_file`

```python
namelist_output_file(nam_file)
```

Name of the output file declared in a FSM2oshd namelist (out_file = '...'). 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_fsm_ds_with_mask`

```python
read_fsm_ds_with_mask(path)
```

Read FSM netcdf outputs (file pattern) as one dataset along point_ind, with a dummy point -9999 for the masked area. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fsm2oshd_sim_parallel`

```python
fsm2oshd_sim_parallel(
    simulation_path='./fsm_sim',
    fsm_forest_nam='fsm__forest*.nam',
    fsm_open_nam='fsm__open*.nam',
    fsm_exec='./FSM_OSHD',
    n_cores=6,
    delete_nlst_files=False,
    keep_txt=False
)
```

Function to run parallelised simulations of FSM 



**Args:**
 
 - <b>`simulation_path`</b> (str):  folder of the namelists. Default './fsm_sim' 
 - <b>`fsm_forest_nam`</b> (str):  file pattern of namelist file for forest setup. Default = 'fsm__forest*.nam' 
 - <b>`fsm_open_nam`</b> (str):  file pattern of namelist file for open setup. Default = 'fsm__open*.nam'. If None, then only performs forest simulation 
 - <b>`fsm_exec`</b> (str):  path to FSM executable. Default = './FSM_OSHD' 
 - <b>`n_cores`</b> (int):  number of cores. Default = 6 
 - <b>`delete_nlst_files`</b> (bool):  delete simulation configuration files or not. Default=False 
 - <b>`keep_txt`</b> (bool):  keep the text outputs of FSM after their conversion to netcdf. Default=False 



**Returns:**
 
 - <b>`list`</b>:  namelists whose simulation failed (empty if all succeeded). A warning lists them. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `txt2ds`

```python
txt2ds(fname)
```

Function to read a single FSM text file output as a xarray dataset 

**Args:**
 
 - <b>`fname`</b> (str):  filename 



**Returns:**
 xarray dataset of dimension (time), with the point index as coordinate point_ind 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `output_to_ds`

```python
output_to_ds(
    fname_pattern,
    complevel=9,
    remove_file=False,
    fout='fsm_outputs',
    n_digits=3,
    n_cores=1
)
```

Copy FSM netcdf outputs to files named {fout}_{point_ind}.nc 



**Args:**
 
 - <b>`fname_pattern`</b> (str):  pattern of the FSM netcdf outputs 
 - <b>`complevel`</b> (int):  compression level 
 - <b>`remove_file`</b> (bool):  remove the original files 
 - <b>`fout`</b> (str):  prefix of the output files 
 - <b>`n_digits`</b> (int):  number of digits of the point index in the file names 
 - <b>`n_cores`</b> (int):  number of cores 



**Returns:**
 
 - <b>`list`</b>:  written files 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_netcdf`

```python
to_netcdf(fname_fsm_sim, complevel=9)
```

Function to convert a single FSM simulation output file (.txt) to a compressed netcdf file (.nc) 



**Args:**
 
 - <b>`fname_fsm_sim`</b> (str):  filename to convert from txt to nc 
 - <b>`complevel`</b> (int):  Compression level. 1-9 



**Returns:**
 
 - <b>`str`</b>:  name of the netcdf file 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `aggregate_all_open_forest`

```python
aggregate_all_open_forest(
    fname_df_forest='fsm_sim/df_forest.pckl',
    fname_forest_outputs='fsm_sim/fsm__outputs_forest_*.nc',
    fname_open_outputs='fsm_sim/fsm__outputs_open_*.nc',
    fout='fsm_sim/fsm_outputs',
    n_cores=6,
    n_digits=3,
    remove_tmp_outputs=True,
    forest_threshold=0.1
)
```

Combine the forest and open outputs of all points (paired by point index, not by file order). 



**Returns:**
 
 - <b>`list`</b>:  written files 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_pt_fsm2oshd`

```python
read_pt_fsm2oshd(fname)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/sim_fsm2oshd.py#L322"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_pt_met_fsm2oshd`

```python
read_pt_met_fsm2oshd(fname)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
