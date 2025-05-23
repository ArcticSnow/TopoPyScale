<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_da`
Data assimilation methods including ensemble simulation, satelite data retrival and processing and data assimilation algorithms J. Fiddes, March 2022 



**TODO:**
 
    - method to mosaic (gdal) multiple tiles before cropping 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lognormDraws_kris`

```python
lognormDraws_kris(n, med, s)
```

Function to generate n random draws from a log normal distribution 



**Args:**
 
 - <b>`n`</b>:  integer number of draws to make 
 - <b>`med`</b>:  median of distribrition 
 - <b>`s`</b>:  standard deviation of distribution 



**Returns:**
 Vector of draws 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `normDraws`

```python
normDraws(n, m, s)
```

Function to generate n random draws from a normal distribution 



**Args:**
 
 - <b>`n`</b>:  integer number of draws to make 
 - <b>`m`</b>:  mean of distribrition 
 - <b>`s`</b>:  standard deviation of distribution 



**Returns:**
 Vector of draws 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ensemble_pars_gen`

```python
ensemble_pars_gen(n)
```

Function to generate perturbabion parameters to create an ensemble of meteorological forcings 



**Args:**
 
 - <b>`n`</b>:  size of the ensemble (integer) 



**Returns:**
 array of perturbation factors of dimension n 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ensemble_meteo_gen`

```python
ensemble_meteo_gen(ds, perturb, N_ens, ensemb_type)
```

Function to perturb downscaled meteorological timeseries 



**Args:**
 
 - <b>`ds`</b>:  downscaled_pts 
 - <b>`perturb`</b>:  matrix of perturbation parameters 
 - <b>`N_ens`</b>:  ensemble number (integer) 
 - <b>`ensemb_type`</b>:  string describing which meteo variables to perturb: "T" = Tair, "P" = Precip, "S" = Swin, "L" = Lwin 



**Returns:**
 ds: 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `construct_HX`

```python
construct_HX(wdir, startDA, endDA, ncol)
```

Function to generate HX matrix of ensembleSamples x day 



**Args:**
 
 - <b>`wdir`</b>:  working directory 
 - <b>`DAyear`</b>:  year of data assimilation yyyy 
 - <b>`startDA`</b>:  start of data assimilation window  yyyy-mm-dd 
 - <b>`endDA`</b>:  end of data assimilation window yyyy-mm-dd 



**Returns:**
 
 - <b>`ensembleRes`</b>:  2D pandas dataframe with dims cols (samples * ensembles) and rows (days / simulation timesteps) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `modis_tile`

```python
modis_tile(latmax, latmin, lonmax, lonmin)
```

Function to retrieve MODIS tile indexes based on a lon/lat bbox 

DEPRECATED - DEPRECATED! Does not conside rectangle v rhombus spatial footprint intersection. 

use get_modis_tile_list 



**Args:**
 





**Returns:**
 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_modis_wrapper`

```python
get_modis_wrapper(vertmax, horizmax, vertmin, horizmin, startDate, endDate)
```

Function to iterate through several MODIS tile downloads. 



**Args:**
 





**Returns:**
 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_modis_tile_list`

```python
get_modis_tile_list(bbox_n, bbox_s, bbox_e, bbox_w)
```

Retrieves a list of MODIS tiles that intersect with a given bounding box. 



**Parameters:**
 
- bbox_n (float): North latitude of the bounding box. 
- bbox_s (float): South latitude of the bounding box. 
- bbox_e (float): East longitude of the bounding box. 
- bbox_w (float): West longitude of the bounding box. 



**Returns:**
 
- list: List of MODIS tiles (e.g., ['h23v04', 'h24v05']) that intersect with the bounding box. 

Usage:   tiles_to_get = get_modis_tile_list( 42.305,41.223, 69.950, 71.981)    


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pymodis_download`

```python
pymodis_download(wdir, tile, STARTDATE, ENDDATE)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L544"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `projFromLandform`

```python
projFromLandform(pathtolandform)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `process_modis`

```python
process_modis(wdir, epsg, bbox, layer=0)
```

Function to convert hdf to tiff, extract layer, and reproject 



**Args:**
 
 - <b>`modpath`</b>:  path to 'modis' directory 
 - <b>`layer`</b>:  layer to extract from hdf 
 - <b>`epsg`</b>:  reprojected target epsg code 
 - <b>`bbox`</b>:  output bounds as list [minX, minY, maxX, maxY] in target 



**Returns:**
 Spatially subsetted reprojected tiffs of data layer specified (still NDSI) 




 - <b>`Docs`</b>:  https://nsidc.org/sites/nsidc.org/files/technical-references/C6.1_MODIS_Snow_User_Guide.pdf 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L603"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `process_modis2`

```python
process_modis2(modpath, epsg, bbox, layer=0)
```

Function to convert hdf to tiff, extract layer, and reproject 



**Args:**
 
 - <b>`modpath`</b>:  path to 'modis' directory 
 - <b>`layer`</b>:  layer to extract from hdf 
 - <b>`epsg`</b>:  reprojected target epsg code 
 - <b>`bbox`</b>:  output bounds as list [minX, minY, maxX, maxY] in target 



**Returns:**
 Spatially subsetted reprojected tiffs of data layer specified (still NDSI) 




 - <b>`Docs`</b>:  https://nsidc.org/sites/nsidc.org/files/technical-references/C6.1_MODIS_Snow_User_Guide.pdf 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L651"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_fsca_timeseries`

```python
extract_fsca_timeseries(wdir, plot=True)
```

Function to extract mean NDSI from domain and converts to fSCA using the empirical formula based on landsat https://modis-snow-ice.gsfc.nasa.gov/uploads/C6_MODIS_Snow_User_Guide.pdf pp.22 

**Args:**
 
 - <b>`wdir`</b>:  working directory 
 - <b>`plot`</b>:  booleen whether to plot dataframe 



**Returns:**
 
 - <b>`df_sort`</b>:  dataframe of julian dates and fSCA values 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L823"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `particle_batch_smoother`

```python
particle_batch_smoother(Y, HX, R)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L847"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `EnKA`

```python
EnKA(prior, obs, pred, alpha, R)
```

EnKA: Implmentation of the Ensemble Kalman Analysis Inputs:  prior: Prior ensemble matrix (n x N array)  obs: Observation vector (m x 1 array)  pred: Predicted observation ensemble matrix (m x N array)  alpha: Observation error inflation coefficient (scalar)  R: Observation error covariance matrix (m x m array, m x 1 array, or scalar) Outputs:  post: Posterior ensemble matrix (n x N array) Dimensions:  N is the number of ensemble members, n is the number of state variables and/or  parameters, and m is the number of boservations. 

For now, we have impelemnted the classic (stochastic) version of the Ensemble Kalman analysis that involves perturbing the observations. Deterministic variants (Sakov and Oke, 2008) also exist that may be worth looking into. This analysis step can be used both for filtering, i.e. Ensemble Kalman filter, when observations are assimilated sequentially or smoothing, i.e. Ensemble (Kalman) smoother, when also be carried out in a multiple data assimilation (i.e. iterative) approach. 

Since the author rusty in python it is possible that this code can be improved considerably. Below are also some examples of further developments that may be worth looking into: 

To do/try list: (i) Using a Bessel correction (Ne-1 normalization) for sample covariances. (ii) Perturbing the predicted observations rather than the observations  following van Leeuwen (2020, doi: 10.1002/qj.3819) which is the formally  when defining the covariances matrices. (iii) Looking into Sect. 4.5.3. in Evensen (2019, doi: 10.1007/s10596-019-9819-z) (iv) Using a deterministic rather than stochastic analysis step. (v) Augmenting with artificial momentum to get some inertia. (vi) Better understand/build on the underlying tempering/annealing idea in ES-MDA. (vii) Versions that may scale better for large n such as those working in the  ensemble subspace (see e.g. formulations in Evensen, 2003, Ocean Dynamics). 

Code by: K. Aalstad (10.12.2020) based on an earlier Matlab version. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L947"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `PBS`

```python
PBS(obs, pred, R)
```

PBS: Implmentation of the Particle Batch Smoother Inputs:  obs: Observation vector (m x 1 array)  pred: Predicted observation ensemble matrix (m x N array)  R: Observation error covariance 'matrix' (m x 1 array, or scalar) Outputs:  w: Posterior weights (N x 1 array) Dimensions:  N is the number of ensemble members and m is the number of boservations. 

Here we have implemented the particle batch smoother, which is a batch-smoother version of the particle filter (i.e. a particle filter without resampling), described in Margulis et al. (2015, doi: 10.1175/JHM-D-14-0177.1). As such, this routine can also be used for particle filtering with sequential data assimilation. This scheme is obtained by using a particle (mixture of Dirac delta functions) representation of the prior and applying this directly to Bayes theorem. In other words, it is just an application of importance sampling with the prior as the proposal density (importance distribution). It is also the same as the Generalized Likelihood Uncertainty Estimation (GLUE) technique (with a formal Gaussian likelihood) which is widely used in hydrology. 

This particular scheme assumes that the observation errors are additive Gaussian white noise (uncorrelated in time and space). The "logsumexp" trick is used to deal with potential numerical issues with floating point operations that occur when dealing with ratios of likelihoods that vary by many orders of magnitude. 

To do/try list:  (i) Implement an option for correlated observation errors if R is specified  as a matrix (this would slow down this function).  (ii) Consier other likelihoods and observation models.  (iii) Look into iterative versions of importance sampling. 



Code by: K. Aalstad (14.12.2020) based on an earlier Matlab version. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L1021"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `da_plots`

```python
da_plots(HX1, HX2, noDA, W, mydates, myobs)
```

Function to extract mean fSSCA from domain 



**Args:**
  HX1:  HX2: 
 - <b>`weights`</b>:  weights output from PBS (dims 1 x NENS) 
 - <b>`dates`</b>:  datetime object 
 - <b>`myobs`</b>:  timeseries of observtions 



**Returns:**
 plot 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L1099"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fsca_plots`

```python
fsca_plots(wdir, plotDay, df_mean)
```

Function to plot side by side observed fSCA 500m [0-1] and modelled SCA 30 m [0,1] 



**Args:**
 
 - <b>`wdir`</b>:  sim directory 
 - <b>`plotDay`</b>:  date to plot as YYYY-mm-dd 
 - <b>`df_mean`</b>:  timeseries mean object given by sim.timeseries_means_period() 



**Returns:**
 plot 

Usage: df = sim.agg_by_var_fsm(6) df2 = sim.agg_by_var_fsm_ensemble(6, W) df_mean = sim.timeseries_means_period(df, "2018-06-01", "2018-06-01") da.fsca_plots("/home/joel/sim/topoPyscale_davos/", "2018150", df_mean) 



**TODO:**
 
    - create modelled fSCA option from agreggated 30m to 500m pixels 
    - accept datetime object instead of julday. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L1224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `da_compare_plot`

```python
da_compare_plot(wdir, plotDay, df_mean_open, df_mean_da)
```

Function to plot side by side observed fSCA 500m [0-1] and modelled SCA 30 m [0,1] 



**Args:**
 
 - <b>`wdir`</b>:  sim directory 
 - <b>`plotDay`</b>:  date to plot as YYYY-mm-dd 
 - <b>`df_mean`</b>:  timeseries mean object given by sim.timeseries_means_period() 



**Returns:**
 plot 

Usage: df = sim.agg_by_var_fsm(6) df2 = sim.agg_by_var_fsm_ensemble(6, W) df_mean = sim.timeseries_means_period(df, "2018-06-01", "2018-06-01") da.fsca_plots("/home/joel/sim/topoPyscale_davos/", "2018150", df_mean) 



**TODO:**
 
    - create modelled fSCA option from agreggated 30m to 500m pixels 
    - accept datetime object instead of julday. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L1331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_modis_cube`

```python
build_modis_cube(wdir)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_da.py#L1365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `getModisbbox`

```python
getModisbbox(Npixel, arr)
```

Function to find obtain bbox of modis pixel CRS is projected UTM 



**Args:**
 
 - <b>`Npixel`</b>:  pixel nuber as integer of 0 - (xdim x ydim) (the flattened grid) eg for a 10 x 10 grid Npixel would have a range of 0-100 
 - <b>`arr`</b>:  xarray DataArray with dimensions "x" and "y" and attribute "res" 



**Returns:**
 
 - <b>`bbox`</b>:  bbox of pixel as list [xmax,xmin, ymax, ymin] 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
