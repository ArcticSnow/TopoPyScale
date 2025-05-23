<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.topo_sub`
Clustering routines for TopoSUB 

S. Filhol, Oct 2021 



**TODO:**
 
- explore other clustering methods available in scikit-learn: https://scikit-learn.org/stable/modules/clustering.html 
- look into DBSCAN and its relative 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ds_to_indexed_dataframe`

```python
ds_to_indexed_dataframe(ds)
```

Function to convert dataset to dataframe 

See definition of function in topo_utils.py 

**Args:**
 
 - <b>`ds`</b> (dataset):  xarray dataset N * 2D Dataarray 



**Returns:**
 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `scale_df`

```python
scale_df(
    df_param,
    scaler=StandardScaler(),
    features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1}
)
```

Function to scale features of a pandas dataframe 



**Args:**
 
 - <b>`df_param`</b> (dataframe):  features to scale 
 - <b>`scaler`</b> (scaler object):  Default is StandardScaler() 
 - <b>`features`</b> (dict):  dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1} 



**Returns:**
 
 - <b>`dataframe`</b>:  scaled data 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `inverse_scale_df`

```python
inverse_scale_df(
    df_scaled,
    scaler,
    features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1}
)
```

Function to inverse feature scaling of a pandas dataframe 



**Args:**
 
 - <b>`df_scaled`</b> (dataframe):  scaled data to transform back to original (inverse transfrom) 
 - <b>`scaler`</b> (scaler object):  original scikit learn scaler 
 - <b>`features`</b> (dict):  dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1} 



**Returns:**
 
 - <b>`dataframe`</b>:  data in original format 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `kmeans_clustering`

```python
kmeans_clustering(
    df_param,
    n_clusters=100,
    features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
    seed=None,
    **kwargs
)
```

Function to perform K-mean clustering 



**Args:**
 
 - <b>`df_param`</b> (dataframe):  features 
 - <b>`features`</b> (dict):  dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1} 
 - <b>`n_clusters`</b> (int):  number of clusters 
 - <b>`seed`</b> (int):  None or int for random seed generator 

**kwargs:**
 



**Returns:**
 
 - <b>`dataframe`</b>:  df_centers 
 - <b>`kmean object`</b>:  kmeans 
 - <b>`dataframe`</b>:  df_param 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `minibatch_kmeans_clustering`

```python
minibatch_kmeans_clustering(
    df_param,
    n_clusters=100,
    features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
    n_cores=4,
    seed=None,
    **kwargs
)
```

Function to perform mini-batch K-mean clustering 



**Args:**
 
 - <b>`df_param`</b> (dataframe):  features 
 - <b>`n_clusters`</b> (int):   number of clusters 
 - <b>`features`</b> (dict):  dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1} 
 - <b>`n_cores`</b> (int):  number of processor core 

**kwargs:**
 



**Returns:**
 
 - <b>`dataframe`</b>:  centroids 
 - <b>`kmean object`</b>:  kmean model 
 - <b>`dataframe`</b>:  labels of input data 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `search_number_of_clusters`

```python
search_number_of_clusters(
    df_param,
    method='minibatchkmean',
    cluster_range=array([100, 300, 500, 700, 900]),
    features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
    scaler_type=StandardScaler(),
    scaler=None,
    seed=2,
    plot=True
)
```

Function to help identify an optimum number of clusters using the elbow method 

**Args:**
 
 - <b>`df_param`</b> (dataframe):  pandas dataframe containing input variable to the clustering method 
 - <b>`method`</b> (str):  method for clustering. Currently available: ['minibatchkmean', 'kmeans'] 
 - <b>`range_n_clusters`</b> (array int):  array of number of clusters to derive scores for 
 - <b>`features`</b> (dict):  dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1} 
 - <b>`scaler_type`</b> (scikit_learn obj):  type of scaler to use: e.g. StandardScaler() or RobustScaler() 
 - <b>`scaler`</b> (scikit_learn obj):  fitted scaler to dataset. Implies that df_param is already scaled 
 - <b>`seed`</b> (int):  random seed for kmeans clustering 
 - <b>`plot`</b> (bool):  plot results or not 



**Returns:**
 
 - <b>`dataframe`</b>:  wcss score, Davies Boulding score, Calinsky Harabasz score 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_center_clusters`

```python
plot_center_clusters(
    dem_file,
    ds_param,
    df_centers,
    var='elevation',
    cmap=<matplotlib.colors.ListedColormap object at 0x7f669fe9c8e0>,
    figsize=(14, 10)
)
```

Function to plot the location of the cluster centroids over the DEM 



**Args:**
 
 - <b>`dem_file`</b> (str):  path to dem raster file 
 - <b>`ds_param`</b> (dataset):  topo_param parameters ['elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf'] 
 - <b>`df_centers`</b> (dataframe):  containing cluster centroid parameters ['x', 'y', 'elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf'] 
 - <b>`var`</b> (str):  variable to plot as background 
 - <b>`cmap`</b> (pyplot cmap):  pyplot colormap to represent the variable. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/topo_sub.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `write_landform`

```python
write_landform(
    dem_file,
    df_param,
    project_directory='./',
    out_dir: Optional[str, Path] = None,
    out_name: Optional[str] = None
) → Union[str, Path]
```

Function to write a landform file which maps cluster ids to dem pixels 



**Args:**
 
 - <b>`dem_file`</b> (str):  path to dem raster file 
 - <b>`ds_param`</b> (dataset):  topo_param parameters ['elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf'] 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
