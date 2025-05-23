# `Topo_sub` Clustering Methods 

## Feature Importance

Clusters are derived using a set of features which are by default: `[x,y,elevation, slope, aspect_cos, aspect_sin, svf]`. Additional features may be added too. The clustering methods in `topo_sub` accepts any features that are indicated in the `dict` of the config file. Values for the features must be available in the variable `ds_param`. The value in `clustering_features` is a multiplier to scale the feature importance relative to the other. This may be useful for instance to emphasize the role of elevation in case of sharp terrain. Here the elevation as a factor of 4 relative to other features. 

```python
clustering_features: {'x':1, 'y':1, 'elevation':4, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1}
```

If you add new features specific to your project, be aware that all features are standardize using the `scikit-learn` function `StandardScaler()`. Other types of scaler not yet available.

## Optimize Number of Clusters
`Topo_sub` includes a method to search for the optimal number of clusters: `search_optimum_number_of_clusters()`. This may be used in the following manner:

```python
from TopoPyScale import topoclass as tc
import numpy as np

config_file = './config.yml'
mp = tc.Topoclass(config_file)
mp.compute_dem_param()
df = mp.search_optimum_number_of_clusters(cluster_range=np.arange(100,1000,50),plot=False)
```

This will compute the scores Within-Cluster-Sum-of-Squares (WCSS) Davies-Bouldin, Calinski-Harabasz, and the RMSE between the original DEM elevation and the clustered DEM elevation. Any of these may be used to decide of an appropriate number of clusters