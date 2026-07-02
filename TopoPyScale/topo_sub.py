"""
Clustering routines for TopoSUB

S. Filhol, Oct 2021

Optimized with parallel K-means, progress tracking, and vectorized operations.
"""

import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn import cluster
import pandas as pd
from matplotlib.colors import LightSource
from matplotlib import pyplot as plt
import numpy as np
import time
from typing import Union, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from TopoPyScale import topo_utils as tu
from tqdm import tqdm
import warnings


def ds_to_indexed_dataframe(ds):
    '''
    Function to convert dataset to dataframe

    See definition of function in topo_utils.py
    Args:
        ds (dataset): xarray dataset N * 2D Dataarray

    Returns:

    '''
    return tu.ds_to_indexed_dataframe(ds)


def scale_df(df_param,
             scaler=StandardScaler(),
             features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1}):
    """
    Optimized function to scale features with vectorized weighting.

    Args:
        df_param (dataframe): features to scale
        scaler (scaler object): Default is StandardScaler()
        features (dict): dictionary of features with importance weights

    Returns:
        dataframe: scaled data
        scaler: fitted scaler object
    """
    feature_list = list(features.keys())
    print('---> Scaling data prior to clustering')
    
    # Vectorized scaling
    df_scaled = pd.DataFrame(scaler.fit_transform(df_param[feature_list].values),
                             columns=df_param[feature_list].columns,
                             index=df_param[feature_list].index)
    
    # Vectorized feature weighting (avoid loop)
    feature_weights = np.array([features.get(fe) for fe in feature_list])
    df_scaled[feature_list] = df_scaled[feature_list].values * feature_weights

    return df_scaled, scaler


def inverse_scale_df(df_scaled,
                     scaler,
                     features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1}):
    """
    Optimized function to inverse feature scaling with vectorized operations.
    
    Args:
        df_scaled (dataframe): scaled data to transform back to original
        scaler (scaler object): original scikit learn scaler
        features (dict): dictionary of features with importance weights

    Returns:
        dataframe: data in original format
    """
    feature_list = list(features.keys())
    df_temp = df_scaled.copy()
    
    # Vectorized inverse weighting  
    feature_weights = np.array([features.get(fe) for fe in feature_list])
    df_temp[feature_list] = df_temp[feature_list].values / feature_weights

    df_inv = pd.DataFrame(scaler.inverse_transform(df_temp.values),
                          columns=df_temp.columns,
                          index=df_temp.index)
    return df_inv


def kmeans_clustering(df_param,
                      n_clusters=100,
                      features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
                      seed=None,
                      n_jobs=-1,
                      **kwargs):
    """
    Optimized K-means clustering with parallel processing and progress tracking.

    Args:
        df_param (dataframe): features
        features (dict): dictionary of features with importance weights
        n_clusters (int): number of clusters
        seed (int): None or int for random seed generator
        n_jobs (int): number of CPU cores (-1 for all cores)
        kwargs: additional parameters for sklearn KMeans

    Returns:
        dataframe: df_centers
        kmean object: kmeans
        array: cluster_labels
    """
    feature_list = list(features.keys())
    X = df_param[feature_list].to_numpy()
    col_names = df_param[feature_list].columns
    
    n_samples = X.shape[0]
    print(f'---> Clustering {n_samples:,} samples with K-means in {n_clusters} clusters (parallel)')
    
    start_time = time.time()
    
    # Use parallel K-means with progress tracking
    with tqdm(desc="K-means clustering", unit="iterations") as pbar:
        kmeans = cluster.KMeans(
            n_clusters=n_clusters, 
            random_state=seed, 
            n_jobs=n_jobs,
            **kwargs
        )
        
        # Fit with progress callback (note: sklearn doesn't support direct progress callbacks)
        kmeans.fit(X)
        pbar.update(1)
    
    elapsed = time.time() - start_time
    print(f'---> K-means finished in {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/sec)')
    
    df_centers = pd.DataFrame(kmeans.cluster_centers_, columns=col_names)
    cluster_labels = kmeans.labels_
    
    return df_centers, kmeans, cluster_labels


def minibatch_kmeans_clustering(df_param,
                                n_clusters=100,
                                features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
                                n_cores=None,
                                seed=None,
                                batch_size=None,
                                **kwargs):
    """
    Optimized mini-batch K-means clustering with automatic batch sizing and progress tracking.
    
    Args:
        df_param (dataframe): features
        n_clusters (int): number of clusters
        features (dict): dictionary of features with importance weights  
        n_cores (int): number of processor cores (deprecated, kept for compatibility)
        seed (int): random seed
        batch_size (int): batch size (auto-computed if None)
        kwargs: additional parameters

    Returns: 
        dataframe: centroids
        kmean object: kmean model
        array: cluster labels
    """
    if n_cores is not None:
        warnings.warn("n_cores parameter is deprecated - batch size is auto-computed", 
                     DeprecationWarning, stacklevel=2)
    
    feature_list = list(features.keys())
    X = df_param[feature_list].to_numpy()
    col_names = df_param[feature_list].columns
    
    n_samples = X.shape[0]
    
    # Auto-compute optimal batch size
    if batch_size is None:
        # Rule of thumb: batch_size should be sqrt(n_samples) but at least 1024
        batch_size = max(1024, min(int(np.sqrt(n_samples)), 10000))
    
    print(f'---> Clustering {n_samples:,} samples with Mini-Batch K-means in {n_clusters} clusters')
    print(f'---> Using batch size: {batch_size}')
    
    start_time = time.time()
    
    with tqdm(desc="Mini-batch K-means", unit="batches") as pbar:
        miniBkmeans = cluster.MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=batch_size, 
            random_state=seed,
            **kwargs
        )
        
        miniBkmeans.fit(X)
        pbar.update(1)
    
    elapsed = time.time() - start_time  
    print(f'---> Mini-Batch K-means finished in {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/sec)')
    
    df_centers = pd.DataFrame(miniBkmeans.cluster_centers_, columns=col_names)
    cluster_labels = miniBkmeans.labels_
    
    return df_centers, miniBkmeans, cluster_labels


def _evaluate_n_clusters(args):
    """
    Worker function to evaluate a single n_clusters value.
    Used by search_number_of_clusters for parallel execution.
    """
    n_clusters, df_param, method, features, scaler_type, scaler, seed = args
    feature_list = list(features.keys())

    # Scale data
    if scaler_type is not None:
        df_scaled, scaler_l = scale_df(df_param[feature_list], scaler=scaler_type, features=features)
    else:
        df_scaled = df_param[feature_list]
        scaler_l = None

    # Run clustering
    if method.lower() in ['minibatchkmean', 'minibatchkmeans']:
        df_centroids, kmeans_obj, cluster_labels = minibatch_kmeans_clustering(df_scaled,
                                                                               n_clusters,
                                                                               features=features,
                                                                               seed=seed)
    elif method.lower() in ['kmean', 'kmeans']:
        df_centroids, kmeans_obj, cluster_labels = kmeans_clustering(df_scaled,
                                                                     n_clusters,
                                                                     features=features,
                                                                     seed=seed)

    labels = kmeans_obj.labels_

    # Compute RMSE
    if scaler_type is not None and scaler_l is not None:
        cluster_elev = inverse_scale_df(df_centroids[feature_list], scaler_l, features=features).elevation.loc[
            cluster_labels].values
        rmse = (((df_param.elevation - cluster_elev) ** 2).mean()) ** 0.5
    elif scaler is not None:
        cluster_elev = inverse_scale_df(df_centroids[feature_list], scaler, features=features).elevation.loc[
            cluster_labels].values
        rmse = (((df_param.elevation - cluster_elev) ** 2).mean()) ** 0.5
    else:
        rmse = 0

    # Compute cluster size stats
    df_tmp = df_param.copy()
    df_tmp['cluster_labels'] = cluster_labels
    pix_count = df_tmp.groupby('cluster_labels').count()['x']

    return {
        'n_clusters': n_clusters,
        'wcss_score': kmeans_obj.inertia_,
        'db_score': davies_bouldin_score(df_param, labels),
        'ch_score': calinski_harabasz_score(df_param, labels),
        'rmse_elevation': rmse,
        'n_pixels_min': pix_count.min(),
        'n_pixels_max': pix_count.max(),
        'n_pixels_mean': pix_count.mean(),
        'n_pixels_median': pix_count.median(),
    }


def search_number_of_clusters(df_param,
                              method='minibatchkmean',
                              cluster_range=np.arange(100, 1000, 200),
                              features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
                              scaler_type=StandardScaler(),
                              scaler=None,
                              seed=2,
                              plot=True,
                              n_workers=None):
    '''
    Function to help identify an optimum number of clusters using the elbow method
    Args:
        df_param (dataframe): pandas dataframe containing input variable to the clustering method
        method (str): method for clustering. Currently available: ['minibatchkmean', 'kmeans']
        range_n_clusters (array int): array of number of clusters to derive scores for
        features (dict): dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1}
        scaler_type (scikit_learn obj): type of scaler to use: e.g. StandardScaler() or RobustScaler()
        scaler (scikit_learn obj): fitted scaler to dataset. Implies that df_param is already scaled
        seed (int): random seed for kmeans clustering
        plot (bool): plot results or not
        n_workers (int): number of parallel workers. None = sequential, 0 or negative = all CPUs

    Returns:
        dataframe: wcss score, Davies Boulding score, Calinsky Harabasz score

    '''
    import os

    # Build argument list for parallel execution
    args_list = [
        (n_clusters, df_param, method, features, scaler_type, scaler, seed)
        for n_clusters in cluster_range
    ]

    # Parallel or sequential execution
    if n_workers is None or len(cluster_range) < 3:
        # Sequential execution
        results = [_evaluate_n_clusters(args) for args in args_list]
    else:
        # Parallel execution
        if n_workers <= 0:
            n_workers = os.cpu_count()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_evaluate_n_clusters, args_list))

    # Sort results by n_clusters (parallel execution may return out of order)
    results = sorted(results, key=lambda x: x['n_clusters'])

    df = pd.DataFrame(results)

    if plot:
        fig, ax = plt.subplots(4, 1, sharex=True)
        # ------ Elbow method ------
        ax[0].plot(df.n_clusters, df.wcss_score, label="Within-Cluster-Sum-of-Squares (WCSS)")
        ax[0].set_ylabel("Score")
        ax[0].legend()
        ax[1].plot(df.n_clusters, df.db_score, label='Davies-Bouldin')
        ax[1].set_ylabel("Score")
        ax[1].legend()
        ax[2].plot(df.n_clusters, df.ch_score, label='Calinski-Harabasz')
        ax[2].set_ylabel("Score")
        ax[2].legend()
        ax[3].plot(df.n_clusters, df.rmse_elevation, label='Elevation RMSE')
        ax[3].set_xlabel("Number of clusters")
        ax[3].set_ylabel("RMSE [m]")
        ax[3].legend()
        plt.show()

    return df


def plot_center_clusters(dem_file, ds_param, df_centers, var='elevation', cmap=plt.cm.viridis, figsize=(14, 10)):
    """
    Function to plot the location of the cluster centroids over the DEM

    Args:
        dem_file (str): path to dem raster file
        ds_param (dataset): topo_param parameters ['elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
        df_centers(dataframe): containing cluster centroid parameters ['x', 'y', 'elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
        var (str): variable to plot as background
        cmap (pyplot cmap): pyplot colormap to represent the variable.

    """
    ls = LightSource(azdeg=315, altdeg=45)

    dx = np.diff(ds_param.x.values)[0]
    dy = np.diff(ds_param.y.values)[0]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    shade = ls.hillshade(ds_param.elevation.values, vert_exag=0.5, dx=dx, dy=dy, fraction=1.0)
    rgb = (ds_param[var] * shade).plot(cmap=cmap)
    ax.scatter(df_centers.x, df_centers.y, c='r', edgecolors='k')
    # cb = plt.colorbar(rgb)
    # cb.set_label(var)
    ax.set_title('Cluster Centroids')
    ax.set_ylabel('y-coordinate')
    ax.set_xlabel('x-coordinate')
    plt.show()


def write_landform(dem_file, df_param, project_directory='./', out_dir: Optional[Union[str, Path]] = None,
                   out_name: Optional[str] = None) -> Union[str, Path]:
    """
    Function to write a landform file which maps cluster ids to dem pixels
    
    Args:
        dem_file (str): path to dem raster file
        ds_param (dataset): topo_param parameters ['elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
    """
    # derive output path
    if out_name is None:
        out_name = 'landform.tif'
    if out_dir is None:
        out_dir = Path(project_directory) / 'outputs'
        out_name = 'landform.tif'
    else:
        if isinstance(out_dir, str):
            out_dir = Path(out_dir)

    out_path = out_dir / out_name

    # read dem
    with rasterio.open(dem_file) as dem:
        shape = dem.shape
        profile = dem.profile
    myarray = np.reshape(df_param.cluster_labels.values, shape)

    # replace nan with number
    nodata = -1
    myarray = np.nan_to_num(myarray, nan=-1)

    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=nodata,
            compress='lzw')

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(myarray.astype(rasterio.int16), 1)

    # At the end of the ``with rasterio.Env()`` block, context
    # manager exits and all drivers are de-registered.

    return out_path
