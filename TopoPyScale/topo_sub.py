"""
Clustering routines for TopoSUB

S. Filhol, Oct 2021

TODO:
- explore other clustering methods available in scikit-learn: https://scikit-learn.org/stable/modules/clustering.html
- look into DBSCAN and its relative
"""

import rasterio
from rasterio import plot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn import cluster
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.colors import LightSource
from matplotlib import pyplot as plt
import numpy as np
import time
from typing import Union, Optional
from pathlib import Path
from TopoPyScale import topo_utils as tu


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
    Function to scale features of a pandas dataframe

    Args:
        df_param (dataframe): features to scale
        scaler (scaler object): Default is StandardScaler()
        features (dict): dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1}

    Returns:
        dataframe: scaled data
    """
    feature_list = features.keys()
    print('---> Scaling data prior to clustering')
    df_scaled = pd.DataFrame(scaler.fit_transform(df_param[feature_list].values),
                             columns=df_param[feature_list].columns,
                             index=df_param[feature_list].index)
    for fe in feature_list:
        df_scaled[fe] *= features.get(fe)

    return df_scaled, scaler


def inverse_scale_df(df_scaled,
                     scaler,
                     features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1}):
    """
    Function to inverse feature scaling of a pandas dataframe
    
    Args:
        df_scaled (dataframe): scaled data to transform back to original (inverse transfrom)
        scaler (scaler object): original scikit learn scaler
        features (dict): dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1}

    Returns:
        dataframe: data in original format
    """
    feature_list = features.keys()
    for fe in feature_list:
        df_scaled[fe] /= features.get(fe)

    df_inv = pd.DataFrame(scaler.inverse_transform(df_scaled.values),
                          columns=df_scaled.columns,
                          index=df_scaled.index)
    return df_inv


def kmeans_clustering(df_param,
                      n_clusters=100,
                      features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
                      seed=None,
                      **kwargs):
    """
    Function to perform K-mean clustering

    Args:
        df_param (dataframe): features
        features (dict): dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1}
        n_clusters (int): number of clusters
        seed (int): None or int for random seed generator
        kwargs:

    Returns:
        dataframe: df_centers
        kmean object: kmeans
        dataframe: df_param

    """
    feature_list = features.keys()
    X = df_param[feature_list].to_numpy()
    col_names = df_param[feature_list].columns
    print('---> Clustering with K-means in {} clusters'.format(n_clusters))
    start_time = time.time()
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed, **kwargs).fit(X)
    print('---> Kmean finished in {}s'.format(np.round(time.time() - start_time), 0))
    df_centers = pd.DataFrame(kmeans.cluster_centers_, columns=col_names)
    cluster_labels = kmeans.labels_
    return df_centers, kmeans, cluster_labels


def minibatch_kmeans_clustering(df_param,
                                n_clusters=100,
                                features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
                                n_cores=4,
                                seed=None,
                                **kwargs):
    """
    Function to perform mini-batch K-mean clustering
    
    Args:
        df_param (dataframe): features
        n_clusters (int):  number of clusters
        features (dict): dictionnary of features to use as predictors with their respect importance. {'x':1, 'y':1}
        n_cores (int): number of processor core
        kwargs:

    Returns: 
        dataframe: centroids
        kmean object: kmean model
        dataframe: labels of input data
    """
    feature_list = features.keys()
    X = df_param[feature_list].to_numpy()
    col_names = df_param.columns
    print('---> Clustering with Mini-Batch K-means in {} clusters'.format(n_clusters))
    start_time = time.time()
    miniBkmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=256 * n_cores, random_state=seed,
                                          **kwargs).fit(X)
    print('---> Mini-Batch Kmean finished in {}s'.format(np.round(time.time() - start_time), 0))
    df_centers = pd.DataFrame(miniBkmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = miniBkmeans.labels_
    return df_centers, miniBkmeans, df_param['cluster_labels']


def search_number_of_clusters(df_param,
                              method='minibatchkmean',
                              cluster_range=np.arange(100, 1000, 200),
                              features={'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1},
                              scaler_type=StandardScaler(),
                              scaler=None,
                              seed=2,
                              plot=True):
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

    Returns:
        dataframe: wcss score, Davies Boulding score, Calinsky Harabasz score

    '''
    feature_list = features.keys()
    wcss = []  # Define a list to hold the Within-Cluster-Sum-of-Squares (WCSS)
    db_scores = []
    ch_scores = []
    rmse_elevation = []
    n_pixels_median = []
    n_pixels_min = []
    n_pixels_max = []
    n_pixels_mean = []

    for n_clusters in cluster_range:
        if scaler_type is not None:
            df_scaled, scaler_l = scale_df(df_param[feature_list], scaler=scaler_type, features=features)
        else:
            df_scaled = df_param[feature_list]
        
        if method.lower() in ['minibatchkmean', 'minibatchkmeans']:
            df_centroids, kmeans_obj, df_param['cluster_labels'] = minibatch_kmeans_clustering(df_scaled,
                                                                                               n_clusters,
                                                                                               features=features,
                                                                                               seed=seed)
        elif method.lower() in ['kmean', 'kmeans']:
            df_centroids, kmeans_obj, df_param['cluster_labels'] = kmeans_clustering(df_scaled,
                                                                                     n_clusters,
                                                                                     features=features,
                                                                                     seed=seed)

        labels = kmeans_obj.labels_
        if scaler_type is not None:
            cluster_elev = inverse_scale_df(df_centroids[feature_list], scaler_l, features=features).elevation.loc[
                df_param.cluster_labels].values
            rmse = (((df_param.elevation - cluster_elev) ** 2).mean()) ** 0.5
        elif scaler is not None:
            cluster_elev = inverse_scale_df(df_centroids[feature_list], scaler, features=features).elevation.loc[
                df_param.cluster_labels].values
            rmse = (((df_param.elevation - cluster_elev) ** 2).mean()) ** 0.5
        else:
            rmse = 0

        # compute scores
        wcss.append(kmeans_obj.inertia_)
        db_scores.append(davies_bouldin_score(df_param, labels))
        ch_scores.append(calinski_harabasz_score(df_param, labels))
        rmse_elevation.append(rmse)

        # compute stats on cluster sizes
        pix_count = df_param.groupby('cluster_labels').count()['x']
        n_pixels_min.append(pix_count.min())
        n_pixels_max.append(pix_count.max())
        n_pixels_mean.append(pix_count.mean())
        n_pixels_median.append(pix_count.median())

    df = pd.DataFrame({'n_clusters': cluster_range,
                       'wcss_score': wcss,
                       'db_score': db_scores,
                       'ch_score': ch_scores,
                       'rmse_elevation': rmse_elevation,
                       'n_pixels_min': n_pixels_min,
                       'n_pixels_median': n_pixels_median,
                       'n_pixels_mean': n_pixels_mean,
                       'n_pixels_max': n_pixels_max})

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
