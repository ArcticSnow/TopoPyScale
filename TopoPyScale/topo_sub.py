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


def ds_to_indexed_dataframe(ds):
    """
    Function to convert an Xarray dataset with multi-dimensions to indexed dataframe (and not a multilevel indexed dataframe).
    WARNING: this only works if the variable of the dataset have all the same dimensions!

    By default the ds.to_dataframe() returns a multi-index dataframe. Here the coordinates are transfered as columns in the dataframe
    
    Args:
        ds (dataset): xarray dataset with all variable of same number of dimensions
    
    Returns:
        pandas dataframe: 
    """
    df = ds.to_dataframe()
    n_levels = df.index.names.__len__()
    return df.reset_index(level=list(range(0, n_levels)))

def scale_df(df_param,
             scaler=StandardScaler(),
             features=None):
    """
    Function to scale features of a pandas dataframe

    Args:
        df_param (dataframe): features to scale
        scaler (scaler object): Default is StandardScaler()

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
                     features=None):
    """
    Function to inverse feature scaling of a pandas dataframe
    
    Args:
        df_scaled (dataframe): scaled data to transform back to original (inverse transfrom)
        scaler (scaler object): original scikit learn scaler

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
                      feature_list=['x','y', 'elevation', 'slope', 'svf', 'aspect_cos', 'aspect_sin'],
                      n_clusters=100,
                      seed=None,
                      **kwargs):
    """
    Function to perform K-mean clustering

    Args:
        df_param (dataframe): features
        n_clusters (int): number of clusters
        seed (int): None or int for random seed generator
        kwargs:

    Returns:
        dataframe: df_centers
        kmean object: kmeans
        dataframe: df_param

    """
    X = df_param[feature_list].to_numpy()
    col_names = df_param[feature_list].columns
    print('---> Clustering with K-means in {} clusters'.format(n_clusters))
    start_time = time.time()
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed, **kwargs).fit(X)
    print('---> Kmean finished in {}s'.format(np.round(time.time()-start_time), 0))
    df_centers = pd.DataFrame(kmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = kmeans.labels_
    return df_centers, kmeans, df_param


def minibatch_kmeans_clustering(df_param, n_clusters=100, n_cores=4, seed=None, **kwargs):
    """
    Function to perform mini-batch K-mean clustering
    
    Args:
        df_param (dataframe): features
        n_clusters (int):  number of clusters
        n_cores (int): number of processor core
        kwargs:

    Returns: 
        dataframe: centroids
        kmean object: kmean model
        dataframe: labels of input data
    """
    X = df_param.to_numpy()
    col_names = df_param.columns
    print('---> Clustering with Mini-Batch K-means in {} clusters'.format(n_clusters))
    start_time = time.time()
    miniBkmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=256*n_cores, random_state=seed, **kwargs).fit(X)
    print('---> Mini-Batch Kmean finished in {}s'.format(np.round(time.time()-start_time), 0))
    df_centers = pd.DataFrame(miniBkmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = miniBkmeans.labels_
    return df_centers, miniBkmeans, df_param['cluster_labels']


def search_number_of_clusters(df_param,
                            method='minibatchkmean',
                            cluster_range=np.arange(100, 1000, 200),
                            plot=True):
    '''
    Function to help identify an optimum number of clusters using the elbow method
    Args:
        df_param (dataframe): pandas dataframe containing input variable to the clustering method
        method (str): method for clustering. Currently available: ['minibatchkmean', 'kmeans']
        range_n_clusters (array int): array of number of clusters to derive scores for
        plot (bool): plot results or not

    Returns:
        dataframe: wcss score, Davies Boulding score, Calinsky Harabasz score

    '''
    wcss = [] # Define a list to hold the Within-Cluster-Sum-of-Squares (WCSS)
    db_scores = []
    ch_scores = []
    rmse_elevation = []
    n_pixels_median = []
    n_pixels_min = []
    n_pixels_max = []
    n_pixels_mean = []

    for n_clusters in cluster_range:
        if 'cluster_labels' in df_param.columns:
            df_param = df_param.drop('cluster_labels', axis=1)

        if method == 'minibatchkmean':
            df_centroids, kmeans_obj, df_param['cluster_labels'] = minibatch_kmeans_clustering(
                            df_param,
                            n_clusters,
                            seed=2)
        elif method == 'kmean':
            df_centroids, kmeans_obj, df_param['cluster_labels'] = kmeans_clustering(
                            df_param,
                            n_clusters,
                            seed=2)

        labels = kmeans_obj.labels_
        cluster_elev = ts.inverse_scale_df(df_centroids, scaler).elevation.loc[df_param.cluster_labels].values
        rmse = (((df_param.elevation - cluster_elev)**2).mean())**0.5

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

    df = pd.DataFrame({'n_clusters':cluster_range,
                       'wcss_score': wcss,
                       'db_score':db_scores,
                       'ch_score':ch_scores,
                       'rmse_elevation':rmse_elevation,
                       'n_pixels_min':n_pixels_min,
                       'n_pixels_median':n_pixels_median,
                       'n_pixels_mean':n_pixels_mean,
                       'n_pixels_max':n_pixels_max})

    if plot:
        fig, ax = plt.subplots(4,1,sharex=True)
        #------ Elbow method ------
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



def plot_center_clusters(dem_file, ds_param, df_centers, var='elevation', cmap=plt.cm.viridis, figsize=(14,10)):
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
    #cb = plt.colorbar(rgb)
    #cb.set_label(var)
    ax.set_title('Cluster Centroids')
    ax.set_ylabel('y-coordinate')
    ax.set_xlabel('x-coordinate')
    plt.show()


def plot_pca_clusters(dem_file, df_param, df_centroids, scaler, n_components, subsample=3):
    pca = PCA(n_components)
    col_names = ['x', 'y', 'elevation', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
    dfs = scale_df(df_param[col_names], scaler=scaler)[0]
    param_pca = pca.fit_transform(dfs[col_names])

    miniBkmeans = cluster.MiniBatchKMeans(n_clusters=50, batch_size=256*4).fit(param_pca)
    centroid = miniBkmeans.cluster_centers_
    print(centroid.shape)
    df_param['cluster_labels'] = miniBkmeans.labels_
    #centroid = pca.transform(scale_df(df_centroids[['x', 'y', 'elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']], scaler=scaler)[0])

    thinner = np.arange(0, df_param.shape[0], subsample)
    fig, ax = plt.subplots(2,1)
    ax[0].scatter(param_pca[thinner, 0], param_pca[thinner, 1], c=df_param.cluster_labels.iloc[thinner], cmap=plt.cm.hsv, alpha=0.3)
    ax[0].scatter(centroid[:, 0], centroid[:, 1], c=np.arange(0,50,1), edgecolor='k', s=100, cmap=plt.cm.hsv)
    ax[0].set_xlabel('PCA-1')
    ax[0].set_ylabel('PCA-2')

    with rasterio.open(dem_file) as dem:
        shape = dem.shape
        extent = plot.plotting_extent(dem)


    ax[1].imshow(np.reshape(df_param.cluster_labels.values, shape), extent=extent, cmap=plt.cm.hsv)
    plt.show()

def write_landform(dem_file, df_param, project_directory='./'):
    """
    Function to write a landform file which maps cluster ids to dem pixels
    
    Args:
        dem_file (str): path to dem raster file
        ds_param (dataset): topo_param parameters ['elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
    """
    with rasterio.open(dem_file) as dem:
        shape = dem.shape
        profile = dem.profile
    myarray = np.reshape(df_param.cluster_labels.values, shape)

    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata= -999,
            compress='lzw')

        with rasterio.open(project_directory + 'outputs/landform.tif', 'w', **profile) as dst:
            dst.write(myarray.astype(rasterio.int16), 1)

    # At the end of the ``with rasterio.Env()`` block, context
    # manager exits and all drivers are de-registered.