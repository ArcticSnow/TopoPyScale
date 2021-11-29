'''
Clustering routines for TopoSUB
S. Filhol, Oct 2021

TODO:
- explore other clustering methods available in scikit-learn: https://scikit-learn.org/stable/modules/clustering.html
- look into DBSCAN and its relative
'''

import rasterio
from rasterio import plot
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.colors import LightSource
from matplotlib import pyplot as plt
import numpy as np
import time

def ds_to_indexed_dataframe(ds):
    '''
    Function to convert an Xarray dataset with multi-dimensions to indexed dataframe (and not a multilevel indexed dataframe).
    WARNING: this only works if the variable of the dataset have all the same dimensions!

    By default the ds.to_dataframe() returns a multi-index dataframe. Here the coordinates are transfered as columns in the dataframe
    :param ds: xarray dataset with all variable of same number of dimensions
    :return: pandas dataframe
    '''
    df = ds.to_dataframe()
    n_levels = df.index.names.__len__()
    return df.reset_index(level=list(range(0, n_levels)))

def scale_df(df_param, scaler=StandardScaler()):
    '''
    Function to scale features of a pandas dataframe

    :param df_param: pandas dataframe with features to scale
    :param scaler: scikit learn scaler. Default is StandardScaler()
    :return: scaled pandas dataframe
    '''
    print('---> Scaling data prior to clustering')
    df_scaled = pd.DataFrame(scaler.fit_transform(df_param.values),
                             columns=df_param.columns, index=df_param.index)
    return df_scaled, scaler

def inverse_scale_df(df_scaled, scaler):
    '''
    Function to inverse feature scaling of a pandas dataframe

    :param df_scaled: pandas dataframe to rescale to original (inverse transfrom
    :param scaler: original scikit learn scaler
    :return:
    '''
    df_inv = pd.DataFrame(scaler.inverse_transform(df_scaled.values),
                          columns=df_scaled.columns, index=df_scaled.index)
    return df_inv


def kmeans_clustering(df_param, n_clusters=100, seed=None, **kwargs):
    '''
    Function to perform K-mean clustering

    :param df_param: pandas dataframe with features
    :param n_clusters: number of clusters
    :param seed: None or int for random seed generator
    :param kwargs:
    :return:
    '''
    X = df_param.to_numpy()
    col_names = df_param.columns
    print('---> Clustering with K-means in {} clusters'.format(n_clusters))
    start_time = time.time()
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed, **kwargs).fit(X)
    print('---> Kmean finished in {}s'.format(np.round(time.time()-start_time), 0))
    df_centers = pd.DataFrame(kmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = kmeans.labels_
    return df_centers, kmeans, df_param


def minibatch_kmeans_clustering(df_param, n_clusters=100, n_cores=4, seed=None, **kwargs):
    '''
    Function to perform mini-batch K-mean clustering

    :param df_param: pandas dataframe with features
    :param n_clusters: (int) number of clusters
    :param n_cores: (int) number of processor core
    :param kwargs:
    :return: centroids, kmean-object, and labels of input data
    '''
    X = df_param.to_numpy()
    col_names = df_param.columns
    print('---> Clustering with Mini-Batch K-means in {} clusters'.format(n_clusters))
    start_time = time.time()
    miniBkmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=256*n_cores, random_state=seed, **kwargs).fit(X)
    print('---> Mini-Batch Kmean finished in {}s'.format(np.round(time.time()-start_time), 0))
    df_centers = pd.DataFrame(miniBkmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = miniBkmeans.labels_
    return df_centers, miniBkmeans, df_param['cluster_labels']


def plot_center_clusters(dem_file, ds_param, df_centers, var='elevation', cmap=plt.cm.viridis, figsize=(14,10)):
    '''
    Function to plot the location of the cluster centroids over the DEM

    :param dem_file: path to dem raster file
    :param ds_param: xarray dataset containing topo_param parameters ['elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
    :param df_centers: dataframe containing cluster centroid parameters ['x', 'y', 'elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
    :param var: variable to plot as background
    :param cmap: pyplot colormap to represent the variable.
    :return:
    '''
    ls = LightSource(azdeg=315, altdeg=45)

    dx = np.diff(ds_param.x.values)[0]
    dy = np.diff(ds_param.y.values)[0]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    shade = ls.hillshade(ds_param.elevation.values, vert_exag=0.5, dx=dx, dy=dy, fraction=1.0)
    rgb = (ds_param[var] * shade).plot(cmap=cmap)
    ax.scatter(df_centers.x, df_centers.y, c='r', edgecolors='k')
    cb = plt.colorbar(rgb)
    cb.set_label(var)
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

