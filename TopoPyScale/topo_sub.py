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
import pandas as pd
from matplotlib.colors import LightSource
from matplotlib import pyplot as plt
import numpy as np
import time

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


def kmeans_clustering(df_param, n_clusters=100, **kwargs):
    '''
    Function to perform K-mean clustering

    :param df_param: pandas dataframe with features
    :param n_clusters: number of clusters
    :param kwargs:
    :return:
    '''
    X = df_param.to_numpy()
    col_names = df_param.columns
    print('---> Clustering with K-means in {} clusters'.format(n_clusters))
    start_time = time.time()
    kmeans = cluster.KMeans(n_clusters=n_clusters, **kwargs).fit(X)
    print('---> Kmean finished in {}s'.format(np.round(time.time()-start_time), 0))
    df_centers = pd.DataFrame(kmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = kmeans.labels_
    return df_centers, kmeans, df_param


def minibatch_kmeans_clustering(df_param, n_clusters=100, n_cores=4, **kwargs):
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
    miniBkmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=256*n_cores, **kwargs).fit(X)
    print('---> Mini-Batch Kmean finished in {}s'.format(np.round(time.time()-start_time), 0))
    df_centers = pd.DataFrame(miniBkmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = miniBkmeans.labels_
    return df_centers, miniBkmeans, df_param['cluster_labels']


def plot_center_clusters(dem_file, df_param, df_centers, var='elev', cmap=plt.cm.viridis):
    '''
    Function to plot the location of the cluster centroids over the DEM

    :param dem_file: path to dem raster file
    :param df_param: dataframe containing topo_param parameters ['x', 'y', 'elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
    :param df_centers: dataframe containing cluster centroid parameters ['x', 'y', 'elev', 'slope', 'aspect_cos', 'aspect_sin', 'svf']
    :param var: variable to plot as background
    :param cmap: pyplot colormap to represent the variable.
    :return:
    '''
    ls = LightSource(azdeg=315, altdeg=45)

    with rasterio.open(dem_file) as dem:
        shape = dem.shape
        extent = plot.plotting_extent(dem)
        dx = dem.transform[0]
        dy = -dem.transform[4]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 9))
    shade = ls.hillshade(np.reshape(df_param['elev'].values, shape), vert_exag=0.5, dx=dx, dy=dy, fraction=1.0)
    rgb = ax.imshow(np.reshape(df_param[var].values, shape)*shade, extent=extent, cmap=cmap)
    ax.scatter(df_centers.x, df_centers.y, c='r', edgecolors='k')
    cb = plt.colorbar(rgb)
    cb.set_label(var)
    ax.set_title('Cluster Centroids')
    ax.set_ylabel('y-coordinate')
    ax.set_xlabel('x-coordinate')
    plt.show()

    def plot_pca_clusters():
        # write here function to reduce input data with PCA, plot first 2 dimension, and color by labels to visualized clusters
        # thin data for plotting
        return