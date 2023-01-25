"""
Collectio of plotting functions for TopoPyScale
S. Filhol, December 2021

"""

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np

def map_terrain(ds_param,
                var='elevation',
                hillshade=True,
                cmap=plt.cm.viridis,
                **kwargs):
    '''
    Function to plot terrain parameters

    Args:
        ds_param:
        var:
        hillshade:

    Returns:

    '''
    plt.figure()
    alpha=1
    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        shade = ls.hillshade(ds_param.elevation.values, vert_exag=0.5,
                             dx=ds_param.x.diff('x')[0].values,
                             dy=ds_param.y.diff('y')[0].values,
                             fraction=1.0)


        plt.imshow(shade,
                   extent=[ds_param.x.min(), ds_param.x.max(), ds_param.y.min(), ds_param.y.max()],
                   cmap=plt.cm.gray)
        alpha=0.5
    plt.imshow(ds_param[var],
               cmap=cmap,
               extent=[ds_param.x.min(), ds_param.x.max(), ds_param.y.min(), ds_param.y.max()],
               alpha=alpha, **kwargs)
    plt.colorbar(label=var)
    plt.show()


def map_variable(ds_down,
                 ds_param,
                 time_step=1,
                 time=None,
                 var='t',
                 ax=None,
                 cmap=plt.cm.RdBu_r,
                 hillshade=True,
                 **kwargs):
    """
    Function to plot unclustered downscaled points given that each point corresponds to a cluster label

    Args:
        ds_down (dataset): TopoPyScale downscaled point object with coordinates (time, point_id)
        ds_param (dataset): TopoPyScale toposub dataset with coordinates (x,y)
        time_step (int): (optional) time step to plot.
        time (str): (optional) time slice to plot. time overule time_step
        var (str): variable from ds_down to plot. e.g. 't' for temperature
        cmap (obj): pyplot colormap object
        hillshade (bool): add terrain hillshade in background
        **kwargs: kwargs to pass to imshow() in dict format.

    TODO:
    - if temperature, divergent colorscale around 0degC
    """
    if ax is None:
        fig, ax = plt.subplots(1,1)

    if time is None:
        time = ds_down.time.isel(time=time_step).data

    if var=='t':
        print('Suggestion: use divergent colormap plt.cm.RdBl_r centered around 0C')

    alpha=1
    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        shade = ls.hillshade(ds_param.elevation.values, vert_exag=0.5,
                             dx=ds_param.x.diff('x')[0].values,
                             dy=ds_param.y.diff('y')[0].values,
                             fraction=1.0)
        ax.imshow(shade,
                   extent=[ds_param.x.min(), ds_param.x.max(), ds_param.y.min(), ds_param.y.max()],
                   cmap=plt.cm.gray)
        alpha=0.5

    ds_down[var].sel(point_id=ds_param.cluster_labels).sel(time=time).plot.imshow(alpha=alpha, cmap=cmap, **kwargs)

    return ax

def map_clusters(ds_down,
                 ds_param,
                 df_centroids=None,
                 **kwargs):
    ds_down.point_id.sel(point_id=ds_param.cluster_labels).plot.imshow(**kwargs)
    if df_centroids is not None:
        plt.scatter(df_centroids.x, df_centroids.y, c='k', s=2)
