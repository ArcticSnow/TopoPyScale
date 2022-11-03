"""
Collectio of plotting functions for TopoPyScale
S. Filhol, December 2021

"""

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import numpy as np

def map__terrain(ds_param,
                     var='elevation',
                     hillshade=True,
                     ):
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


def map_unclustered(ds_down,
                         ds_param,
                         time_step=1,
                         var='t',
                         cmap=plt.cm.RdBu_r,
                         hillshade=True,
                         **kwargs):
    """
    Function to plot unclustered downscaled points given that each point corresponds to a cluster label

    Args:
        ds_down:
        ds_param:
        time_step:
        var:
        cmap:
        **kwargs: kwargs to pass to imshow() in dict format.

    TODO:
    - if temperature, divergent colorscale around 0degC
    - see if we can remain within xarray dataset/dataarray to reshape and include x,y
    """

    ds_mini = ds_down.isel(time=time_step).copy()
    dc = ds_mini.sel(point_id=ds_param.cluster_labels.values.flatten())
    val = np.reshape(dc[var].values, ds_param.elevation.shape)

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
    plt.imshow(val,
               cmap=cmap,
               extent=[ds_param.x.min(), ds_param.x.max(), ds_param.y.min(), ds_param.y.max()],
               alpha=alpha, **kwargs)
    plt.colorbar()
    plt.show()
