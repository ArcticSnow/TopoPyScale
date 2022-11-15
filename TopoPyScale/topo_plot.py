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


def map_unclustered(ds_down,
                    ds_param,
                    time_step=1,
                    var='t',
                    ax=None,
                    cmap=plt.cm.RdBu_r,
                    hillshade=True,
                    colorbar=True,
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
    if ax is None:
        fig, ax = plt.subplots(1,1)

    ds_mini = ds_down[var].isel(time=time_step)
    nclust = ds_mini.shape[0]
    lookup = np.arange(nclust, dtype=np.uint16)

    for i in range(0, nclust):
        lookup[i] = ds_mini[i].values
    val = lookup[ds_param.cluster_labels.astype(int).values]

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
    ax.imshow(val,
               cmap=cmap,
               extent=[ds_param.x.min(), ds_param.x.max(), ds_param.y.min(), ds_param.y.max()],
               alpha=alpha, **kwargs)
    if colorbar:
        plt.colorbar()

    return ax

def map_variable(ds_down,
                 ds_param):
    """
    Function to plot unclustered downscaled points given that each point corresponds to a cluster label

    Args:
        ds_down (dataset): Slice of single timestep for a variable of interest.  e.g. mp.downscaled_pts['t'].sel(time='2020-01-01 12:00')
        ds_param (dataset): mapping of the clusters. From topoclass().toposub.ds_param

    Returns:
        Numpy 2D Array of the variable mapped to individual clusters

    """
    arr = ds_down.values
    nclust = ds_down.shape[0]
    lookup = np.arange(nclust)
    mapping = ds_param.cluster_labels.values

    for i in range(0, nclust):
        lookup[i] = ds_down[i]
    val = lookup[mapping]

    return val