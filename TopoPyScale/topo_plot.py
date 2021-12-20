'''
Collectio of plotting functions for TopoPyScale
S. Filhol, December 2021

'''

import matplotlib.pyplot as plt

def plot_unclustered_map(ds_down, ds_param, time_step=1, var='t', cmap=plt.cm.RdBu_r):
    '''
    Function to plot unclustered downscaled points given that each point corresponds to a cluster label
    :param ds_down:
    :param ds_param:
    :param time_step:
    :param var:
    :param cmap:
    :return:

    TODO:
    - if temperature, divergent colorscale around 0degC
    - see if we can remain within xarray dataset/dataarray to reshape and include x,y
    '''

    ds_mini = ds_down.isel(time=time_step).copy()
    dc = ds_mini.sel(point_id=ds_param.cluster_labels.values.flatten())
    val = np.reshape(dc[var].values, ds_param.elevation.shape)

    plt.figure()
    plt.imshow(val)
    plt.colorbar()
    plt.show()