'''
Collectio of plotting functions for TopoPyScale
S. Filhol, December 2021

'''

import matplotlib.pyplot as plt

def plot_unclustered_map(ds_down, ds_param, time_step, var='t', cmap=plt.cm.RdBu_r):
    ds_mini = ds_down.isel(time=time_step).copy()
    dc = ds_mini.sel(point_id=ds_param.cluster_labels.values.flatten())
    val = np.reshape(dc[var].values, ds_param.elevation.shape)

    plt.figure()
    plt.imshow(val)
    plt.colorbar()
    plt.show()
