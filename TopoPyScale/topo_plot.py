'''
Collectio of plotting functions for TopoPyScale
S. Filhol, December 2021

'''

import matplotlib.pyplot as plt

def plot_unclustered_map(ds_down, ds_param, time_step, var='t', cmap=plt.cm.RdBu_r):
    ds_mini = ds_down.isel(time=time_step).copy()
    ds_mini.sel(point_id=ds_paran.cluster_labels.values.flatten())

    plt.figure()
