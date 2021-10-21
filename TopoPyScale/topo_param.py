'''
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- write function to get ERA5 x,y,z at each point list
'''

import rasterio
import pandas as pd
import numpy as np
import xarray as xr
from topocalc import gradient
from topocalc import viewf
from topocalc import horizon
import time


def compute_dem_param(dem_file):
    '''
    Function to compute and derive DEM parameters: slope, aspect, sky view factor
    :param dem_file: path to raster file (geotif)
    :return: pandas dataframe containing x, y, elev, slope, aspect, svf
    '''
    print('\n---> Extracting DEM parameters (slope, aspect, svf)')
    with rasterio.open(dem_file) as rf:
        dem_arr = rf.read()
        geot = rf.read_transform()
        bounds = rf.bounds
        if dem.shape.__len__() == 3:
            dem = dem[0, :, :]
        dx, dy = geot[1], -geot[-1]
        x = np.linspace(rf.bounds.left + np.round(geot[1], 3), rf.bounds.right, rf.shape[1])
        y = np.linspace(rf.bounds.bottom + np.round(geot[5], 3), rf.bounds.top, rf.shape[0])

    dem_param = pd.DataFrame()
    Xs, Ys = np.meshgrid(x, y)

    slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
    dem_param['x'] = Xs.flatten()
    dem_param['y'] = Ys.flatten()
    dem_param['elev'] = dem_arr.flatten()
    dem_param['slope'] = slope.flatten()
    dem_param['aspect_cos'] = np.cos(np.deg2rad(aspect.flatten()))
    dem_param['aspect_sin'] = np.sin(np.deg2rad(aspect.flatten()))
    Xs, Ys, slope, aspect = None, None, None, None

    print('---> Computing sky view factor')
    start_time = time.time()
    svf = viewf.viewf(np.double(dem_arr), dx)[0]
    dem_param['svf'] = svf.flatten()
    print('---> Sky-view-factor finished in {}s'.format(np.round(time.time()-start_time), 0))
    dem_arr = None
    return dem_param


def extract_ERA5_xyz(ds, df_centroids):
    return


def compute_horizon(dem_file, azimuth_inc=30):
    '''
    Function to compute horizon angles for
    :param dem_file:
    :param azimuth_inc:
    :return:
    '''

    with rasterio.open(dem_file) as rf:
        dem = rf.read()
        geot = rf.read_transform()
        if dem.shape.__len__() == 3:
            dem = dem[0, :, :]
        dx = geot[1]
        x = np.linspace(rf.bounds.left + np.round(geot[1], 3), rf.bounds.right, rf.shape[1])
        y = np.linspace(rf.bounds.bottom + np.round(geot[5], 3), rf.bounds.top, rf.shape[0])
    azimuth = np.arange(-180, 180, azimuth_inc)
    arr = np.empty((azimuth.shape[0], dem.shape[0], dem.shape[1]))
    for i, azi in enumerate(azimuth):
        arr[i, :, :] = horizon.horizon(azi, dem.astype(float), dx)

    da = xr.DataArray(data=arr,
                      coords={
                          "y": y,
                          "z": x,
                          "azimuth": azimuth
                      },
                      dims=["azimuth", "y", "x"]
                      )

    arr, dem = None, None
    return da
