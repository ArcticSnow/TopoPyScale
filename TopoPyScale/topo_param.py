'''
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- an improvement could be to first copmute horizons, and then SVF to avoid computing horizon twice
'''

import rasterio
import pandas as pd
import numpy as np
import xarray as xr
from topocalc import gradient
from topocalc import viewf
from topocalc import horizon
import time


def extract_pts_param(df_pts, ds_param, method='nearest'):
    '''
    Function to use a list point as input rather than cluster centroids from DEM segmentation (topo_sub.py/self.clustering_dem()).
    :param df_pts: pandas DataFrame
    :param ds_param: xarray dataset of dem parameters
    :return:

    TODO:
    - add option to return xarray dataset from compute_dem_param()
    - use the xarray dataset to sample just like in topo_scale the dataset using weights
    '''
    for i, row in df_pts.iterrows():
        ind_lat = np.abs(ds_param.latitude-row.y).argmin()
        ind_lon = np.abs(ds_param.longitude-row.x).argmin()
        ds_param_pt = ds_param.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])

        Xs, Ys = np.meshgrid(ds_param_pt.longitude.values, ds_param_pt.latitude.values)

        if method == 'nearest':
            print('TBI')
        if method == 'idw':
            print('TBI')
            dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
            idw = 1/(dist**2)
            weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1
            # see line 103 of topo_scale.py for weights
        if method == 'linear':
            print('TBI')
            dist = np.array([np.abs(row.x - Xs).values, np.abs(row.y - Ys).values])
            weights = dist / np.sum(dist, axis=0)
            # see line 103 of topo_scale.py for weights




# sample methods: nearest, inverse



#return self.toposub.df_centroids

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
        if dem_arr.shape.__len__() == 3:
            dem_arr = dem_arr[0, :, :]
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
    dem_param['aspect'] = np.deg2rad(aspect.flatten() - 180)
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


def compute_horizon(dem_file, azimuth_inc=30):
    '''
    Function to compute horizon angles for
    :param dem_file:
    :param azimuth_inc:
    :return: xarray dataset containing the
    '''

    with rasterio.open(dem_file) as rf:
        dem = rf.read()
        geot = rf.read_transform()
        if dem.shape.__len__() == 3:
            dem = dem[0, :, :]
        dx = geot[1]
        x = np.linspace(rf.bounds.left + np.round(geot[1], 3), rf.bounds.right, rf.shape[1])
        y = np.linspace(rf.bounds.bottom + np.round(geot[5], 3), rf.bounds.top, rf.shape[0])
    azimuth = np.arange(-180 + azimuth_inc / 2, 180, azimuth_inc) # center the azimuth in middle of the bin
    arr = np.empty((azimuth.shape[0], dem.shape[0], dem.shape[1]))
    #dem = np.flip(dem.astype(float))
    dem = dem.astype(float)
    for i, azi in enumerate(azimuth):
        arr[i, :, :] = horizon.horizon(azi, dem, dx)

    da = xr.DataArray(data= np.flip(np.pi/2 - np.arccos(arr), 1),
                      coords={
                          "y": y,
                          "x": x,
                          "azimuth": azimuth
                      },
                      dims=["azimuth", "y", "x"]
                      )
    arr, dem = None, None
    return da


