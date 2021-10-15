'''
Set of functions to work with DEMs
S. Filhol, Oct 2021
'''

import rasterio
import pandas as pd
import numpy as np
from topocalc import gradient
from topocalc import viewf

def compute_DEM_param(dem_file):
    '''
    Function to compute and derive DEM parameters: slope, aspect, sky view factor
    :param dem_file: path to raster file (geotif)
    :return: pandas dataframe containing x, y, elev, slope, aspect, svf
    '''
    dem = rasterio.open(dem_file)

    dem_param = pd.DataFrame()
    geot = dem.read_transform()

    dx, dy = geot[1], -geot[-1]

    x = np.linspace(dem.bounds.left + np.round(geot[1],3), dem.bounds.right, dem.shape[1])
    y = np.linspace(dem.bounds.bottom + np.round(geot[5],3), dem.bounds.top, dem.shape[0])

    Xs, Ys = np.meshgrid(x,y)

    dem_arr = dem.read()
    if dem_arr.shape.__len__() == 3:
        dem_arr = dem_arr[0,:,:]

    slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
    dem_param['x'] = Xs.flatten()
    dem_param['y'] = Ys.flatten()
    dem_param['elev'] = dem_arr.flatten()
    dem_param['slope'] = slope.flatten()
    dem_param['aspect'] = aspect.flatten()
    Xs, Ys, slope, aspect = None, None, None, None

    svf = viewf.viewf(np.double(dem_arr), dx)[0]
    dem_param['svf'] = svf.flatten()
    dem = None
    dem_arr = None
    return dem_param