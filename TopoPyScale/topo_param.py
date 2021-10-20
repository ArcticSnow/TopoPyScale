'''
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- figure out xarray coordinates for horizon function
- write function to get ERA5 x,y,z at each point list
- set to np.nan all values below 0m in the DEM for the horizon output. Horizin does not accept NaN
'''

import rasterio
import pandas as pd
import numpy as np
import xarray as xr
from topocalc import gradient
from topocalc import viewf
from topocalc import horizon

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

    x = np.linspace(dem.bounds.left + np.round(geot[1], 3), dem.bounds.right, dem.shape[1])
    y = np.linspace(dem.bounds.bottom + np.round(geot[5], 3), dem.bounds.top, dem.shape[0])

    Xs, Ys = np.meshgrid(x,y)

    dem_arr = dem.read()
    if dem_arr.shape.__len__() == 3:
        dem_arr = dem_arr[0, :, :]

    slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
    dem_param['x'] = Xs.flatten()
    dem_param['y'] = Ys.flatten()
    dem_param['elev'] = dem_arr.flatten()
    dem_param['slope'] = slope.flatten()
    dem_param['aspect_cos'] = np.cos(np.deg2rad(aspect.flatten()))
    dem_param['aspect_sin'] = np.sin(np.deg2rad(aspect.flatten()))
    Xs, Ys, slope, aspect = None, None, None, None

    svf = viewf.viewf(np.double(dem_arr), dx)[0]
    dem_param['svf'] = svf.flatten()
    dem = None
    dem_arr = None
    return dem_param


def extract_ERA5_xyz(ds, df_centroids):
    return


def compute_horizon(dem_file, azimtuh=np.arange(-180, 180, 30)):
    return

'''

dem_file = '/home/arcticsnow/Desktop/topoTest/inputs/dem/dem_90m_SRTM_proj.tif'
with rasterio.open(dem_file) as rf:
    dem = rf.read()
    geot = rf.read_transform()
    if dem.shape.__len__() == 3:
        dem = dem[0, :, :]
    dx = geot[1]
    x = np.linspace(rf.bounds.left + np.round(geot[1], 3), rf.bounds.right, rf.shape[1])
    y = np.linspace(rf.bounds.bottom + np.round(geot[5], 3), rf.bounds.top, rf.shape[0])
Xs, Ys = np.meshgrid(x,y)
azimuth = np.arange(-180, 180, 30)
arr = np.empty((dem.shape[0], dem.shape[1], azimuth.shape[0]))
for i, azi in enumerate(azimuth):
    arr[:, :, i] = horizon.horizon(azi, dem.astype(float), dx)

da = xr.DataArray(data = arr,
                  coords={
                      "lat": (["x", "y"], Ys),
                      "lon": (["x", "y"], Xs),
                      "azimuth": azimuth
                  },
                  dims=["x", "y", "azimuth"]
    )


ds = xr.Dataset(
        {
            "horizon": (["y", "x", "azimuth"], arr)
        },
        coords={
            "lat": (["y", "x"], Ys),
            "lon": (["y", "x"], Xs),
            "azimuth": azimuth
        }
    )
arr = None
dem = None
'''