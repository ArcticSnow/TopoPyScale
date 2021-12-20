'''
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- an improvement could be to first copmute horizons, and then SVF to avoid computing horizon twice
'''

import sys
import rasterio
from pyproj import Transformer
import pandas as pd
import numpy as np
import xarray as xr
from topocalc import gradient
from topocalc import viewf
from topocalc import horizon
import time

def convert_epsg_pts(xs,ys, epsg_src=4326, epsg_tgt=3844):
    '''
    Simple function to convert a list fo poitn from one projection to another oen using PyProj
    :param xs: 1D array with X-coordinate expressed in the source EPSG
    :param ys: 1D array with Y-coordinate expressed in the source EPSG
    :param epsg_src: source projection EPSG code
    :param epsg_tgt: target projection EPSG code
    :return: Xs, Ys 1D arrays of the point coordinates expressed in the target projection
    '''
    print('Convert coordinates from EPSG:{} to EPSG:{}'.format(epsg_src, epsg_tgt))
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:{}".format(epsg_tgt), always_xy=True)
    Xs, Ys = trans.transform(xs, ys)
    return Xs, Ys


def get_extent_latlon(dem_file, epsg_src):
    '''
    Function to extract DEM extent in Lat/Lon
    :param dem_file: path to DEM file (GeoTiFF)
    :param epsg_src: int, EPSG projection code
    :return: dict, extent in lat/lon, {latN, latS, lonW, lonE}
    '''
    with rasterio.open(dem_file) as rf:
        xs, ys = [rf.bounds.left, rf.bounds.right], [rf.bounds.bottom, rf.bounds.top]
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:4326", always_xy=True)
    lons, lats = trans.transform(xs, ys)
    extent = {'latN': lats[1],
              'latS': lats[0],
              'lonW': lons[0],
              'lonE': lons[1]}
    return extent


def extract_pts_param(df_pts, ds_param, method='nearest'):
    '''
    Function to sample DEM parameters for a list point. This is used as an alternative the the TopoSub method, to perform downscaling at selected locations (x,y)
    WARNING: the projection and coordiante system of the EDM and point coordinates MUST be the same!

    :param df_pts: pandas DataFrame containing a list of points coordinates with coordiantes in (x,y).
    :param ds_param: xarray dataset of dem parameters
    :param method: sampling method. Supported 'nearest', 'linear' interpolation, 'idw' interpolation (inverse-distance weighted)
    :return: df_pts updated with new columns ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']
    '''
    print('\n---> Extracting DEM parameters for the given list of point coordinates')
    # delete columns in case they already exist
    df_pts = df_pts.drop(['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf'], errors='ignore')
    # create columns, filled with 0
    df_pts[['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = 0

    if method == 'nearest':
        for i, row in df_pts.iterrows():
            d_mini = ds_param.sel(x=row.x, y=row.y, method='nearest')
            df_pts.loc[i, ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = np.array((d_mini.elevation.values,
                                                                                                   d_mini.slope.values,
                                                                                                   d_mini.aspect.values,
                                                                                                   d_mini.aspect_cos,
                                                                                                   d_mini.aspect_sin,
                                                                                                   d_mini.svf.values))
    elif method == 'idw' or method == 'linear':
        for i, row in df_pts.iterrows():
            ind_lat = np.abs(ds_param.y-row.y).argmin()
            ind_lon = np.abs(ds_param.x-row.x).argmin()
            ds_param_pt = ds_param.isel(y=[ind_lat-1, ind_lat, ind_lat+1], x=[ind_lon-1, ind_lon, ind_lon+1])
            Xs, Ys = np.meshgrid(ds_param_pt.x.values, ds_param_pt.y.values)

            if method == 'idw':
                dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
                idw = 1/(dist**2)
                weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1

            if method == 'linear':
                dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
                weights = dist / np.sum(dist)
            da_idw = xr.DataArray(data=weights,
                                  coords={
                                    "y": ds_param_pt.y.values,
                                    "x": ds_param_pt.x.values,
                              },
                              dims=["y", "x"]
                              )
            dw = xr.core.weighted.DatasetWeighted(ds_param_pt, da_idw)
            d_mini = dw.sum(['x', 'y'], keep_attrs=True)
            df_pts.loc[i, ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = np.array((d_mini.elevation.values,
                                                                                                   d_mini.slope.values,
                                                                                                   d_mini.aspect.values,
                                                                                                   d_mini.aspect_cos,
                                                                                                   d_mini.aspect_sin,
                                                                                                 d_mini.svf.values))
    else:
        print('ERROR: Method not implemented. Only nearest, linear or idw available')
    return df_pts

def compute_dem_param(dem_file):
    '''
    Function to compute and derive DEM parameters: slope, aspect, sky view factor
    :param dem_file: path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM)
    :return:  xarray dataset containing x, y, elev, slope, aspect, svf
    '''
    print('\n---> Extracting DEM parameters (slope, aspect, svf)')
    ds = xr.open_rasterio(dem_file).to_dataset('band')
    ds = ds.rename({1: 'elevation'})
    dx = ds.x.diff('x').median().values
    dy = ds.y.diff('y').median().values
    dem_arr = ds.elevation.values
    slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
    print('---> Computing sky view factor')
    start_time = time.time()
    svf = viewf.viewf(np.double(dem_arr), dx)[0]
    print('---> Sky-view-factor finished in {}s'.format(np.round(time.time()-start_time), 0))
    ds['slope'] = (["y", "x"], slope)
    ds['aspect'] = (["y", "x"], np.deg2rad(aspect - 180))
    ds['aspect_cos'] = (["y", "x"], np.cos(np.deg2rad(aspect)))
    ds['aspect_sin'] = (["y", "x"], np.sin(np.deg2rad(aspect)))
    ds['svf'] = (["y", "x"], svf)
    ds.attrs = dict(description="DEM input parameters to TopoSub",
                   author="TopoPyScale, https://github.com/ArcticSnow/TopoPyScale")
    ds.x.attrs = {'units': 'm'}
    ds.y.attrs = {'units': 'm'}
    ds.elevation.attrs = {'units': 'm'}
    ds.slope.attrs = {'units': 'rad'}
    ds.aspect.attrs = {'units': 'rad'}
    ds.aspect_cos.attrs = {'units': 'cosinus'}
    ds.aspect_sin.attrs = {'units': 'sinus'}
    ds.svf.attrs = {'units': 'ratio', 'standard_name': 'svf', 'long_name': 'Sky view factor'}

    return ds


def compute_horizon(dem_file, azimuth_inc=30):
    '''
    Function to compute horizon angles for
    :param dem_file:
    :param azimuth_inc:
    :return: xarray dataset containing the
    '''
    print('\n---> Computing horizons with {} degree increment'.format(azimuth_inc))
    ds = xr.open_rasterio(dem_file).to_dataset('band')
    ds = ds.rename({1: 'elevation'})
    dx = ds.x.diff('x').median().values

    azimuth = np.arange(-180 + azimuth_inc / 2, 180, azimuth_inc) # center the azimuth in middle of the bin
    arr = np.empty((azimuth.shape[0], ds.elevation.shape[0], ds.elevation.shape[1]))
    for i, azi in enumerate(azimuth):
        arr[i, :, :] = horizon.horizon(azi, ds.elevation.values, dx)

    da = xr.DataArray(data=np.pi/2 - np.arccos(arr),
                      coords={
                          "y": ds.y.values,
                          "x": ds.x.values,
                          "azimuth": azimuth
                      },
                      dims=["azimuth", "y", "x"]
                      )
    return da


