'''
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- an improvement could be to first copmute horizons, and then SVF to avoid computing horizon twice
'''
import pdb

import rasterio
from pyproj import Transformer
import pandas as pd
import numpy as np
import xarray as xr
from topocalc import gradient
from topocalc import viewf
from topocalc import horizon
import time

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
            #pdb.set_trace()
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
    print('---> Computing sky view factor')
    start_time = time.time()
    svf = viewf.viewf(np.double(dem_arr), dx)[0]
    dem_param['svf'] = svf.flatten()
    print('---> Sky-view-factor finished in {}s'.format(np.round(time.time()-start_time), 0))
    # Build an xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            elevation=(["y", "x"], np.flip(dem_arr, 0)),
            slope=(["y", "x"], np.flip(slope, 0)),
            aspect=(["y", "x"], np.flip(np.deg2rad(aspect - 180), 0)),
            aspect_cos=(["y", "x"], np.flip(np.cos(np.deg2rad(aspect)), 0)),
            aspect_sin=(["y", "x"], np.flip(np.sin(np.deg2rad(aspect)), 0)),
            svf=(["y", "x"], np.flip(svf, 0))
        ),
        coords={'x': x, 'y': y}
        ,
        attrs=dict(description="DEM input parameters to TopoSub",
                   author="TopoPyScale, https://github.com/ArcticSnow/TopoPyScale"),
        )
    ds.x.attrs = {'units':'m'}
    ds.y.attrs = {'units':'m'}
    ds.elevation.attrs = {'units':'m'}
    ds.slope.attrs = {'units':'rad'}
    ds.aspect.attrs = {'units':'rad'}
    ds.aspect_cos.attrs = {'units':'cosinus'}
    ds.aspect_sin.attrs = {'units':'sinus'}
    ds.svf.attrs = {'units':'ratio', 'standard_name':'svf', 'long_name':'Sky view factor'}

    Xs, Ys, slope, aspect = None, None, None, None
    dem_arr = None
    return ds


def compute_horizon(dem_file, azimuth_inc=30):
    '''
    Function to compute horizon angles for
    :param dem_file:
    :param azimuth_inc:
    :return: xarray dataset containing the
    '''
    print('---> Computing horizons with {} degree increment'.format(azimuth_inc))
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


