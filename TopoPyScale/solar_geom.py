"""
Function to compute Solar angles
S. Filhol, Oct 2021

"""

import pvlib
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import Transformer
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
from TopoPyScale import topo_export as te


def get_solar_geom(df_position, start_date, end_date, tstep, sr_epsg="4326", num_threads=None, fname='ds_solar.nc', project_directory='./'):
    """
    Function to compute solar position for each location given in the dataframe
    azimuth is define with 0 towards South, negative in W-dir, and posiive towards E-dir

    Args:
        df_position (dataframe): point_id as index, latitude, longitude
        start_date (str): start date  "2014-05-10"
        end_date (str: end date   "2015-05-10"
        tstep (str): time step, ex: '6H'
        sr_epsg (str): source EPSG code for the input coordinate
        num_threads (int): number of threads to parallelize computation on. default is number of core -2
        fname (str): name of netcdf file to store solar geometry
        project_directory (str): path to project root directory

    Returns: 
        dataset: solar angles in degrees

    """
    print('\n---> Computing solar geometry')
    if (int(sr_epsg) != "4326") or ('longitude' not in df_position.columns):
        trans = Transformer.from_crs("epsg:" + sr_epsg, "epsg:4326", always_xy=True)
        df_position['longitude'], df_position['latitude'] = trans.transform(df_position.x.values, df_position.y.values)
    # tstep_dict = {'1H': 1, '3H': 3, '6H': 6}

    times = pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq=tstep, tz='UTC', closed='left')


    df_pool = pd.DataFrame()
    df_pool[['latitude', 'longitude', 'elevation']] = df_position[['latitude', 'longitude', 'elevation']]
    df_pool['times'] = df_pool.latitude.apply(lambda x: times)

    def compute_solar_geom(time_step, latitude, longitude, elevation):
        arr = pvlib.solarposition.get_solarposition(time_step, latitude, longitude, elevation)[['zenith', 'azimuth', 'elevation']].values.T
        return arr

    if num_threads is None:
        pool = ThreadPool(mproc.cpu_count() - 2)
    else:
        pool = ThreadPool(num_threads)

    arr = pool.starmap(compute_solar_geom, zip(list(df_pool.times),
                                                   list(df_pool.latitude),
                                                   list(df_pool.longitude),
                                                   list(df_pool.elevation)))
    pool.close()
    pool.join()

    arr_val = np.empty((df_position.shape[0], 3, times.shape[0]))
    for i, a in enumerate(arr):
        arr_val[i,:,:]=a

    ds = xr.Dataset(
        {
            "zenith": (["point_id", "time"], np.deg2rad(arr_val[:, 0, :])),
            "azimuth": (["point_id", "time"], np.deg2rad(arr_val[:,1,:] - 180)),
            "elevation": (["point_id", "time"], np.deg2rad(arr_val[:, 2, :])),
        },
        coords={
            "point_id": df_position.index,
            "time": pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq=tstep, closed='left'),
            "reference_time": pd.Timestamp(start_date),
        },
    )

    ds['mu0'] = np.cos(ds.zenith) * (np.cos(ds.zenith)>0)
    S0 = 1370 # Solar constat (total TOA solar irradiance) [Wm^-2] used in ECMWF's IFS
    ds['SWtoa'] = S0 * ds.mu0
    ds['sunset'] = ds.mu0 < np.cos(89 * np.pi/180)
    ds.SWtoa.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave radiations downward top of the atmosphere'}
    ds.sunset.attrs = {'units': 'bool', 'standard_name': 'Sunset'}

    te.to_netcdf(ds, fname=f'{project_directory}outputs/{fname}')

    return ds

