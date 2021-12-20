'''
Function to compute Solar angles
S. Filhol, Oct 2021

'''

import pvlib
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import Transformer
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc


def get_solar_geom(df_position, start_date, end_date, tstep, sr_epsg="4326", num_threads=None):
    '''
    Function to compute solar position for each location given in the dataframe
    azimuth is define with 0 towards South, negative in W-dir, and posiive towards E-dir
    :param df_position: dataframe with point_id as index, latitude, longitude
    :param start_date: start date in string "2014-05-10"
    :param end_date: end date in string     "2015-05-10"
    :param tstep: time step to use (str)    '6H'
    :param sr_epsg: source EPSG code for the input coordinate
    :param num_threads: int, number of threads to parallelize computation on. default is number of core -2
    :return: xarray dataset of   solar angles in degrees

    TODO:
        - [ ] implement pooling instead of the for loop to compute simultaneously multiple points
        - [ ] remove avg computation by doubling the computation of.
        - [ ] assume 1H timestep only
    '''
    print('\n---> Computing solar geometry')
    if (int(sr_epsg) != "4326") or ('longitude' not in df_position.columns):
        trans = Transformer.from_crs("epsg:" + sr_epsg, "epsg:4326", always_xy=True)
        df_position['longitude'], df_position['latitude'] = trans.transform(df_position.x.values, df_position.y.values)
    tstep_dict = {'1H': 1, '3H': 3, '6H': 6}

    times = pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq='1H', tz='UTC', closed='left')


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
    return ds

