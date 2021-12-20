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



def get_solar_geom(df_position, start_date, end_date, tstep, sr_epsg="4326"):
    '''
    Function to compute solar position for each location given in the dataframe
    azimuth is define with 0 towards South, negative in W-dir, and posiive towards E-dir
    :param df_position: dataframe with point_id as index, latitude, longitude
    :param start_date: start date in string "2014-05-10"
    :param end_date: end date in string     "2015-05-10"
    :param tstep: time step to use (str)    '6H'
    :param sr_epsg: source EPSG code for the input coordinate
    :return: xarray dataset of   solar angles in degrees

    TODO:
        - [ ] implement pooling instead of the for loop to compute simultaneously multiple points
        - [ ] remove avg computation by doubling the computation of
    '''
    print('\n---> Computing solar geometry')
    if (int(sr_epsg) != "4326") or ('longitude' not in df_position.columns):
        trans = Transformer.from_crs("epsg:" + sr_epsg, "epsg:4326", always_xy=True)
        df_position['longitude'], df_position['latitude'] = trans.transform(df_position.x.values, df_position.y.values)
    tstep_dict = {'1H': 1, '3H': 3, '6H': 6}

    times = pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq='1H', tz='UTC', closed='left')
    tstep_vec = pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq=tstep, tz='UTC', closed='left')
    arr_avg = np.empty((df_position.shape[0], 4, tstep_vec.shape[0]))

    num_threads=4
    pool = ThreadPool(num_threads)

    def compute_solar_geom(arr_avg, times,  ):
        df = pvlib.solarposition.get_solarposition(times, row.latitude, row.longitude, row.elevation)[['zenith', 'azimuth', 'elevation']]
        df['cos_az'] = np.cos((df.azimuth - 180) * np.pi / 180)
        df['sin_az'] = np.sin((df.azimuth - 180) * np.pi / 180)
        arr_avg[i, :, :] = df[['zenith', 'cos_az', 'sin_az', 'elevation']].resample(tstep).mean().values.T


    for i, row in tqdm(df_position.iterrows(), total=df_position.shape[0]):

        # compute cos and sin of azimuth to get avg (to avoid discontinuity at North)
        df = pvlib.solarposition.get_solarposition(times, row.latitude, row.longitude, row.elevation)[['zenith', 'azimuth', 'elevation']]
        df['cos_az'] = np.cos((df.azimuth - 180) * np.pi / 180)
        df['sin_az'] = np.sin((df.azimuth - 180) * np.pi / 180)
        arr_avg[i, :, :] = df[['zenith', 'cos_az', 'sin_az', 'elevation']].resample(tstep).mean().values.T



    # add buffer start date by tstep that later aeraging does not produce NaNs
    #start_date = pd.Timestamp(start_date) - pd.Timedelta(tstep)

    ds = xr.Dataset(
        {
            "zenith_avg": (["point_id", "time"], np.deg2rad(arr_avg[:, 0, :])),
            "azimuth_avg": (["point_id", "time"], np.arctan2(arr_avg[:, 1, :], arr_avg[:, 2, :])),
            "elevation_avg": (["point_id", "time"], np.deg2rad(arr_avg[:, 3, :])),
        },
        coords={
            "point_id": df_position.index,
            "time": pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq=tstep, closed='left'),
            "reference_time": pd.Timestamp(start_date),
        },
    )
    return ds

