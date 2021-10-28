'''
Function to compute Solar angles
S. Filhol, Oct 2021

'''

import pvlib
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import Transformer


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
        - check if degrees is the correct unit
    '''

    if (int(sr_epsg) != "4326") or ('longitude' not in df_position.columns):
        trans = Transformer.from_crs("epsg:" + sr_epsg, "epsg:4326", always_xy=True)
        df_position['longitude'], df_position['latitude'] = trans.transform(df_position.x, df_position.y)
    tstep_dict = {'1H':1, '3H':3, '6H':6}

    times = pd.date_range(start_date, end_date, freq='1H', tz='UTC')
    tstep_vec = pd.date_range(start_date, end_date, freq=tstep, tz='UTC')
    arr_val = np.empty((df_position.shape[0], 2, tstep_vec.shape[0]))
    arr_avg = np.empty((df_position.shape[0], 2, tstep_vec.shape[0]))

    for i, row in df_position.iterrows():
        arr_val[i, :, :] = pvlib.solarposition.get_solarposition(tstep_vec, row.latitude, row.longitude)[['zenith', 'azimuth']].values.T
        arr_avg[i, :, :] = pvlib.solarposition.get_solarposition(times, row.latitude, row.longitude)[['zenith', 'azimuth']].resample(tstep).mean().values.T



    # add buffer start date by tstep that later aeraging does not produce NaNs
    #start_date = pd.Timestamp(start_date) - pd.Timedelta(tstep)

    ds = xr.Dataset(
        {
            "zenith": (["point_id", "time"], arr_val[:, 0, :]),
            "azimuth": (["point_id", "time"], arr_val[:, 1, :] - 180),
            "zenith_avg": (["point_id", "time"], arr_avg[:, 0, :]),
            "azimuth_avg": (["point_id", "time"], arr_avg[:, 0, :] - 180),
            "latitude": (["point_id", "time"], np.reshape(np.repeat(df_position.latitude.values, tstep_vec.shape[0]),
                                                         (df_position.latitude.shape[0], tstep_vec.shape[0]))),
            "longitude": (["point_id", "time"], np.reshape(np.repeat(df_position.longitude.values, tstep_vec.shape[0]),
                                                         (df_position.longitude.shape[0], tstep_vec.shape[0]))),
        },
        coords={
            "point_id": df_position.index,
            "time": pd.date_range(start_date, end_date, freq=tstep),
            "reference_time": pd.Timestamp(start_date),
        },
    )
    return ds

