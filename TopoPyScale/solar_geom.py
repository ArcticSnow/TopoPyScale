'''
Function to compute Solar angles
S. Filhol, Oct 2021

'''

import pvlib
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import Transformer


def get_solar_geom(df_position, start_date, end_date, tstep, sr_epsg=4326):
    '''
    Function to compute solar position for each location given in the dataframe

    :param df_position: dataframe with point_id as index, latitude, longitude
    :param start_date: start date in string "2014-05-10"
    :param end_date: end date in string     "2015-05-10"
    :param tstep: time step to use (str)    '6H'
    :param sr_epsg: source EPSG code for the input coordinate
    :return: xarray dataset of solar angles in degrees

    TODO:
        - check if degrees is the correct unit
    '''

    if (int(sr_epsg) != 4326) or ('longitude' not in df_position.columns):
        trans = Transformer.from_proj(int(sr_epsg), 4326)
        df_position['longitude'], df_position['latitude'] = trans.transform(df_position.x, df_position.y)

    times = pd.date_range(start_date, end_date, freq=tstep, tz='UTC')
    arr = np.empty((2, df_position.shape[0], times.shape[0]))

    for i, row in df_position.iterrows():
        arr[:, i,:] = pvlib.solarposition.get_solarposition(times, row.latitude, row.longitude)[['zenith', 'azimuth']].values.T

    ds = xr.Dataset(
        {
            "zenith": (["point_id", "time"], arr[0, :, :]),
            "azimuth": (["point_id", "time"], arr[1, :, :]),
            "latitude": (["point_id", "time"], np.reshape(np.repeat(df_position.latitude.values, times.shape[0]),
                                                         (df_position.latitude.shape[0], times.shape[0]))),
            "longitude": (["point_id", "time"], np.reshape(np.repeat(df_position.longitude.values, times.shape[0]),
                                                         (df_position.longitude.shape[0], times.shape[0]))),
        },
        coords={
            "point_id": df_position.index,
            "time": times,
            "reference_time": pd.Timestamp(start_date),
        },
    )
    return ds

