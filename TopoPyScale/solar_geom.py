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
    tstep_dict = {'1H': 1, '3H': 3, '6H': 6}

    times = pd.date_range(start_date, end_date, freq='1H', tz='UTC')
    tstep_vec = pd.date_range(start_date, end_date, freq=tstep, tz='UTC')
    arr_val = np.empty((df_position.shape[0], 3, tstep_vec.shape[0]))
    arr_avg = np.empty((df_position.shape[0], 4, tstep_vec.shape[0]))

    for i, row in df_position.iterrows():
        arr_val[i, :, :] = pvlib.solarposition.get_solarposition(tstep_vec, row.latitude, row.longitude, row.elev)[['zenith', 'azimuth', 'elevation']].values.T

        # compute cos and sin of azimuth to get avg (to avoid discontinuity at North)
        df = pvlib.solarposition.get_solarposition(times, row.latitude, row.longitude, row.elev)[['zenith', 'azimuth', 'elevation']]
        df['cos_az'] = np.cos((df.azimuth - 180) * np.pi / 180)
        df['sin_az'] = np.sin((df.azimuth - 180) * np.pi / 180)
        arr_avg[i, :, :] = df[['zenith', 'cos_az', 'sin_az', 'elevation']].resample(tstep).mean().values.T



    # add buffer start date by tstep that later aeraging does not produce NaNs
    #start_date = pd.Timestamp(start_date) - pd.Timedelta(tstep)

    ds = xr.Dataset(
        {
            "zenith": (["point_id", "time"], np.deg2rad(arr_val[:, 0, :])),
            "azimuth": (["point_id", "time"], np.deg2rad(arr_val[:, 1, :])),
            "elevation": (["point_id", "time"], np.deg2rad(arr_val[:, 2, :])),
            "zenith_avg": (["point_id", "time"], np.deg2rad(arr_avg[:, 0, :])),
            "azimuth_avg": (["point_id", "time"], np.arctan2(arr_avg[:, 1, :], arr_avg[:, 2, :])),
            "elevation_avg": (["point_id", "time"], np.deg2rad(arr_avg[:, 3, :])),
        },
        coords={
            "point_id": df_position.index,
            "time": pd.date_range(start_date, end_date, freq=tstep),
            "reference_time": pd.Timestamp(start_date),
        },
    )
    return ds

