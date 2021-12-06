'''
Functions to export topo_scale output to formats compatible with existing models (e.g. CROCUS, Cryogrid, Snowmodel, ...)
S. Filhol, December 2021

'''

import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
from scipy import io

def partition_snow(ds):
    '''
    Function to partition precipitation in rain vs snow based on temperature threshold and mixing
    :param ds:
    :return:

    TODO:
    - all
    '''
    return rain, snow

# Save data to micromet format
def to_micromet_single_station(fname, stn_df, stn_east, stn_north, stn_elev, stn_id, na_values=-9999, headers=False):
    '''
    Function to write data into a format compatible with micromet of snowmodel Liston
    :param fname: file name and path
    :param stn_df: data frame containing the data. WARNING: columns names must be compatible
    :param stn_east: East coordinate of the station [m]
    :param stn_north: North coordinate of the station [m]
    :param stn_elev: Elevation of the station [m]
    :param stn_id: Station ID
    :return: -

    TODO: ADAPT everythong

    Example of the compatible format for micromet:

     year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip
    (yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)

     2002   10    1   12.00    101   426340.0  4411238.0  3598.0     0.92    57.77     4.80   238.29 -9999.00
     2002   10    2   12.00    101   426340.0  4411238.0  3598.0    -3.02    78.77 -9999.00 -9999.00 -9999.00
     2002   10    8   12.00    101   426340.0  4411238.0  3598.0    -5.02    88.77 -9999.00 -9999.00 -9999.00
    '''

    try:
        df = pd.DataFrame()
        df['year'] = stn_df.index.year
        df['mo']  = stn_df.index.month
        df['dy']  = stn_df.index.day
        df['hr']  = stn_df.index.hour
        df['stn_id'] = np.repeat(stn_id, df.shape[0])
        df['easting'] = np.repeat(stn_east, df.shape[0])
        df['northing'] = np.repeat(stn_north, df.shape[0])
        df['elevation'] = np.repeat(stn_elev, df.shape[0])
        df['Tair'] = stn_df.Tair.__array__()
        df['RH'] = stn_df.RH.__array__()
        df['speed'] = stn_df.speed.__array__()
        df['dir'] = stn_df.dir.__array__()
        df['precip'] = stn_df.precip.__array__()

        df.to_csv(fname, index=False, header=False, sep=' ', na_rep=na_values)
        if headers:
            header = 'year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip\n(yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)\n'

            with open(fname, 'r+') as f:
                f_data = f.read()
                f.seek(0,0)
                f.write(header + '\n' + f_data)

        print(fname, ' Saved')
    except:
        print('something wrong')

def to_crocus(foutput, ds, scale_precip=1):
    '''
    Function to
    :param ds:
    :return:

    TODO:
    - partition precip in between snow and rain
    - add point_id in file name (with sero padded)
    '''
    # create one file per point_id
    for pt in ds.point_id.values:
        ds_pt = ds.sel(point_id=pt).copy()
        df = pd.DataFrame()
        df['time'] = ds_pt.time.data
        df['Tair'] = ds_pt.t.data.flatten()
        df['DIR_SWdown'] = ds_pt.SW.data.flatten()
        df['LWdown'] = ds_pt.LW.data.flatten()
        df['PSurf'] = ds_pt.p.data.flatten()
        #df['HUMREL'] = ds_pt.relative_humidity_2m.data.flatten()
        df['xwind'] = ds_pt.u.data.flatten()
        df['ywind'] = ds_pt.v.data.flatten()
        df['precip'] = ds_pt.tp.data.flatten() / 3600 * scale_precip
        #df['Snowf'] =ds_pt.Snowf.data.flatten() * scale_precip
        #df['Rainf'] = df.precip - df.Snowf

            # Derive variables: Q- humidity, WD - wind direction (deg), and WS
        df['Qair'] = ds_pt.q.data.flatten()
        df['Wind'], df['Wind_DIR'] = ds_pt.ws.values, np.rad2deg(ds.wd.values)
        df['NEB'] = np.zeros(df.shape[0])
        df['CO2air'] = np.zeros(df.shape[0])
        df['SCA_SWdown'] = np.zeros(df.shape[0])
        ds_pt.close()

        df = df.loc[(df.time>=pd.Timestamp(start)) & (df.time<pd.Timestamp(end)+pd.Timedelta('1D'))]

        # Build netcdf forcing file for CROCUS
        fo = xr.Dataset.from_dataframe(df.set_index('time'))
        fo = df.set_index(['time']).to_xarray()
        fo.Tair.attrs = {'units':'K', 'standard_name':'Tair', 'long_name':'Near Surface Air Temperature', '_FillValue': -9999999.0}
        fo.Qair.attrs = {'units':'kg/kg', 'standard_name':'Qair', 'long_name':'Near Surface Specific Humidity', '_FillValue': -9999999.0}
        #fo.HUMREL.attrs = {'units':'%', 'standard_name':'HUMREL', 'long_name':'Relative Humidity', '_FillValue': -9999999.0}
        fo.Wind.attrs = {'units':'m/s', 'standard_name':'Wind', 'long_name':'Wind Speed', '_FillValue': -9999999.0}
        fo.Wind_DIR.attrs = {'units':'deg', 'standard_name':'Wind_DIR', 'long_name':'Wind Direction', '_FillValue': -9999999.0}
        fo.Rainf.attrs = {'units':'kg/m2/s', 'standard_name':'Rainf', 'long_name':'Rainfall Rate', '_FillValue': -9999999.0}
        fo.Snowf.attrs = {'units':'kg/m2/s', 'standard_name':'Snowf', 'long_name':'Snowfall Rate', '_FillValue': -9999999.0}
        fo.DIR_SWdown.attrs = {'units':'W/m2', 'standard_name':'DIR_SWdown', 'long_name':'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
        fo.LWdown.attrs = {'units':'W/m2', 'standard_name':'LWdown', 'long_name':'Surface Incident Longtwave Radiation', '_FillValue': -9999999.0}
        fo.PSurf.attrs = {'units':'Pa', 'standard_name':'PSurf', 'long_name':'Surface Pressure', '_FillValue': -9999999.0}
        fo.SCA_SWdown.attrs = {'units':'W/m2', 'standard_name':'SCA_SWdown', 'long_name':'Surface Incident Diffuse Shortwave Radiation', '_FillValue': -9999999.0}
        fo.CO2air.attrs = {'units':'kg/m3', 'standard_name':'CO2air', 'long_name':'Near Surface CO2 Concentration', '_FillValue': -9999999.0}
        fo.NEB.attrs = {'units':'between 0 and 1', 'standard_name':'NEB', 'long_name':'Nebulosity', '_FillValue': -9999999.0}


        fo.attrs = {'title':'Forcing for SURFEX crocus',
                    'source': 'data from AROME MEPS, ready for CROCUS forcing',
                    'creator_name':'Simon Filhol',
                    'creator_email':'simon.filhol@geo.uio.no',
                    'creator_url':'https://www.mn.uio.no/geo/english/people/aca/geohyd/simonfi/index.html',
                    'institution': 'Department of Geosciences, University of Oslo, Norway',
                    'date_created':dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

        fo['ZS'] = np.float64(df_pts.iloc[pt].elevation)
        fo['aspect'] = np.float64(np.rad2deg(df_pts.iloc[pt].aspect))
        fo['slope'] = np.float64(np.rad2deg(df_pts.iloc[pt].slope))
        fo['massif_number'] = np.float64(0)
        fo['LAT'] = df_pts.iloc[pt].latitude
        fo['LON'] = df_pts.iloc[pt].longitude
        fo.LAT.attrs = {'units':'degrees_north', 'standard_name':'LAT', 'long_name':'latitude', '_FillValue': -9999999.0}
        fo.LON.attrs = {'units':'degrees_east', 'standard_name':'LON', 'long_name':'longitude', '_FillValue': -9999999.0}
        fo['ZREF'] = np.float64(2)
        fo['UREF'] = np.float64(10)
        fo.ZREF.attrs = {'units':'m', 'standard_name':'ZREF', 'long_name':'Reference_Height', '_FillValue': -9999999.0}
        fo.UREF.attrs = {'units':'m', 'standard_name':'UREF', 'long_name':'Reference_Height_for_Wind', '_FillValue': -9999999.0}
        fo.aspect.attrs = {'units':'degree from north', 'standard_name':'aspect', 'long_name':'slope aspect', '_FillValue': -9999999.0}
        fo.slope.attrs = {'units':'degrees from horizontal', 'standard_name':'slope', 'long_name':'slope angle', '_FillValue': -9999999.0}
        fo.ZS.attrs = {'units':'m', 'standard_name':'ZS', 'long_name':'altitude', '_FillValue': -9999999.0}
        fo.massif_number.attrs = {'units':'', 'standard_name':'massif_number', 'long_name':'SAFRAN massif number in SOPRANO world', '_FillValue': -9999999.0}
        fo.time.attrs = {'standard_name':'time', 'long_name':'time', '_FillValue': -9999999.0}
        fo['FRC_TIME_STP'] = np.float64(3600)
        fo.FRC_TIME_STP.attrs = {'units':'s','standard_name':'FRC_TIME_STP', 'long_name':'Forcing_Time_Step', '_FillValue': -9999999.0}
        fo['FORCE_TIME_STEP'] = np.float64(3600)
        fo.FORCE_TIME_STEP.attrs = {'units':'s','standard_name':'FORCE_TIME_STEP', 'long_name':'Forcing_Time_Step', '_FillValue': -9999999.0}

        fo = fo.expand_dims('Number_of_points')
        fo = fo.transpose()


        fo.to_netcdf(foutput, mode='w',format='NETCDF3_CLASSIC',  unlimited_dims={'time':True},
                     encoding={'time': {'dtype': 'float64', 'calendar':'standard'},
                     'Tair': {'dtype': 'float64'},
                     'Qair': {'dtype': 'float64'},
                     'xwind': {'dtype': 'float64'},
                     'ywind': {'dtype': 'float64'},
                     'precip': {'dtype': 'float64'},
                     'Snowf': {'dtype': 'float64'},
                     'Rainf': {'dtype': 'float64'},
                     'DIR_SWdown': {'dtype': 'float64'},
                     'LWdown': {'dtype': 'float64'},
                     'PSurf': {'dtype': 'float64'},
                     'HUMREL': {'dtype': 'float64'},
                     'Wind': {'dtype': 'float64'},
                     'Wind_DIR': {'dtype': 'float64'},
                     'CO2air': {'dtype': 'float64'},
                     'NEB': {'dtype': 'float64'},
                     'SCA_SWdown': {'dtype': 'float64'},
                     'ZREF': {'dtype': 'float64'},
                     'UREF': {'dtype': 'float64'}})

