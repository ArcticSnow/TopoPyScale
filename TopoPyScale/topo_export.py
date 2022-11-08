"""
Functions to export topo_scale output to formats compatible with existing models (e.g. CROCUS, Cryogrid, Snowmodel, ...)

S. Filhol, December 2021

TODO;
- SPHY forcing (grids)
"""
import pdb
import sys
import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
from scipy import io
from TopoPyScale import meteo_util as mu
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc


def compute_scaling_and_offset(da, n=16):
    """
    Compute offset and scale factor for int conversion

    Args:
        da (dataarray): of a given variable
        n (int): number of digits to account for
    """
    vmin = float(da.min().values)
    vmax = float(da.max().values)

    # stretch/compress data to the available packed range
    scale_factor = (vmax - vmin) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = vmin + 2 ** (n - 1) * scale_factor

    return scale_factor, add_offset

def to_musa(ds,
            df_pts,
            da_label,
            fname_met='musa_met.nc',
            fname_labels='musa_labels.nc',
            path='outputs/',
            climate_dataset_name='ERA5',
            project_authors='S. Filhol'
            ):
    """
    Function to export to MuSa standard.
    Args:
        ds:
        df_pts:
        da_labels:
        fname:
        path:

    Returns:
        Save downscaled timeseries and toposub cluster mapping
    """

    da_label.to_netcdf(path + fname_labels)

    fo = xr.Dataset()
    fo['tair'] = ds['t'].T
    fo['q'] = ds['q'].T
    fo['p'] = ds['p'].T
    fo['lw'] = ds['LW'].T
    fo['sw'] = ds['SW'].T
    fo['ws'] = ds['ws'].T
    fo['tp'] = ds['tp'].T
    fo['rh'] = (('time', 'point_id'), mu.q_2_rh(fo.tair.values, fo.p.values, fo.q.values))
    fo['precip'] = (('time', 'point_id'), fo.tp.values / 3600)  # convert from mm/hr to mm/s
    fo = fo.drop_vars(['q', 'tp'])
    fo = fo[['tair', 'rh', 'p', 'lw', 'sw', 'ws', 'precip']].expand_dims({'dummy': 1}, axis=2)

    fo.tair.attrs = {'units':'C', 'standard_name':'tair', 'long_name':'Near Surface Air Temperature', '_FillValue': -9999999.0}
    fo.rh.attrs = {'units':'kg/kg', 'standard_name':'rh', 'long_name':'Near Surface Relative Humidity', '_FillValue': -9999999.0}
    fo.ws.attrs = {'units':'m/s', 'standard_name':'ws', 'long_name':'Wind Speed', '_FillValue': -9999999.0}
    fo.precip.attrs = {'units':'mm/s', 'standard_name':'precip', 'long_name':'Precipitation', '_FillValue': -9999999.0}
    fo.sw.attrs = {'units':'W/m2', 'standard_name':'sw', 'long_name':'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
    fo.lw.attrs = {'units':'W/m2', 'standard_name':'lw', 'long_name':'Surface Incident Longtwave Radiation', '_FillValue': -9999999.0}
    fo.p.attrs = {'units':'Pa', 'standard_name':'p', 'long_name':'Surface Pressure', '_FillValue': -9999999.0}

    fo.attrs = {'title':'Forcing for MuSa assimilation scheme',
                'source': 'Data from {} downscaled with TopoPyScale'.format(climate_dataset_name),
                'creator_name':'Dataset created by {}'.format(project_authors),
                'date_created':dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    encod_dict = {}
    for var in list(fo.keys()):
        scale_factor, add_offset = compute_scaling_and_offset(fo[var], n=10)
        encod_dict.update({var:{"zlib": True,
                               "complevel": 9,
                               'dtype':'int16',
                               'scale_factor':scale_factor,
                               'add_offset':add_offset}})

    fo['latitude'] = (('point_id'), df_pts.latitude.values)
    fo['longitude'] = (('point_id'), df_pts.longitude.values)
    fo['elevation'] = (('point_id'), df_pts.elevation.values)
    fo.latitude.attrs = {'units':'deg', 'standard_name':'latitude', 'long_name':'Cluster latitude'}
    fo.longitude.attrs = {'units':'deg', 'standard_name':'longitude', 'long_name':'Cluster longitude'}
    fo.elevation.attrs = {'units':'m', 'standard_name':'elevation', 'long_name':'Cluster elevation'}

    fo.to_netcdf(path + fname_met, encoding=encod_dict)
    print('---> File {} saved'.format(fname_met))




def to_cryogrid(ds,
                df_pts,
                fname_format='Cryogrid_pt_*.nc',
                path='outputs/',
                label_map=False,
                da_label=None,
                snow_partition_method='continuous',
                climate_dataset_name='ERA5',
                project_author='S. Filhol'):
    """
    Function to export TopoPyScale downscaled dataset in a netcdf format compatible for Cryogrid-community model.

    Args:
        ds (dataset): downscaled values from topo_scale
        df_pts (dataframe): with metadata of all point of downscaling
        fname_format (str): file name for output
        path (str): path where to export files
        label_map (bool): export cluster_label map
        da_label (dataarray):(optional) dataarray containing label value for each pixel of the DEM
    
    """
    # Add logic to save maps of cluster labels in case of usage of topo_sub
    if label_map:
        if da_label is None:
            print('ERROR: no cluster label (da_label) provided')
            sys.exit()
        else:
            da_label.to_netcdf(path + 'cluster_labels_map.nc')


    n_digits = len(str(ds.point_id.values.max()))
    for pt in ds.point_id.values:
        foutput = path + fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
        fo = xr.Dataset()
        fo['time'] = ds_pt.time
        fo['Tair'] = ('time', ds_pt.t.values - 273.15)
        fo['q'] = ('time', ds_pt.q.values)
        fo['wind'] = ('time', ds_pt.ws.values)
        fo['Sin'] = ('time', ds_pt.SW.values)
        fo['Lin'] = ('time', ds_pt.LW.values)
        fo['p'] = ('time', ds_pt.p.values)
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values, method=snow_partition_method)
        fo['rainfall'], fo['snowfall'] = ('time', rain * 24), ('time', snow * 24)  # convert from mm/hr to mm/day

        fo.Tair.attrs = {'units':'C', 'standard_name':'Tair', 'long_name':'Near Surface Air Temperature', '_FillValue': -9999999.0}
        fo.q.attrs = {'units':'kg/kg', 'standard_name':'q', 'long_name':'Near Surface Specific Humidity', '_FillValue': -9999999.0}
        fo.wind.attrs = {'units':'m/s', 'standard_name':'wind', 'long_name':'Wind Speed', '_FillValue': -9999999.0}
        fo.rainfall.attrs = {'units':'mm/day', 'standard_name':'rainfall', 'long_name':'Rainfall Rate', '_FillValue': -9999999.0}
        fo.snowfall.attrs = {'units':'mm/day', 'standard_name':'snowfall', 'long_name':'Snowfall Rate', '_FillValue': -9999999.0}
        fo.Sin.attrs = {'units':'W/m2', 'standard_name':'Sin', 'long_name':'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
        fo.Lin.attrs = {'units':'W/m2', 'standard_name':'Lin', 'long_name':'Surface Incident Longtwave Radiation', '_FillValue': -9999999.0}
        fo.p.attrs = {'units':'Pa', 'standard_name':'p', 'long_name':'Surface Pressure', '_FillValue': -9999999.0}

        fo.attrs = {'title':'Forcing for Cryogrid Community model',
                    'source': 'Data from {} downscaled with TopoPyScale'.format(climate_dataset_name),
                    'creator_name':'Dataset created by {}'.format(project_author),
                    'date_created':dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
        encod_dict = {}
        for var in list(fo.keys()):
            scale_factor, add_offset = compute_scaling_and_offset(fo[var], n=10)
            encod_dict.update({var:{"zlib": True,
                                   "complevel": 9,
                                   'dtype':'int16',
                                   'scale_factor':scale_factor,
                                   'add_offset':add_offset}})
        fo['latitude'] = df_pts.latitude.iloc[pt]
        fo['longitude'] = df_pts.longitude.iloc[pt]
        fo['elevation'] = df_pts.elevation.iloc[pt]
        fo.latitude.attrs = {'units':'deg', 'standard_name':'latitude'}
        fo.longitude.attrs = {'units':'deg', 'standard_name':'longitude'}
        fo.elevation.attrs = {'units':'m', 'standard_name':'elevation'}
        fo.to_netcdf(foutput, encoding=encod_dict)
        print('---> File {} saved'.format(foutput))


def to_fsm(ds, fname_format='FSM_pt_*.tx', snow_partition_method='continuous'):
    """
    Function to export data for FSM.

    Args:
        ds (dataset): downscaled_pts,
        df_pts (dataframe): toposub.df_centroids,
        fname_format (str pattern): output format of filename
        snow_partition_method (str): snow/rain partitioning method: default 'jennings2018_trivariate'

    format is a text file with the following columns
    year month  day   hour  SW      LW      Sf         Rf     Ta  RH   Ua    Ps
    (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa)

    See README.md file from FSM source code for further details

    TODO: 
    - Check unit DONE jf
    - Check format is compatible with compiled model DONE jf
    - ensure ds.point_id.values always
    """

    #n_digits = len(str(ds.point_id.values.max()))
    # always set this as 3  simplifies parsing files later on
    n_digits = 3

    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['month']  = pd.to_datetime(ds_pt.time.values).month
        df['day']  = pd.to_datetime(ds_pt.time.values).day
        df['hr']  = pd.to_datetime(ds_pt.time.values).hour
        df['SW'] = ds_pt.SW.values
        df['LW'] = ds_pt.LW.values
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values, method=snow_partition_method)
        df['snowfall'] = snow / 3600
        df['rainfall'] = rain / 3600
        df['Tair'] = np.round(ds_pt.t.values, 2)
        df['RH'] = rh * 100
        df['speed'] = ds_pt.ws.values
        df['p'] = ds_pt.p.values

        df.to_csv(foutput, index=False, header=False, sep=' ')
        print('---> File {} saved'.format(foutput))


def to_micromet_single_station(ds,
                               df_pts,
                               fname_format='Snowmodel_pt_*.csv',
                               na_values=-9999,
                               headers=False):
    """
    Function to export TopoScale output in the format for Listn's Snowmodel (using Micromet). One CSV file per point_id

    Args:
        ds (dataset): TopoPyScale xarray dataset, downscaled product
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_id is inserted where * is
        na_values (int): na_value default
        headers (bool): add headers to file


    Example of the compatible format for micromet (headers should be removed to run the model:

     year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip
    (yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)

     2002   10    1   12.00    101   426340.0  4411238.0  3598.0     0.92    57.77     4.80   238.29 -9999.00
     2002   10    2   12.00    101   426340.0  4411238.0  3598.0    -3.02    78.77 -9999.00 -9999.00 -9999.00
     2002   10    8   12.00    101   426340.0  4411238.0  3598.0    -5.02    88.77 -9999.00 -9999.00 -9999.00
    """

    n_digits = len(str(ds.point_id.values.max()))
    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['mo']  = pd.to_datetime(ds_pt.time.values).month
        df['dy']  = pd.to_datetime(ds_pt.time.values).day
        df['hr']  = pd.to_datetime(ds_pt.time.values).hour
        df['stn_id'] = pt
        df['easting'] = np.round(df_pts.x.iloc[pt], 2)
        df['northing'] = np.round(df_pts.y.iloc[pt], 2)
        df['elevation'] = np.round(df_pts.elevation.iloc[pt], 2)
        df['Tair'] = np.round(ds_pt.t.values - 273.15, 2)
        df['RH'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values) * 100
        df['speed'] = ds_pt.ws.values
        df['dir'] = np.round(np.rad2deg(ds_pt.wd.values), 1)
        df['precip'] = ds_pt.tp.values

        df.to_csv(foutput, index=False, header=False, sep=' ', na_rep=na_values)
        if headers:
            header = 'year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip\n(yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)\n'
            with open(foutput, 'r+') as f:
                f_data = f.read()
                f.seek(0,0)
                f.write(header + '\n' + f_data)
        print('---> File {} saved'.format(foutput))


def to_crocus(ds,
              df_pts,
              fname_format='CROCUS_pt_*.nc',
              scale_precip=1,
              climate_dataset_name='ERA5',
              project_author='S. Filhol',
              snow_partition_method='continuous'):
    """
    Functiont to export toposcale output to CROCUS netcdf format. Generates one file per point_id

    Args:
        ds (dataset): Toposcale downscaled dataset.
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_id is inserted where * is
        scale_precip (float): scaling factor to apply on precipitation. Default is 1
        climate_dataset_name (str): name of original climate dataset. Default 'ERA5',
        project_author (str): name of project author(s)
        snow_partition_method (str): snow/rain partitioning method: default 'jennings2018_trivariate'

    """
    # create one file per point_id
    n_digits = len(str(ds.point_id.values.max()))
    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
        df = pd.DataFrame()
        df['time'] = ds_pt.time.values
        df['Tair'] = ds_pt.t.values
        df['DIR_SWdown'] = ds_pt.SW.values
        df['LWdown'] = ds_pt.LW.values
        df['PSurf'] = ds_pt.p.values
        df['HUMREL'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)*100
        df['xwind'] = ds_pt.u.values
        df['ywind'] = ds_pt.v.values
        df['precip'] = ds_pt.tp.values / 3600 * scale_precip  # convert from mm/hr to mm/s
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        df['Rainf'], df['Snowf'] = mu.partition_snow(df.precip, ds_pt.t.values, rh, ds_pt.p.values, method=snow_partition_method)

        # Derive variables: Q- humidity, WD - wind direction (deg), and WS
        df['Qair'] = ds_pt.q.values
        df['Wind'], df['Wind_DIR'] = ds_pt.ws.values, np.rad2deg(ds_pt.wd.values)
        df['NEB'] = np.zeros(df.shape[0])
        df['CO2air'] = np.zeros(df.shape[0])
        df['SCA_SWdown'] = np.zeros(df.shape[0])

        df.drop(columns=['precip','xwind','ywind'], inplace=True)
        ds_pt.close()

        #df = df.loc[(df.time>=pd.Timestamp(start)) & (df.time<pd.Timestamp(end)+pd.Timedelta('1D'))]

        # Build netcdf forcing file for CROCUS
        fo = xr.Dataset.from_dataframe(df.set_index('time'))
        fo = df.set_index(['time']).to_xarray()
        fo.Tair.attrs = {'units':'K', 'standard_name':'Tair', 'long_name':'Near Surface Air Temperature', '_FillValue': -9999999.0}
        fo.Qair.attrs = {'units':'kg/kg', 'standard_name':'Qair', 'long_name':'Near Surface Specific Humidity', '_FillValue': -9999999.0}
        fo.HUMREL.attrs = {'units':'%', 'standard_name':'HUMREL', 'long_name':'Relative Humidity', '_FillValue': -9999999.0}
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


        fo.attrs = {'title':'Forcing for SURFEX CROCUS',
                    'source': 'data from {} downscaled with TopoPyScale ready for CROCUS'.format(climate_dataset_name),
                    'creator_name':'Dataset created by {}'.format(project_author),
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
        print('---> File {} saved'.format(foutput))

def to_snowpack(ds, fname_format='smet_pt_*.tx'):
    """
    Function to export data for snowpack model as smet.
    https://models.slf.ch/docserver/meteoio/SMET_specifications.pdf

    Args:
        ds (dataset): downscaled_pts,
        df_pts (dataframe): toposub.df_centroids,
        fname_format (str pattern): output format of filename

    format is a text file with the following columns (coulumns can vary)

    SMET 1.1 ASCII
    [HEADER]
    station_id       = meteoc1
    station_name     = WFJ2
    latitude         = 46.829650
    longitude        = 9.809328
    altitude         = 2539.0
    easting          = 780851.861845
    northing         = 189232.420554
    epsg             = 21781
    nodata           = -999
    tz               = 0
    plot_unit        = time K - m/s ° W/m2 W/m2 kg/m2 -
    plot_description = time air_temperature relative_humidity wind_velocity wind_direction incoming_short_wave_radiation incoming_long_wave_radiation water_equivalent_precipitation_sum -
    plot_color       = 0x000000 0x8324A4 0x50CBDB 0x297E24 0x64DD78 0xF9CA25 0xD99521 0x2431A4 0xA0A0A0
    plot_min         = -999 253.15 0 0 0 0 150 0 -999
    plot_max         = -999 283.15 1 30 360 1400 400 20 -999
    fields           = timestamp TA RH VW DW ISWR ILWR PSUM PINT
    [DATA]
    2014-09-01T00:00:00   271.44   1.000    5.9   342     67    304  0.000    0.495
    2014-09-01T01:00:00   271.50   0.973    6.7   343    128    300  0.166    0.663
    2014-09-01T02:00:00   271.46   0.968    7.4   343    244    286  0.197    0.788
    2014-09-01T03:00:00   271.41   0.975    7.6   345    432    273  0.156    0.626
    2014-09-01T04:00:00   271.57   0.959    7.8   347    639    249  0.115    0.462
    2014-09-01T05:00:00   271.52   0.965    8.2   350    632    261  0.081    0.323

    """

    #n_digits = len(str(ds.point_id.values.max()))
    # always set this as 3  simplifies parsing files later on
    n_digits = 3

    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
        df = pd.DataFrame()
        df['timestamp'] = ds_pt.time
        df['Tair'] = np.round(ds_pt.t.values, 2)
        df['RH'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        df['VW'] = ds_pt.ws.values
        df['DW'] = ds_pt.wd.values
        df['SW'] = ds_pt.SW.values
        df['LW'] = ds_pt.LW.values
        df['PSUM'] = ds_pt.tp.values

        df.to_csv(foutput, index=False, header=False, sep=' ')
        print('---> File {} saved'.format(foutput))

        # if headers:
        #     header = 'year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip\n(yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)\n'
        #     with open(foutput, 'r+') as f:
        #         f_data = f.read()
        #         f.seek(0,0)
        #         f.write(header + '\n' + f_data)
        # print('---> File {} saved'.format(foutput))

def to_geotop(ds, fname_format='geotop_pt_*.txt'):
    """
    Function to export data for snowpack model as smet.
    https://models.slf.ch/docserver/meteoio/SMET_specifications.pdf

    Args:
        ds (dataset): downscaled_pts,
        df_pts (dataframe): toposub.df_centroids,
        fname_format (str pattern): output format of filename


    Date format = DD/MM/YYYY hh:mm
    Air temperature = Degree Celsius
    Relative Humidity = %
    Radiations components such SWin, SWout, LWin = W/m2
    Precipitation = mm/hr
    Wind speed = m/s
    Wind direction = degree
    Air pressure = mbar

    The format of the file is in *.txt format.



    format is a text file with the following columns (coulumns can vary)

    Date,AirT,WindS,WindDr,RelHum,Swglob,Swout,Lwin,Lwout,Iprec,AirPressure
    01/09/2015 00:00,1.5,0.7,165,59,0,0,223.9,315.5,0,581.02897
    01/09/2015 01:00,1.2,0.6,270,59,0,0,261.8,319,0,581.02897
    01/09/2015 02:00,0.8,0.5,152,61,0,0,229.9,312.2,0,581.02897
    01/09/2015 03:00,0.8,0.5,270,60,0,0,226.1,310.8,0,581.02897
    01/09/2015 04:00,0.3,0.8,68,60,0,0,215.3,309.6,0,581.02897
    01/09/2015 05:00,0.2,0.6,270,62,0,0,230.2,309.1,0,581.02897
    01/09/2015 06:00,0.3,0.6,35,62,0,0,222.8,306.7,0,581.02897
    01/09/2015 07:00,-0.2,0.3,270,65,0,0,210,305.5,0,581.02897
    01/09/2015 08:00,0,0.6,52,59,114,23,218.3,312.3,0,581.02897
    01/09/2015 09:00,1.9,1.5,176,57,173,35,220.2,322.8,0,581.02897
    01/09/2015 10:00,3.4,1.9,183,47,331,67,245.8,372.6,0,581.02897


    """

    #n_digits = len(str(ds.point_id.values.max()))
    # always set this as 3  simplifies parsing files later on
    n_digits = 3

    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
        df = pd.DataFrame()
        dt = pd.to_datetime(ds_pt.time.values)
        df['Date'] = dt.strftime( "%d/%m/%Y %H:%M")
        # df['Date'] = str(pd.to_datetime(ds_pt.time.values).day) +   "/" + str(pd.to_datetime(ds_pt.time.values).month) +  "/" + str(pd.to_datetime(ds_pt.time.values).year) +  " " + str(pd.to_datetime(ds_pt.time.values).hour) + ":" + str(pd.to_datetime(ds_pt.time.values).minute)
        df['AirT'] = np.round(ds_pt.t.values-273.15, 2)
        df['RelHum'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values) * 100
        df['WindS'] = ds_pt.ws.values
        df['WIndDr'] = ds_pt.wd.values * 180/np.pi
        df['Swglob'] = ds_pt.SW.values
        df['Lwin'] = ds_pt.LW.values
        df['Iprec'] = ds_pt.tp.values

        df.to_csv(foutput, index=False, header=False, sep=',', quoting=csv.QUOTE_NONE)


        header = 'Date, AirT, RelHum, WindS, WindDr, Swglob, Lwin, Iprec'
        with open(foutput, 'r+') as f:
            f_data = f.read()
            f.seek(0,0)
            f.write(header + '\n' + f_data)
        print('---> File {} saved'.format(foutput))



        # if headers:
        #     header = 'year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip\n(yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)\n'
        #     with open(foutput, 'r+') as f:
        #         f_data = f.read()
        #         f.seek(0,0)
        #         f.write(header + '\n' + f_data)
        # print('---> File {} saved'.format(foutput))