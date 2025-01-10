"""
Functions to export topo_scale output to formats compatible with existing models (e.g. CROCUS, Cryogrid, Snowmodel, ...)

S. Filhol, December 2021

TODO;
- SPHY forcing (grids)
"""
import csv
import datetime as dt
import pdb
import sys, os

import numpy as np
import pandas as pd
import xarray as xr

from TopoPyScale import meteo_util as mu
from TopoPyScale import topo_utils as tu
from TopoPyScale import topo_param as tp
from TopoPyScale import topo_sub as ts

from pathlib import Path


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


def to_netcdf(ds, fname='output.nc', variables=None, complevel=9):
    """
    Generic function to save a datatset to one single compressed netcdf file

    Args:
        fname (str): name of export file
        variables (list str): list of variable to export. Default exports all variables
        complevel (int): Compression level. 1-9
    """

    encod_dict = {}
    if variables is None:
        variables = list(ds.keys())

    for var in variables:
        scale_factor, add_offset = compute_scaling_and_offset(ds[var], n=10)
        if str(ds[var].dtype)[:3] == 'int' or var == 'cluster_labels' or var == 'point_name':
            encod_dict.update({var: {
                'dtype': ds[var].dtype}})
        else:
            encod_dict.update({var: {"zlib": True,
                                     "complevel": complevel,
                                     'dtype': 'int16',
                                     'scale_factor': scale_factor,
                                     'add_offset': add_offset}})
    ds[variables].to_netcdf(fname, encoding=encod_dict, engine='h5netcdf')

    print(f'---> File {fname} saved')


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
    ver_dict = tu.get_versionning()

    if da_label is not None:
        da_label.to_netcdf(str(path + fname_labels))

    da_label.to_netcdf(str(path / fname_labels))

    fo = xr.Dataset()
    fo['tair'] = ds['t'].T
    fo['q'] = ds['q'].T
    fo['p'] = ds['p'].T
    fo['lw'] = ds['LW'].T
    fo['sw'] = ds['SW'].T
    fo['ws'] = ds['ws'].T
    fo['tp'] = ds['tp'].T
    fo['rh'] = (('time', 'point_name'), mu.q_2_rh(fo.tair.values, fo.p.values, fo.q.values))
    fo['precip'] = (('time', 'point_name'), fo.tp.values / 3600)  # convert from mm/hr to mm/s
    fo = fo.drop_vars(['q', 'tp'])
    fo = fo[['tair', 'rh', 'p', 'lw', 'sw', 'ws', 'precip']].expand_dims({'dummy': 1}, axis=2)

    fo.tair.attrs = {'units': 'C', 'standard_name': 'tair', 'long_name': 'Near Surface Air Temperature',
                     '_FillValue': -9999999.0}
    fo.rh.attrs = {'units': 'kg/kg', 'standard_name': 'rh', 'long_name': 'Near Surface Relative Humidity',
                   '_FillValue': -9999999.0}
    fo.ws.attrs = {'units': 'm/s', 'standard_name': 'ws', 'long_name': 'Wind Speed', '_FillValue': -9999999.0}
    fo.precip.attrs = {'units': 'mm/s', 'standard_name': 'precip', 'long_name': 'Precipitation',
                       '_FillValue': -9999999.0}
    fo.sw.attrs = {'units': 'W/m^2', 'standard_name': 'sw', 'long_name': 'Surface Incident Direct Shortwave Radiation',
                   '_FillValue': -9999999.0}
    fo.lw.attrs = {'units': 'W/m^2', 'standard_name': 'lw', 'long_name': 'Surface Incident Longtwave Radiation',
                   '_FillValue': -9999999.0}
    fo.p.attrs = {'units': 'Pa', 'standard_name': 'p', 'long_name': 'Surface Pressure', '_FillValue': -9999999.0}

    fo.attrs = {'title': 'Forcing for MuSa assimilation scheme',
                'source': 'Data from {} downscaled with TopoPyScale'.format(climate_dataset_name),
                'creator_name': 'Dataset created by {}'.format(project_authors),
                'package_version': ver_dict.get('package_version'),
                'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                'git_commit': ver_dict.get('git_commit'),
                'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    encod_dict = {}
    for var in list(fo.keys()):
        scale_factor, add_offset = compute_scaling_and_offset(fo[var], n=10)
        encod_dict.update({var: {"zlib": True,
                                 "complevel": 9,
                                 'dtype': 'int16',
                                 'scale_factor': scale_factor,
                                 'add_offset': add_offset}})

    fo['latitude'] = (('point_name'), df_pts.latitude.values)
    fo['longitude'] = (('point_name'), df_pts.longitude.values)
    fo['elevation'] = (('point_name'), df_pts.elevation.values)
    fo.latitude.attrs = {'units': 'deg', 'standard_name': 'latitude', 'long_name': 'Cluster latitude'}
    fo.longitude.attrs = {'units': 'deg', 'standard_name': 'longitude', 'long_name': 'Cluster longitude'}
    fo.elevation.attrs = {'units': 'm', 'standard_name': 'elevation', 'long_name': 'Cluster elevation'}

    fo.to_netcdf(str(path / fname_met), encoding=encod_dict)
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

    ver_dict = tu.get_versionning()
    if label_map:
        if da_label is None:
            print('ERROR: no cluster label (da_label) provided')
            sys.exit()
        else:
            da_label.to_netcdf(str(path / 'cluster_labels_map.nc'))

    n_digits = len(str(len(ds.point_name))) + 1

    for pt_ind, pt_name in enumerate(ds.point_name.values):
        foutput = str(path) + fname_format.split('*')[0] + str(pt_ind).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_name=pt_name).copy()
        fo = xr.Dataset()
        fo['time'] = ds_pt.time
        fo['Tair'] = ('time', ds_pt.t.values - 273.15)
        fo['q'] = ('time', ds_pt.q.values)
        fo['wind'] = ('time', ds_pt.ws.values)
        fo['Sin'] = ('time', ds_pt.SW.values)
        fo['Lin'] = ('time', ds_pt.LW.values)
        fo['p'] = ('time', ds_pt.p.values)
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values,
                                       method=snow_partition_method)
        fo['rainfall'], fo['snowfall'] = ('time', rain * 24), ('time', snow * 24)  # convert from mm/hr to mm/day

        fo.Tair.attrs = {'units': '°C', 'standard_name': 'Tair', 'long_name': 'Near Surface Air Temperature',
                         '_FillValue': -9999999.0}
        fo.q.attrs = {'units': 'kg/kg', 'standard_name': 'q', 'long_name': 'Near Surface Specific Humidity',
                      '_FillValue': -9999999.0}
        fo.wind.attrs = {'units': 'm/s', 'standard_name': 'wind', 'long_name': 'Wind Speed', '_FillValue': -9999999.0}
        fo.rainfall.attrs = {'units': 'mm/d', 'standard_name': 'rainfall', 'long_name': 'Rainfall Rate',
                             '_FillValue': -9999999.0}
        fo.snowfall.attrs = {'units': 'mm/d', 'standard_name': 'snowfall', 'long_name': 'Snowfall Rate',
                             '_FillValue': -9999999.0}
        fo.Sin.attrs = {'units': 'W/m^2', 'standard_name': 'Sin',
                        'long_name': 'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
        fo.Lin.attrs = {'units': 'W/m^2', 'standard_name': 'Lin', 'long_name': 'Surface Incident Longtwave Radiation',
                        '_FillValue': -9999999.0}
        fo.p.attrs = {'units': 'Pa', 'standard_name': 'p', 'long_name': 'Surface Pressure', '_FillValue': -9999999.0}

        fo.attrs = {'title': 'Forcing for Cryogrid Community model',
                    'source': 'Data from {} downscaled with TopoPyScale'.format(climate_dataset_name),
                    'creator_name': 'Dataset created by {}'.format(project_author),
                    'package_version': ver_dict.get('package_version'),
                    'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                    'git_commit': ver_dict.get('git_commit'),
                    'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
        encod_dict = {}
        for var in list(fo.keys()):
            scale_factor, add_offset = compute_scaling_and_offset(fo[var], n=10)
            encod_dict.update({var: {"zlib": True,
                                     "complevel": 9,
                                     'dtype': 'int16',
                                     'scale_factor': scale_factor,
                                     'add_offset': add_offset}})
        fo['latitude'] = df_pts.latitude.iloc[pt_ind]
        fo['longitude'] = df_pts.longitude.iloc[pt_ind]
        fo['elevation'] = df_pts.elevation.iloc[pt_ind]
        fo.latitude.attrs = {'units': 'deg', 'standard_name': 'latitude'}
        fo.longitude.attrs = {'units': 'deg', 'standard_name': 'longitude'}
        fo.elevation.attrs = {'units': 'm', 'standard_name': 'elevation'}
        fo.to_netcdf(foutput, encoding=encod_dict)
        print('---> File {} saved'.format(foutput))


def to_fsm2(ds_down,
                fsm_param,
                simulation_path='fsm_sim',
                fname_format='fsm_',
                namelist_options=None,
                n_digits=None,
                snow_partition_method='continuous',
                cluster_method=True,
                epsg_ds_param=2056,
                temperature_correction=0,
                forest_param_scaler={'hcan':100, 'lai':100}):
    """
    Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd).
    FSM2oshd includes canopy structures processes
    one simulation consists of 2 driving file:
       - met.txt with variables:
           year, month, day, hour, SWdir, SWdif, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt
       - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10

    Args:
        ds_down:  Downscaled weather variable dataset
        fsm_param:  terrain and canopy parameter dataset
        df_centroids:  cluster centroids statistics (terrain + canopy)
        ds_tvt (dataset, int, float, or str):  transmisivity. Can be a dataset, a constant or 'svf_for'
        simulation_path (str): 'fsm_sim'
        fname_format (str):'fsm_'
        namelist_options (dict): {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]}
        n_digits (int): Number of digits (for filename system)
        snow_partition_method (str): method for snow partitioning. Default: 'continuous'
        cluster_method (bool): boolean to be True is using cluster appraoch
        epsg_ds_param (int): epsg code of ds_parma: example: 2056

    """

    print(" \n##### WARNING ######\n")
    print(" Method is not fully implemented")
    print(" \n#####         ######\n")


    def write_fsm2_compilation_nc(file_compil_nc='compil_nc.sh',
                                    ALBEDO=2, 
                                    CANMOD=2,  
                                    CANRAD=2,  
                                    CONDCT=1,
                                    DENSITY=2,
                                    EXCHNG=1,
                                    HYDROL=2,
                                    SGRAIN=2, 
                                    SNFRAC=2,
                                    DRIV1D=1):       
        compil_part1 = f"""
#!/bin/bash

FC=gfortran
cd src

cat > OPTS.h << EOF
/* Process options                                  : Possible values */
#define ALBEDO {ALBEDO}   /* snow albedo                   : 1, 2            */
#define CANMOD {CANMOD}   /* forest canopy layers          : 1, 2            */
#define CANRAD {CANRAD}   /* canopy radiative properties   : 1, 2            */
#define CONDCT {CONDCT}   /* snow thermal conductivity     : 0, 1            */
#define DENSTY {DENSITY}  /* snow density                  : 0, 1, 2         */
#define EXCHNG {EXCHNG}   /* turbulent exchange            : 0, 1            */
#define HYDROL {HYDROL}   /* snow hydraulics               : 0, 1, 2         */
#define SGRAIN {SGRAIN}   /* snow grain growth             : 1, 2            */
#define SNFRAC {SNFRAC}   /* snow cover fraction           : 1, 2            */
/* Driving data options                             : Possible values */
#define DRIV1D {DRIV1D}   /* 1D driving data format        : 1, 2            */
#define SETPAR 1   /* parameter inputs              : 0, 1            */
#define SWPART 0   /* SW radiation partition        : 0, 1            */
#define ZOFFST 1   /* measurement height offset     : 0, 1            */
/* Output options                                   : Possible values */
#define PROFNC 1   /* netCDF output                 : 0, 1            */
EOF

"""

        compil_part2 = """
files=(FSM2_MODULES.F90 FSM2.F90 FSM2_DRIVE.F90 FSM2_MAP.F90          \
       FSM2_PARAMS.F90 FSM2_PREPNC.F90 FSM2_TIMESTEP.F90              \
       FSM2_WRITENC.F90                                               \
       CANOPY.F90 INTERCEPT.F90 LUDCMP.F90 PSIMH.F90 QSAT.F90         \
       SNOW.F90 SOIL.F90 SOLARPOS.F90 SRFEBAL.F90 SWRAD.F90           \
       THERMAL.F90 TRIDIAG.F90)
> FSM2_temp.f90
for file in ${files[*]}
do
  $FC -cpp -E -P -o out $file
  cat --squeeze-blank out >> FSM2_temp.f90
done

$FC -O3 -c -I/usr/lib64/gfortran/modules FSM2_temp.f90
$FC -o FSM2 FSM2_temp.o -L/usr/lib64 -lnetcdff
mv FSM2 ../FSM2

rm out
rm *.mod
rm *.o
rm FSM2_temp.f90
cd ..
"""

        with open(file_compil_nc, "w") as file:
                file.write(compil_part1 + compil_part2)



    def write_fsm2_namelist(row,
                                pt_ind,
                                n_digits,
                                fname_format='fsm_sim/fsm_',
                                namelist_param=None,
                                scaler={ 'hcan':100, 'lai':100, 'svf_for':1}):
        # Function to write namelist file (.nam) for each point where to run FSM.

        file_namelist = str(fname_format) + f'_{mode}_' + str(pt_ind).zfill(n_digits) + '.nam'
        file_met = str(fname_format) + '_met_' + str(pt_ind).zfill(n_digits) + '.txt'
        file_output = str(fname_format) + f'_outputs_{mode}_' + str(pt_ind).zfill(n_digits) + '.nc'

        # populate namelist parameters
        max_sd = namelist_param.get('max_sd')
        z_snow = namelist_param.get('z_snow')
        z_soil = namelist_param.get('z_soil')
        z_tair = namelist_param.get('z_tair')
        z_wind = namelist_param.get('z_wind')
        soil_moist_init = namelist_param.get('soil_moist_init')
        soil_temp_init = namelist_param.get('soil_temp_init')
        latitude = namelist_param.get('latitude')


        # Values compatible with 'forest' mode
        albedo_snowfree = 0.2
        hcan = row.hcan/scaler.get('hcan')
        lai = row.lai5/scaler.get('lai')
        svf_for = row.lai5/scaler.get('svf_for')


        if os.path.exists(file_met):
            nlst = f"""
&params
  asmn = {namelist_param.get('asmn')}
  asmx = {namelist_param.get('asmx')}
  eta0 = {namelist_param.get('eta0')}
  hfsn = {namelist_param.get('hfsn')}
  kfix = {namelist_param.get('kfix')}
  rcld = {namelist_param.get('rcld')}
  rfix = {namelist_param.get('rfix')}
  rgr0 = {namelist_param.get('rgr0')}
  rhof = {namelist_param.get('rhof')}
  rhow = {namelist_param.get('rhow')}
  rmlt = {namelist_param.get('rmlt')}
  Salb = {namelist_param.get('Salb')}
  snda = {namelist_param.get('snda')}
  Talb = {namelist_param.get('Talb')}
  tcld = {namelist_param.get('tcld')}
  tmlt = {namelist_param.get('tmlt')}
  trho = {namelist_param.get('trho')}
  Wirr = {namelist_param.get('Wirr')}
  z0sn = {namelist_param.get('z0sn')}
  acn0 = {namelist_param.get('acn0')}
  acns = {namelist_param.get('acns')}
  avg0 = {namelist_param.get('avg0')}
  avgs = {namelist_param.get('avgs')}
  cvai = {namelist_param.get('cvai')}
  gsnf = {namelist_param.get('gsnf')}
  hbas = {namelist_param.get('hbas')}
  kext = {namelist_param.get('kext')}
  rveg = {namelist_param.get('rveg')}
  svai = {namelist_param.get('svai')}
  tunl = {namelist_param.get('tunl')}
  wcan = {namelist_param.get('wcan')}
  fcly = {namelist_param.get('fcly')}
  fsnd = {namelist_param.get('fsnd')}
  gsat = {namelist_param.get('gsat')}
  z0sf = {namelist_param.get('z0sf')}

/
&gridpnts
  Ncols = 1
  Nrows = 1
  Nsmax = {len(z_snow)}
  Nsoil = {len(z_soil)}

/
&gridlevs
  Dzsnow = {', '.join(str(i) for i in z_snow)}          ! Minimum snow layer thicknesses (m)
  Dzsoil = {', '.join(str(i) for i in z_soil)}          ! Soil layer thicknesses (m)
  fvg1 = 0.5                                            ! Fraction of vegetation in upper layer (CANMOD=2)
  zsub = 1.5                                            ! Subcanopy wind speed diagnostic height
/
&drive
  met_file = {file_met}
  dt = 3600                     ! Timestep
  zT = {z_tair}                 ! Temperature and humidity measurement height
  zU = {z_wind}                 ! Wind speed measurement height
  lat = {latitude}
  noon = {solar_noon}                     ! Time of solar noon (for SWPART=1)
/
&veg
  ab0 = 0.2
  fsky = {svf_for}
  vegh = {hcan}
  VAI =  {lai}
/
&initial
  fsat = {soil_moist_init}
  Tprf = {soil_temp_init}
/
&outputs
  runid = {path_to_store}
  dump_file = {file_output}
/
  """

            with open(file_namelist, "w") as nlst_file:
                nlst_file.write(nlst)
        else:
            print(f'ERROR: met_file: {file_met} not found')
            return


    def write_fsm2_met(ds_pt,
                           pt_name,
                           pt_ind,
                           n_digits,
                           fname_format='fsm_sim/fsm_',
                           temperature_correction=0,
                           DRIV1D=1):
        """
        Function to write meteorological forcing for FSM2


        for DRIV1D=1   (FSM format)

            year month  day   hour  SW  LW  Sf  Rf  Ta  RH   Ua    Ps    
            (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa)

        for DRIV1D=2    (ESM-SnowMIP format)

            year month  day   hour  SW  LW  Sf  Rf  Ta  Qa RH   Ua    Ps    
            (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (kg/kg) (RH 0-100) (m/s) (Pa)

        """
        
        # for storage optimization tvt is stored in percent.


        foutput = str(fname_format) + '_met_' + str(pt_ind).zfill(n_digits) + '.txt'
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['month']  = pd.to_datetime(ds_pt.time.values).month
        df['day']  = pd.to_datetime(ds_pt.time.values).day
        df['hr']  = pd.to_datetime(ds_pt.time.values).hour
        df['SWdir'] = np.round(ds_pt.SW_direct.values,2)              # change to direct SW
        df['SWdif'] = np.round(ds_pt.SW_diffuse.values,2)                # change to diffuse SW
        df['LW'] = np.round(ds_pt.LW.values,2)
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values, method=snow_partition_method)
        df['snowfall'] = np.round(snow, 5)
        df['rainfall'] = np.round(rain, 5)
        df['Tair'] = np.round(ds_pt.t.values, 2) + temperature_correction
        if DRIV1D == 2:
            df['Qa'] = np.round(ds_pt.q.values, 2)
        df['RH'] = np.round(rh * 100,2)
        df['speed'] = np.round(ds_pt.ws.values,2)
        df['p'] = np.round(ds_pt.p.values,2)

        df.to_csv(foutput, index=False, header=False, sep=' ')
        print(f'---> Met file {foutput} saved')


    # 0. Add print statements about the need of data currently computed extrenally to TopoPyScale.
    # 1. Convert Clare's canopy output to FSM2oshd parametrization
    # 2. Extract cluster canopy and forest cover weights from ds_param_canopy
    # 3. loop through clusters and write met_file and namelist files
    # - 2 sets of FSM run, one with forest, another one without. Need to combine both output into a final value for the cluster centroid

    # ----- unpack and overwrite namelist_options ----
    # default namelist_param values
    namelist_param = { 'max_sd':4,
                       'z_snow':[0.1, 0.2, 0.4],
                       'z_soil':[0.1, 0.2, 0.4, 0.8],
                       'z_tair':2,
                       'z_wind':10,
                       'latitude':42,
                       'soil_moist_init':0.5,
                       'soil_temp_init':285,
                       'fsat':0.5,
                       'Tprf':285,
                       'asmn':0.1,
                       'asmx':0.4,
                       'eta0':0.21,
                       'hfsn':0.6,
                       'kfix':3.6e4,
                       'rcld':300,
                       'rfix':300,
                       'rgr0':5e-5,
                       'rhof':100,
                       'rhow':300,
                       'rmlt':500,
                       'Salb':10,
                       'snda':2.8e-6,
                       'Talb':-2,
                       'tcld':1000,
                       'tmlt':100,
                       'trho':200,
                       'Wirr':0.03,
                       'z0sn':0.01,
                       'acn0':0.1,
                       'acns':0.4,
                       'avg0':0.21,
                       'avgs':0.6,
                       'cvai':3.6e4,
                       'gsnf':0.01,
                       'hbas':2,
                       'kext':0.5,
                       'rveg':20,
                       'svai':4.4,
                       'tunl':240,
                       'wcan':2.5,
                       'fcly':0.3,
                       'fsnd':0.6,
                       'gsat':0.01,
                       'z0sf':0.1
                       }

    if namelist_options is not None:
        namelist_param.update(namelist_options)

    if n_digits is None:
        n_digits = len(str(len(ds_down.point_name))) + 1

    if cluster_method:
        # extract FSM forest parameters for each clusters
        # Aggregate forest parameters only to fores area

        dx = np.abs(np.diff(fsm_param.x)[0])
        dy = np.abs(np.diff(fsm_param.y)[0])
        df_forest = tu.ds_to_indexed_dataframe(fsm_param.drop(['point_name', 'cluster_labels'])).groupby('point_ind').mean()
        df_forest['cluster_total_area'] = fsm_param.elevation.groupby(fsm_param.point_ind).count().values * dx * dy
        df_forest['proportion_with_forest'] = fsm_param.isfor.groupby(fsm_param.point_ind).mean().values
        df_forest['cluster_domain_size'] = np.sqrt(df_forest.cluster_total_area)
        #df_forest['cluster_domain_size'] = np.sqrt(fsm_param.drop('point_name').groupby(fsm_param.point_name).count().to_dataframe().LAI5)*dx
        df_forest['forest_cover'] = fsm_param.forcov.groupby(fsm_param.point_ind).mean().values
        df_forest.forest_cover.loc[df_forest.proportion_with_forest<0.01] = 0
        df_forest['lon'], df_forest['lat'] = tp.convert_epsg_pts(df_forest.x, df_forest.y, epsg_ds_param, 4326)

    else:
        pass

    p = Path(simulation_path)
    # rename variable columns to match namelist functino varnames
    new_name = {'LAI5':'lai', 'svf_for':'vfhp', 'CH5':'hcan'}
    df_forest = df_forest.rename(columns=new_name)
    print(df_forest)
 
    # ----- Loop through all points-------
    # NOTE: eventually this for loop could be parallelized to several cores -----
    for pt_ind, pt_name in enumerate(ds_down.point_name.values):

        ds_pt = ds_down.sel(point_name=pt_name).copy()

        row_forest = df_forest.iloc[pt_ind]
        write_fsm2_compilation_nc(file_compil_nc='compil_nc.sh',
                                    ALBEDO=2, 
                                    CANMOD=2,  
                                    CANRAD=2,  
                                    CONDCT=1,
                                    DENSITY=2,
                                    EXCHNG=1,
                                    HYDROL=2,
                                    SGRAIN=2, 
                                    SNFRAC=2,
                                    DRIV1D=1)
        write_fsm2_met(ds_pt,
                           n_digits=n_digits,
                           pt_name=pt_name,
                           pt_ind=pt_ind,
                           fname_format=p/fname_format,
                           temperature_correction=temperature_correction)
        write_fsm2_namelist(row_forest,
                                pt_ind=pt_ind,
                                n_digits=n_digits,
                                fname_format=p/fname_format,
                                mode='forest',
                                namelist_param=namelist_param,
                                scaler=forest_param_scaler) # write forest namelist

        if cluster_method:
            row_open = df_forest.iloc[pt_ind]
            write_fsm2oshd_namelist(row_open,
                                    pt_ind=pt_ind,
                                    n_digits=n_digits,
                                    fname_format=p/fname_format,
                                    mode='open',
                                    namelist_param=namelist_param) # write open namelist

        ds_pt = None
        # [ ] add logic to computed weighted average outputs based on forest cover fraction per point.

    df_forest.to_pickle(p / 'df_forest.pckl')
    return



def to_fsm2oshd(ds_down,
                fsm_param,
                ds_tvt,
                simulation_path='fsm_sim',
                fname_format='fsm_',
                namelist_options=None,
                n_digits=None,
                snow_partition_method='continuous',
                cluster_method=True,
                epsg_ds_param=2056,
                temperature_correction=0,
                forest_param_scaler={'vfhp':100, 'fveg':100, 'fves':100, 'hcan':100, 'lai5':100, 'lai50':100}):
    """
    Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd).
    FSM2oshd includes canopy structures processes
    one simulation consists of 2 driving file:
       - met.txt with variables:
           year, month, day, hour, SWdir, SWdif, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt
       - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10

    Args:
        ds_down:  Downscaled weather variable dataset
        fsm_param:  terrain and canopy parameter dataset
        df_centroids:  cluster centroids statistics (terrain + canopy)
        ds_tvt (dataset, int, float, or str):  transmisivity. Can be a dataset, a constant or 'svf_for'
        simulation_path (str): 'fsm_sim'
        fname_format (str):'fsm_'
        namelist_options (dict): {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]}
        n_digits (int): Number of digits (for filename system)
        snow_partition_method (str): method for snow partitioning. Default: 'continuous'
        cluster_method (bool): boolean to be True is using cluster appraoch
        epsg_ds_param (int): epsg code of ds_parma: example: 2056

    """

    def write_fsm2oshd_namelist(row,
                                pt_ind,
                                n_digits,
                                fname_format='fsm_sim/fsm_',
                                mode='forest',
                                namelist_param=None,
                                modconf=None,
                                scaler={'vfhp':100, 'fveg':100, 'fves':100, 'hcan':100, 'lai5':100, 'lai50':100}):
        # Function to write namelist file (.nam) for each point where to run FSM.

        file_namelist = str(fname_format) + f'_{mode}_' + str(pt_ind).zfill(n_digits) + '.nam'
        file_met = str(fname_format) + '_met_' + str(pt_ind).zfill(n_digits) + '.txt'
        file_output = str(fname_format) + f'_outputs_{mode}_' + str(pt_ind).zfill(n_digits) + '.txt'

        if modconf is None:
            modconf = {
                'albedo':2,
                'condct':1,
                'density':3,
                'hydrol':2
            }

        # populate namelist parameters
        precip_multi = namelist_param.get('precip_multi')
        max_sd = namelist_param.get('max_sd')
        z_snow = namelist_param.get('z_snow')
        z_soil = namelist_param.get('z_soil')
        z_tair = namelist_param.get('z_tair')
        z_wind = namelist_param.get('z_wind')
        diag_var_outputs = namelist_param.get('diag_var_outputs')
        state_var_outputs = namelist_param.get('state_var_outputs')


        if mode == 'forest':
            # Values compatible with 'forest' mode
            canmod = 1
            turbulent_exchange = 2
            z_offset = 1
            fveg = row.fveg/scaler.get('fveg')
            hcan = row.hcan/scaler.get('hcan')
            lai5 = row.lai5/scaler.get('lai5')
            vfhp = row.vfhp/scaler.get('vfhp')
            fves = row.fves/scaler.get('fves')

        elif mode == 'open':
            # default values for 'open' mode
            canmod = 0
            turbulent_exchange = 1
            z_offset, fveg, fves, hcan, lai5, vfhp= 0, 0, 0, 0, 0, 1
        else:
            raise ValueError('mode is not available. Must be open or forest')
        if os.path.exists(file_met):
            nlst = f"""
&nam_grid
  NNx = 1,                          ! Grid x-indexing. Not relevant with this implementation
  NNy = 1,                          ! Grid y-indexing. Not relevant with this implementation
  NNsmax = {len(z_snow)},           ! Number snow layers. Default 3-layers
  NNsoil = {len(z_soil)},           ! Number soil layers. Default 4-layers
/
&nam_layers
  DDzsnow = {', '.join(str(i) for i in z_snow)},    ! Minimum thickness to define a new snow layer
  DDzsoil = {', '.join(str(i) for i in z_soil)},    ! soil layer thickness
/
&nam_driving
  zzT = {z_tair},                   ! Height of temperature forcing (2 or 10m)
  zzU = {z_wind},                   ! Height of wind forcing (2 or 10m)
  met_file = '{file_met}',            ! name of met file associatied to this namelist
  out_file = '{file_output}',         ! name of output file for this FSM2oshd simulation
/
&nam_modconf
  NALBEDO = {modconf.get('albedo')},           ! albedo parametrization (special one for oshd)
  NCANMOD = {canmod},               ! canopy is switch ON/OFF. must sync to CTILE param (open vs. forest).
  NCONDCT = {modconf.get('condct')},           ! thermal conductivity (from original FSM)
  NDENSTY = {modconf.get('density')},          ! densification module 
  NEXCHNG = {turbulent_exchange},   ! option for turbulent heat exchange (need in sync  with in forest or in open)
  NHYDROL = {modconf.get('hydrol')},           ! water transport through snowpack
  NSNFRAC = 3,                      ! fractional snow cover (consider pathciness during melt or not). relevant for large scale run
  NRADSBG = 0,                      ! ignore
  NZOFFST = {z_offset},             ! 0 for above ground, 1 for above canopy. Z-offset for temperature and wind forcing. in sync with 'open' or 'forest'
  NOSHDTN = 0,                      ! switch if compensate for elevation precipitation lapse rate. Relevant for swiss operational model. Turn on if systematic bias of not enough snow in high elevation
  LHN_ON  = .FALSE.,                ! irrelevant for none operational setup. keep to false
  LFOR_HN = .FALSE.,                ! irrelevant for none operational setup. keep to false
/
&nam_modpert
  LZ0PERT = .FALSE.,                ! switch for perturbation run. Not available, in development...
/
&nam_modtile
  CTILE = '{mode}',               !  open or forest mode to run FSM
  rtthresh = 0.1,                   ! threshold of forest cover fraction at which to run forest mode. relevant when run in a grid setup. Not relevant here.
/
&nam_results                        ! https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/src/core/MODULES.F90#L80
   CLIST_DIAG_RESULTS = {', '.join(f"'{i}'" for i in diag_var_outputs)},   ! need all?   
   CLIST_STATE_RESULTS = {', '.join(f"'{i}'" for i in state_var_outputs)}, ! try if can be deleted
/
&nam_location
  fsky_terr = {np.round(row.svf,3)},            ! terrain svf 
  slopemu = {np.round(row.slope,3)},            ! slope in rad
  xi = 0,                           ! to be ignored. relevant coarse scale run. see Nora's paper
  Ld = {np.round(row.cluster_domain_size,3)},              ! grid cell size in meters (used in snow fractional cover) linked to Nora's paper
  lat = {np.round(row.lat,3)},             ! DD.DDD
  lon = {np.round(row.lon,3)},            ! DD.DDD
  dem = {np.round(row.elevation,0)},            ! elevation
  pmultf = {precip_multi},          ! precip multiplier default 1
  fveg = {np.round(fveg,3)},                    ! local canopy cover fraction (set to 0 in open)
  hcan = {np.round(hcan,3)},                    ! canopy height (meters) (set to 0 in open)
  lai = {np.round(lai5,2)},                      ! Leaf area index  (set to 0 in open)
  vfhp = {np.round(vfhp,3)},                    ! sky view fraction of canopy and terrain(set to 1 in open case)
  fves = {np.round(fves,3)},                ! canopy cover fraction (larger area)
/
  """

            with open(file_namelist, "w") as nlst_file:
                nlst_file.write(nlst)
        else:
            print(f'ERROR: met_file: {file_met} not found')
            return


    def write_fsm2oshd_met(ds_pt,
                           ds_tvt,
                           pt_name,
                           pt_ind,
                           n_digits,
                           fname_format='fsm_sim/fsm_',
                           temperature_correction=0):
        """
        Function to write meteorological forcing for FSM

        Format of the text file is:
            2021 9 1 6 61 45.01 206.7 0 0 275.02 74.08 0.29 74829 0 0.5
            2021 9 1 7 207.9 85.9 210.3 0 0 275.84 66.92 0.39 74864 0 0.5

        year month  day   hour  SWdir   SWdiff  LW  Sf  Rf     Ta  RH   Ua    Ps    Sf24 Tvt
        (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa) (mm) (-)

        """
        
        # for storage optimization tvt is stored in percent.


        foutput = str(fname_format) + '_met_' + str(pt_ind).zfill(n_digits) + '.txt'
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['month']  = pd.to_datetime(ds_pt.time.values).month
        df['day']  = pd.to_datetime(ds_pt.time.values).day
        df['hr']  = pd.to_datetime(ds_pt.time.values).hour
        df['SWdir'] = np.round(ds_pt.SW_direct.values,2)              # change to direct SW
        df['SWdif'] = np.round(ds_pt.SW_diffuse.values,2)                # change to diffuse SW
        df['LW'] = np.round(ds_pt.LW.values,2)
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values, method=snow_partition_method)
        df['snowfall'] = np.round(snow, 5)
        df['rainfall'] = np.round(rain, 5)
        df['Tair'] = np.round(ds_pt.t.values, 2) + temperature_correction
        df['RH'] = np.round(rh * 100,2)
        df['speed'] = np.round(ds_pt.ws.values,2)
        df['p'] = np.round(ds_pt.p.values,2)

        arr = df.snowfall.rolling(24).sum()  # hardcoded sum over last 24h [mm]
        arr.loc[np.isnan(arr)] = 0
        df['sf24'] = np.round(arr,3)

        #ds_pt['t_iter'] = ds_pt.time.dt.month*10000 + ds_pt.time.dt.day*100 + ds_pt.time.dt.hour
        if type(ds_tvt) in [int, float, np.float64, np.float16, np.float32]:
            print('Warning: tvt is constant')
            if ds_tvt>1:
                scale_tvt = 100
            else:
                scale_tvt = 1
            df['tvt'] = ds_tvt / scale_tvt
        else:
            if ds_tvt.for_tau.max()>10:
                scale_tvt = 100
            else:
                scale_tvt = 1
            df['tvt'] = np.round(ds_tvt.sel(point_name=pt_name).for_tau.values,4)/scale_tvt

        df.to_csv(foutput, index=False, header=False, sep=' ')
        print(f'---> Met file {foutput} saved')


    # 0. Add print statements about the need of data currently computed extrenally to TopoPyScale.
    # 1. Convert Clare's canopy output to FSM2oshd parametrization
    # 2. Extract cluster canopy and forest cover weights from ds_param_canopy
    # 3. loop through clusters and write met_file and namelist files
    # - 2 sets of FSM run, one with forest, another one without. Need to combine both output into a final value for the cluster centroid

    # ----- unpack and overwrite namelist_options ----
    # default namelist_param values
    namelist_param = { 'precip_multi':1,
                       'max_sd':4,
                       'z_snow':[0.1, 0.2, 0.4],
                       'z_soil':[0.1, 0.2, 0.4, 0.8],
                       'z_tair':2,
                       'z_wind':10,
                       'diag_var_outputs' : ['rotc', 'hsnt', 'swet', 'slqt',  'romc', 'sbsc'],
                       'state_var_outputs':['Tsrf', 'Sveg', 'Ds']}
    if namelist_options is not None:
        namelist_param.update(namelist_options)

    if n_digits is None:
        n_digits = len(str(len(ds_down.point_name))) + 1

    if cluster_method:
        # extract FSM forest parameters for each clusters
        # Aggregate forest parameters only to fores area

        dx = np.abs(np.diff(fsm_param.x)[0])
        dy = np.abs(np.diff(fsm_param.y)[0])
        df_forest = tu.ds_to_indexed_dataframe(fsm_param.drop(['point_name', 'cluster_labels'])).groupby('point_ind').mean()
        df_forest['cluster_total_area'] = fsm_param.elevation.groupby(fsm_param.point_ind).count().values * dx * dy
        df_forest['proportion_with_forest'] = fsm_param.isfor.groupby(fsm_param.point_ind).mean().values
        df_forest['cluster_domain_size'] = np.sqrt(df_forest.cluster_total_area)
        #df_forest['cluster_domain_size'] = np.sqrt(fsm_param.drop('point_name').groupby(fsm_param.point_name).count().to_dataframe().LAI5)*dx
        df_forest['forest_cover'] = fsm_param.forcov.groupby(fsm_param.point_ind).mean().values
        df_forest.forest_cover.loc[df_forest.proportion_with_forest<0.01] = 0
        df_forest['lon'], df_forest['lat'] = tp.convert_epsg_pts(df_forest.x, df_forest.y, epsg_ds_param, 4326)

    else:
        pass

    p = Path(simulation_path)
    # rename variable columns to match namelist functino varnames
    new_name = {'LAI5':'lai5', 'LAI50':'lai50', 'svf_for':'vfhp', 'CC5':'fveg', 'CC50':'fves', 'CH5':'hcan'}
    df_forest = df_forest.rename(columns=new_name)
    print(df_forest)
 
    # ----- Loop through all points-------
    # NOTE: eventually this for loop could be parallelized to several cores -----
    for pt_ind, pt_name in enumerate(ds_down.point_name.values):

        ds_pt = ds_down.sel(point_name=pt_name).copy()

        # [ ] Add checking of NaNs in ds_tvt. If NaN present stop process and send ERROR message

        if type(ds_tvt) in [int, float, np.float64, np.float16, np.float32]:
            tvt_pt = ds_tvt
        elif ds_tvt == 'svf_for':
            tvt_pt = df_forest.vfhp.iloc[pt_ind]
        else:
            tvt_pt = ds_tvt.sel(point_name=pt_name).copy()


        row_forest = df_forest.iloc[pt_ind]
        write_fsm2oshd_met(ds_pt,
                           ds_tvt=tvt_pt,
                           n_digits=n_digits,
                           pt_name=pt_name,
                           pt_ind=pt_ind,
                           fname_format=p/fname_format,
                           temperature_correction=temperature_correction)
        write_fsm2oshd_namelist(row_forest,
                                pt_ind=pt_ind,
                                n_digits=n_digits,
                                fname_format=p/fname_format,
                                mode='forest',
                                namelist_param=namelist_param,
                                scaler=forest_param_scaler) # write forest namelist

        if cluster_method:
            row_open = df_forest.iloc[pt_ind]
            write_fsm2oshd_namelist(row_open,
                                    pt_ind=pt_ind,
                                    n_digits=n_digits,
                                    fname_format=p/fname_format,
                                    mode='open',
                                    namelist_param=namelist_param) # write open namelist

        ds_pt = None
        tvt_pt = None
        # [ ] add logic to computed weighted average outputs based on forest cover fraction per point.

    df_forest.to_pickle(p / 'df_forest.pckl')
    return


def to_fsm(ds, fname_format='FSM_pt_*.tx', snow_partition_method='continuous', n_digits=None):
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
    - ensure ds.point_name.values always
    """
    if n_digits is None:
        n_digits = len(str(len(ds.point_name)))

    for pt_ind, pt_name in enumerate(ds.point_name.values):
        foutput = str(fname_format).split('*')[0] + str(pt_ind).zfill(n_digits) + str(fname_format).split('*')[1]
        ds_pt = ds.sel(point_name=pt_name).copy()
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['month'] = pd.to_datetime(ds_pt.time.values).month
        df['day'] = pd.to_datetime(ds_pt.time.values).day
        df['hr'] = pd.to_datetime(ds_pt.time.values).hour
        df['SW'] = ds_pt.SW.values
        df['LW'] = ds_pt.LW.values
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values,
                                       method=snow_partition_method)
        df['snowfall'] = snow / 3600
        df['rainfall'] = rain / 3600
        df['Tair'] = np.round(ds_pt.t.values, 2)
        df['RH'] = rh * 100
        df['speed'] = ds_pt.ws.values
        df['p'] = ds_pt.p.values

        df.to_csv(foutput, index=False, header=False, sep=' ')
        print('---> File {} saved'.format(foutput))


def to_TC(ds, fname_format='pt_*.tx'):
    """
    Function to export data for the T&C model.
    https://hyd.ifu.ethz.ch/research-data-models/t-c.html

    Args:
        ds (dataset): downscaled_pts,
        df_pts (dataframe): toposub.df_centroids,
        fname_format (str pattern): output format of filename

    format is a text file with the following columns
    datetime    TA  P       PRESS RH  SWIN    LWIN    WS   WDIR
    (datetime) (°C) (mm/h)  (Pa)  (%) (W/m2)  (W/m2)  (m/s) (°)

    """

    # set n_digits
    n_digits = len(str(len(ds.point_name))) + 1

    for pt_ind, pt_name in enumerate(ds.point_name.values):
        foutput = str(fname_format).split('*')[0] + str(pt_ind).zfill(n_digits) + str(fname_format).split('*')[1]
        ds_pt = ds.sel(point_name=pt_name).copy()
        df = pd.DataFrame()
        df['datetime'] = pd.to_datetime(ds_pt.time.values)
        df['TA'] = np.round(ds_pt.t.values, 2)  # air temperature
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        df['RH'] = rh * 100  # relative humidity
        df['PP'] = ds_pt.tp.values  # precipitation
        df['PRESS'] = ds_pt.p.values  # pressure
        df['SWIN'] = ds_pt.SW.values  # shortwave
        df['LWIN'] = ds_pt.LW.values  # longwave
        df['WS'] = ds_pt.ws.values  # wind speed
        df['WDIR'] = ds_pt.wd.values * (180 / np.pi)  # wind direction

        # convert units
        df.TA = df.TA - 273.15  # to °C (from Kelvin)
        df.PRESS = df.PRESS / 100  # to hPa (from Pascal)

        # save
        df.to_csv(foutput, index=False, header=True, sep=',')
        print('---> File {} saved'.format(foutput))


def to_micromet_single_station(ds,
                               df_pts,
                               fname_format='Snowmodel_pt_*.csv',
                               na_values=-9999,
                               headers=False):
    """
    Function to export TopoScale output in the format for Listn's Snowmodel (using Micromet). One CSV file per point_name

    Args:
        ds (dataset): TopoPyScale xarray dataset, downscaled product
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_name is inserted where * is
        na_values (int): na_value default
        headers (bool): add headers to file


    Example of the compatible format for micromet (headers should be removed to run the model:

     year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip
    (yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)

     2002   10    1   12.00    101   426340.0  4411238.0  3598.0     0.92    57.77     4.80   238.29 -9999.00
     2002   10    2   12.00    101   426340.0  4411238.0  3598.0    -3.02    78.77 -9999.00 -9999.00 -9999.00
     2002   10    8   12.00    101   426340.0  4411238.0  3598.0    -5.02    88.77 -9999.00 -9999.00 -9999.00
    """

    n_digits = len(str(len(ds.point_name))) + 1

    for pt_ind, pt_name in enumerate(ds.point_name.values):
        foutput = str(fname_format).split('*')[0] + str(pt_ind).zfill(n_digits) + str(fname_format).split('*')[1]
        ds_pt = ds.sel(point_name=pt_name).copy()
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['mo'] = pd.to_datetime(ds_pt.time.values).month
        df['dy'] = pd.to_datetime(ds_pt.time.values).day
        df['hr'] = pd.to_datetime(ds_pt.time.values).hour
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
                f.seek(0, 0)
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
    Functiont to export toposcale output to CROCUS netcdf format. Generates one file per point_name

    Args:
        ds (dataset): Toposcale downscaled dataset.
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_name is inserted where * is
        scale_precip (float): scaling factor to apply on precipitation. Default is 1
        climate_dataset_name (str): name of original climate dataset. Default 'ERA5',
        project_author (str): name of project author(s)
        snow_partition_method (str): snow/rain partitioning method: default 'jennings2018_trivariate'
    """
    # create one file per point_name
    n_digits = len(str(ds.point_name.values.max()))
    ver_dict = tu.get_versionning()

    for pt_ind, pt_name in enumerate(ds.point_name.values):
        foutput = str(fname_format).split('*')[0] + str(pt_ind).zfill(n_digits) + str(fname_format).split('*')[1]
        ds_pt = ds.sel(point_name=pt_name).copy()
        df = pd.DataFrame()
        df['time'] = ds_pt.time.values
        df['Tair'] = ds_pt.t.values
        df['DIR_SWdown'] = ds_pt.SW.values
        df['LWdown'] = ds_pt.LW.values
        df['PSurf'] = ds_pt.p.values
        df['HUMREL'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values) * 100
        df['xwind'] = ds_pt.u.values
        df['ywind'] = ds_pt.v.values
        df['precip'] = ds_pt.tp.values / 3600 * scale_precip  # convert from mm/hr to mm/s
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        df['Rainf'], df['Snowf'] = mu.partition_snow(df.precip, ds_pt.t.values, rh, ds_pt.p.values,
                                                     method=snow_partition_method)

        # Derive variables: Q- humidity, WD - wind direction (deg), and WS
        df['Qair'] = ds_pt.q.values
        df['Wind'], df['Wind_DIR'] = ds_pt.ws.values, np.rad2deg(ds_pt.wd.values)
        df['NEB'] = np.zeros(df.shape[0])
        df['CO2air'] = np.zeros(df.shape[0])
        df['SCA_SWdown'] = np.zeros(df.shape[0])

        df.drop(columns=['precip', 'xwind', 'ywind'], inplace=True)
        ds_pt.close()

        # df = df.loc[(df.time>=pd.Timestamp(start)) & (df.time<pd.Timestamp(end)+pd.Timedelta('1D'))]

        # Build netcdf forcing file for CROCUS
        fo = xr.Dataset.from_dataframe(df.set_index('time'))
        fo = df.set_index(['time']).to_xarray()
        fo.Tair.attrs = {'units': 'K', 'standard_name': 'Tair', 'long_name': 'Near Surface Air Temperature',
                         '_FillValue': -9999999.0}
        fo.Qair.attrs = {'units': 'kg/kg', 'standard_name': 'Qair', 'long_name': 'Near Surface Specific Humidity',
                         '_FillValue': -9999999.0}
        fo.HUMREL.attrs = {'units': '%', 'standard_name': 'HUMREL', 'long_name': 'Relative Humidity',
                           '_FillValue': -9999999.0}
        fo.Wind.attrs = {'units': 'm/s', 'standard_name': 'Wind', 'long_name': 'Wind Speed', '_FillValue': -9999999.0}
        fo.Wind_DIR.attrs = {'units': 'deg', 'standard_name': 'Wind_DIR', 'long_name': 'Wind Direction',
                             '_FillValue': -9999999.0}
        fo.Rainf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Rainf', 'long_name': 'Rainfall Rate',
                          '_FillValue': -9999999.0}
        fo.Snowf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Snowf', 'long_name': 'Snowfall Rate',
                          '_FillValue': -9999999.0}
        fo.DIR_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'DIR_SWdown',
                               'long_name': 'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
        fo.LWdown.attrs = {'units': 'W/m2', 'standard_name': 'LWdown',
                           'long_name': 'Surface Incident Longtwave Radiation', '_FillValue': -9999999.0}
        fo.PSurf.attrs = {'units': 'Pa', 'standard_name': 'PSurf', 'long_name': 'Surface Pressure',
                          '_FillValue': -9999999.0}
        fo.SCA_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'SCA_SWdown',
                               'long_name': 'Surface Incident Diffuse Shortwave Radiation', '_FillValue': -9999999.0}
        fo.CO2air.attrs = {'units': 'kg/m3', 'standard_name': 'CO2air', 'long_name': 'Near Surface CO2 Concentration',
                           '_FillValue': -9999999.0}
        fo.NEB.attrs = {'units': 'between 0 and 1', 'standard_name': 'NEB', 'long_name': 'Nebulosity',
                        '_FillValue': -9999999.0}

        fo.attrs = {'title': 'Forcing for SURFEX CROCUS',
                    'source': 'data from {} downscaled with TopoPyScale ready for CROCUS'.format(climate_dataset_name),
                    'creator_name': 'Dataset created by {}'.format(project_author),
                    'package_version': ver_dict.get('package_version'),
                    'git_commit': ver_dict.get('git_commit'),
                    'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                    'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

        fo['ZS'] = np.float64(df_pts.iloc[pt].elevation)
        fo['aspect'] = np.float64(np.rad2deg(df_pts.iloc[pt].aspect))
        fo['slope'] = np.float64(np.rad2deg(df_pts.iloc[pt].slope))
        fo['massif_number'] = np.float64(0)
        fo['LAT'] = df_pts.iloc[pt].latitude
        fo['LON'] = df_pts.iloc[pt].longitude
        fo.LAT.attrs = {'units': 'degrees_north', 'standard_name': 'LAT', 'long_name': 'latitude',
                        '_FillValue': -9999999.0}
        fo.LON.attrs = {'units': 'degrees_east', 'standard_name': 'LON', 'long_name': 'longitude',
                        '_FillValue': -9999999.0}
        fo['ZREF'] = np.float64(2)
        fo['UREF'] = np.float64(10)
        fo.ZREF.attrs = {'units': 'm', 'standard_name': 'ZREF', 'long_name': 'Reference_Height',
                         '_FillValue': -9999999.0}
        fo.UREF.attrs = {'units': 'm', 'standard_name': 'UREF', 'long_name': 'Reference_Height_for_Wind',
                         '_FillValue': -9999999.0}
        fo.aspect.attrs = {'units': 'degree from north', 'standard_name': 'aspect', 'long_name': 'slope aspect',
                           '_FillValue': -9999999.0}
        fo.slope.attrs = {'units': 'degrees from horizontal', 'standard_name': 'slope', 'long_name': 'slope angle',
                          '_FillValue': -9999999.0}
        fo.ZS.attrs = {'units': 'm', 'standard_name': 'ZS', 'long_name': 'altitude', '_FillValue': -9999999.0}
        fo.massif_number.attrs = {'units': '', 'standard_name': 'massif_number',
                                  'long_name': 'SAFRAN massif number in SOPRANO world', '_FillValue': -9999999.0}
        fo.time.attrs = {'standard_name': 'time', 'long_name': 'time', '_FillValue': -9999999.0}
        fo['FRC_TIME_STP'] = np.float64(3600)
        fo.FRC_TIME_STP.attrs = {'units': 's', 'standard_name': 'FRC_TIME_STP', 'long_name': 'Forcing_Time_Step',
                                 '_FillValue': -9999999.0}
        fo['FORCE_TIME_STEP'] = np.float64(3600)
        fo.FORCE_TIME_STEP.attrs = {'units': 's', 'standard_name': 'FORCE_TIME_STEP', 'long_name': 'Forcing_Time_Step',
                                    '_FillValue': -9999999.0}

        fo = fo.rename_dims({'point_name':'Number_of_points'})
        fo = fo.transpose()

        fo.to_netcdf(foutput, mode='w', format='NETCDF3_CLASSIC', unlimited_dims={'time': True},
                     encoding={'time': {'dtype': 'float64', 'calendar': 'standard'},
                               'Tair': {'dtype': 'float64'},
                               'Qair': {'dtype': 'float64'},
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


def to_surfex(ds,
              df_pts,
              fname_format='FORCING*.nc',
              scale_precip=1,
              climate_dataset_name='ERA5',
              project_author='S. Filhol',
              snow_partition_method='continuous',
              year_start='2023-08-01 00:00:00',
              year_end='2024-06-30 23:00:00'):
    """
    Function to export toposcale output to Surfex forcing netcdf format. Generates one file per year

    Args:
        ds (dataset): Toposcale downscaled dataset.
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_name is inserted where * is
        scale_precip (float): scaling factor to apply on precipitation. Default is 1
        climate_dataset_name (str): name of original climate dataset. Default 'ERA5',
        project_author (str): name of project author(s)
        snow_partition_method (str): snow/rain partitioning method: default 'jennings2018_trivariate'
        year_start (str): start of year for yearly Surfex forcing

    ToDo:
    - [x] split to yearly files,

    """
    # create one file per point_name
    n_digits = len(str(ds.point_name.max().values))
    ver_dict = tu.get_versionning()

    tvec_start, tvec_end = tu.time_vecs(start_date=year_start, end_date=year_end)
    
    for year_id, start in enumerate(tvec_start):
        fo = ds.sel(time=slice(start, tvec_end[year_id])).copy()
        start_str = start.strftime('%Y%m%d%H')
        end_str = tvec_end[year_id].strftime('%Y%m%d%H')
        foutput = f"{str(fname_format).split('*')[0]}_{start_str}_{end_str}.nc"
        fo = fo.rename_dims({'point_name':'Number_of_points'})
        fo = fo.rename({'t':'Tair', 
                              'SW':'DIR_SWdown', 
                              'LW':'LWdown',
                              'p':'PSurf',
                              'q':'Qair',
                              'ws':'Wind'})
        fo['Wind_DIR'] = fo.wd * 180 / np.pi
        fo['HUMREL'] = (('Number_of_points','time'), mu.q_2_rh(fo.Tair.values, fo.PSurf.values, fo.Qair.values) * 100)
        fo['precip'] = (('Number_of_points','time'), fo.tp.values / 3600 * scale_precip)  # convert from mm/hr to mm/s
        rh = mu.q_2_rh(fo.Tair.values, fo.PSurf.values, fo.Qair.values)
        Rainf, Snowf = mu.partition_snow(fo.precip.values, fo.Tair.values, rh, fo.PSurf.values,
                                                     method=snow_partition_method)
        fo['Rainf'] = (('Number_of_points','time'), Rainf)
        fo['Snowf'] = (('Number_of_points','time'), Snowf)
                
        fo['NEB'] = (('Number_of_points','time'), np.zeros(fo.Tair.shape))
        fo['CO2air'] = (('Number_of_points','time'), np.zeros(fo.Tair.shape))
        fo['SCA_SWdown'] = (('Number_of_points','time'), np.zeros(fo.Tair.shape))

        fo = fo.drop_vars(['precip', 'u', 'v', 'w', 'wd', 'vp', 'cse','SW_direct', 'cos_illumination', 'SW_diffuse', 'precip_lapse_rate', 'tp' ])

        fo.Tair.attrs = {'units': 'K', 'standard_name': 'Tair', 'long_name': 'Near Surface Air Temperature',
                         '_FillValue': -9999999.0}
        fo.Qair.attrs = {'units': 'kg/kg', 'standard_name': 'Qair', 'long_name': 'Near Surface Specific Humidity',
                         '_FillValue': -9999999.0}
        fo.HUMREL.attrs = {'units': '%', 'standard_name': 'HUMREL', 'long_name': 'Relative Humidity',
                           '_FillValue': -9999999.0}
        fo.Wind.attrs = {'units': 'm/s', 'standard_name': 'Wind', 'long_name': 'Wind Speed', '_FillValue': -9999999.0}
        fo.Wind_DIR.attrs = {'units': 'deg', 'standard_name': 'Wind_DIR', 'long_name': 'Wind Direction',
                             '_FillValue': -9999999.0}
        fo.Rainf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Rainf', 'long_name': 'Rainfall Rate',
                          '_FillValue': -9999999.0}
        fo.Snowf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Snowf', 'long_name': 'Snowfall Rate',
                          '_FillValue': -9999999.0}
        fo.DIR_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'DIR_SWdown',
                               'long_name': 'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
        fo.LWdown.attrs = {'units': 'W/m2', 'standard_name': 'LWdown',
                           'long_name': 'Surface Incident Longtwave Radiation', '_FillValue': -9999999.0}
        fo.PSurf.attrs = {'units': 'Pa', 'standard_name': 'PSurf', 'long_name': 'Surface Pressure',
                          '_FillValue': -9999999.0}
        fo.SCA_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'SCA_SWdown',
                               'long_name': 'Surface Incident Diffuse Shortwave Radiation', '_FillValue': -9999999.0}
        fo.CO2air.attrs = {'units': 'kg/m3', 'standard_name': 'CO2air', 'long_name': 'Near Surface CO2 Concentration',
                           '_FillValue': -9999999.0}
        fo.NEB.attrs = {'units': 'between 0 and 1', 'standard_name': 'NEB', 'long_name': 'Nebulosity',
                        '_FillValue': -9999999.0}

        fo.attrs = {'title': 'Forcing for SURFEX CROCUS',
                    'source': 'data from {} downscaled with TopoPyScale ready for CROCUS'.format(climate_dataset_name),
                    'creator_name': 'Dataset created by {}'.format(project_author),
                    'package_version': ver_dict.get('package_version'),
                    'git_commit': ver_dict.get('git_commit'),
                    'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                    'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

        ds_pts = xr.Dataset(df_pts).rename_dims({'dim_0':'Number_of_points'})
        fo['ZS'] = (('Number_of_points'), np.float64(ds_pts.elevation.sel(Number_of_points=fo.Number_of_points).values))
        fo['aspect'] = (('Number_of_points'), np.float64(np.rad2deg(ds_pts.aspect.sel(Number_of_points=fo.Number_of_points).values)))
        fo['slope'] = (('Number_of_points'), np.float64(np.rad2deg(ds_pts.slope.sel(Number_of_points=fo.Number_of_points).values)))
        fo['massif_number'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(0))
        fo['LAT'] = (('Number_of_points'), ds_pts.latitude.sel(Number_of_points=fo.Number_of_points).values)
        fo['LON'] = (('Number_of_points'), ds_pts.longitude.sel(Number_of_points=fo.Number_of_points).values)
        ds_pts=None
        fo.LAT.attrs = {'units': 'degrees_north', 'standard_name': 'LAT', 'long_name': 'latitude',
                        '_FillValue': -9999999.0}
        fo.LON.attrs = {'units': 'degrees_east', 'standard_name': 'LON', 'long_name': 'longitude',
                        '_FillValue': -9999999.0}
        fo['ZREF'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(2))
        fo['UREF'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(10))
        fo.ZREF.attrs = {'units': 'm', 'standard_name': 'ZREF', 'long_name': 'Reference_Height',
                         '_FillValue': -9999999.0}
        fo.UREF.attrs = {'units': 'm', 'standard_name': 'UREF', 'long_name': 'Reference_Height_for_Wind',
                         '_FillValue': -9999999.0}
        fo.aspect.attrs = {'units': 'degree from north', 'standard_name': 'aspect', 'long_name': 'slope aspect',
                           '_FillValue': -9999999.0}
        fo.slope.attrs = {'units': 'degrees from horizontal', 'standard_name': 'slope', 'long_name': 'slope angle',
                          '_FillValue': -9999999.0}
        fo.ZS.attrs = {'units': 'm', 'standard_name': 'ZS', 'long_name': 'altitude', '_FillValue': -9999999.0}
        fo.massif_number.attrs = {'units': '', 'standard_name': 'massif_number',
                                  'long_name': 'SAFRAN massif number in SOPRANO world', '_FillValue': -9999999.0}
        fo.time.attrs = {'standard_name': 'time', 'long_name': 'time', '_FillValue': -9999999.0}
        fo['FRC_TIME_STP'] = ((), np.float64(3600))
        fo.FRC_TIME_STP.attrs = {'units': 's', 'standard_name': 'FRC_TIME_STP', 'long_name': 'Forcing_Time_Step',
                                 '_FillValue': -9999999.0}
        #fo['FORCE_TIME_STEP'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(3600))
        #fo.FORCE_TIME_STEP.attrs = {'units': 's', 'standard_name': 'FORCE_TIME_STEP', 'long_name': 'Forcing_Time_Step',
        #                            '_FillValue': -9999999.0}

        fo = fo.transpose()

        fo.to_netcdf(foutput, mode='w', format='NETCDF3_CLASSIC', unlimited_dims={'time': True},
                     encoding={'time': {'dtype': 'float64', 'calendar': 'standard'},
                               'Tair': {'dtype': 'float64'},
                               'Qair': {'dtype': 'float64'},
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
        # clear memory
        fo = None














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

    n_digits = len(str(len(ds.point_name))) + 1

    for pt_ind, pt_name in enumerate(ds.point_name.values):
        foutput = str(fname_format).split('*')[0] + str(pt_ind).zfill(n_digits) + str(fname_format).split('*')[1]
        ds_pt = ds.sel(point_name=pt_name).copy()
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

    n_digits = len(str(len(ds.point_name))) + 1

    for pt_ind, pt_name in enumerate(ds.point_name.values):
        foutput = str(fname_format).split('*')[0] + str(pt_ind).zfill(n_digits) + str(fname_format).split('*')[1]
        ds_pt = ds.sel(point_name=pt_name).copy()
        df = pd.DataFrame()
        dt = pd.to_datetime(ds_pt.time.values)
        df['Date'] = dt.strftime("%d/%m/%Y %H:%M")
        # df['Date'] = str(pd.to_datetime(ds_pt.time.values).day) +   "/" + str(pd.to_datetime(ds_pt.time.values).month) +  "/" + str(pd.to_datetime(ds_pt.time.values).year) +  " " + str(pd.to_datetime(ds_pt.time.values).hour) + ":" + str(pd.to_datetime(ds_pt.time.values).minute)
        df['AirT'] = np.round(ds_pt.t.values - 273.15, 2)
        df['RelHum'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values) * 100
        df['WindS'] = ds_pt.ws.values
        df['WIndDr'] = ds_pt.wd.values * 180 / np.pi
        df['Swglob'] = ds_pt.SW.values
        df['Lwin'] = ds_pt.LW.values
        df['Iprec'] = ds_pt.tp.values

        df.to_csv(foutput, index=False, header=False, sep=',', quoting=csv.QUOTE_NONE)

        header = 'Date, AirT, RelHum, WindS, WindDr, Swglob, Lwin, Iprec'
        with open(foutput, 'r+') as f:
            f_data = f.read()
            f.seek(0, 0)
            f.write(header + '\n' + f_data)
        print('---> File {} saved'.format(foutput))

        # if headers:
        #     header = 'year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip\n(yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)\n'
        #     with open(foutput, 'r+') as f:
        #         f_data = f.read()
        #         f.seek(0,0)
        #         f.write(header + '\n' + f_data)
        # print('---> File {} saved'.format(foutput))
