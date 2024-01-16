"""
Functions to export topo_scale output to formats compatible with existing models (e.g. CROCUS, Cryogrid, Snowmodel, ...)

S. Filhol, December 2021

TODO;
- SPHY forcing (grids)
"""
import csv
import datetime as dt
import sys

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
        if str(ds[var].dtype)[:3] == 'int' or var == 'cluster_labels':
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
        da_label.to_netcdf(path + fname_labels)

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

    fo['latitude'] = (('point_id'), df_pts.latitude.values)
    fo['longitude'] = (('point_id'), df_pts.longitude.values)
    fo['elevation'] = (('point_id'), df_pts.elevation.values)
    fo.latitude.attrs = {'units': 'deg', 'standard_name': 'latitude', 'long_name': 'Cluster latitude'}
    fo.longitude.attrs = {'units': 'deg', 'standard_name': 'longitude', 'long_name': 'Cluster longitude'}
    fo.elevation.attrs = {'units': 'm', 'standard_name': 'elevation', 'long_name': 'Cluster elevation'}

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

    ver_dict = tu.get_versionning()
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
        fo['latitude'] = df_pts.latitude.iloc[pt]
        fo['longitude'] = df_pts.longitude.iloc[pt]
        fo['elevation'] = df_pts.elevation.iloc[pt]
        fo.latitude.attrs = {'units': 'deg', 'standard_name': 'latitude'}
        fo.longitude.attrs = {'units': 'deg', 'standard_name': 'longitude'}
        fo.elevation.attrs = {'units': 'm', 'standard_name': 'elevation'}
        fo.to_netcdf(foutput, encoding=encod_dict)
        print('---> File {} saved'.format(foutput))

def to_fsm2oshd(ds_down,
                fsm_param,
                ds_tvt,
                simulation_path='fsm_sim',
                fname_format='fsm_',
                namelist_options=None,
                n_digits=None,
                snow_partition_method='continuous',
                cluster_method=True,
                epsg_ds_param=2056):
    '''
    Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd).
    FSM2oshd includes canopy structures processes
    one simulation consists of 2 driving file:
       - met.txt with variables:
           year, month, day, hour, SWb, SWd, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt
       - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10

    Args:
        ds_down:  Downscaled weather variable dataset
        fsm_param:  terrain and canopy parameter dataset
        df_centroids:  cluster centroids statistics (terrain + canopy)
        ds_tvt (dataset):  transmisivity dataset
        namelist_param (dict): {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]}

    '''

    def write_fsm2oshd_namelist(row,
                                pt_name,
                                n_digits,
                                fname_format='fsm_sim/fsm_',
                                mode='forest',
                                namelist_param=None,
                                modconf=None):
        # Function to write namelist file (.nam) for each point where to run FSM.

        file_namelist = str(fname_format) + f'_{mode}_' + str(pt_name).zfill(n_digits) + '.nam'
        file_met = str(fname_format) + '_met_' + str(pt_name).zfill(n_digits) + '.txt'
        file_output = str(fname_format) + f'_outputs_{mode}_' + str(pt_name).zfill(n_digits) + '.txt'

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
        
        if (row.fveg.max()>10) | (row.lai5 > 10):
            scale = 100
        else:
            scale = 1


        if mode == 'forest':
            # Values compatible with 'forest' mode
            canmod = 1
            turbulent_exchange = 2
            z_offset = 1
            fveg = row.fveg/scale
            hcan = row.hcan/scale
            lai = row.lai5/scale
            vfhp = row.vfhp/scale

        else:
            # default values for 'open' mode
            canmod = 0
            turbulent_exchange = 1
            z_offset, fveg, fves, hcan, lai, vfhp= 0, 0, 0, 0, 0, 1

        #pdb.set_trace()
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
  lai = {np.round(lai,2)},                      ! Leaf area index  (set to 0 in open)
  vfhp = {np.round(vfhp,3)},                    ! sky view fraction of canopy and terrain(set to 1 in open case)
  fves = {np.round(row.fves/scale,3)},                ! canopy cover fraction (larger area)
/
  """

            with open(file_namelist, "w") as nlst_file:
                nlst_file.write(nlst)
        else:
            print(f'ERROR: met_file: {file_met} not found')
            return


# ToDo fix issues with to_fsm2oshd (a lot of unused parameters, wrongly named parameters (e.g. fsm_mode) etc. -> Would make the whole topopyscale not work & wasn't able to quick fix. So I outcommented.
# def to_fsm2oshd(ds_down,
#                 ds_param_canop,
#                 df_centroids,
#                 ds_tvt,
#                 path='outputs/',
#                 namelist_param=None):
#     '''
#     Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd).
#     FSM2oshd includes canopy structures processes
#     one simulation consists of 2 driving file:
#        - met.txt with variables:
#            year, month, day, hour, SWb, SWd, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt
#        - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10
#
#     Args:
#         ds_down:  Downscaled weather variable dataset
#         ds_param_canop:  terrain and canopy parameter dataset
#         df_centroids:  cluster centroids statistics (terrain + canopy)
#         ds_tvt (dataset):  transmisivity dataset
#         namelist_param (dict): {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]}
#
#     '''
#
#     def write_fsm2oshd_namelist(row_centroids,
#                                 file_namelist,
#                                 file_met,
#                                 file_output,
#                                 mode='forest',
#                                 namelist_options=None):
#         # Function to write namelist file (.nam) for each point where to run FSM.
#
#         for k, v in namelist_param.items(): exec(k + '=v')
#
#         if mode == 'forest':
#             # Values compatible with 'forest' mode
#             canmod = 1
#             turbulent_exchange = 2
#             z_offset = 0
#             fveg = row.fveg
#             hcan = row.hcan
#             lai = row.lai
#             vfhp = row.vfhp
#
#         else:
#             # default values for 'open' mode
#             canmod = 0
#             turbulent_exchange = 1
#             z_offset, fveg, hcan, lai, vfhd = 0, 0, 0, 0, 1
#
#         if os.path.exists(file_met):
#             nlst = f"""
# &nam_grid
#   NNx = 1,                          ! Grid x-indexing. Not relevant with this implementation
#   NNy = 1,                          ! Grid y-indexing. Not relevant with this implementation
#   NNsmax = {len(z_snow)},           ! Number snow layers. Default 3-layers
#   NNsoil = {len(z_soil)},           ! Number soil layers. Default 4-layers
# /
# &nam_layers
#   DDzsnow = {', '.join(z_snow)},    ! Minimum thickness to define a new snow layer
#   DDzsoil = {', '.join(z_soil)},    ! soil layer thickness
# /
# &nam_driving
#   zzT = {z_tair},                   ! Height of temperature forcing (2 or 10m)
#   zzU = {z_wind},                   ! Height of wind forcing (2 or 10m)
#   met_file = {file_met},            ! name of met file associatied to this namelist
#   out_file = {output_file},         ! name of output file for this FSM2oshd simulation
# /
# &nam_modconf
#   NALBEDO = {row.albedo},           ! albedo parametrization (special one for oshd)
#   NCANMOD = {canmod},               ! canopy is switch ON/OFF. must sync to CTILE param (open vs. forest).
#   NCONDCT = {row.condct},           ! thermal conductivity (from original FSM)
#   NDENSTY = {row.density},          ! densification module
#   NEXCHNG = {turbulent_exchange},   ! option for turbulent heat exchange (need in sync  with in forest or in open)
#   NHYDROL = {row.hydrol},           ! water transport through snowpack
#   NSNFRAC = 3,                      ! fractional snow cover (consider pathciness during melt or not). relevant for large scale run
#   NRADSBG = 0,                      ! ignore
#   NZOFFST = {z_offset},             ! 0 for above ground, 1 for above canopy. Z-offset for temperature and wind forcing. in sync with 'open' or 'forest'
#   NOSHDTN = 0,                      ! switch if compensate for elevation precipitation lapse rate. Relevant for swiss operational model. Turn on if systematic bias of not enough snow in high elevation
#   LHN_ON  = .FALSE.,                ! irrelevant for none operational setup. keep to false
#   LFOR_HN = .FALSE.,                ! irrelevant for none operational setup. keep to false
# /
# &nam_modpert
#   LZ0PERT = .FALSE.,                ! switch for perturbation run. Not available, in development...
# /
# &nam_modtile
#   CTILE = {fsm_mode},               !  open or forest mode to run FSM
#   rtthresh = 0.1,                   ! threshold of forest cover fraction at which to run forest mode. relevant when run in a grid setup. Not relevant here.
# /
# &nam_results                        ! https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/src/core/MODULES.F90#L80
#    CLIST_DIAG_RESULTS = {', '.join(diag_var_outputs)},   ! need all?
#    CLIST_STATE_RESULTS = {', '.join(state_var_outputs)}, ! try if can be deleted
# /
# &nam_location
#   fsky_terr = {row.svf},            ! terrain svf
#   slopemu = {row.slope},            ! slope in rad
#   xi = 0,                           ! to be ignored. relevant coarse scale run. see Nora's paper
#   Ld = {row.gridsize},              ! grid cell size in meters (used in snow fractional cover) linked to Nora's paper
#   lat = {row.latitude},             ! DD.DDD
#   lon = {row.longitude},            ! DD.DDD
#   dem = {row.elevation},            ! elevation
#   pmultf = {precip_multi},          ! precip multiplier default 1
#   fveg = {fveg},                    ! local canopy cover fraction (set to 0 in open)
#   hcan = {hcan},                    ! canopy height (meters) (set to 0 in open)
#   lai = {lai},                      ! Leaf area index  (set to 0 in open)
#   vfhp = {vfhp},                    ! sky view fraction of canopy and terrain(set to 1 in open case)
#   fves = {row.fves},                ! canopy cover fraction (larger area)
# /
#   """
#
#             with open(file_namelist, "w") as nlst_file:
#                 nlst_file.write(nlst)
#         else:
#             print(f'ERROR: met_file: {file_met} not found')
#             return
#
#     def write_fsm2oshd_met(ds_pt,
#                            ds_param,
#                            df_centroids,
#                            df_canopy):
#         '''
#         Function to write meteorological forcing for FSM
#
#         Format of the text file is:
#             2021 9 1 6 61 45.01 206.7 0 0 275.02 74.08 0.29 74829 0 0.5
#             2021 9 1 7 207.9 85.9 210.3 0 0 275.84 66.92 0.39 74864 0 0.5
#
#         year month  day   hour  SWb   SWd  LW  Sf  Rf     Ta  RH   Ua    Ps    Sf24 Tvt
#     (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa) (mm) (-)
#
#         '''
#         print('TBI')
#         n_digits = 2
#
#         foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
#         ds_pt = ds.sel(point_id=pt).copy()
#         df = pd.DataFrame()
#         df['year'] = pd.to_datetime(ds_pt.time.values).year
#         df['month'] = pd.to_datetime(ds_pt.time.values).month
#         df['day'] = pd.to_datetime(ds_pt.time.values).day
#         df['hr'] = pd.to_datetime(ds_pt.time.values).hour
#         df['SWb'] = ds_pt.SW.values  # change to direct SW
#         df['SWd'] = ds_pt.SW.values  # change to diffuse SW
#         df['LW'] = ds_pt.LW.values
#         rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
#         rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values,
#                                        method=snow_partition_method)
#         df['snowfall'] = snow / 3600
#         df['rainfall'] = rain / 3600
#         df['Tair'] = np.round(ds_pt.t.values, 2)
#         df['RH'] = rh * 100
#         df['speed'] = ds_pt.ws.values
#         df['p'] = ds_pt.p.values
#         df['sf24'] = df.snowfall.rolling('24').sum()  # hardcoded sum over last 24h [mm]
#
#         # code to pick transmissivity based on Julian date, and cluster/point ID, (think about leap years)
#         df['tvt'] = ds_param.transmissivity.sel()
#
#         df.to_csv(foutput, index=False, header=False, sep=' ')
#         print('---> Met file {} saved'.format(foutput))
#
#     # 0. Add print statements about the need of data currently computed extrenally to TopoPyScale.
#     # 1. Convert Clare's canopy output to FSM2oshd parametrization
#     # 2. Extract cluster canopy and forest cover weights from ds_param_canopy
#     # 3. loop through clusters and write met_file and namelist files
#     # - 2 sets of FSM run, one with forest, another one without. Need to combine both output into a final value for the cluster centroid
#
#     # ----- sample canopy propoerties from df_canopy -----
#
#     # ----- unpack and overwrite namelist_options ----
#     # default namelist_param values
#     namelist_param = {'precip_multi': 1,
#                       'max_sd': 4,
#                       'z_snow': [0.1, 0.2, 0.4],
#                       'z_soil': [0.1, 0.2, 0.4, 0.8],
#                       'z_tair': 2,
#                       'z_wind': 10,
#                       'diag_var_outputs': ['rotc', 'hsnt', 'swet', 'slqt', 'romc', 'sbsc'],
#                       'state_var_outputs': ['']}
#     if namelist_options is not None:
#         namelist_param.update(namelist_options)
#
#     # ----- Loop through all points -----
#     for pt in ds.point_id.values:
#         fname_namlst_forest = f'fsm_namlst_{str(pt).zfill(n_digits)}.nam'
#
#         ds_pt = ds.sel(point_id=pt).copy()
#         row = df_centroids.loc[df_centroids.pt]
#         write_fsm2oshd_met(ds_pt, ds_param, pt_name)
#         write_fsm2oshd_namelist(row,, mode = 'open', namelist_param = namelist_param)  # write open namelist
#         write_fsm2oshd_namelist(mode='forest', namelist_param=namelist_param)  # write forest namelist
#
#     return


    def write_fsm2oshd_met(ds_pt,
                           ds_tvt,
                           pt_name,
                           n_digits,
                           fname_format='fsm_sim/fsm_'):
        '''
        Function to write meteorological forcing for FSM

        Format of the text file is:
            2021 9 1 6 61 45.01 206.7 0 0 275.02 74.08 0.29 74829 0 0.5
            2021 9 1 7 207.9 85.9 210.3 0 0 275.84 66.92 0.39 74864 0 0.5

        year month  day   hour  SWb   SWd  LW  Sf  Rf     Ta  RH   Ua    Ps    Sf24 Tvt
        (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa) (mm) (-)

        '''
        
        # for storage optimization tvt is stored in percent.
        if ds_tvt.for_tau.max()>10:
            scale_tvt = 100
        else:
            scale_tvt = 1

        foutput = str(fname_format) + '_met_' + str(pt_name).zfill(n_digits) + '.txt'
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['month']  = pd.to_datetime(ds_pt.time.values).month
        df['day']  = pd.to_datetime(ds_pt.time.values).day
        df['hr']  = pd.to_datetime(ds_pt.time.values).hour
        df['SWb'] = np.round(ds_pt.SW_direct.values,2)              # change to direct SW
        df['SWd'] = np.round(ds_pt.SW_diffuse.values,2)                # change to diffuse SW
        df['LW'] = np.round(ds_pt.LW.values,2)
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh, ds_pt.p.values, method=snow_partition_method)
        df['snowfall'] = np.round(snow, 5)
        df['rainfall'] = np.round(rain, 5)
        df['Tair'] = np.round(ds_pt.t.values, 2)
        df['RH'] = np.round(rh * 100,2)
        df['speed'] = np.round(ds_pt.ws.values,2)
        df['p'] = np.round(ds_pt.p.values,2)

        arr = df.snowfall.rolling(24).sum()  # hardcoded sum over last 24h [mm]
        arr.loc[np.isnan(arr)] = 0
        df['sf24'] = np.round(arr,3)

        #ds_pt['t_iter'] = ds_pt.time.dt.month*10000 + ds_pt.time.dt.day*100 + ds_pt.time.dt.hour
        df['tvt'] = np.round(ds_tvt.sel(cluster_labels=pt_name).for_tau.values,4)/scale_tvt

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
        n_digits = len(str(ds_down.point_id.values.max())) + 1

    if cluster_method:
        # extract FSM forest parameters for each clusters
        # Aggregate forest parameters only to fores area
        fsm_df = ts.ds_to_indexed_dataframe(fsm_param)
        fsm_df['lon'], fsm_df['lat'] = tp.convert_epsg_pts(fsm_df.x, fsm_df.y, epsg_ds_param, 4326)
        df_forest = fsm_df.where(fsm_df.forcov>0.).dropna().groupby('cluster_labels').mean()
        df_open = fsm_df.where(fsm_df.forcov==0.).dropna().groupby('cluster_labels').mean()

        dx = np.abs(np.diff(fsm_param.x)[0])
        dy = np.abs(np.diff(fsm_param.y)[0])
        #pdb.set_trace()
        df_forest['cluster_total_area'] = fsm_df.groupby('cluster_labels').count().elevation.values * dx * dy
        df_forest['proportion_with_forest'] = fsm_df.where(fsm_df.forcov > 0.).groupby('cluster_labels').count().elevation.values / fsm_df.groupby('cluster_labels').count().elevation.values
        df_forest['cluster_domain_size'] = np.sqrt(df_forest.cluster_total_area)
        #df_forest['cluster_domain_size'] = np.sqrt(fsm_param.drop('cluster_labels').groupby(fsm_param.cluster_labels).count().to_dataframe().LAI5)*dx
        df_forest['forest_cover'] = fsm_param.drop('cluster_labels').groupby(fsm_param.cluster_labels).mean().forcov.values
    else:
        pass

    p = Path(simulation_path)
    # rename variable columns to match namelist functino varnames
    new_name = {'LAI5':'lai5', 'LAI50':'lai50', 'svf_for':'vfhp', 'CC5':'fveg', 'CC50':'fves', 'CH5':'hcan'}
    df_forest = df_forest.rename(columns=new_name)
    print(df_forest)
 
    # ----- Loop through all points-------
    # NOTE: eventually this for loop could be parallelized to several cores -----
    for pt in ds_down.point_id.values:

        ds_pt = ds_down.sel(point_id=pt).copy()
        tvt_pt = ds_tvt.sel(cluster_labels=pt).copy()
        row_forest = df_forest.loc[pt]
        write_fsm2oshd_met(ds_pt,
                           ds_tvt=ds_tvt,
                           n_digits=n_digits,
                           pt_name=pt,
                           fname_format=p/fname_format)
        write_fsm2oshd_namelist(row_forest,
                                pt_name=pt,
                                n_digits=n_digits,
                                fname_format=p/fname_format,
                                mode='forest',
                                namelist_param=namelist_param) # write forest namelist

        if cluster_method:
            row_open = df_forest.loc[pt]
            write_fsm2oshd_namelist(row_open,
                                    pt_name=pt,
                                    n_digits=n_digits,
                                    fname_format=p/fname_format,
                                    mode='open',
                                    namelist_param=namelist_param) # write open namelist

        ds_pt = None
        tvt_pt = None
        # [ ] add logic to computed weighted average outputs based on forest cover fraction per point.

    df_forest.to_pickle(p/'df_forest.pckl')
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
    - ensure ds.point_id.values always
    """
    if n_digits is None:
        n_digits = len(str(ds.point_id.values.max())) + 1

    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
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

    Args:
        ds (dataset): downscaled_pts,
        df_pts (dataframe): toposub.df_centroids,
        fname_format (str pattern): output format of filename

    format is a text file with the following columns
    datetime    TA  P       PRESS RH  SWIN    LWIN    WS   WDIR
    (datetime) (°C) (mm/h)  (Pa)  (%) (W/m2)  (W/m2)  (m/s) (°)

    """

    # set n_digits
    n_digits = len(str(max(ds.point_id.values)))

    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
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
    ver_dict = tu.get_versionning()

    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
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

        fo = fo.expand_dims('Number_of_points')
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

    # n_digits = len(str(ds.point_id.values.max()))
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

    # n_digits = len(str(ds.point_id.values.max()))
    # always set this as 3  simplifies parsing files later on
    n_digits = 3

    for pt in ds.point_id.values:
        foutput = fname_format.split('*')[0] + str(pt).zfill(n_digits) + fname_format.split('*')[1]
        ds_pt = ds.sel(point_id=pt).copy()
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
