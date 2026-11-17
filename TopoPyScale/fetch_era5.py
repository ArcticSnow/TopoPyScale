"""
Retrieve ecmwf data with cdsapi.

- J. Fiddes, Origin implementation
- S. Filhol adapted in 2021

"""
# !/usr/bin/env python
import pandas as pd
import xarray as xr
import numpy as np
import dask
try:
    from zarr.codecs import BloscCodec
except ImportError:
    # For newer versions of zarr
    try:
        from zarr.codecs._blosc import BloscCodec
    except ImportError:
        # Fallback for older versions or different structure
        from zarr import Blosc as BloscCodec

import cdsapi, os, sys
from dateutil.relativedelta import *
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime, timedelta
from TopoPyScale import topo_utils as tu
from pathlib import Path
import zipfile
import shutil
import glob
from cdo import *
from munch import DefaultMunch


# [ ] remove this dependencie era5_downloader bringing little value
# [ ] create a class to execute download


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


var_surf_name_google = {'geopotential_at_surface':'z',
                '2m_dewpoint_temperature':'d2m', 
                'surface_thermal_radiation_downwards':'strd',
                'surface_solar_radiation_downwards':'ssrd',
                'surface_pressure':'sp',
                'total_precipitation':'tp',
                '2m_temperature':'t2m'}

var_plev_name_google = {'geopotential':'z', 
                'temperature':'t', 
                'u_component_of_wind':'u',
                'v_component_of_wind':'v', 
                'specific_humidity':'q'}



class FetchERA5():
    """
    Class to fetch data from ERA5 based on config of loaded by the main class Topoclass.config or config loaded directly from yml file.

    Args:
        config (str or DefaultMunch): point to a config object (inherited from Topoclass, or filepath to yml)
        varoi_surf (list): list of surface variables to download. Default is None which will fetch only necessary for TPS 
        varoi_plev (list): list of pressure level variables to download. Default is None which will fetch only necessary for TPS
        verify_ssl (bool): boolean flag to stop ssl certificate. DO NOT USE!!! unless needed.  
    """
    
    def __init__(self, config, varoi_surf=None, varoi_plev=None, verify_ssl=True):

        if type(config) is DefaultMunch:
            self.config = config
            print("---> Config loaded")
        elif type(config) is str:
            self.config = self.load_config(config_file=config)
            print(f"---> Config from {config} loaded")
        else:
            raise ValueError('Cannot find config. Check path to config, or config is a Munch.DefaultMunch object.')

        self.varoi_surf = varoi_surf
        self.varoi_plev = varoi_plev
        self.verify_flag = verify_ssl

        os.makedirs(self.config.climate.path / 'yearly', exist_ok=True)
        os.makedirs(self.config.climate.path/ 'daily', exist_ok=True)

        # if keyword exists in config set realtime to value (True/False)
        if self.config.climate[self.config.project.climate].realtime:
            self.realtime = self.config.climate[self.config.project.climate].realtime
            # make sure end date is correct
            lastdate = return_last_fullday()
            self.config.project.end = self.config.project.end.replace(year=lastdate.year, 
                                                                        month=lastdate.month,
                                                                        day=lastdate.day)

            # remove existing results (temporary results are ok and updated but cannot append to final files eg .outputs/downscaled/down_pt_0.nc)
            try:
                shutil.rmtree(self.config.outputs.path / "downscaled")
                print('---> Downscaled directory cleaned')
                os.makedirs(self.config.outputs.path / "downscaled")
            except:
                os.makedirs(self.config.outputs.path / "downscaled")

        # else set realtime to False
        else:
            self.realtime = False
                    
        self.realtime = False # always false now as redownload of current month handled elsewhere ? key required only to dynamically set config.project.end

    @staticmethod
    def load_config(config_file):
        try:
            with open(config_file, 'r') as f:
                return DefaultMunch.fromYAML(f)
        except IOError:
            print(f'ERROR: config file does not exist. \n\t Current file path: {config_file}\n\t Current working directory: {os.getcwd()}')


    def go_fetch(self, surf_plev='all'):
            """
            Funtion to call fetching of ERA5 data

            """
            if surf_plev.lower() == 'surf':
                self.get_plev = False
                self.get_surf = True 
            elif surf_plev.lower() == 'plev':
                self.get_plev = True 
                self.get_surf = False 

            elif surf_plev.lower() == 'all':
                self.get_plev = True 
                self.get_surf = True 

            lonW = self.config.project.extent.get('lonW') - 0.4
            lonE = self.config.project.extent.get('lonE') + 0.4
            latN = self.config.project.extent.get('latN') + 0.4
            latS = self.config.project.extent.get('latS') - 0.4

            # if keyword exists in config set out_format to value or default to netcdf
            if self.config.climate[self.config.project.climate].cds_output_format:
                output_format = self.config.climate[self.config.project.climate].cds_output_format
            else:
                output_format = 'netcdf'

            if self.config.climate[self.config.project.climate].cds_download_format:
                download_format = self.config.climate[self.config.project.climate].cds_download_format
            else: 
                download_format = 'unarchived'

            # Check data repository is currently supported
            clim = self.config.climate[self.config.project.climate]
            data_repository = self.config.climate[self.config.project.climate].data_repository or 'cds'
            if data_repository == 'gdex':
                if clim.realtime:
                    raise ValueError('realtime needs ERA5T from the CDS: GDEX is updated months later')
                if clim.product not in [None, {}, 'reanalysis']:
                    raise ValueError(f'GDEX only holds the ERA5 reanalysis, not {clim.product}')
                retrieve_era5_gdex(self.config.project.start, self.config.project.end, self.config.climate.path,
                                   latN, latS, lonE, lonW, step=clim.timestep or '1h', surf_plev=surf_plev.lower(),
                                   plevels=clim.plevels, num_processes=clim.download_threads or 4,
                                   root=self.config.climate[self.config.project.climate].gdex_root,
                                   rm_daily=bool(clim.rm_daily))
                return
            elif data_repository != 'cds':
                raise ValueError(f"Data repository {data_repository} not supported: use cds or gdex")

            if self.get_surf:
                # retreive ERA5 surface data
                retrieve_era5(
                    self.config.climate[self.config.project.climate].product,
                    self.config.project.start,
                    self.config.project.end,
                    self.config.climate.path,
                    latN, latS, lonE, lonW,
                    self.config.climate[self.config.project.climate].timestep,
                    self.config.climate[self.config.project.climate].download_threads,
                    surf_plev='surf',
                    realtime=self.realtime,
                    output_format=output_format,
                    download_format=download_format,
                    new_CDS_API=True,
                    rm_daily=self.config.climate[self.config.project.climate].rm_daily,
                    surf_varoi=self.varoi_surf,
                    plev_varoi=self.varoi_plev,
                    verify_flag=self.verify_flag
                )
            if self.get_plev:
                # retrieve era5 plevels
                retrieve_era5(
                    self.config.climate[self.config.project.climate].product,
                    self.config.project.start,
                    self.config.project.end,
                    self.config.climate.path,
                    latN, latS, lonE, lonW,
                    self.config.climate[self.config.project.climate].timestep,
                    self.config.climate[self.config.project.climate].download_threads,
                    surf_plev='plev',
                    plevels=self.config.climate[self.config.project.climate].plevels,
                    realtime=self.realtime,
                    output_format=output_format,
                    download_format=download_format,
                    new_CDS_API=True,
                    rm_daily=self.config.climate[self.config.project.climate].rm_daily,
                    surf_varoi=self.varoi_surf,
                    plev_varoi=self.varoi_plev,
                    verify_flag=self.verify_flag
                )


    def get_era5_snowmapper(self, surf_plev, lastday):
        """
        Funtion to call fetching of ERA5 data
        TODO:
        - merge monthly data into one file (cdo?)- this creates massive slow down!
        """
        lonW = self.config.project.extent.get('lonW') - 0.4
        lonE = self.config.project.extent.get('lonE') + 0.4
        latN = self.config.project.extent.get('latN') + 0.4
        latS = self.config.project.extent.get('latS') - 0.4


        # if keyword exists in config set out_format to value or default to netcdf
        if self.config.climate[self.config.project.climate].output_format:
            output_format = self.config.climate[self.config.project.climate].output_format
        # else set realtime to False
        else:
            output_format = 'netcdf'

        if surf_plev == 'surf':
            # retreive ERA5 surface data
            era5_request_surf_snowmapper(
                lastday,
                latN, latS, lonE, lonW,
                self.config.climate.path,
                output_format=output_format
            )

        if surf_plev == 'plev':            
            # retrieve era5 plevels
            era5_request_plev_snowmapper(
                lastday,
                latN, latS, lonE, lonW,
                self.config.climate.path,
                plevels=self.config.climate[self.config.project.climate].plevels,
                output_format=output_format
            )

    def to_zarr(self, zarr_store=None, time_chunk=720):
        """
        Convert the yearly PLEV/SURF netcdf files of the climate folder to the zarr store used for downscaling.

        Args:
            zarr_store (str): name of the store in the climate folder. Default: climate.<source>.zarr_store or ERA5.zarr
            time_chunk (int): number of time steps per chunk
        """
        name = zarr_store or self.config.climate[self.config.project.climate].zarr_store or 'ERA5.zarr'
        netcdf_to_zarr(path_to_netcdfs=climate_netcdf_dir(self.config.climate.path),
                       zarrout=self.config.climate.path / name,
                       plev_name='PLEV_*.nc',
                       surf_name='SURF_*.nc',
                       time_chunk=time_chunk)


def climate_netcdf_dir(climate_path, plev_name='PLEV_*.nc'):
    """
    Folder of the PLEV/SURF netcdf files to convert to zarr: climate_path/yearly (where get_era5() merges the daily
    files) if it holds some, else climate_path itself (files placed there by hand, as in the examples).
    """
    climate_path = Path(climate_path)
    yearly = climate_path / 'yearly'
    return yearly if any(yearly.glob(plev_name)) else climate_path


def append_nc_to_zarr_region(ncfile, zarrstore, surf=False):
    ncfile = str(ncfile)
    print(f"---> Appending {ncfile} to {zarrstore}")
    dp = xr.open_dataset(ncfile, chunks='auto')
    if surf:
        dp = dp.rename({'z':'z_surf'})
    dp.to_zarr(zarrstore, mode='a',region='auto', align_chunks=True)
    dp = None

def _standardize_era5(ds):
    """Harmonize ERA5 netcdf files: CDS-beta dimension names, ERA5/ERA5T expver, ascending pressure levels."""
    ds = ds.rename({k: v for k, v in {'valid_time': 'time', 'pressure_level': 'level'}.items()
                    if k in ds.variables or k in ds.dims})
    if 'expver' in ds.dims:
        # ERA5 (first expver) completed by ERA5T for the most recent dates
        parts = [ds.isel(expver=i, drop=True) for i in range(ds.sizes['expver'])]
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.combine_first(part)
        ds = merged
    ds = ds.drop_vars([c for c in ['number', 'expver'] if c in ds.variables])
    if 'level' in ds.dims:
        ds = ds.sortby('level')
    return ds


def netcdf_to_zarr(path_to_netcdfs='inputs/climate/yearly',
                   zarrout='inputs/climate/ERA5.zarr',
                   plev_name='PLEV_*.nc',
                   surf_name='SURF_*.nc',
                   time_chunk=720,
                   chunks=None,
                   max_chunk_mb=16,
                   compressor=None):
    """
    Convert pairs of PLEV and SURF netcdf files (e.g. yearly files) into one zarr store, file by file.

    Each pair is opened with xr.open_dataset, merged (the surface geopotential is renamed z_surf), cast to float32
    and appended along time. The store holds z, t, q, u, v, r (time, level, latitude, longitude) and z_surf, sp, t2m,
    d2m, tp, ssrd, strd (time, latitude, longitude), the layout read by topo_scale_zarr.ClimateDownscaler.

    Args:
        path_to_netcdfs (str): folder of the netcdf files
        zarrout (str): output zarr store (overwritten)
        plev_name (str): pattern of the pressure level files, * standing for the period (e.g. the year)
        surf_name (str): pattern of the surface files
        time_chunk (int): number of time steps per chunk
        chunks (dict): optional chunk sizes per dimension, overriding the automatic choice
        max_chunk_mb (float): target size of a pressure-level chunk, used to choose the spatial chunk size
        compressor (obj): zarr codec. Default: BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle')
    """
    pn, pz = Path(path_to_netcdfs), Path(zarrout)

    def by_period(pattern):
        prefix, suffix = pattern.split('*')
        return {f.name[len(prefix):len(f.name) - len(suffix)]: f for f in pn.glob(pattern)}

    plev_files, surf_files = by_period(plev_name), by_period(surf_name)
    periods = sorted(set(plev_files) & set(surf_files))
    unmatched = sorted(set(plev_files) ^ set(surf_files))
    if unmatched:
        warnings.warn(f'Files without PLEV/SURF counterpart are skipped: {unmatched}')
    if not periods:
        raise FileNotFoundError(f'No pair of {plev_name} and {surf_name} files in {pn}')
    if compressor is None:
        compressor = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)

    last_time = None
    for i, period in enumerate(periods):
        print(f'---> Converting {plev_files[period].name} and {surf_files[period].name} to {pz.name}')
        with xr.open_dataset(plev_files[period]) as dp, xr.open_dataset(surf_files[period]) as du:
            dp = _standardize_era5(dp)
            du = _standardize_era5(du).rename({'z': 'z_surf'})
            try:
                ds = xr.merge([dp, du], join='exact', combine_attrs='drop_conflicts')
            except ValueError as e:
                raise ValueError(f'{plev_files[period].name} and {surf_files[period].name} do not share the same '
                                 f'time, latitude and longitude coordinates: {e}')
            ds = ds.assign({v: ds[v].astype('float32') for v in ds.data_vars}).load()
        if last_time is not None and ds.time.values[0] <= last_time:
            raise ValueError(f'{plev_files[period].name} overlaps the previous file (starts at {ds.time.values[0]})')
        last_time = ds.time.values[-1]
        if i == 0:
            nlev = ds.sizes['level']
            spatial = max(1, int(np.sqrt(max_chunk_mb * 1e6 / (time_chunk * nlev * 4))))
            ch = {'time': time_chunk, 'level': nlev, 'latitude': min(ds.sizes['latitude'], spatial),
                  'longitude': min(ds.sizes['longitude'], spatial)}
            ch.update(chunks or {})
            encoding = {v: {'compressors': compressor, 'chunks': tuple(ch[d] for d in ds[v].dims)}
                        for v in ds.data_vars}
            ds.to_zarr(pz, mode='w', zarr_format=3, encoding=encoding)
        else:
            ds.to_zarr(pz, mode='a', append_dim='time')
    print(f'---> {pz} written ({len(periods)} periods)')
    return pz


def to_zarr(ds, fout, chuncks={'x':3, 'y':3, 'time':8760, 'level':7}, compressor=None, mode='w'):
    """
    Function to convert a dataset to a zarr store.

    Args:
        ds (dataset): dataset to convert to zarr
        fout (str): name of the zarr archive
        chuncks (dict): {'x':3, 'y':3, 'time':8760, 'level':7}
        compressor (obj): default is None, otherwise use an encoder compatible with zarr3. Blosc is default
        mode (str): mode to write to zarr store. See the xarray to_zarr() documentation
    """
    vars = list(ds.keys())

    if compressor is None:
        compressor = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)

    encoder = dict(zip(vars, [{'compressors': compressor}]*len(vars)))
    ds.chunk(chuncks).to_zarr(fout,  zarr_format=3, encoding=encoder, mode=mode)



def fetch_era5_google_from_zarr(eraDir, startDate, endDate, lonW, latS, lonE, latN, plevels, bucket='gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'):
    '''
    Function to download data from Zarr repository on Google Cloud Storage (https://github.com/google-research/arco-era5/tree/main)

    '''
    print('!!!!!!!!!WORK IN PROGRESS!!!!!!!!!')
    ds = xr.open_zarr(
        bucket,
        chunks=None,
        storage_options=dict(token='anon'),
    )
    ds_plev = ds[[
                'geopotential', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'specific_humidity'
            ]].sel(
        time=slice(startDate, endDate), 
        latitude=slice(latN,latS), 
        longitude=slice(lonE, lonW),
        level=plevels)
    ds_plev.to_zarr('PLEV.zarr')    
    ds_plev = None
    ds_surf = ds.sel(
        time=slice(startDate, endDate), 
        latitude=slice(latN,latS), 
        longitude=slice(lonE, lonW))[['geopotential_at_surface','2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                      'surface_solar_radiation_downwards','surface_pressure',
                      'total_precipitation', '2m_temperature', 'toa_incident_solar_radiation',
                      'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux'
                      ]]
    ds_surf.to_zarr('SURF.zarr')




# ---------------------------------------------------------------------------------------------------------------
# NCAR GDEX (Geoscience Data Exchange), dataset d633000: ERA5 on the 0.25 deg grid, the same data as the CDS.
# Surface analyses are zarr stores. Pressure levels (daily files), accumulated fluxes (half-month files of forecasts)
# and the surface geopotential are netCDF4 files: they are read in place, only the box is kept.
# ---------------------------------------------------------------------------------------------------------------
GDEX_POSIX_ROOT = '/gdex/data/d633000'                                   # NCAR systems
GDEX_OSDF_ROOT = 'https://osdf-director.osg-htc.org/ncar/gdex/d633000'   # anywhere, over HTTPS (OSDF)

GDEX_PLEV = {'z': '128_129_z.ll025sc', 't': '128_130_t.ll025sc', 'u': '128_131_u.ll025uv',
             'v': '128_132_v.ll025uv', 'q': '128_133_q.ll025sc'}
GDEX_SURF_AN = {'t2m': ('2t', '128_167_2t.ll025sc'), 'd2m': ('2d', '128_168_2d.ll025sc'),
                'sp': ('sp', '128_134_sp.ll025sc')}
# accumulations over the hour ending at the valid time (forecast start + step, steps 1 to 12), as on the CDS
GDEX_SURF_ACCUMU = {'ssrd': '128_169_ssrd.ll025sc', 'strd': '128_175_strd.ll025sc', 'tp': '128_228_tp.ll025sc'}
GDEX_INVARIANT_Z = 'e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'


def gdex_root(root=None):
    """Root of GDEX dataset d633000: root if given, else the local mount on NCAR systems, else OSDF over HTTPS."""
    if root not in [None, {}, '']:
        return str(root).rstrip('/')
    return GDEX_POSIX_ROOT if os.path.isdir(GDEX_POSIX_ROOT) else GDEX_OSDF_ROOT


def _is_remote(path):
    return str(path).startswith(('http://', 'https://'))


def _require_aiohttp():
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        raise ImportError('Reading GDEX over HTTPS needs aiohttp: pip install aiohttp')


def _data_var(ds):
    """The variable of a GDEX file (each file or store holds one, plus utc_date)."""
    names = [v for v in ds.data_vars if v != 'utc_date']
    if len(names) != 1:
        raise ValueError(f'Expected one data variable, found {names}')
    return ds[names[0]]


def _gdex_load(path, select):
    """Open a GDEX netCDF4 file (local path or URL), apply select to its variable, load the result, close the file."""
    if _is_remote(path):
        _require_aiohttp()
        import fsspec
        with fsspec.open(path, 'rb') as f, xr.open_dataset(f, engine='h5netcdf') as ds:
            return select(_data_var(ds)).load()
    with xr.open_dataset(path, engine='h5netcdf') as ds:
        return select(_data_var(ds)).load()


def _gdex_box(da, latN, latS, lonW, lonE):
    """Subset to the box. GDEX longitudes are 0..360; returned in -180..180, ascending, as the CDS does."""
    lat = da.latitude.values
    da = da.sel(latitude=slice(latN, latS) if lat[0] > lat[-1] else slice(latS, latN))
    w, e = lonW % 360, lonE % 360
    if w <= e:
        da = da.sel(longitude=slice(w, e))
    else:  # across the Greenwich meridian
        da = xr.concat([da.sel(longitude=slice(w, None)), da.sel(longitude=slice(None, e))], dim='longitude')
    lon = da.longitude.values
    lon = np.where(lon > 180, lon - 360, lon)
    if np.any(np.diff(lon) <= 0):
        raise ValueError('Domains across the 180 deg meridian are not supported')
    return da.assign_coords(longitude=lon)


def _accumu_init_step(times):
    """Forecast start (06 or 18 UTC) and step (1 to 12 h) giving each valid time, as ECMWF maps CDS times to MARS."""
    t = pd.DatetimeIndex(times)
    offset = np.where(t.hour >= 19, 18, np.where(t.hour >= 7, 6, -6))
    init = t.floor('D') + pd.to_timedelta(offset, unit='h')
    return init, ((t - init) / pd.Timedelta('1h')).astype(int)


def _accumu_file(root, stem, init):
    """Half-month file holding the forecasts started at init."""
    m = init.strftime('%Y%m')
    if init.day <= 15:
        span = f'{m}0106_{m}1606'
    else:
        span = f'{m}1606_{(init + pd.offsets.MonthBegin(1)).strftime("%Y%m")}0106'
    return f'{root}/e5.oper.fc.sfc.accumu/{m}/e5.oper.fc.sfc.accumu.{stem}.{span}.nc'


def _gdex_accumu(root, stem, times, box):
    init, step = _accumu_init_step(times)
    files = pd.Series([_accumu_file(root, stem, i) for i in init])
    parts = []
    for path in files.unique():
        k = np.flatnonzero(files.values == path)
        sel = dict(forecast_initial_time=xr.DataArray(init[k].values, dims='time'),
                   forecast_hour=xr.DataArray(step[k], dims='time'))
        da = _gdex_load(path, lambda d: _gdex_box(d.sel(**sel), *box))
        parts.append(da.drop_vars(['forecast_initial_time', 'forecast_hour'], errors='ignore')
                     .assign_coords(time=pd.DatetimeIndex(times)[k]))
    return xr.concat(parts, dim='time').sortby('time')


def _gdex_surf_an(root, short, stem, times, box):
    """Surface analysis from the zarr store, or from the monthly netCDF4 file where the store ends."""
    store = f'{root}/e5.oper.an.sfc.zarr/e5.oper.an.sfc.{short}.zarr'
    if _is_remote(store):
        _require_aiohttp()
    times = pd.DatetimeIndex(times)
    try:
        with xr.open_zarr(store, chunks=None) as ds:
            da = _data_var(ds)
            if da.time.values[-1] >= times[-1].to_datetime64():
                return _gdex_box(da.sel(time=times), *box).load()
    except FileNotFoundError:
        pass
    day = times[0]
    last = (day + pd.offsets.MonthEnd(0)).day
    path = (f'{root}/e5.oper.an.sfc/{day:%Y%m}/e5.oper.an.sfc.{stem}.'
            f'{day:%Y%m}0100_{day:%Y%m}{last:02d}23.nc')
    return _gdex_load(path, lambda d: _gdex_box(d.sel(time=times), *box))


def _gdex_day(root, kind, day, step, box, plevels, z_surf, target):
    """Write the daily file of one day (dSURF_ or dPLEV_), with the variables and layout of the CDS files."""
    times = pd.date_range(day, periods=24, freq='1h')[::int(pd.Timedelta(step) / pd.Timedelta('1h'))]
    if kind == 'plev':
        out = {}
        for name, stem in GDEX_PLEV.items():
            path = (f'{root}/e5.oper.an.pl/{day:%Y%m}/e5.oper.an.pl.{stem}.'
                    f'{day:%Y%m%d}00_{day:%Y%m%d}23.nc')
            out[name] = _gdex_load(path, lambda d: _gdex_box(d.sel(time=times, level=list(plevels)), *box))
        ds = xr.merge([v.rename(k) for k, v in out.items()], join='exact').sortby('level')
    else:
        out = {name: _gdex_surf_an(root, short, stem, times, box) for name, (short, stem) in GDEX_SURF_AN.items()}
        out.update({name: _gdex_accumu(root, stem, times, box) for name, stem in GDEX_SURF_ACCUMU.items()})
        out['z'] = z_surf.expand_dims(time=times)
        ds = xr.merge([v.rename(k) for k, v in out.items()], join='exact')
    ds = ds.drop_vars([c for c in ds.coords if c not in ('time', 'level', 'latitude', 'longitude')])
    ds.attrs = {'source': f'NCAR GDEX d633000 (ERA5), {root}'}
    tmp = Path(str(target) + '.part')
    ds.to_netcdf(tmp, encoding={v: {'dtype': 'float32', 'zlib': True, 'complevel': 4} for v in ds.data_vars})
    os.replace(tmp, target)
    return str(target)


def _gdex_day_star(args):
    return _gdex_day(*args)


def retrieve_era5_gdex(startDate, endDate, eraDir, latN, latS, lonE, lonW, step='1h', surf_plev='all',
                       plevels=None, num_processes=4, root=None, rm_daily=False):
    """
    Fetch ERA5 from NCAR GDEX (dataset d633000) instead of the CDS, into the same daily and yearly files.

    Writes eraDir/daily/dSURF_YYYYMMDD.nc and dPLEV_YYYYMMDD.nc, then eraDir/yearly/SURF_YYYY.nc and PLEV_YYYY.nc,
    with the variables, units and layout of the CDS files: z, t, u, v, q on pressure levels (ascending) and z, t2m,
    d2m, sp, ssrd, strd, tp at the surface; latitude descending, longitude in -180..180. Accumulated fluxes are
    over the hour ending at the time stamp, as on the CDS. Days already in eraDir/daily are skipped.

    On NCAR systems the files are read from /gdex/data. Elsewhere they are read over HTTPS (OSDF), which needs
    aiohttp. Pressure-level files are chunked as one global field per hour and per variable: whatever the size of
    the box, about 1 GB per variable and day is transferred.

    Args:
        startDate, endDate: first and last day
        eraDir (Path): climate folder
        latN, latS, lonE, lonW (float): box [deg]
        step (str): '1h', '3h' or '6h'
        surf_plev (str): 'surf', 'plev' or 'all'
        plevels (list): pressure levels [hPa]
        num_processes (int): days fetched in parallel (processes, as HDF5 reads hold a global lock)
        root (str): GDEX root (path or URL). Default: /gdex/data/d633000 if it exists, else OSDF over HTTPS
        rm_daily (bool): remove the daily folder once merged into yearly files
    """
    root = gdex_root(root)
    eraDir = Path(eraDir)
    kinds = ['surf', 'plev'] if surf_plev == 'all' else [surf_plev.lower()]
    if 'plev' in kinds and not plevels:
        raise ValueError('plevels are required for pressure-level data')
    days = pd.date_range(pd.Timestamp(startDate).normalize(), pd.Timestamp(endDate).normalize(), freq='D')
    (eraDir / 'daily').mkdir(parents=True, exist_ok=True)
    (eraDir / 'yearly').mkdir(parents=True, exist_ok=True)
    box = (latN, latS, lonW, lonE)

    jobs = [(kind, day, eraDir / 'daily' / f'd{kind.upper()}_{day:%Y%m%d}.nc') for kind in kinds for day in days]
    todo = [j for j in jobs if not j[2].is_file()]
    print(f'---> ERA5 from NCAR GDEX ({root}): {len(todo)} daily files to fetch, {len(jobs) - len(todo)} present')
    if _is_remote(root) and any(k == 'plev' for k, _, _ in todo):
        n = sum(k == 'plev' for k, _, _ in todo) / (pd.Timedelta(step) / pd.Timedelta('1h'))
        print(f'     Pressure levels are stored as global hourly fields: about {5 * n:.0f} GB to transfer '
              f'whatever the size of the domain. On NCAR systems they are read locally')

    z_surf = None
    if any(k == 'surf' for k, _, _ in todo):
        z_surf = _gdex_load(f'{root}/{GDEX_INVARIANT_Z}',
                            lambda d: _gdex_box(d.isel(time=0, drop=True) if 'time' in d.dims else d, *box))
    args = [(root, kind, day, step, box, plevels, z_surf, target) for kind, day, target in todo]
    if num_processes and num_processes > 1 and len(args) > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(min(num_processes, len(args))) as pool:
            for f in pool.imap_unordered(_gdex_day_star, args):
                print(f'     {Path(f).name}')
    else:
        for a in args:
            print(f'     {Path(_gdex_day(*a)).name}')

    for kind in kinds:
        for year in sorted(set(days.year)):
            _merge_daily_year(eraDir, kind, year)
    if rm_daily:
        shutil.rmtree(eraDir / 'daily')
        print('---> Daily files removed')
    print('===> ERA5 netcdf files ready')


def _merge_daily_year(eraDir, kind, year):
    """Concatenate the daily files of one year into yearly/SURF_YYYY.nc or PLEV_YYYY.nc, file by file."""
    prefix = kind.upper()
    files = sorted((Path(eraDir) / 'daily').glob(f'd{prefix}_{year:04d}*.nc'))
    if not files:
        return
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(_standardize_era5(ds).load())
    out = Path(eraDir) / 'yearly' / f'{prefix}_{year:04d}.nc'
    xr.concat(parts, dim='time').sortby('time').to_netcdf(out)
    print(f'---> {out.name} written from {len(files)} daily files')


def retrieve_era5(product, startDate, endDate, eraDir, latN, latS, lonE, lonW, step, 
                    num_threads=10, surf_plev='surf', plevels=None, realtime=False, 
                    output_format='netcdf', download_format="unarchived", new_CDS_API=True, 
                    rm_daily=False, surf_varoi=None, plev_varoi=None, verify_flag=True):
    """ Sets up era5 surface retrieval.
    * Creates list of year/month pairs to iterate through.
    * MARS retrievals are most efficient when subset by time.
    * Identifies preexisting downloads if restarted.
    * Calls api using parallel function.

    Args:
        product: "reanalysis" (HRES) or "ensemble_members" (EDA)
        startDate:
        endDate:
        eraDir: directory to write output
        latN: north latitude of bbox
        latS: south latitude of bbox
        lonE: easterly lon of bbox
        lonW: westerly lon of bbox
        step: timestep to use: 1, 3, 6
        num_threads: number of threads to use for downloading data
        surf_plev: download surface single level or pressure level product: 'surf' or 'plev'
        output_format (str): default is "netcdf", can be "grib".
        download_format (str): default "unarchived". Can be "zip"
        new_CDS_API: flag to handle new formating of SURF files with the new CDS API (2024).
        rm_daily: remove folder containing all daily ERA5 file. Option to clear space of data converted to yearly files.
        surf_varoi: list of surface variable to download. Default is None, automatically assigning minimum required for the downscaling
        plev_varoi: list of pressure level variables to download. . Default is None, automatically assigning minimum required for the downscaling
        verify_flag: bypass ssl certifcate flag. Default must be True
        
    Returns:
        Monthly era surface files stored in disk.

    """
    print('\n')
    print('---> Loading ERA5 {} climate forcing'.format(surf_plev))
    bbox = [str(latN), str(lonW), str(latS), str(lonE)]
    time_step_dict = {'1h': ['00:00', '01:00', '02:00',
                             '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00',
                             '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00',
                             '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00',
                             '21:00', '22:00', '23:00'],
                    '3h': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
                    '6h': ['00:00', '06:00', '12:00', '18:00']}

    df = pd.DataFrame()
    # date_range will make sure to include the month of the latest date (endDate) provided
    #df['dates'] = pd.date_range(startDate, pd.Timestamp(endDate)-pd.offsets.Day()+pd.offsets.MonthEnd(), freq='M', inclusive='both')
    df['dates'] = pd.date_range(startDate, pd.Timestamp(endDate), freq='D', inclusive='both')
    df['month'] = df.dates.dt.month
    df['day'] = df.dates.dt.day
    df['year'] = df.dates.dt.year

    os.makedirs(eraDir / "daily", exist_ok=True)

    if surf_plev.lower() == 'surf':
        df['dataset'] = 'reanalysis-era5-single-levels'
        if output_format == "netcdf":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dSURF_%04d%02d%02d.nc" % (x.year, x.month, x.day)))
        if output_format == "grib":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dSURF_%04d%02d%02d.grib" % (x.year, x.month, x.day)))
    
    elif surf_plev.lower() == 'plev':
        df['dataset'] = 'reanalysis-era5-pressure-levels'
        if output_format == "netcdf":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dPLEV_%04d%02d%02d.nc" % (x.year, x.month, x.day)))
        if output_format == "grib":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dPLEV_%04d%02d%02d.grib" % (x.year, x.month, x.day)))
        loc_list = []
        loc_list.extend([plevels]*df.shape[0])
        df['plevels'] = loc_list
    
    else:
        sys.exit('ERROR: surf_plev can only be surf or plev')
    
    df['file_exist'] = 0
    df.file_exist = df.target_file.apply(lambda x: os.path.isfile(x)*1)

    # if file does not exist, check if data already combined into yearly netcdf
    print("TBI: automatically check if daily data already in yearly concatenated netcdf files")
    print("--- If data already dowloaded, do not fetch data ")

    df['step'] = step
    df['time_steps'] = df.step.apply(lambda x: time_step_dict.get(x))
    df['bbox'] = df.step.apply(lambda x: bbox)
    df['product_type'] = product
    df['output_format'] = output_format
    df['download_format'] = download_format
    df['verify_flag'] = verify_flag

    if surf_varoi is None:
        surf_varoi = ['geopotential', 
                    '2m_dewpoint_temperature',
                    'surface_thermal_radiation_downwards',
                    'surface_solar_radiation_downwards',
                    'surface_pressure',
                    'total_precipitation',
                    '2m_temperature']
    if plev_varoi is None: 
        plev_varoi = ['geopotential',
                 'temperature',
                 'u_component_of_wind',
                 'v_component_of_wind',
                 'specific_humidity',
                 'relative_humidity']
    df['surf_varoi'] = df.step.apply(lambda x: surf_varoi)
    df['plev_varoi'] = df.step.apply(lambda x: plev_varoi)

    print("Start = ", df.dates[0].strftime('%Y-%m-%d'))
    print("End = ", df.dates[len(df.dates) - 1].strftime('%Y-%m-%d'))

    if df.file_exist.sum() > 0:
        print("ECWMF daily {} data found:".format(surf_plev.upper()))
        print(df.target_file.loc[df.file_exist == 1].apply(lambda x: x.name))

    if (df.file_exist == 0).sum() > 0:
        print("Downloading daily {} from ECWMF:".format(surf_plev.upper()))
        print(df.target_file.loc[df.file_exist == 0].apply(lambda x: x.name))

    download = df.loc[df.file_exist == 0]
    if download.shape[0] > 0:
        if surf_plev == 'surf':
            pool = ThreadPool(num_threads)
            pool.starmap(era5_request_surf, zip(list(download.dataset),
                                                list(download.year),
                                                list(download.month),
                                                list(download.day),
                                                list(download.bbox),
                                                list(download.target_file),
                                                list(download.product_type),
                                                list(download.time_steps),
                                                list(download.verify_flag),
                                                list(download.output_format),
                                                list(download.download_format),
                                                list(download.surf_varoi),))
            pool.close()
            pool.join()
        elif surf_plev == 'plev':
            pool = ThreadPool(num_threads)
            pool.starmap(era5_request_plev, zip(list(download.dataset),
                                                list(download.year),
                                                list(download.month),
                                                list(download.day),
                                                list(download.bbox),
                                                list(download.target_file),
                                                list(download.product_type),
                                                list(download.time_steps),
                                                list(download.plevels),
                                                list(download.verify_flag),
                                                list(download.output_format),
                                                list(download.download_format),
                                                list(download.plev_varoi)))
            pool.close()
            pool.join()
        else:
            sys.exit('ERROR: surf_plev can only be surf or plev')
        #else:
        #	sys.exit('ERROR: Some forcing files are missing given the date range provided\n ---> or implement a method to modify start/end date of project to file available')

    if realtime:
        if surf_plev == 'surf':
            # redownload current month to catch missing days in realtime mode.
            era5_realtime_surf(eraDir / "daily", df.dataset[0], df.bbox[0], df.product_type[0])

        if surf_plev == 'plev':
            # redownload current month to catch missing days in realtime mode.
            era5_realtime_plev(eraDir / "daily", df.dataset[0], df.bbox[0], df.product_type[0], df.plevels[0])

    if new_CDS_API:
        # Since mid-2024, CDS API has changed variable name and the way it packages SURF variables. 
        # These two functions apply a small post process to get things in order.
        if surf_plev == 'surf':
            remap_CDSbeta(str(eraDir / "daily" / "dSURF*.nc"), file_type='surf')
        if surf_plev == 'plev':
            remap_CDSbeta(str(eraDir / "daily" / "dPLEV*.nc"), file_type='plev')

    # code to merge daily files to monthly
    # - [ ] write option earlier that checks also if monthly files already exist. improve logic that does not require to have both monthly and daily files
    cdo = Cdo()
    for year in df.year.unique():
        print(f"---> Merging daily {surf_plev.upper()} files from {year} to yearly file")
        
        if surf_plev == 'surf':
            fpat = str(eraDir / "daily" / ("dSURF_%04d*.nc" % (year)))
            fout = str(eraDir / "yearly" / ("SURF_%04d.nc" % (year)))


            try:
                cdo.mergetime(input=fpat, output=fout)
            except Exception as e:
                print(f"CDO warning (non-fatal): {e}")
                # Check if output file was actually created despite warnings
                if os.path.exists(fout):
                    print(f"Merge completed successfully: {fout}")
                else:
                    raise e  # Re-raise if it's a real error



        if surf_plev == 'plev':
            fpat = str(eraDir / "daily" / ("dPLEV_%04d*.nc" % (year)))
            fout = str(eraDir / "yearly" / ("PLEV_%04d.nc" % (year)))

            try:
                cdo.mergetime(input=fpat, output=fout)
            except Exception as e:
                print(f"CDO warning (non-fatal): {e}")
                # Check if output file was actually created despite warnings
                if os.path.exists(fout):
                    print(f"Merge completed successfully: {fout}")
                else:
                    raise e  # Re-raise if it's a real error
            

    if rm_daily:
        try:
            shutil.rmtree(eraDir / "daily")
            print('---> Daily files removed')
        except:
            raise IOError('deletion of daily files issue')

    print("===> ERA5 netcdf files ready")


def era5_request_surf(dataset, year, month, day, bbox, target, product, time, verify_flag=True, output_format= "netcdf", download_format="unarchived", varoi=None):
    """CDS surface api call

    Args:
        dataset (str): copernicus dataset (era5)
        year (str or list): year of interest
        month (str or list): month of interest
        bbox (list): bonding box in lat-lon
        target (str): filename
        product (str): type of model run. defaul: reanalysis
        time (str or list): hours for which to download data
        output_format (str): default is "netcdf", can be "grib".
        download_format (str): default "unarchived". Can be "zip"
        varoi (list): list of variable of interest to download. Default to None which fallsback to minimum needed for TPS

    Returns:
        Store to disk dataset as indicated

    TODO: 
    - [ ] change SURF request to the ZARR repository for timeseries retrieval.

    """
    if varoi is None:
        varoi = ['geopotential', 
                    '2m_dewpoint_temperature',
                    'surface_thermal_radiation_downwards',
                    'surface_solar_radiation_downwards','surface_pressure',
                    'total_precipitation',
                    '2m_temperature']
    
    c = cdsapi.Client(verify=verify_flag)
    c.retrieve(
        dataset,
        {'variable': varoi,
         'product_type': [product],
         "area": bbox,
         'year': year,
         'month': '%02d'%(month),
         'day': '%02d'%(day),
         'time': time,
         'grid': "0.25/0.25",
         'data_format': output_format,
         'download_format': download_format
         },
        target)
    print(str(target) + " complete")
    unzip_file(str(target))


def era5_request_plev(dataset, year, month, day, bbox, target, product, time, plevels, verify_flag=True, output_format= "netcdf", download_format="unarchived", varoi=None):
    """CDS plevel api call

    Args:
        dataset (str): copernicus dataset (era5)
        year (str or list): year of interest
        month (str or list): month of interest
        bbox (list): bonding box in lat-lon
        target (str): filename
        product (str): type of model run. defaul: reanalysis
        time (str or list): hours to query
        plevels (str or list): pressure levels to query
        output_format (str): default is "netcdf", can be "grib".
        download_format (str): default "unarchived". Can be "zip"
        varoi (list): list of variable of interest to download. Default to None which fallsback to minimum needed for TPS

    Returns:
        Store to disk dataset as indicated

    """
    if varoi is None:
        varoi = ['geopotential',
                 'temperature',
                 'u_component_of_wind',
                 'v_component_of_wind',
                 'specific_humidity',
                 'relative_humidity']


    c = cdsapi.Client(verify=verify_flag)
    c.retrieve(
        dataset,
        {
            'product_type': [product],
            "area": bbox,
            'variable': varoi,
            'pressure_level': plevels,
            'year': year,
            'month': '%02d'%(month),
            'day': '%02d'%(day),
            'time': time,
            'grid': "0.25/0.25",
            'data_format': output_format,
            'download_format': download_format
        },
        target)
    print(str(target) + " complete")
    unzip_file(str(target))



def era5_realtime_surf(eraDir, dataset, bbox, product ):
    # a small routine thaty simply redownloads the current month
    # this is a simple (not most efficient) way of ensuring we have latest data for realtime applications
    # a better way woulfd be to compare date that data is available for with date in latest files, download the difference and append - tobe implemented!

    # get current month
    currentMonth = datetime.now().month
    currentYear = datetime.now().year
    print("Running realtime download " + str(currentYear) + "_" + str(currentMonth))

    time = ['00:00', '01:00', '02:00',
                             '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00',
                             '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00',
                             '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00',
                             '21:00', '22:00', '23:00']

    target = str(eraDir / "SURF_%04d%02d.nc" % (currentYear, currentMonth))
    era5_request_surf(dataset, currentYear, currentMonth, bbox, target, product, time)


def era5_realtime_plev(eraDir, dataset, bbox, product,plevels ):
    # a small routine thaty simply redownloads the current month
    # this is a simple (not most efficient) way of ensuring we have latest data for realtime applications
    # a better way woulfd be to compare date that data is available for with date in latest files, download the difference and append - tobe implemented!

    # get current month
    currentMonth = datetime.now().month
    currentYear = datetime.now().year
    print("Running realtime download " + str(currentYear) + "_" + str(currentMonth))

    time = ['00:00', '01:00', '02:00',
                             '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00',
                             '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00',
                             '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00',
                             '21:00', '22:00', '23:00']

    target = str(eraDir / "PLEV_%04d%02d.nc" % (currentYear, currentMonth))
    plevels = plevels
    era5_request_plev(dataset, currentYear, currentMonth, bbox, target, product, time, plevels)



def return_last_fullday():
    """
    TODO: NEED docstring and explanation 
    """


    # Get current UTC time
    current_time_utc = datetime.utcnow()

    # Round down to the nearest hour
    rounded_time_utc = current_time_utc.replace(minute=0, second=0, microsecond=0)

    # Get time 6 days ago in UTC
    six_days_ago_utc = rounded_time_utc - timedelta(days=6)

    # Format the time strings
    current_time_utc_str = rounded_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    six_days_ago_utc_str = six_days_ago_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Print the results
    six_days_ago_utc = datetime.strptime(six_days_ago_utc_str, "%Y-%m-%d %H:%M:%S UTC")

    # # if 5 days ago is an incomplete day h<23, Subtract one day from the datetime object to get last full day of data
    # if six_days_ago_utc.hour < 23:
    #     last_fullday_data = six_days_ago_utc - timedelta(days=1)
    # else:
    #     last_fullday_data = six_days_ago_utc
    last_fullday_data_str = six_days_ago_utc.strftime("%Y-%m-%d")
    print("Current time (rounded down) in UTC:", current_time_utc_str)
    print("Last full day ERA5T data in UTC:", last_fullday_data_str)

    return six_days_ago_utc


def grib2netcdf(gribname, outname=None):
    """
    Function to convert grib file to netcdf

    Args:
        gribname: filename fo grib file to convert
    """
    if outname is None:
        outname = gribname[:-5] + ".nc"

    cdo = Cdo()
    cdo.copy(option='-f nc4', input=gribname, output=outname)


def unzip_file(file_path):
    """
    Function to unpack and repack as NETCDF data sent by the new CDS API. CDS can send either weird zip file shipped as netcdf file for Surface data. 
    The option of zip format can be used to download the data. 
    
    Args:
        file_path: path of era5 data. Typically in TopoPyScale project it will be at: ./inputs/climate/daily
    """
        
    # Step 1: Try to open as a NetCDF file
    try:
        with xr.open_dataset(file_path) as ds:
            print(f"{file_path} is a valid NetCDF file. No processing needed.")

    except Exception:
        print(f"{file_path} is not a valid NetCDF file. Checking if it's a ZIP file.")
    
        # Step 2: Check if it's a ZIP file
        if zipfile.is_zipfile(file_path):
            if file_path[-3:]=='.nc':
                print(f"{file_path} is neither a valid NetCDF nor a ZIP file.")
            
                # Step 3: Rename the file if it's actually a ZIP
                zip_file_path = file_path.replace('.nc', '.zip')
                os.rename(file_path, zip_file_path)
                print(f"Renamed {file_path} to {zip_file_path} for processing.")
        
            # Step 4: Unzip the file
            unzip_dir = zip_file_path.replace('.zip', '')
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            print(f"Unzipped {zip_file_path} to {unzip_dir}.")
            
            # Step 5: Merge `.nc` files inside the unzipped directory
            nc_files = [os.path.join(unzip_dir, f) for f in os.listdir(unzip_dir) if f.endswith('.nc')]
            if not nc_files:
                print(f"No .nc files found in {unzip_dir}.")
                return
            
            merged_file_path = os.path.join(os.path.dirname(zip_file_path), os.path.basename(zip_file_path).replace('.zip', '.nc'))
            try:
                # Combine all `.nc` files
                datasets = [xr.open_dataset(nc_file).load() for nc_file in sorted(nc_files)]
                merged_ds = xr.merge(datasets, compat='override')  # override conflicting metadata like 'expver'
                [d.close() for d in datasets]
                merged_ds.to_netcdf(merged_file_path)
                print(f"Merged .nc files into {merged_file_path}.")
            except:
                raise IOError("Unzipping failing.")

            # Step 6: Clean up
            os.remove(zip_file_path)
            shutil.rmtree(unzip_dir)
            print(f"Deleted {zip_file_path} and {unzip_dir}.")


def remap_CDSbeta(file_pattern, file_type='SURF'):
    """
    Remapping of variable names from CDS beta to CDS legacy standard.
    
    Args:
        file_pattern (str): path and pattern of the file to be remapped. example: 'inputs/climate/daily/dSURF_*.nc'
        file_type (str): type of file: SURF or PLEV.  
    """

    flist = glob.glob(file_pattern)

    if file_type.lower() == 'plev':
        for nc_file in flist:
            try:
                ds = xr.open_dataset(nc_file)
                ds = ds.rename({ 'pressure_level': 'level', 'valid_time' : 'time'})
                ds = ds.sortby('level', ascending=True)  # sort levels ascending (300 to 1000)

                try:
                    ds = ds.drop_vars('number')
                except:
                    print("Coordinate 'number' not found")

                try:
                    ds = ds.drop_vars('expver')
                except:
                    print("Coordinate 'expver' not found")

                ds.to_netcdf(nc_file+ "_remap", mode='w')
                # move remap back to orig name
                os.rename(nc_file + "_remap", nc_file)
            except:
                print(nc_file+" already remapped or fc file.")

    if file_type.lower() == 'surf':
        for nc_file in flist:
            try:
                ds = xr.open_dataset(nc_file)
                ds = ds.rename({ 'valid_time' : 'time'})

                try:
                    ds = ds.drop_vars('number')
                except:
                    print("Coordinate 'number' not found")

                try:
                    ds = ds.drop_vars('expver')
                except:
                    print("Coordinate 'expver' not found")

                ds.to_netcdf(nc_file + "_remap", mode='w')
                # move remap back to orig name
                os.rename(nc_file + "_remap", nc_file)

            except:
                print(nc_file+" already remapped or fc file.")







def era5_request_surf_snowmapper( today, latN, latS, lonE, lonW, eraDir, output_format= "netcdf"):
    """CDS surface api call

    Args:
        dataset (str): copernicus dataset (era5)
        today (datetime object): day to download
        target (str): filename
        product (str): type of model run. defaul: reanalysis
        time (str or list): hours for which to download data
        format (str): "grib" or "netcdf"

    Returns:
        Store to disk dataset as indicated

    """
    time = ['00:00', '01:00', '02:00',
                             '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00',
                             '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00',
                             '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00',
                             '21:00', '22:00', '23:00']

    bbox = [str(latN), str(lonW), str(latS), str(lonE)]

    target = str(eraDir) + "/forecast/SURF_%04d%02d%02d.nc" % (today.year, today.month, today.day)

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {'variable': ['geopotential', '2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                      'surface_solar_radiation_downwards','surface_pressure',
                      'Total precipitation', '2m_temperature', 'TOA incident solar radiation',
                      'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux'
                      ],
         'product_type': ['reanalysis'],
         "area": bbox,
         'year': [today.year],
         'month': '%02d'%(today.month),
         'day': [today.day],
         'time': time,
         'grid': [0.25, 0.25],
         'data_format': output_format,
         'download_format': 'unarchived'
         },
        target)
    print(target + " complete")


def era5_request_plev_snowmapper(today, latN, latS, lonE, lonW, eraDir, plevels, output_format= "netcdf"):
    """CDS plevel api call

    Args:
        dataset (str): copernicus dataset (era5)
        year (str or list): year of interest
        month (str or list): month of interest
        bbox (list): bonding box in lat-lon
        target (str): filename
        product (str): type of model run. defaul: reanalysis
        time (str or list): hours to query
        plevels (str or list): pressure levels to query

    Returns:
        Store to disk dataset as indicated

    """
    time = ['00:00', '01:00', '02:00',
                             '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00',
                             '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00',
                             '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00',
                             '21:00', '22:00', '23:00']

    bbox = [str(latN), str(lonW), str(latS), str(lonE)]

    target = str(eraDir) + "/forecast/PLEV_%04d%02d%02d.nc" % (today.year, today.month, today.day)


    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': ['reanalysis'],
            "area": bbox,
            'variable': [
                'geopotential', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'relative_humidity', 'specific_humidity'
            ],
            'pressure_level': plevels,
            'year': [today.year],
            'month': '%02d'%(today.month),
            'day': [today.day],
            'time': time,
            'grid': [0.25, 0.25],
            'data_format': output_format,
            'download_format': 'unarchived'
        },
        target)
    print(target + " complete")
