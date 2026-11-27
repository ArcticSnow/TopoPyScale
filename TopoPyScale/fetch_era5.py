"""
Retrieve ERA5 from the CDS (cdsapi) or NCAR GDEX into the zarr store read by the downscaling (see era5_store.py).

- J. Fiddes, Origin implementation
- S. Filhol adapted in 2021

"""
# !/usr/bin/env python
import functools
import threading

import pandas as pd
import xarray as xr
import numpy as np
try:
    from zarr.codecs import BloscCodec
except ImportError:
    # For newer versions of zarr
    try:
        from zarr.codecs._blosc import BloscCodec
    except ImportError:
        # Fallback for older versions or different structure
        from zarr import Blosc as BloscCodec

import cdsapi, os
from datetime import datetime, timedelta, timezone
from TopoPyScale import era5_store as es
from pathlib import Path
import zipfile
import shutil
import glob
from munch import DefaultMunch

import warnings
import logging

logger = logging.getLogger(__name__)


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
            logger.info("---> Config loaded")
        elif type(config) is str:
            self.config = self.load_config(config_file=config)
            logger.info(f"---> Config from {config} loaded")
        else:
            raise ValueError('Cannot find config. Check path to config, or config is a Munch.DefaultMunch object.')

        self.varoi_surf = varoi_surf
        self.varoi_plev = varoi_plev
        self.verify_flag = verify_ssl
        os.makedirs(self.config.climate.path, exist_ok=True)

        # realtime: the period ends on the last full day of ERA5T. The store records which time steps it holds, so
        # the next run fetches the days added since (the current month is requested again, as one month)
        self.realtime = bool(self.config.climate[self.config.project.climate].realtime)
        if self.realtime:
            lastdate = return_last_fullday()
            self.config.project.end = self.config.project.end.replace(year=lastdate.year, month=lastdate.month,
                                                                        day=lastdate.day)
            # downscaled outputs are rewritten
            shutil.rmtree(self.config.outputs.path / "downscaled", ignore_errors=True)
            os.makedirs(self.config.outputs.path / "downscaled", exist_ok=True)
            logger.info('---> Downscaled directory cleaned')

    @staticmethod
    def load_config(config_file):
        if not Path(config_file).is_file():
            raise FileNotFoundError(f'Config file {config_file} not found (current directory: {os.getcwd()})')
        with open(config_file, 'r') as f:
            return DefaultMunch.fromYAML(f)


    def go_fetch(self, surf_plev='all'):
        """
        Fetch ERA5 into the zarr store of the climate folder (climate.<source>.zarr_store, default ERA5.zarr), from
        the CDS or NCAR GDEX (climate.<source>.data_repository). See fetch_era5_to_zarr().

        Args:
            surf_plev (str): 'all', 'surf' or 'plev'

        Returns:
            Path: the store
        """
        streams = {'all': ('plev', 'surf'), 'surf': ('surf',), 'plev': ('plev',)}.get(surf_plev.lower())
        if streams is None:
            raise ValueError(f'surf_plev must be all, surf or plev, not {surf_plev}')
        clim = self.config.climate[self.config.project.climate]
        source = clim.data_repository or 'cds'
        if source not in ('cds', 'gdex'):
            raise ValueError(f"Data repository {source} not supported: use cds or gdex")
        if source == 'gdex' and clim.realtime:
            raise ValueError('realtime needs ERA5T from the CDS: GDEX is updated months later')
        if clim.cds_output_format not in [None, {}, 'netcdf']:
            logger.info(f'     cds_output_format={clim.cds_output_format} ignored: netcdf is downloaded and written to '
                  f'the zarr store')
        if clim.rm_daily:
            logger.info('     rm_daily ignored: ERA5 is written to the zarr store, without daily files')
        extent = self.config.project.extent
        return fetch_era5_to_zarr(self.config.project.start, self.config.project.end, self.config.climate.path,
                                  latN=extent.get('latN') + 0.4, latS=extent.get('latS') - 0.4,
                                  lonE=extent.get('lonE') + 0.4, lonW=extent.get('lonW') - 0.4,
                                  step=clim.timestep or '1h', plevels=clim.plevels, streams=streams, source=source,
                                  product=clim.product or 'reanalysis', zarr_store=clim.zarr_store or 'ERA5.zarr',
                                  n_workers=clim.download_threads or None, plev_variables=self.varoi_plev,
                                  surf_variables=self.varoi_surf, gdex_root=clim.gdex_root or None,
                                  download_format=clim.cds_download_format or 'unarchived', verify=self.verify_flag)

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
    logger.info(f"---> Appending {ncfile} to {zarrstore}")
    dp = xr.open_dataset(ncfile, chunks='auto')
    if surf:
        dp = dp.rename({'z':'z_surf'})
    dp.to_zarr(zarrstore, mode='a',region='auto', align_chunks=True)
    dp = None

_standardize_era5 = es.standardize


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
        logger.info(f'---> Converting {plev_files[period].name} and {surf_files[period].name} to {pz.name}')
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
    logger.info(f'---> {pz} written ({len(periods)} periods)')
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
    logger.info('!!!!!!!!!WORK IN PROGRESS!!!!!!!!!')
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


gdex_root_path = gdex_root


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


def _gdex_day(root, kind, day, step, box, plevels, z_surf, target, acc_hours=1):
    """
    Write the file of one day and stream ('plev' or 'surf') to target, with the variables and layout of the CDS
    files. The surface geopotential z_surf is added to surface files unless None. With acc_hours > 1, the accumulated
    fields are read at every hour of the windows of acc_hours hours ending at the time steps (from the previous day),
    to be summed by era5_store.
    """
    times = pd.date_range(day, periods=24, freq='1h')[::int(pd.Timedelta(step) / pd.Timedelta('1h'))]
    acc_times = times if acc_hours == 1 else pd.DatetimeIndex(np.unique(es.window_hours(times, acc_hours).ravel()))
    if kind == 'plev':
        out = {}
        for name, stem in GDEX_PLEV.items():
            path = (f'{root}/e5.oper.an.pl/{day:%Y%m}/e5.oper.an.pl.{stem}.'
                    f'{day:%Y%m%d}00_{day:%Y%m%d}23.nc')
            out[name] = _gdex_load(path, lambda d: _gdex_box(d.sel(time=times, level=list(plevels)), *box))
        ds = xr.merge([v.rename(k) for k, v in out.items()], join='exact').sortby('level')
    else:
        out = {name: _gdex_surf_an(root, short, stem, times, box) for name, (short, stem) in GDEX_SURF_AN.items()}
        out.update({name: _gdex_accumu(root, stem, acc_times, box) for name, stem in GDEX_SURF_ACCUMU.items()})
        if z_surf is not None:
            out['z'] = z_surf.expand_dims(time=times)
        ds = xr.merge([v.rename(k) for k, v in out.items()], join='exact' if acc_hours == 1 else 'outer')
    ds = ds.drop_vars([c for c in ds.coords if c not in ('time', 'level', 'latitude', 'longitude')])
    ds.attrs = {'source': f'NCAR GDEX d633000 (ERA5), {root}'}
    tmp = Path(str(target) + '.part')
    ds.to_netcdf(tmp, encoding={v: {'dtype': 'float32', 'zlib': True, 'complevel': 4} for v in ds.data_vars})
    os.replace(tmp, target)
    return str(target)


# ---------------------------------------------------------------------------------------------------------------
# ERA5 to the zarr store, month by month from the CDS or day by day from GDEX (see era5_store.py)
# ---------------------------------------------------------------------------------------------------------------
CDS_DATASETS = {'plev': 'reanalysis-era5-pressure-levels', 'surf': 'reanalysis-era5-single-levels',
                'zsurf': 'reanalysis-era5-single-levels'}
CDS_MAX_PARALLEL = 10   # requests of one user the CDS runs at the same time; more only wait in its queue
_cds_local = threading.local()


def _cds_client(verify=True):
    """CDS client of the calling thread (one per thread, reused by its requests)."""
    if getattr(_cds_local, 'client', None) is None or _cds_local.verify != verify:
        _cds_local.client = cdsapi.Client(verify=verify, progress=False, quiet=True)
        _cds_local.verify = verify
    return _cds_local.client


def cds_request(task, variables, area, step, plevels=None, product='reanalysis', download_format='unarchived',
                hours=None):
    """
    CDS API request of a task: its days (one month at most), all time steps (or the given hours, e.g. [0, 1, ...]),
    the box on the 0.25 deg grid.
    """
    days = sorted(task.days)
    if task.stream == 'zsurf':
        days, hours = days[:1], ['00:00']
    elif hours is not None:
        hours = [f'{int(h):02d}:00' for h in hours]
    else:
        hours = [f'{h:02d}:00' for h in range(0, 24, es.STEP_HOURS[step])]
    if len({(d.year, d.month) for d in days}) > 1:
        raise ValueError(f'{task.key}: a CDS request covers one month at most')
    request = {'product_type': [product], 'variable': list(variables), 'year': [f'{days[0].year}'],
               'month': [f'{days[0].month:02d}'], 'day': [f'{d.day:02d}' for d in days], 'time': hours,
               'area': [float(a) for a in area], 'grid': [es.GRID, es.GRID], 'data_format': 'netcdf',
               'download_format': download_format}
    if task.stream == 'plev':
        request['pressure_level'] = [str(int(p)) for p in sorted(plevels)]
    return request


def _cds_cost_limit(e):
    text = str(e).lower()
    return 'cost limits exceeded' in text or 'request is too large' in text


def _cds_fatal(e):
    """Errors that a new attempt does not cure: client errors (4xx) except timeouts and rate limits."""
    status = getattr(getattr(e, 'response', None), 'status_code', None)
    return _cds_cost_limit(e) or (status is not None and 400 <= status < 500 and status not in (408, 429))


def _container(path):
    """'nc' or 'zip' from the first bytes of a download; IOError for anything else (an error page, a cut file)."""
    path = Path(path)
    with open(path, 'rb') as f:
        head = f.read(8)
    if head.startswith(b'\x89HDF') or head.startswith(b'CDF'):
        return 'nc'
    if zipfile.is_zipfile(path):
        return 'zip'
    raise IOError(f'{path.name} is not a netCDF or zip file (starts with {head!r})')


def _cds_download(dataset, request, stem, verify):
    """Retrieve to stem.part; renamed stem.nc or stem.zip once complete, so a cut download is never taken as done."""
    part = Path(f'{stem}.part')
    _cds_client(verify).retrieve(dataset, request, str(part))
    final = Path(f'{stem}.{_container(part)}')
    os.replace(part, final)
    return final


def _cds_fetch(task, raw_dir, variables, area, step, plevels, product, download_format, verify, retries,
               retry_wait, stem=None, hours=None):
    """
    Download a task to raw_dir (runs in a worker thread: no netCDF reading here). Files left by an interrupted run
    are reused. A request over the CDS cost limit is split in two, by days, then by variables.
    """
    stem = Path(stem or Path(raw_dir) / task.key)
    for ext in ('nc', 'zip'):
        if Path(f'{stem}.{ext}').is_file():
            return [Path(f'{stem}.{ext}')]
    request = cds_request(task, variables, area, step, plevels, product, download_format, hours)
    try:
        return [es.retry(lambda: _cds_download(CDS_DATASETS[task.stream], request, stem, verify), retries,
                         retry_wait, fatal=_cds_fatal, label=stem.name)]
    except Exception as e:
        if not _cds_cost_limit(e):
            raise
        if len(task.days) > 1:
            k = len(task.days) // 2
            parts = [(es.Task(task.key, task.stream, task.days[:k]), variables),
                     (es.Task(task.key, task.stream, task.days[k:]), variables)]
        elif len(variables) > 1:
            k = len(variables) // 2
            parts = [(task, variables[:k]), (task, variables[k:])]
        else:
            raise
        logger.info(f'     {stem.name}: over the CDS cost limit, split in two requests')
        return [f for i, (t, v) in enumerate(parts)
                for f in _cds_fetch(t, raw_dir, v, area, step, plevels, product, download_format, verify, retries,
                                    retry_wait, stem=f'{stem}_{i}', hours=hours)]


def _cds_fetch_task(task, plev_variables, surf_variables, acc_hours=1, **kwargs):
    """
    Download a task. With acc_hours > 1, surface fields are fetched hourly, with the last acc_hours - 1 hours of the
    previous day for the accumulated fields (a second request), to be summed over each time step by era5_store.
    """
    variables = {'plev': plev_variables, 'surf': surf_variables, 'zsurf': ['geopotential']}[task.stream]
    if task.stream != 'surf' or acc_hours == 1:
        return _cds_fetch(task, variables=variables, **kwargs)
    files = _cds_fetch(task, variables=variables, hours=range(24), **kwargs)
    acc = es.accumulated_names(variables)
    if acc:
        lead_day = pd.Timestamp(min(task.days)) - pd.Timedelta('1D')
        lead = es.Task(f'{task.key}_lead', 'surf', (lead_day,))
        files += _cds_fetch(lead, variables=acc, hours=range(25 - acc_hours, 24), **kwargs)
    return files


def _gdex_fetch(task, raw_dir, root, area, step, plevels, retries, retry_wait, acc_hours=1):
    """Read a task from GDEX and write it to raw_dir (runs in a spawned process, HDF5 reads hold a global lock)."""
    target = Path(raw_dir) / f'{task.key}.nc'
    if target.is_file():
        return [target]
    n, w, s, e = area
    box = (n, s, w, e)

    def get():
        if task.stream != 'zsurf':
            return Path(_gdex_day(root, task.stream, task.days[0], step, box, plevels, None, target, acc_hours))
        z = _gdex_load(f'{root}/{GDEX_INVARIANT_Z}',
                       lambda d: _gdex_box(d.isel(time=0, drop=True) if 'time' in d.dims else d, *box))
        tmp = Path(f'{target}.part')
        z.rename('z').to_dataset().drop_vars([c for c in z.coords if c not in z.dims]).to_netcdf(tmp)
        os.replace(tmp, target)
        return target
    return [es.retry(get, retries, retry_wait, fatal=lambda e: isinstance(e, (FileNotFoundError, KeyError,
                                                                                 ValueError)), label=task.key)]


def fetch_era5_to_zarr(start, end, eraDir, latN, latS, lonE, lonW, step='1h', plevels=None, streams=('plev', 'surf'),
                       source='cds', product='reanalysis', zarr_store='ERA5.zarr', n_workers=None,
                       plev_variables=None, surf_variables=None, gdex_root=None, download_format='unarchived',
                       verify=True, retries=3, retry_wait=60., time_chunk=720):
    """
    Fetch ERA5 for a period and a box into the zarr store eraDir/zarr_store, read by the downscaling.

    From the CDS, one request per stream and month (as ECMWF advises), at most 10 at a time (the CDS per-user limit).
    From NCAR GDEX, one day per stream and task, read in spawned processes. Workers only download; the main thread
    reads each file as it arrives, checks it and writes it to the store (see era5_store.py). The surface geopotential
    is fetched once. Time steps already in the store are skipped, so an interrupted or failed run is completed by
    running again, and a longer period only fetches what is new. Data of the earlier layout (yearly/PLEV_*.nc,
    daily/dPLEV_*.nc...) are read instead of downloaded again.

    Args:
        start, end: first and last day
        eraDir (Path): climate folder
        latN, latS, lonE, lonW (float): box [deg], reduced to the nodes of the 0.25 deg grid inside it
        step (str): '1h', '3h' or '6h'. At 3h and 6h the accumulated surface fields are fetched hourly and stored as
            their sums over the time step ending at the stamp
        plevels (list): pressure levels [hPa]
        streams (list): 'plev' and/or 'surf'
        source (str): 'cds' or 'gdex'
        product (str): CDS product type, 'reanalysis'
        zarr_store (str): store name in eraDir
        n_workers (int): parallel downloads. Default 8 for the CDS (at most 10), 4 processes for GDEX
        plev_variables, surf_variables (list): CDS variable names. Default: what the downscaling reads
        gdex_root (str): GDEX root (path or URL). Default: /gdex/data/d633000 if it exists, else OSDF over HTTPS
        download_format (str): CDS download format, 'unarchived' or 'zip' (both are read)
        verify (bool): verify the SSL certificate of the CDS
        retries (int): new attempts of a failed download, after retry_wait, 2 retry_wait, 4 retry_wait... seconds
        retry_wait (float): seconds before the first new attempt
        time_chunk (int): time steps per chunk of the store

    Returns:
        Path: the store

    Raises RuntimeError listing the downloads that failed, once all others are in the store.
    """
    eraDir = Path(eraDir)
    streams = [s for s in es.STREAMS if s in streams]
    if source not in ('cds', 'gdex'):
        raise ValueError(f'ERA5 source {source} not supported: use cds or gdex')
    if step not in es.STEP_HOURS:
        raise ValueError(f'step {step} not supported, use 1h, 3h or 6h')
    if 'plev' in streams and not plevels:
        raise ValueError('plevels are required for pressure-level data')
    if source == 'gdex':
        if product not in [None, 'reanalysis']:
            raise ValueError(f'GDEX only holds the ERA5 reanalysis, not {product}')
        if (plev_variables and list(plev_variables) != es.PLEV_VARIABLES) or \
                (surf_variables and [v for v in surf_variables if v != 'geopotential'] != es.SURF_VARIABLES):
            raise ValueError('GDEX gives the variables of the downscaling only: leave varoi_plev and varoi_surf empty')
    area = es.snap_area(latN, latS, lonW, lonE)
    plev_variables = list(plev_variables or es.PLEV_VARIABLES)
    # the surface geopotential is static: fetched once, not with every month
    surf_variables = [v for v in (surf_variables or es.SURF_VARIABLES)
                      if v not in ('geopotential', 'geopotential_at_surface')]
    request = {'product': product, 'area': area, 'levels': sorted(float(p) for p in (plevels or [])),
               'timestep': step, 'plev_variables': plev_variables, 'surf_variables': surf_variables}
    acc_hours = es.STEP_HOURS[step]
    if acc_hours > 1:
        # accumulated fields stored as sums over the time step, from hourly data
        request['accumulation_hours'] = acc_hours
    store = es.Era5Store(eraDir / zarr_store, request, es.time_axis(start, end, step), time_chunk=time_chunk)

    make_tasks = es.month_tasks if source == 'cds' else es.day_tasks
    tasks = ([es.Task('zsurf', 'zsurf', (pd.Timestamp(start).normalize(),))] if 'surf' in streams else []) \
        + make_tasks(start, end, streams)
    todo = store.missing(tasks)
    logger.info(f'---> ERA5 from {source.upper()} to {store.path}: {len(todo)} of {len(tasks)} downloads to do '
          f'(box N{area[0]} W{area[1]} S{area[2]} E{area[3]})')
    todo = es.import_legacy(store, todo, es.LegacyFiles(eraDir), area, plev_variables, surf_variables)
    if not todo:
        logger.info(f'===> {store.path.name} complete')
        return store.path

    raw = eraDir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    if source == 'cds':
        n = int(n_workers or 8)
        if n > CDS_MAX_PARALLEL:
            logger.info(f'     download_threads={n}: the CDS runs {CDS_MAX_PARALLEL} requests of a user at a time and '
                  f'lowers the priority of users sending more. Using {CDS_MAX_PARALLEL}')
            n = CDS_MAX_PARALLEL
        _cds_client(verify)   # a missing or wrong ~/.cdsapirc fails here, before any thread starts
        fetch = functools.partial(_cds_fetch_task, plev_variables=plev_variables, surf_variables=surf_variables,
                                  raw_dir=raw, area=area, step=step, plevels=plevels, product=product,
                                  download_format=download_format, verify=verify, retries=retries,
                                  retry_wait=retry_wait, acc_hours=acc_hours)
        es.run(store, todo, fetch, n_workers=n, processes=False)
    else:
        root = gdex_root_path(gdex_root)
        if _is_remote(root) and any(t.stream == 'plev' for t in todo):
            n_days = sum(t.stream == 'plev' for t in todo) / es.STEP_HOURS[step]
            logger.info(f'     Pressure levels are stored as global hourly fields: about {5 * n_days:.0f} GB to transfer '
                  f'whatever the size of the domain. On NCAR systems they are read locally')
        fetch = functools.partial(_gdex_fetch, raw_dir=raw, root=root, area=area, step=step, plevels=plevels,
                                  retries=retries, retry_wait=retry_wait, acc_hours=acc_hours)
        es.run(store, todo, fetch, n_workers=n_workers or 4, processes=True)
    shutil.rmtree(raw, ignore_errors=True)
    logger.info(f'===> {store.path.name} complete')
    return store.path


def return_last_fullday():
    """
    TODO: NEED docstring and explanation 
    """


    # Get current UTC time
    current_time_utc = datetime.now(timezone.utc).replace(tzinfo=None)

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
    logger.info(f"Current time (rounded down) in UTC: {current_time_utc_str}")
    logger.info(f"Last full day ERA5T data in UTC: {last_fullday_data_str}")

    return six_days_ago_utc


def unzip_file(file_path):
    """
    Repack as one netCDF file, in place, a CDS download that is a zip of netCDF files (the CDS sends one for surface
    fields of different kinds, instantaneous and accumulated). The result is standardized (see remap_CDSbeta) and
    named like the input with the .nc extension. A netCDF file is left unchanged.

    Args:
        file_path (str): downloaded file

    Returns:
        str: path of the netCDF file
    """
    path = Path(file_path)
    if not zipfile.is_zipfile(path):
        return str(path)
    ds = es.open_raw([path])
    out = path.with_suffix('.nc')
    tmp = path.with_name(path.name + '.part')
    ds.to_netcdf(tmp)
    if path != out:
        path.unlink()
    os.replace(tmp, out)
    logger.info(f'---> {path.name} unzipped to {out.name}')
    return str(out)


def remap_CDSbeta(file_pattern, file_type='SURF'):
    """
    Rewrite netCDF files of the CDS with the names of the earlier CDS (time, level), ascending levels, and without
    the number and expver coordinates (see era5_store.standardize). Files already remapped are left as they are.

    Args:
        file_pattern (str): path and pattern of the files, e.g. 'inputs/climate/daily/dSURF_*.nc'
        file_type (str): SURF or PLEV (kept for compatibility, both are handled the same way)
    """
    for nc_file in sorted(glob.glob(str(file_pattern))):
        with xr.open_dataset(nc_file) as ds:
            renamed = any(n in ds.variables or n in ds.dims for n in ('valid_time', 'pressure_level', 'number',
                                                                     'expver'))
            out = es.standardize(ds).load() if renamed else None
        if out is None:
            continue
        tmp = nc_file + '.part'
        out.to_netcdf(tmp)
        os.replace(tmp, nc_file)


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
    logger.info(target + " complete")


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
    logger.info(target + " complete")
