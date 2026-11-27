"""
ERA5 zarr store filled as the downloads arrive, whatever the source (CDS or NCAR GDEX).

The store (inputs/climate/ERA5.zarr by default) is created by the first data written, with the time axis of the
whole period allocated. Chunks not yet written read as NaN.

- z, t, q, u, v (time, level, latitude, longitude): pressure levels, float32
- t2m, d2m, sp, tp, ssrd, strd (time, latitude, longitude): surface fields, float32
- z_surf (latitude, longitude): surface geopotential, static
- filled_plev, filled_surf (time): 1 where the time step was written, after validation

Workers only transfer files. Every netCDF file is read by the main thread, one at a time, as netCDF-C and HDF5 are
not thread safe. Each file is standardized (CDS names, ERA5/ERA5T expver, level order), checked (time steps,
variables, levels, grid, missing values), written to its time slice of the store, then deleted.

The filled flags make the store its own record of what was fetched: a rerun only fetches the missing time steps, a
longer period extends the store, and the downscaling refuses a store with gaps.

S. Filhol, J. Fiddes and contributors
"""
import json
import shutil
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr

try:
    from zarr.codecs import BloscCodec
except ImportError:
    from zarr.codecs._blosc import BloscCodec
import logging

logger = logging.getLogger(__name__)

GRID = 0.25
STREAMS = ('plev', 'surf')
STEP_HOURS = {'1h': 1, '3h': 3, '6h': 6}

# CDS variable names and the names they get in the netCDF files
SHORT_NAMES = {'geopotential': 'z', 'temperature': 't', 'u_component_of_wind': 'u', 'v_component_of_wind': 'v',
               'specific_humidity': 'q', 'relative_humidity': 'r', 'vertical_velocity': 'w',
               'fraction_of_cloud_cover': 'cc', 'specific_cloud_liquid_water_content': 'clwc',
               'specific_cloud_ice_water_content': 'ciwc', 'divergence': 'd', 'vorticity': 'vo',
               'potential_vorticity': 'pv', 'ozone_mass_mixing_ratio': 'o3',
               '2m_temperature': 't2m', '2m_dewpoint_temperature': 'd2m', 'surface_pressure': 'sp',
               'total_precipitation': 'tp', 'surface_solar_radiation_downwards': 'ssrd',
               'surface_thermal_radiation_downwards': 'strd', 'toa_incident_solar_radiation': 'tisr',
               '10m_u_component_of_wind': 'u10', '10m_v_component_of_wind': 'v10', 'snowfall': 'sf',
               'total_cloud_cover': 'tcc', 'mean_sea_level_pressure': 'msl', 'skin_temperature': 'skt',
               'snow_depth': 'sd', 'friction_velocity': 'zust',
               'instantaneous_surface_sensible_heat_flux': 'ishf', 'instantaneous_moisture_flux': 'ie'}

# what the downscaling reads (topo_scale_zarr.PLEV_VARS and SURF_VARS)
PLEV_VARIABLES = ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'specific_humidity']
SURF_VARIABLES = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure', 'total_precipitation',
                  'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards']
REQUIRED = {'plev': ('z', 't', 'q', 'u', 'v'), 'surf': ('t2m', 'd2m', 'sp', 'tp', 'ssrd', 'strd'),
            'zsurf': ('z_surf',)}
DIMS = {'plev': ('time', 'level', 'latitude', 'longitude'), 'surf': ('time', 'latitude', 'longitude'),
        'zsurf': ('latitude', 'longitude')}
REQUEST_KEYS = ('product', 'area', 'levels', 'timestep', 'plev_variables', 'surf_variables')


@dataclass(frozen=True)
class Task:
    """
    One download: a stream over whole days. The stream is 'plev', 'surf' or 'zsurf' (the static surface geopotential,
    fetched once).
    """
    key: str
    stream: str
    days: tuple

    def times(self, step):
        if self.stream == 'zsurf':
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=24 // STEP_HOURS[step], freq=step).values
                                                for d in self.days]))


def time_axis(start, end, step):
    """Time steps from the first hour of start to the last time step of end."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return pd.date_range(start, end + pd.Timedelta('1D'), freq=step, inclusive='left')


def snap_area(latN, latS, lonW, lonE, res=GRID):
    """
    Box [N, W, S, E] on the nodes of the ERA5 grid inside it: the nodes the CDS and GDEX return for the box, so that
    every source and the store agree on the grid.
    """
    def snap(x, op):
        return float(op(np.round(x / res, 6)) * res)
    area = [snap(latN, np.floor), snap(lonW, np.ceil), snap(latS, np.ceil), snap(lonE, np.floor)]
    if area[0] < area[2] or area[3] < area[1]:
        raise ValueError(f'The box N={latN} S={latS} W={lonW} E={lonE} holds no node of the {res} deg ERA5 grid')
    return area


def grid_of(area, res=GRID):
    """Latitudes (descending) and longitudes (ascending) of the nodes of a box from snap_area()."""
    n, w, s, e = area
    lat = np.round(n - res * np.arange(int(round((n - s) / res)) + 1), 6)
    lon = np.round(w + res * np.arange(int(round((e - w) / res)) + 1), 6)
    return lat, lon


def _days(start, end):
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq='D')


def month_tasks(start, end, streams=STREAMS):
    """One task per stream and calendar month (or part of it within the period), the request size ECMWF advises."""
    days = _days(start, end)
    months = days.to_period('M')
    return [Task(f'{stream}_{month}', stream, tuple(days[months == month]))
            for month in months.unique() for stream in streams]


def day_tasks(start, end, streams=STREAMS):
    """One task per stream and day."""
    return [Task(f'{stream}_{day:%Y-%m-%d}', stream, (day,)) for day in _days(start, end) for stream in streams]


def standardize(ds):
    """
    Harmonize an ERA5 netCDF file: CDS names (valid_time, pressure_level), ERA5 completed by ERA5T where expver is a
    dimension, extra coordinates dropped (number, expver...), levels ascending, latitude descending, longitude
    ascending.
    """
    ds = ds.rename({k: v for k, v in {'valid_time': 'time', 'pressure_level': 'level'}.items()
                    if k in ds.variables or k in ds.dims})
    if 'expver' in ds.dims:
        # ERA5 (first expver) completed by ERA5T for the most recent dates
        parts = [ds.isel(expver=i, drop=True) for i in range(ds.sizes['expver'])]
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.combine_first(part)
        ds = merged
    if 'number' in ds.dims:
        if ds.sizes['number'] > 1:
            raise ValueError('Ensemble members (dimension number) are not supported')
        ds = ds.isel(number=0, drop=True)
    ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])
    for dim, ascending in [('level', True), ('latitude', False), ('longitude', True)]:
        if dim in ds.dims:
            ds = ds.sortby(dim, ascending=ascending)
    return ds


def _load(path):
    try:
        with xr.open_dataset(path) as ds:
            return standardize(ds).load()
    except (OSError, RuntimeError) as e:
        raise IOError(f'{Path(path).name} is not a readable netCDF file: {e}') from e


def open_raw(paths):
    """
    Read downloaded files into memory, standardized: netCDF files, or zip files of netCDF files (the CDS sends one
    for surface fields of different kinds, instantaneous and accumulated). Several files (a request split in parts)
    are combined by time and variable.
    """
    parts = []
    for path in map(Path, paths):
        if zipfile.is_zipfile(path):
            tmp = path.with_name(path.name + '.unzip')
            shutil.rmtree(tmp, ignore_errors=True)
            with zipfile.ZipFile(path) as z:
                names = sorted(n for n in z.namelist() if n.endswith('.nc'))
                if not names:
                    raise IOError(f'{path.name} holds no netCDF file')
                z.extractall(tmp, members=names)
            try:
                pieces = [_load(tmp / n) for n in names]
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            parts.append(xr.merge(pieces, join='exact', compat='override', combine_attrs='drop_conflicts'))
        else:
            parts.append(_load(path))
    if len(parts) == 1:
        return parts[0]
    return xr.combine_by_coords(parts, data_vars='all', coords='minimal', compat='override', join='outer',
                                combine_attrs='drop_conflicts')


def retry(fun, retries=3, wait=60., fatal=None, label=''):
    """
    Call fun(); on failure call it again after wait, 2 wait, 4 wait... seconds, unless fatal(error) is True.
    """
    for attempt in range(retries + 1):
        try:
            return fun()
        except Exception as e:
            if attempt == retries or (fatal is not None and fatal(e)):
                raise
            delay = wait * 2 ** attempt
            logger.info(f'     {label}: {_first_line(e)}. Retrying in {delay:.0f} s')
            time.sleep(delay)


def _first_line(e):
    text = str(e).strip()
    return f'{type(e).__name__}: {text.splitlines()[0] if text else ""}'


class Era5Store:
    """
    ERA5 zarr store written by time slices (see the module docstring).

    Args:
        path (str): store path
        request (dict): what the store holds: product, area [N, W, S, E] on the grid, levels (hPa), timestep,
            plev_variables and surf_variables (CDS names). An existing store must hold the same.
        times (DatetimeIndex): time steps of the period. An existing store is extended to cover them.
        time_chunk (int): time steps per chunk
        max_chunk_mb (float): target size of a pressure-level chunk, used to choose the spatial chunk size
        compressor (obj): zarr codec. Default: BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle')
    """

    def __init__(self, path, request, times, time_chunk=720, max_chunk_mb=16, compressor=None):
        self.path = Path(path)
        self.request = _jsonable(request)
        self.step = self.request['timestep']
        self.times = pd.DatetimeIndex(times)
        self.time_chunk, self.max_chunk_mb = int(time_chunk), max_chunk_mb
        self.compressor = compressor or BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)
        if self.exists:
            self._check_request()
            self._extend()

    @property
    def exists(self):
        return (self.path / 'zarr.json').is_file()

    # ---------------------------------------------------------------- reading
    def _group(self, mode='r'):
        return zarr.open_group(str(self.path), mode=mode, use_consolidated=False)

    def _time_index(self):
        with xr.open_zarr(str(self.path), consolidated=False) as ds:
            return pd.DatetimeIndex(ds.time.values)

    def _grid(self):
        g = self._group()
        return g['latitude'][:], g['longitude'][:]

    def missing(self, tasks):
        """Tasks with time steps not yet written (and the surface geopotential task if it is missing)."""
        if not self.exists:
            return list(tasks)
        g = self._group()
        index = self._time_index()
        flags = {s: g[f'filled_{s}'][:] for s in STREAMS}
        out = []
        for task in tasks:
            if task.stream == 'zsurf':
                if 'z_surf' not in g or not np.isfinite(g['z_surf'][:]).all():
                    out.append(task)
                continue
            idx = index.get_indexer(task.times(self.step))
            if (idx < 0).any() or not flags[task.stream][idx].all():
                out.append(task)
        return out

    # ---------------------------------------------------------------- checks
    def _check_request(self):
        stored = self._group().attrs.get('era5_request')
        if stored is None:
            raise ValueError(f'{self.path} was not written by era5_store (no era5_request attribute). Delete it, or '
                             f'set climate.<source>.zarr_store to another name')
        stored = json.loads(stored) if isinstance(stored, str) else stored
        diff = [k for k in REQUEST_KEYS if not _same(stored.get(k), self.request.get(k))]
        if diff:
            details = '; '.join(f'{k}: store {stored.get(k)}, now {self.request.get(k)}' for k in diff)
            raise ValueError(f'{self.path.name} holds other ERA5 data than requested ({details}). Delete it, or set '
                             f'climate.<source>.zarr_store to another name')

    def check(self, task, ds):
        """Check a standardized dataset against the task and the store; returned as float32, dimensions ordered."""
        stream = task.stream
        if stream == 'zsurf':
            if 'time' in ds.dims:
                ds = ds.isel(time=0, drop=True)
            ds = ds.rename({'z': 'z_surf'}) if 'z' in ds else ds
            names = ['z_surf']
        else:
            requested = self.request[f'{stream}_variables']
            names = [SHORT_NAMES.get(v) for v in requested]
            if None in names:   # names unknown here: only their number is checked
                names = None if len(ds.data_vars) == len(requested) else requested
        problems = []
        if names is not None and set(ds.data_vars) != set(names):
            problems.append(f'variables {sorted(ds.data_vars)} instead of {sorted(names)}')
        dims = DIMS[stream]
        for v in ds.data_vars:
            if set(ds[v].dims) != set(dims):
                problems.append(f'{v} has dimensions {ds[v].dims}, not {dims}')
        if stream != 'zsurf' and 'time' in ds.dims:
            got, expected = pd.DatetimeIndex(ds.time.values), task.times(self.step)
            if not got.equals(expected):
                missing, extra = expected.difference(got), got.difference(expected)
                problems.append(f'{got.size} time steps instead of {expected.size}'
                                + (f', missing from {missing[0]}' if missing.size else '')
                                + (f', unexpected {extra[0]}' if extra.size else ''))
        if stream == 'plev' and 'level' in ds.dims and not _same(list(ds.level.values), self.request['levels']):
            problems.append(f'levels {list(ds.level.values)} instead of {self.request["levels"]}')
        if self.exists and 'latitude' in ds.dims and 'longitude' in ds.dims:
            lat, lon = self._grid()
            if not (_same(ds.latitude.values, lat) and _same(ds.longitude.values, lon)):
                problems.append(f'grid {ds.latitude.values.max()}N {ds.latitude.values.min()}S '
                                f'{ds.longitude.values.min()}W {ds.longitude.values.max()}E '
                                f'({ds.sizes["latitude"]}x{ds.sizes["longitude"]}) differs from the store '
                                f'({lat.max()}N {lat.min()}S {lon.min()}W {lon.max()}E, {lat.size}x{lon.size})')
        if not problems:
            for v in ds.data_vars:
                finite = np.isfinite(ds[v].values)
                if v in REQUIRED[stream] and not finite.all():
                    problems.append(f'{v} has {(~finite).sum()} missing values')
                elif not finite.any():
                    problems.append(f'{v} has no value')
        if problems:
            raise ValueError(f'{task.key}: ' + '; '.join(problems))
        return xr.Dataset({v: ds[v].transpose(*[d for d in dims if d in ds[v].dims]).astype('float32')
                           for v in ds.data_vars})

    # ---------------------------------------------------------------- writing
    def _to_zarr(self, ds, **kwargs):
        # every write repeats the root attributes, which xarray would otherwise reset; metadata is consolidated
        # after each change of the layout, so readers never see a stale layout
        ds = ds.copy()
        ds.attrs = dict(self._group().attrs) if self.exists else ds.attrs
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Consolidated metadata')
            ds.to_zarr(str(self.path), zarr_format=3, **kwargs)

    def _chunks(self, nlev, nlat, nlon):
        spatial = max(1, int(np.sqrt(self.max_chunk_mb * 1e6 / (self.time_chunk * max(nlev, 1) * 4))))
        return {'time': self.time_chunk, 'level': max(nlev, 1), 'latitude': min(nlat, spatial),
                'longitude': min(nlon, spatial)}

    def _create(self, ds):
        lat, lon = ds.latitude.values, ds.longitude.values
        levels = np.asarray(self.request['levels'], dtype=float)
        nt = self.times.size
        flags = {f'filled_{s}': ('time', da.zeros(nt, dtype='uint8', chunks=self.time_chunk)) for s in STREAMS}
        base = xr.Dataset(flags, coords={'time': self.times, 'level': levels, 'latitude': lat, 'longitude': lon},
                          attrs={'era5_request': json.dumps(self.request),
                                 'description': 'ERA5 for TopoPyScale, see TopoPyScale.era5_store'})
        base.time.encoding.update(units='hours since 1900-01-01', dtype='int64')
        encoding = {v: {'chunks': (self.time_chunk,)} for v in flags}
        self._to_zarr(base, mode='w', compute=False, encoding=encoding)
        logger.info(f'---> {self.path.name} created: {nt} time steps from {self.times[0]} to {self.times[-1]}, '
              f'{lat.size}x{lon.size} nodes, {levels.size} levels')

    def _extend(self):
        index = self._time_index()
        if self.times[0] < index[0]:
            raise ValueError(f'{self.path.name} starts on {index[0]}, after the period start {self.times[0]}. A zarr '
                             f'store cannot grow backwards: delete it, or set climate.<source>.zarr_store to another '
                             f'name')
        if self.times[-1] <= index[-1]:
            return
        new = pd.date_range(index[-1], self.times[-1], freq=self.step)[1:]
        with xr.open_zarr(str(self.path), consolidated=False) as ds:
            block = {}
            for v in ds.data_vars:
                if 'time' not in ds[v].dims:
                    continue
                shape = tuple(new.size if d == 'time' else ds.sizes[d] for d in ds[v].dims)
                fill = 0 if ds[v].dtype.kind == 'u' else np.nan
                block[v] = (ds[v].dims, da.full(shape, fill, dtype=ds[v].dtype, chunks=shape))
        self._to_zarr(xr.Dataset(block, coords={'time': new}), mode='a', append_dim='time', compute=False)
        logger.info(f'---> {self.path.name} extended by {new.size} time steps, to {new[-1]}')

    def _ensure_variables(self, ds):
        """Add the variables of ds missing from the store, empty over the whole time axis."""
        g = self._group()
        new = [v for v in ds.data_vars if v not in g]
        if not new:
            return
        sizes = {'time': g['time'].shape[0], 'level': g['level'].shape[0], 'latitude': g['latitude'].shape[0],
                 'longitude': g['longitude'].shape[0]}
        ch = self._chunks(sizes['level'], sizes['latitude'], sizes['longitude'])
        template, encoding = {}, {}
        for v in new:
            dims = ds[v].dims
            shape = tuple(sizes[d] for d in dims)
            chunks = tuple(ch[d] if d != 'time' else self.time_chunk for d in dims)
            if 'time' not in dims:
                chunks = shape
            template[v] = (dims, da.full(shape, np.nan, dtype='float32', chunks=chunks))
            encoding[v] = {'chunks': chunks, 'compressors': self.compressor}
        self._to_zarr(xr.Dataset(template), mode='a', compute=False, encoding=encoding)

    def write(self, task, ds):
        """Check ds (a standardized dataset of the task) and write it to its time slice, then flag the time steps."""
        ds = self.check(task, ds)
        if not self.exists:
            self._create(ds)
        self._ensure_variables(ds)
        ds = ds.drop_vars(list(ds.coords))
        if task.stream == 'zsurf':
            self._to_zarr(ds, mode='a')
            return
        idx = self._time_index().get_indexer(task.times(self.step))
        if (idx < 0).any() or not (np.diff(idx) == 1).all():
            raise ValueError(f'{task.key}: time steps not contiguous in the store')
        region = slice(int(idx[0]), int(idx[-1]) + 1)
        self._to_zarr(ds, region={'time': region}, mode='r+')
        self._group('r+')[f'filled_{task.stream}'][region] = 1


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _same(a, b):
    if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
        if len(a) != len(b):
            return False
        try:
            return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=1e-6))
        except (TypeError, ValueError):
            return sorted(map(str, a)) == sorted(map(str, b))
    return a == b


def store_gaps(store, start, end):
    """
    What a store lacks to cover the period [start, end]: a list of messages, empty if complete. None for a store
    without filled flags (built from netCDF files by fetch_era5.netcdf_to_zarr), which is not checked.
    """
    g = zarr.open_group(str(store), mode='r')
    if 'filled_plev' not in g:
        return None
    with xr.open_zarr(str(store)) as ds:
        request = ds.attrs.get('era5_request')
        request = json.loads(request) if isinstance(request, str) else (request or {})
        period = time_axis(start, end, request.get('timestep', '1h'))
        index = pd.DatetimeIndex(ds.time.values)
        idx = index.get_indexer(period)
        gaps = []
        if (idx < 0).any():
            gaps.append(f'{(idx < 0).sum()} time steps outside the store ({index[0]} to {index[-1]})')
        inside = idx[idx >= 0]
        for s in STREAMS:
            missing = period[idx >= 0][ds[f'filled_{s}'].values[inside] == 0]
            if missing.size:
                gaps.append(f'{missing.size} {s} time steps (from {missing[0]})')
        if 'z_surf' not in ds or not np.isfinite(ds.z_surf.values).all():
            gaps.append('the surface geopotential')
    return gaps


class LegacyFiles:
    """
    netCDF files of the earlier layout, read instead of downloading again: climate/yearly/PLEV_*.nc and SURF_*.nc,
    the same in the climate folder, and climate/daily/dPLEV_*.nc and dSURF_*.nc.
    """
    PATTERNS = {'plev': ['yearly/PLEV_*.nc', 'PLEV_*.nc', 'daily/dPLEV_*.nc'],
                'surf': ['yearly/SURF_*.nc', 'SURF_*.nc', 'daily/dSURF_*.nc']}

    def __init__(self, folder):
        self.folder = Path(folder)
        self._index = {}
        self.names = {}

    def index(self, stream):
        """(file, first time, last time) of the readable files of a stream."""
        if stream not in self._index:
            rows = []
            for pattern in self.PATTERNS[stream]:
                for f in sorted(self.folder.glob(pattern)):
                    try:
                        with xr.open_dataset(f) as ds:
                            t = ds['valid_time' if 'valid_time' in ds.variables else 'time'].values
                            rows.append((f, pd.Timestamp(t.min()), pd.Timestamp(t.max())))
                    except (OSError, KeyError, ValueError, RuntimeError):
                        continue
            self._index[stream] = rows
        return self._index[stream]

    def read(self, task, step, area):
        """
        Data of the task from the files, standardized; None if the files do not hold all of it on the grid. Reads the
        variables of self.names ({'plev': [...], 'surf': [...]}, short names).
        """
        stream = 'surf' if task.stream == 'zsurf' else task.stream
        files = self.index(stream)
        if not files:
            return None
        lat, lon = grid_of(area)
        try:
            if task.stream == 'zsurf':
                with xr.open_dataset(files[0][0]) as ds:
                    ds = standardize(ds)
                    if 'z' not in ds:
                        return None
                    out = ds[['z']].isel(time=0, drop=True).load()
            else:
                names = self.names
                if not names.get(stream):
                    return None
                times = task.times(step)
                parts = []
                for f, t0, t1 in files:
                    if t0 > times[-1] or t1 < times[0]:
                        continue
                    with xr.open_dataset(f) as ds:
                        ds = standardize(ds)
                        if not set(names[stream]) <= set(ds.data_vars):
                            return None
                        keep = ds.time.to_index().isin(times)
                        if keep.any():
                            parts.append(ds[names[stream]].isel(time=np.flatnonzero(keep)).load())
                if not parts:
                    return None
                out = xr.concat(parts, dim='time', data_vars='all', coords='minimal', compat='override',
                                join='exact')
                out = out.isel(time=~out.time.to_index().duplicated()).sortby('time')
                if not out.time.to_index().equals(times):
                    return None
        except (OSError, KeyError, ValueError, RuntimeError) as e:
            logger.info(f'     {task.key}: earlier netCDF files not used ({_first_line(e)})')
            return None
        if not (_same(out.latitude.values, lat) and _same(out.longitude.values, lon)):
            return None
        return out


def import_legacy(store, tasks, legacy, area, plev_variables, surf_variables):
    """Write the tasks found in earlier netCDF files to the store. Returns the tasks still to download."""
    short = {'plev': [SHORT_NAMES.get(v) for v in plev_variables],
             'surf': [SHORT_NAMES.get(v) for v in surf_variables]}
    if any(None in names for names in short.values()):
        return tasks
    legacy.names = short
    if not any(legacy.index(s) for s in STREAMS):
        return tasks
    left, n = [], 0
    for task in tasks:
        ds = legacy.read(task, store.step, area)
        if ds is not None:
            try:
                store.write(task, ds)
                n += 1
                continue
            except ValueError as e:
                logger.info(f'     {task.key}: earlier netCDF files not used ({_first_line(e)})')
        left.append(task)
    if n:
        logger.info(f'---> {n} of {len(tasks)} downloads found in earlier netCDF files of {legacy.folder}')
    return left


def run(store, tasks, fetch, n_workers=1, processes=False):
    """
    Fetch the tasks in parallel workers, threads (or spawned processes for sources read with HDF5), and write each
    result to the store from this thread as soon as it arrives.

    Args:
        store (Era5Store): store
        tasks (list): tasks to fetch
        fetch (callable): fetch(task) -> list of downloaded files (picklable when processes is True)
        n_workers (int): number of workers
        processes (bool): workers are processes instead of threads

    Raises RuntimeError listing the tasks that failed, after all the others are written.
    """
    if not tasks:
        return
    n = max(1, min(int(n_workers or 1), len(tasks)))
    pool = ProcessPoolExecutor(n, mp_context=get_context('spawn')) if processes else ThreadPoolExecutor(n)
    failed = []
    try:
        futures = {pool.submit(fetch, task): task for task in tasks}
        for future in as_completed(futures):
            task, paths = futures[future], []
            try:
                paths = future.result()
                store.write(task, open_raw(paths))
                logger.info(f'     {task.key} written')
            except Exception as e:
                failed.append((task, e))
                logger.error(f'     {task.key} FAILED: {_first_line(e)}')
            finally:
                for p in paths:
                    Path(p).unlink(missing_ok=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    if failed:
        listed = '; '.join(f'{t.key} ({_first_line(e)})' for t, e in failed[:10])
        raise RuntimeError(f'{len(failed)} of {len(tasks)} ERA5 downloads failed and are missing from '
                           f'{store.path.name}: {listed}. Run again to fetch only what is missing.')
