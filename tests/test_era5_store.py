"""
ERA5 from the CDS to the zarr store (fetch_era5.fetch_era5_to_zarr, era5_store), against a fake CDS client that
answers like the CDS of 2024 onwards: netCDF files with valid_time, pressure_level, number and expver, surface
fields as a zip of an instantaneous and an accumulated file. Values encode variable, time, level and position.
The fake writes netCDF3 with scipy, so the worker threads never touch HDF5, as the real downloads.
"""
import contextlib
import io
import threading
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TopoPyScale import era5_store as es
from TopoPyScale import fetch_era5 as fe

REF = pd.Timestamp('2020-01-01')
BOX = dict(latN=46.6, latS=46.1, lonW=7.1, lonE=7.6)   # nodes 46.5..46.25 N, 7.25..7.5 E
LEVELS = [1000, 700, 850]
CODE = {'z': 1, 't': 2, 'u': 3, 'v': 4, 'q': 5, 't2m': 6, 'd2m': 7, 'sp': 8, 'ssrd': 9, 'strd': 10, 'tp': 11,
        'fdir': 12}
INSTANT, ACCUM = ['t2m', 'd2m', 'sp'], ['tp', 'ssrd', 'strd']


def value(var, times, levels, lat, lon):
    hours = (pd.DatetimeIndex(times) - REF) / pd.Timedelta('1h')
    return (100 * CODE[var] + 0.01 * hours.values[:, None, None, None] + np.asarray(levels)[None, :, None, None] / 1e4
            + lat[None, None, :, None] / 10 + lon[None, None, None, :] / 1000)


class CostLimit(Exception):
    pass


class FakeCDS:
    """Fake CDS client. fail: {request key: number of failures (ConnectionError)}; corrupt: keys answered once
    with an HTML page; max_days: days per request above which the cost limit is exceeded."""

    def __init__(self, fail=None, corrupt=(), max_days=None, drop_day=None):
        self.calls, self.fail, self.corrupt = [], dict(fail or {}), set(corrupt)
        self.max_days, self.drop_day = max_days, drop_day
        self.lock = threading.Lock()

    def retrieve(self, dataset, request, target):
        days = [pd.Timestamp(f"{request['year'][0]}-{request['month'][0]}-{d}") for d in request['day']]
        stream = 'plev' if 'pressure-levels' in dataset else ('zsurf' if request['variable'] == ['geopotential']
                                                               else 'surf')
        key = 'zsurf' if stream == 'zsurf' else f'{stream}_{days[0]:%Y-%m}'
        with self.lock:
            self.calls.append((key, len(days), tuple(request['variable'])))
            if self.fail.get(key, 0) > 0:
                self.fail[key] -= 1
                raise ConnectionError('connection reset by peer')
            if key in self.corrupt:
                self.corrupt.discard(key)
                open(target, 'wb').write(b'<html>Service unavailable</html>')
                return target
        if self.max_days and len(days) > self.max_days:
            raise CostLimit('403 Client Error: Forbidden. cost limits exceeded. Your request is too large')
        n, w, s, e = request['area']
        lat, lon = es.grid_of([n, w, s, e])
        times = pd.DatetimeIndex([d + pd.Timedelta(h + ':00') for d in days for h in request['time']])
        if self.drop_day is not None and key == self.drop_day[0]:
            times = times[times.normalize() != pd.Timestamp(self.drop_day[1])]
        short = [es.SHORT_NAMES[v] for v in request['variable']]
        coords = {'valid_time': times, 'latitude': lat, 'longitude': lon, 'number': 0,
                  'expver': ('valid_time', np.array(['0001'] * times.size))}
        if stream == 'plev':
            levels = sorted((float(p) for p in request['pressure_level']), reverse=True)   # as the CDS: descending
            ds = xr.Dataset({v: (('valid_time', 'pressure_level', 'latitude', 'longitude'),
                                 value(v, times, levels, lat, lon).astype('float32')) for v in short},
                            coords={**coords, 'pressure_level': levels})
            ds.to_netcdf(target, engine='scipy')
            return target
        fields = {v: (('valid_time', 'latitude', 'longitude'), value(v, times, [0.], lat, lon)[:, 0].astype('float32'))
                  for v in short}
        if stream == 'zsurf':
            xr.Dataset({'z': (('valid_time', 'latitude', 'longitude'),
                              np.broadcast_to(1000 * lat[None, :, None], (1, lat.size, lon.size)).astype('float32'))},
                       coords=coords).to_netcdf(target, engine='scipy')
            return target
        # surface: a zip of an instantaneous and an accumulated file, whatever download_format says
        buffers = {}
        for kind, names in [('instant', INSTANT), ('accum', ACCUM + ['fdir'])]:
            part = {v: fields[v] for v in names if v in fields}
            if part:
                buffers[kind] = xr.Dataset(part, coords=coords).to_netcdf(engine='scipy')
        with zipfile.ZipFile(target, 'w') as z:
            for kind, data in buffers.items():
                z.writestr(f'data_stream-oper_stepType-{kind}.nc', bytes(data))
        return target


@pytest.fixture
def cds(monkeypatch):
    def install(**kw):
        fake = FakeCDS(**kw)
        monkeypatch.setattr(fe, '_cds_client', lambda verify=True: fake)
        return fake
    return install


def fetch(workdir, start='2020-01-30', end='2020-02-02', **kw):
    kw = {**dict(plevels=LEVELS, n_workers=4, retries=2, retry_wait=0), **kw}
    with contextlib.redirect_stdout(io.StringIO()):
        return fe.fetch_era5_to_zarr(start, end, workdir, **BOX, **kw)


def check_store(store, start, end):
    ds = xr.open_zarr(store)
    times = es.time_axis(start, end, '1h')
    lat, lon = ds.latitude.values, ds.longitude.values
    np.testing.assert_allclose(lat, [46.5, 46.25])
    np.testing.assert_allclose(lon, [7.25, 7.5])
    assert list(ds.level.values) == [700, 850, 1000]
    assert set(ds.data_vars) == {'z', 't', 'u', 'v', 'q', 't2m', 'd2m', 'sp', 'tp', 'ssrd', 'strd', 'z_surf',
                                 'filled_plev', 'filled_surf'}
    sel = ds.sel(time=times)
    assert sel.filled_plev.values.all() and sel.filled_surf.values.all()
    for v in ['z', 't', 'u', 'v', 'q']:
        np.testing.assert_allclose(sel[v].values, value(v, times, [700, 850, 1000], lat, lon), rtol=1e-6, err_msg=v)
    for v in INSTANT + ACCUM:
        np.testing.assert_allclose(sel[v].values, value(v, times, [0.], lat, lon)[:, 0], rtol=1e-6, err_msg=v)
    assert ds.z_surf.dims == ('latitude', 'longitude')
    np.testing.assert_allclose(ds.z_surf.values, 1000 * lat[:, None] + 0 * lon[None, :])
    assert es.store_gaps(store, start, end) == []
    return ds


def test_monthly_requests_fill_the_store(cds, tmp_path):
    fake = cds()
    store = fetch(tmp_path)
    check_store(store, '2020-01-30', '2020-02-02')
    # one request per stream and month, plus the surface geopotential once
    assert sorted(k for k, _, _ in fake.calls) == ['plev_2020-01', 'plev_2020-02', 'surf_2020-01', 'surf_2020-02',
                                                   'zsurf']
    assert all('relative_humidity' not in v and ('geopotential' not in v or k != 'surf_2020-01')
               for k, _, v in fake.calls)
    assert not (tmp_path / 'raw').exists() and not (tmp_path / 'daily').exists()


def test_request_layout():
    task = es.Task('plev_2020-02', 'plev', tuple(pd.date_range('2020-02-01', '2020-02-29')))
    req = fe.cds_request(task, es.PLEV_VARIABLES, es.snap_area(46.6, 46.1, 7.1, 7.6), '3h', LEVELS)
    assert req['day'] == [f'{d:02d}' for d in range(1, 30)] and req['month'] == ['02'] and req['year'] == ['2020']
    assert req['time'] == ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    assert req['area'] == [46.5, 7.25, 46.25, 7.5] and req['pressure_level'] == ['700', '850', '1000']
    assert req['data_format'] == 'netcdf'
    with pytest.raises(ValueError, match='no node'):
        es.snap_area(46.4, 46.3, 7.3, 7.4)


def test_rerun_and_longer_period_fetch_only_what_is_missing(cds, tmp_path):
    cds()
    fetch(tmp_path)
    fake = cds()
    fetch(tmp_path)
    assert fake.calls == []
    # a longer period: February is incomplete (1-2 only), March is new
    fetch(tmp_path, end='2020-03-01')
    assert sorted(k for k, _, _ in fake.calls) == ['plev_2020-02', 'plev_2020-03', 'surf_2020-02', 'surf_2020-03']
    ds = check_store(tmp_path / 'ERA5.zarr', '2020-01-30', '2020-03-01')
    assert ds.sizes['time'] == 24 * 32
    # a store cannot grow backwards
    with pytest.raises(ValueError, match='cannot grow backwards'):
        fetch(tmp_path, start='2020-01-01')
    # nor hold another request
    with pytest.raises(ValueError, match='levels'):
        fetch(tmp_path, plevels=[700, 850])


def test_transient_errors_are_retried(cds, tmp_path):
    fake = cds(fail={'plev_2020-01': 2}, corrupt={'surf_2020-02'})
    check_store(fetch(tmp_path), '2020-01-30', '2020-02-02')
    assert [k for k, _, _ in fake.calls].count('plev_2020-01') == 3
    assert [k for k, _, _ in fake.calls].count('surf_2020-02') == 2


def test_failures_are_reported_after_the_rest_is_written(cds, tmp_path):
    cds(fail={'plev_2020-02': 10})
    with pytest.raises(RuntimeError, match='1 of 5 ERA5 downloads failed.*plev_2020-02.*ConnectionError'):
        fetch(tmp_path)
    gaps = es.store_gaps(tmp_path / 'ERA5.zarr', '2020-01-30', '2020-02-02')
    assert gaps == ['48 plev time steps (from 2020-02-01 00:00:00)']
    fake = cds()
    check_store(fetch(tmp_path), '2020-01-30', '2020-02-02')
    assert [k for k, _, _ in fake.calls] == ['plev_2020-02']


def test_incomplete_download_is_rejected(cds, tmp_path):
    cds(drop_day=('surf_2020-02', '2020-02-02'))
    with pytest.raises(RuntimeError, match='surf_2020-02.*24 time steps instead of 48, missing from 2020-02-02'):
        fetch(tmp_path)
    assert not list((tmp_path / 'raw').glob('surf_2020-02*'))   # deleted, downloaded again by the next run
    cds()
    check_store(fetch(tmp_path), '2020-01-30', '2020-02-02')


def test_requests_over_the_cost_limit_are_split(cds, tmp_path):
    fake = cds(max_days=10)
    store = fetch(tmp_path, start='2020-01-01', end='2020-01-31', retries=0)
    check_store(store, '2020-01-01', '2020-01-31')
    plev = [n for k, n, _ in fake.calls if k == 'plev_2020-01']
    assert plev == [31, 15, 7, 8, 16, 8, 8]   # 31 days refused, then halves until under the limit


def test_threads_capped_at_the_cds_limit(cds, tmp_path, monkeypatch):
    cds()
    used = []
    run = es.run
    monkeypatch.setattr(es, 'run', lambda *a, **k: used.append(k['n_workers']) or run(*a, **k))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fe.fetch_era5_to_zarr('2020-01-30', '2020-01-31', tmp_path, **BOX, plevels=LEVELS, n_workers=20)
    assert used == [10] and 'Using 10' in out.getvalue()


def test_earlier_netcdf_files_are_reused(cds, tmp_path):
    """yearly/PLEV_*.nc and daily/dSURF_*.nc of the earlier layout, as the new CDS wrote them (with r and z)."""
    lat, lon = es.grid_of(es.snap_area(BOX['latN'], BOX['latS'], BOX['lonW'], BOX['lonE']))
    times = pd.date_range('2020-01-30', '2020-02-02 23:00', freq='1h')
    plev = xr.Dataset({v: (('valid_time', 'pressure_level', 'latitude', 'longitude'),
                           value(v, times, [1000, 850, 700], lat, lon)) for v in ['z', 't', 'u', 'v', 'q']},
                      coords={'valid_time': times, 'pressure_level': [1000., 850., 700.], 'latitude': lat,
                              'longitude': lon, 'number': 0})
    plev['r'] = plev.t * 0 + 50
    (tmp_path / 'yearly').mkdir()
    plev.to_netcdf(tmp_path / 'yearly' / 'PLEV_2020.nc')
    (tmp_path / 'daily').mkdir()
    for day in pd.date_range('2020-01-30', '2020-02-02'):
        t = pd.date_range(day, periods=24, freq='1h')
        surf = xr.Dataset({v: (('valid_time', 'latitude', 'longitude'), value(v, t, [0.], lat, lon)[:, 0])
                           for v in INSTANT + ACCUM}, coords={'valid_time': t, 'latitude': lat, 'longitude': lon})
        surf['z'] = (('valid_time', 'latitude', 'longitude'), np.broadcast_to(1000 * lat[None, :, None],
                                                                              (24, lat.size, lon.size)))
        surf.to_netcdf(tmp_path / 'daily' / f'dSURF_{day:%Y%m%d}.nc')
    fake = cds(fail={k: 99 for k in ['zsurf', 'plev_2020-01', 'plev_2020-02', 'surf_2020-01', 'surf_2020-02']})
    store = fetch(tmp_path)
    assert fake.calls == []
    ds = check_store(store, '2020-01-30', '2020-02-02')
    assert 'r' not in ds


def test_topoclass_refuses_a_store_with_gaps(cds, tmp_path):
    from munch import DefaultMunch
    from TopoPyScale.topoclass import Topoclass
    cds(fail={'surf_2020-01': 10})
    with pytest.raises(RuntimeError):
        fetch(tmp_path)
    obj = object.__new__(Topoclass)
    obj.config = DefaultMunch.fromDict({'project': {'climate': 'era5', 'start': pd.Timestamp('2020-01-30'),
                                                    'end': pd.Timestamp('2020-02-02')},
                                        'climate': {'path': tmp_path, 'era5': {}}})
    with pytest.raises(ValueError, match=r'lacks 48 surf time steps \(from 2020-01-30.*get_era5'):
        obj._climate_store()
    cds()
    fetch(tmp_path)
    assert obj._climate_store() == tmp_path / 'ERA5.zarr'


def test_fetcher_config(cds, tmp_path, monkeypatch):
    from munch import DefaultMunch
    config = DefaultMunch.fromDict({
        'project': {'climate': 'era5', 'start': pd.Timestamp('2020-01-30'), 'end': pd.Timestamp('2020-02-02'),
                    'extent': {'latN': 46.2, 'latS': 46.0, 'lonW': 7.2, 'lonE': 7.5}},
        'climate': {'path': tmp_path, 'era5': {'product': 'reanalysis', 'timestep': '1h', 'plevels': [700, 850],
                                               'download_threads': 3, 'cds_output_format': 'grib',
                                               'zarr_store': 'my.zarr'}}})
    calls = []
    monkeypatch.setattr(fe, 'fetch_era5_to_zarr', lambda *a, **k: calls.append((a, k)))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fe.FetchERA5(config).go_fetch('all')
    (args, kw), = calls
    assert kw['source'] == 'cds' and kw['n_workers'] == 3 and kw['zarr_store'] == 'my.zarr'
    assert kw['streams'] == ('plev', 'surf') and kw['latN'] == pytest.approx(46.6)
    assert 'grib ignored' in out.getvalue()
    config.climate.era5.data_repository = 'google_cloud_storage'
    with pytest.raises(ValueError, match='use cds or gdex'):
        with contextlib.redirect_stdout(io.StringIO()):
            fe.FetchERA5(config).go_fetch('all')


def test_unzip_and_remap_utilities(tmp_path):
    lat, lon = np.array([46.5, 46.25]), np.array([7.25, 7.5])
    t = pd.date_range('2020-01-01', periods=3, freq='1h')
    coords = {'valid_time': t, 'latitude': lat, 'longitude': lon, 'number': 0}
    with zipfile.ZipFile(tmp_path / 'SURF_20200101.nc', 'w') as z:
        for kind, names in [('instant', INSTANT), ('accum', ACCUM)]:
            ds = xr.Dataset({v: (('valid_time', 'latitude', 'longitude'), value(v, t, [0.], lat, lon)[:, 0])
                             for v in names}, coords=coords)
            z.writestr(f'data_stream-oper_stepType-{kind}.nc', bytes(ds.to_netcdf(engine='scipy')))
    with contextlib.redirect_stdout(io.StringIO()):
        out = fe.unzip_file(str(tmp_path / 'SURF_20200101.nc'))
    ds = xr.load_dataset(out)
    assert set(ds.data_vars) == set(INSTANT + ACCUM) and 'time' in ds.dims and 'number' not in ds.variables
    assert fe.unzip_file(out) == out   # netCDF already
    plev = xr.Dataset({'t': (('valid_time', 'pressure_level', 'latitude', 'longitude'),
                             value('t', t, [1000, 500], lat, lon))},
                      coords={**coords, 'pressure_level': [1000., 500.]})
    plev.to_netcdf(tmp_path / 'PLEV_20200101.nc')
    fe.remap_CDSbeta(str(tmp_path / 'PLEV_*.nc'), 'plev')
    fe.remap_CDSbeta(str(tmp_path / 'PLEV_*.nc'), 'plev')   # already remapped: unchanged
    ds = xr.load_dataset(tmp_path / 'PLEV_20200101.nc')
    assert list(ds.t.dims) == ['time', 'level', 'latitude', 'longitude'] and list(ds.level.values) == [500, 1000]


def test_3h_store_holds_window_sums_of_the_hourly_accumulations(cds, tmp_path):
    fake = cds()
    store = fetch(tmp_path, start='2020-01-31', end='2020-02-01', step='3h')
    ds = xr.open_zarr(store)
    times = es.time_axis('2020-01-31', '2020-02-01', '3h')
    lat, lon = ds.latitude.values, ds.longitude.values
    assert (ds.time.values == times.values).all() and ds.attrs['accumulation_hours'] == 3
    for v in INSTANT:
        np.testing.assert_allclose(ds[v].values, value(v, times, [0.], lat, lon)[:, 0], rtol=1e-6, err_msg=v)
    for v in ACCUM:
        expected = sum(value(v, times - pd.Timedelta(f'{h}h'), [0.], lat, lon)[:, 0] for h in range(3))
        np.testing.assert_allclose(ds[v].values, expected, rtol=1e-6, err_msg=v)
    # surface requests: all hours, and the last two hours of the previous day for the accumulated fields
    surf = [c for c in fake.calls if c[0].startswith('surf')]
    assert sorted(c[1] for c in surf) == [1, 1, 1, 1]
    assert sum(set(c[2]) == {'total_precipitation', 'surface_solar_radiation_downwards',
                             'surface_thermal_radiation_downwards'} for c in surf) == 2
    assert es.store_gaps(store, '2020-01-31', '2020-02-01') == []
    # a 3 h store of window sums is not mixed with sampled data
    with pytest.raises(ValueError, match='accumulation_hours'):
        es.Era5Store(store, {**json_request(store), 'accumulation_hours': None}, times)


def json_request(store):
    import json
    return json.loads(xr.open_zarr(store).attrs['era5_request'])


def test_adding_fdir_fetches_the_surface_stream_again(cds, tmp_path):
    cds()
    fetch(tmp_path)
    fake = cds()
    store = fetch(tmp_path, surf_variables=es.SURF_VARIABLES + [es.FDIR])
    assert sorted(k for k, _, _ in fake.calls) == ['surf_2020-01', 'surf_2020-02']
    ds = xr.open_zarr(store)
    times = es.time_axis('2020-01-30', '2020-02-02', '1h')
    np.testing.assert_allclose(ds.fdir.sel(time=times).values,
                               value('fdir', times, [0.], ds.latitude.values, ds.longitude.values)[:, 0], rtol=1e-6)
    assert es.store_gaps(store, '2020-01-30', '2020-02-02') == []
    assert es.FDIR in json_request(store)['surf_variables']
    # requesting fewer variables than the store holds: it keeps all, and nothing is missing
    fake = cds()
    fetch(tmp_path, surf_variables=es.SURF_VARIABLES)
    assert fake.calls == []
    fetch(tmp_path, end='2020-02-03', surf_variables=es.SURF_VARIABLES)
    assert all(es.FDIR in v for k, _, v in fake.calls if k.startswith('surf'))


def test_fetcher_adds_fdir_for_the_era5_split(cds, tmp_path, monkeypatch):
    from munch import DefaultMunch
    seen = {}
    monkeypatch.setattr(fe, 'fetch_era5_to_zarr', lambda *a, **k: seen.update(k))
    config = DefaultMunch.fromDict({'project': {'start': '2020-01-30', 'end': '2020-02-02', 'climate': 'era5',
                                                'extent': {'latN': 46.6, 'latS': 46.1, 'lonW': 7.1, 'lonE': 7.6}},
                                    'climate': {'path': str(tmp_path), 'era5': {'timestep': '1h', 'plevels': LEVELS}},
                                    'toposcale': {'direct_diffuse': 'era5'}})
    with contextlib.redirect_stdout(io.StringIO()):
        fe.FetchERA5(config).go_fetch('all')
    assert seen['surf_variables'] == es.SURF_VARIABLES + [es.FDIR]
