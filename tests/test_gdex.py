"""
ERA5 from NCAR GDEX (fetch_era5.fetch_era5_to_zarr with source='gdex'), against a miniature GDEX tree with the file names, variable
names and layouts of dataset d633000: zarr stores of surface analyses, daily pressure-level files, half-month files
of forecast accumulations (forecast_initial_time, forecast_hour) and the invariant surface geopotential. Values
encode the variable, time, level and position, so the time mapping and the subsetting can be checked exactly.
"""
import contextlib
import functools
import http.server
import io
import os
import threading

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TopoPyScale import fetch_era5 as fe

LAT = np.arange(60.0, 57.99, -0.25)                                   # descending, as GDEX
LON = np.r_[np.arange(0, 2.01, 0.25), np.arange(358.0, 359.99, 0.25)]  # 0..360, near Greenwich only
LEVELS = np.array([500., 700., 850., 925., 1000.])
REF = pd.Timestamp('2020-01-01')
DAYS = pd.date_range('2020-01-31', '2020-02-01', freq='D')            # across a month and a half-month file
BOX = dict(latN=59.6, latS=58.4, lonE=0.6, lonW=-0.6)                  # across the Greenwich meridian
CODE = {'Z': 1, 'T': 2, 'U': 3, 'V': 4, 'Q': 5, 'VAR_2T': 6, 'VAR_2D': 7, 'SP': 8, 'SSRD': 9, 'STRD': 10, 'TP': 11}


def value(var, time, level=0., lat=None, lon=None):
    """Synthetic field: variable, valid time (hours since REF), level and position are all recoverable."""
    hours = (pd.DatetimeIndex(np.atleast_1d(time)) - REF) / pd.Timedelta('1h')
    lon360 = np.mod(lon, 360)
    return (100 * CODE[var] + 0.01 * hours.values[:, None, None, None] + np.asarray(level)[None, :, None, None] / 1000
            + lat[None, None, :, None] / 10 + lon360[None, None, None, :] / 1000)


def write_nc(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine='h5netcdf')


def make_gdex(root):
    root.mkdir(parents=True, exist_ok=True)
    # pressure levels: one file per variable and day, (time, level, latitude, longitude)
    for day in DAYS:
        times = pd.date_range(day, periods=24, freq='1h')
        for var, stem in zip(['Z', 'T', 'U', 'V', 'Q'], fe.GDEX_PLEV.values()):
            da = xr.DataArray(value(var, times, LEVELS, LAT, LON), dims=['time', 'level', 'latitude', 'longitude'],
                              coords={'time': times, 'level': LEVELS, 'latitude': LAT, 'longitude': LON}, name=var)
            write_nc(da.to_dataset().assign(utc_date=('time', np.zeros(24, 'int32'))),
                     root / f'e5.oper.an.pl/{day:%Y%m}/e5.oper.an.pl.{stem}.{day:%Y%m%d}00_{day:%Y%m%d}23.nc')
    # surface analyses: zarr stores ending on Jan 31 (Feb 1 comes from the monthly netCDF file)
    jan = pd.date_range('2020-01-30', '2020-01-31 23:00', freq='1h')
    feb = pd.date_range('2020-02-01', periods=24, freq='1h')
    for var, (short, stem) in zip(['VAR_2T', 'VAR_2D', 'SP'], fe.GDEX_SURF_AN.values()):
        for times, kind in [(jan, 'zarr'), (feb, 'nc')]:
            da = xr.DataArray(value(var, times, [0.], LAT, LON)[:, 0], dims=['time', 'latitude', 'longitude'],
                              coords={'time': times, 'latitude': LAT, 'longitude': LON}, name=var)
            ds = da.to_dataset().assign(utc_date=('time', np.zeros(times.size, 'int32')))
            if kind == 'zarr':
                ds.chunk({'time': 27}).to_zarr(root / f'e5.oper.an.sfc.zarr/e5.oper.an.sfc.{short}.zarr',
                                               zarr_format=2, consolidated=True)
            else:
                write_nc(ds, root / f'e5.oper.an.sfc/202002/e5.oper.an.sfc.{stem}.2020020100_2020022923.nc')
    # accumulations: half-month files of forecasts started at 06 and 18 UTC, steps 1..12
    files = {'2020011606_2020020106': pd.date_range('2020-01-16 06:00', '2020-01-31 18:00', freq='12h'),
             '2020020106_2020021606': pd.date_range('2020-02-01 06:00', '2020-02-15 18:00', freq='12h')}
    steps = np.arange(1, 13)
    for var, stem in zip(['SSRD', 'STRD', 'TP'], fe.GDEX_SURF_ACCUMU.values()):
        for span, inits in files.items():
            valid = inits.values[:, None] + steps[None, :] * np.timedelta64(1, 'h')
            data = value(var, valid.ravel(), [0.], LAT, LON)[:, 0].reshape(inits.size, steps.size, LAT.size, LON.size)
            da = xr.DataArray(data, dims=['forecast_initial_time', 'forecast_hour', 'latitude', 'longitude'],
                              coords={'forecast_initial_time': inits, 'forecast_hour': steps.astype('int32'),
                                      'latitude': LAT, 'longitude': LON}, name=var)
            write_nc(da.to_dataset().assign(utc_date=('forecast_initial_time', np.zeros(inits.size, 'int32'))),
                     root / f'e5.oper.fc.sfc.accumu/{span[:6]}/e5.oper.fc.sfc.accumu.{stem}.{span}.nc')
    # invariant surface geopotential
    z = xr.DataArray(1000 * (LAT[:, None] + 0 * LON[None, :])[None], dims=['time', 'latitude', 'longitude'],
                     coords={'time': [pd.Timestamp('1979-01-01')], 'latitude': LAT, 'longitude': LON}, name='Z')
    write_nc(z.to_dataset(), root / fe.GDEX_INVARIANT_Z)
    return root


@pytest.fixture(scope='module')
def gdex(tmp_path_factory):
    return make_gdex(tmp_path_factory.mktemp('gdex'))


def fetch(root, workdir, **kw):
    kw = {**dict(step='1h', plevels=[1000, 700, 850], n_workers=1), **kw}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        store = fe.fetch_era5_to_zarr(DAYS[0], DAYS[-1], workdir, source='gdex', gdex_root=str(root), retry_wait=0,
                                      **BOX, **kw)
    return xr.load_dataset(store, engine='zarr'), out.getvalue()


def check(ds, step_h=1):
    times = pd.date_range(DAYS[0], DAYS[-1] + pd.Timedelta('23h'), freq=f'{step_h}h')
    lat, lon = ds.latitude.values, ds.longitude.values
    np.testing.assert_allclose(lat, [59.5, 59.25, 59.0, 58.75, 58.5])
    np.testing.assert_allclose(lon, [-0.5, -0.25, 0., 0.25, 0.5])
    assert (ds.time.values == times.values).all()
    assert list(ds.level.values) == [700, 850, 1000]
    assert set(ds.data_vars) == {'z', 't', 'u', 'v', 'q', 'z_surf', 't2m', 'd2m', 'sp', 'ssrd', 'strd', 'tp',
                                 'filled_plev', 'filled_surf'}
    assert ds.filled_plev.values.all() and ds.filled_surf.values.all()
    for name, var in [('z', 'Z'), ('t', 'T'), ('u', 'U'), ('v', 'V'), ('q', 'Q')]:
        np.testing.assert_allclose(ds[name].transpose('time', 'level', 'latitude', 'longitude').values,
                                   value(var, times, [700, 850, 1000], lat, lon), rtol=1e-6, err_msg=name)
    for name, var in [('t2m', 'VAR_2T'), ('d2m', 'VAR_2D'), ('sp', 'SP'), ('ssrd', 'SSRD'), ('strd', 'STRD'),
                      ('tp', 'TP')]:
        np.testing.assert_allclose(ds[name].transpose('time', 'latitude', 'longitude').values,
                                   value(var, times, [0.], lat, lon)[:, 0], rtol=1e-6, err_msg=name)
    assert ds.z_surf.dims == ('latitude', 'longitude')
    np.testing.assert_allclose(ds.z_surf.values, 1000 * lat[:, None] + 0 * lon[None, :], rtol=1e-6)


def test_local_gdex_matches_cds_layout(gdex, tmp_path):
    ds, _ = fetch(gdex, tmp_path)
    check(ds)
    assert not (tmp_path / 'raw').exists() and not (tmp_path / 'daily').exists()
    # a second run finds everything in the store
    _, log = fetch(gdex, tmp_path)
    assert '0 of 5 downloads to do' in log


def test_timestep_and_processes(gdex, tmp_path):
    ds, _ = fetch(gdex, tmp_path, step='3h', n_workers=2)
    check(ds, step_h=3)


def test_accumulation_valid_times():
    t = pd.date_range('2020-02-01', periods=24, freq='1h')
    init, step = fe._accumu_init_step(t)
    assert (init + pd.to_timedelta(step, unit='h') == t).all()
    assert step.min() == 1 and step.max() == 12
    assert set(init.hour) == {6, 18}
    # ECMWF table: 00 UTC is the 18 UTC forecast of the previous day, step 6
    assert init[0] == pd.Timestamp('2020-01-31 18:00') and step[0] == 6
    assert fe._accumu_file('/r', 's', pd.Timestamp('2020-01-31 18:00')).endswith(
        '202001/e5.oper.fc.sfc.accumu.s.2020011606_2020020106.nc')
    assert fe._accumu_file('/r', 's', pd.Timestamp('2020-12-16 06:00')).endswith('2020121606_2021010106.nc')


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP range requests, as served by OSDF caches."""

    def log_message(self, *args):
        pass

    def _serve(self, body):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        start, end, status = 0, size - 1, 200
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            a, b = rng[6:].split(',')[0].split('-')
            start, end = (max(0, size - int(b)), size - 1) if a == '' else (int(a), min(int(b), size - 1) if b else size - 1)
            status = 206
        self.send_response(status)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if body:
            with open(path, 'rb') as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))

    def do_GET(self):
        self._serve(True)

    def do_HEAD(self):
        self._serve(False)


def test_remote_gdex_over_https(gdex, tmp_path):
    pytest.importorskip('aiohttp')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(RangeHandler, directory=str(gdex)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ds, _ = fetch(f'http://127.0.0.1:{server.server_address[1]}', tmp_path)
    finally:
        server.shutdown()
    check(ds)


def test_fetcher_dispatch(tmp_path, monkeypatch):
    from munch import DefaultMunch
    config = DefaultMunch.fromDict({
        'project': {'climate': 'era5', 'start': pd.Timestamp('2020-01-31'), 'end': pd.Timestamp('2020-02-01'),
                    'extent': {'latN': 59.2, 'latS': 58.8, 'lonW': -0.2, 'lonE': 0.2}},
        'climate': {'path': tmp_path, 'era5': {'product': 'reanalysis', 'timestep': '1h', 'plevels': [700, 850],
                                               'download_threads': 3, 'data_repository': 'gdex',
                                               'gdex_root': '/somewhere'}},
        'outputs': {'path': tmp_path / 'outputs'}})
    calls = []
    monkeypatch.setattr(fe, 'fetch_era5_to_zarr', lambda *a, **k: calls.append(k))
    with contextlib.redirect_stdout(io.StringIO()):
        fe.FetchERA5(config).go_fetch('all')
    kw, = calls
    assert kw['source'] == 'gdex' and kw['gdex_root'] == '/somewhere' and kw['n_workers'] == 3
    assert kw['latN'] == pytest.approx(59.6) and kw['lonW'] == pytest.approx(-0.6)    # with the 0.4 margin

    config.climate.era5.realtime = True
    with pytest.raises(ValueError, match='realtime needs ERA5T'):
        with contextlib.redirect_stdout(io.StringIO()):
            fe.FetchERA5(config).go_fetch('all')
    config.climate.era5.realtime = False

    # no repository given: the CDS
    del config.climate.era5['data_repository']
    with contextlib.redirect_stdout(io.StringIO()):
        fe.FetchERA5(config).go_fetch('all')
    assert calls[-1]['source'] == 'cds'
