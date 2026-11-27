"""
Accumulated fields at 3 h time steps (hourly store summed, store of window sums, store sampling the hours) and the
season of the precipitation coefficients in each hemisphere.
"""
import contextlib
import io
import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import synthetic_era5 as syn
from TopoPyScale import topo_scale_zarr as tz
from TopoPyScale import toposcale_core as core

G = 9.81


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


@pytest.fixture(scope='module')
def case():
    rng = np.random.default_rng(3)
    tp = np.where(rng.random(72) < 0.3, rng.gamma(1., 1.5, 72), 0.)
    return syn.make_era5(start='2020-06-18', n_days=3, cloud=[1, .5, .9], tp_mm=tp)


@pytest.fixture(scope='module')
def rows(case):
    return [syn.make_point(case, name='001', dx=800., elevation=1900.),
            syn.make_point(case, name='002', dx=-3000., dy=2500., elevation=2400., slope_deg=30., aspect_deg=20.,
                           svf=0.9)]


def write_store(case, path, step=1, attrs=None):
    plev = case.plev.copy(deep=True)
    plev['z'] = plev.z * G
    surf = case.surf.rename({'z': 'z_surf'}).copy(deep=True)
    surf['z_surf'] = surf.z_surf * G
    ds = xr.merge([plev, surf]).astype('float32')
    if step > 1:
        # window sums (the first window, before the period, rescaled), or the hour ending at the stamp
        acc = ds[tz.ACC_VARS]
        if attrs:
            acc = acc.rolling(time=step, min_periods=1).mean() * step
        ds = ds.drop_vars(tz.ACC_VARS).merge(acc).isel(time=slice(0, None, step))
    ds.attrs = attrs or {}
    ds.to_zarr(path, mode='w', zarr_format=3)
    return path


def downscale(store, rows, wd, tstep, start, end, name):
    df = pd.DataFrame([vars(r) for r in rows])
    cd = tz.ClimateDownscaler(store, wd / 'down', df, syn.horizon_da(rows), None, syn.EPSG, start, end, tstep,
                              precip_lapse_rate_flag=True, store_name=name)
    return quiet(cd.downscale).load()


def test_hourly_store_is_summed_over_3h_steps(case, rows, tmp_path, caplog):
    store = write_store(case, tmp_path / 'hourly.zarr')
    h1 = downscale(store, rows, tmp_path, '1h', case.start, case.end, 'h1.zarr')
    with caplog.at_level(logging.WARNING, logger='TopoPyScale'):
        h3 = downscale(store, rows, tmp_path, '3h', case.start, case.end, 'h3.zarr')
    assert 'first accumulation window starts before' in caplog.text
    assert h3.attrs['accumulation_hours'] == 3 and h3.attrs['accumulation_window'] == '3h'
    # precipitation: the sum of the hourly values of the window ending at each stamp (all but the first window)
    tp1 = h1.tp.values[:, 1:70].reshape(2, 23, 3).sum(axis=2)   # hours 1-3, 4-6, ... 67-69
    np.testing.assert_allclose(h3.tp.values[:, 1:], tp1, rtol=1e-5, atol=1e-6)
    # shortwave: close to the mean of the hourly values over the window
    sw1 = h1.SW.values[:, 1:70].reshape(2, 23, 3).mean(axis=2)
    assert np.abs(h3.SW.values[:, 1:] - sw1).mean() < 0.05 * sw1.mean()
    assert abs(h3.SW.values.mean() - h1.SW.values.mean()) < 0.03 * h1.SW.values.mean()


def test_store_of_window_sums_equals_the_summed_hourly_store(case, rows, tmp_path):
    hourly = write_store(case, tmp_path / 'hourly.zarr')
    sums = write_store(case, tmp_path / 'sums.zarr', step=3, attrs={'accumulation_hours': 3})
    start = '2020-06-18 03:00'    # first stamp whose window is in the hourly store
    a = downscale(hourly, rows, tmp_path, '3h', case.start, case.end, 'a.zarr')
    b = downscale(sums, rows, tmp_path, '3h', case.start, case.end, 'b.zarr')
    for v in ['tp', 'SW', 'SW_direct', 'LW', 't']:
        np.testing.assert_allclose(b[v].sel(time=slice(start, None)).values,
                                   a[v].sel(time=slice(start, None)).values, rtol=1e-4, atol=1e-3, err_msg=v)


def test_store_sampling_the_hours_is_scaled_with_a_warning(case, rows, tmp_path, caplog):
    sampled = write_store(case, tmp_path / 'sampled.zarr', step=3)
    hourly = write_store(case, tmp_path / 'hourly.zarr')
    with caplog.at_level(logging.WARNING, logger='TopoPyScale'):
        s = downscale(sampled, rows, tmp_path, '3h', case.start, case.end, 's.zarr')
    assert 'cover 1 h of each 3 h time step' in caplog.text
    assert s.attrs['accumulation_window'] == '1h'
    h1 = downscale(hourly, rows, tmp_path, '1h', case.start, case.end, 'h1.zarr')
    # the hour ending at each stamp, times 3
    np.testing.assert_allclose(s.tp.values, 3 * h1.tp.values[:, ::3], rtol=1e-5, atol=1e-6)


def test_precipitation_coefficients_follow_the_season_of_each_hemisphere():
    months = np.arange(1, 13)
    assert list(core.seasonal_month(months, [10.])[:, 0]) == list(months)
    assert list(core.seasonal_month(months, [-10.])[:, 0]) == [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    dz = np.full((12, 2), 1000.)
    tp = np.ones((12, 2))
    _, f = core.precipitation(tp, dz, months, 1, True, latitude=[45., -45.])
    np.testing.assert_allclose(f[:, 1], np.roll(f[:, 0], -6))
    assert f[0, 0] > f[0, 1]   # January: winter in the north, summer in the south
