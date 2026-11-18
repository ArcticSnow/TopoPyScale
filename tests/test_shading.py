"""Fractional terrain shading: the direct beam is scaled by the sunlit fraction of the accumulation window."""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import synthetic_era5 as syn
from TopoPyScale import solar_geom as sg
from TopoPyScale import topo_scale_zarr as tz


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


def test_geocentric_sun_matches_spa():
    times = pd.date_range('2021-01-01', '2021-12-31', freq='97min', tz='UTC')
    lat, lon, z = np.array([46.8, -33.9, 69.6]), np.array([9.8, 18.4, 18.9]), np.array([1600., 0., 0.])
    terms = sg.spa_time_terms(times)
    zen, az_n, el = sg.spa_positions(terms, lat, lon, z)
    elev, az = sg.sun_geocentric(terms, lat, lon)
    np.testing.assert_allclose(np.rad2deg(elev), el, atol=0.01)
    up = (el > 0) & (el < 85)
    d_az = (np.rad2deg(az) - (180. - az_n) + 180.) % 360. - 180.
    assert np.abs(d_az[up]).max() < 0.02


@pytest.fixture(scope='module')
def summer_wall(tmp_path_factory):
    """One flat point behind a 25 deg horizon, 3 summer days."""
    wd = tmp_path_factory.mktemp('shading')
    case = syn.make_era5(start='2020-06-20', n_days=3)
    rows = [syn.make_point(case, name='001', elevation=1500.)]
    ds_solar = quiet(syn.solar_geometry, case, rows, wd)
    df = pd.DataFrame([vars(r) for r in rows])
    return case, df, ds_solar, syn.era5_store(case, wd), wd


def run(setup, horizon_deg, n_sub, name, ds_solar=None):
    case, df, sol, store, wd = setup
    cd = tz.ClimateDownscaler(store, wd / 'down', df, syn.horizon_da(list(df.itertuples()), horizon_deg),
                              sol if ds_solar is None else ds_solar, syn.EPSG, case.start, case.end, '1h',
                              store_name=name, shading_substeps=n_sub)
    return quiet(cd.downscale).load().SW_direct.isel(point_name=0).values


def test_fractional_shading_converges_to_fine_reference(summer_wall):
    binary = run(summer_wall, 25., 1, 'b.zarr')
    frac = run(summer_wall, 25., 6, 'f.zarr')
    fine = run(summer_wall, 25., 60, 'r.zarr')
    partial = (frac > 1) & (frac < binary - 1) | (frac > binary + 1)
    assert partial.any()                                       # transition hours change
    assert np.abs(frac - fine).max() < np.abs(binary - fine).max() / 3
    assert abs(frac.sum() / fine.sum() - 1) < 0.01
    np.testing.assert_allclose(frac[(binary == 0) & (frac == 0)], 0)


def test_open_horizon_unchanged(summer_wall):
    np.testing.assert_allclose(run(summer_wall, 0., 6, 'o6.zarr'), run(summer_wall, 0., 1, 'o1.zarr'), atol=1e-4)


def test_window_from_older_solar_files(summer_wall):
    sol = summer_wall[2].copy()
    sol.attrs = {k: v for k, v in sol.attrs.items() if k not in ('accumulation_window', 'solar_substeps')}
    np.testing.assert_allclose(run(summer_wall, 25., None, 'old.zarr', ds_solar=sol),
                               run(summer_wall, 25., 6, 'new.zarr'))
