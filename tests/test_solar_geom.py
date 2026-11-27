"""Tests of the vectorized solar geometry (solar_geom.py) against pvlib."""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pvlib
import pytest
import xarray as xr

from TopoPyScale import solar_geom as sg


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fun(*args, **kwargs)


@pytest.fixture(scope='module')
def points():
    rng = np.random.default_rng(3)
    n = 12
    return pd.DataFrame({'point_name': [f'{i:03d}' for i in range(n)], 'latitude': rng.uniform(-75, 80, n),
                         'longitude': rng.uniform(-180, 180, n), 'elevation': rng.uniform(0, 5000, n)})


@pytest.fixture(scope='module')
def ds_solar(points, tmp_path_factory):
    return quiet(sg.get_solar_geom, points, '2021-12-20', '2021-12-23', '1h',
                 project_ouput=tmp_path_factory.mktemp('solar'))


T_MID = pd.date_range('2021-12-20', '2021-12-24', freq='1h', inclusive='left', tz='UTC') - pd.Timedelta('30min')


# ---------------------------------------------------------------------------------------------------------------------
#  solar geometry
# ---------------------------------------------------------------------------------------------------------------------
def test_angles_match_pvlib_at_window_centre(points, ds_solar):
    for i, r in points.iterrows():
        ref = pvlib.solarposition.get_solarposition(T_MID, r.latitude, r.longitude, r.elevation, method='nrel_numpy')
        np.testing.assert_allclose(np.rad2deg(ds_solar.zenith[i].values), ref.zenith.values, atol=1e-3)
        np.testing.assert_allclose(np.rad2deg(ds_solar.elevation[i].values), ref.elevation.values, atol=1e-3)
        daz = np.angle(np.exp(1j * (ds_solar.azimuth[i].values - np.deg2rad(180 - ref.azimuth.values))), deg=True)
        assert np.abs(daz).max() < 1e-2


def test_swtoa_is_mean_over_accumulation_window(points, ds_solar):
    fine = pd.date_range(T_MID[0] - pd.Timedelta('30min'), periods=T_MID.size * 60, freq='1min', tz='UTC') + \
        pd.Timedelta('30s')
    esd = pvlib.solarposition.nrel_earthsun_distance(fine).values
    for i, r in points.iloc[:6].iterrows():
        sp = pvlib.solarposition.get_solarposition(fine, r.latitude, r.longitude, r.elevation, method='nrel_numpy')
        ref = (1361 * np.clip(np.cos(np.deg2rad(sp.zenith.values)), 0, None) / esd ** 2).reshape(-1, 60).mean(1)
        np.testing.assert_allclose(ds_solar.SWtoa[i].values, ref, atol=2.)  # W m-2


def test_earth_sun_distance_in_swtoa(tmp_path):
    """Same geometry in January (perihelion) and July (aphelion): SWtoa ratio = (R_jul / R_jan)^2 ~ 1.069."""
    df = pd.DataFrame({'point_name': ['eq'], 'latitude': [0.], 'longitude': [0.], 'elevation': [0.]})
    jan = quiet(sg.get_solar_geom, df, '2021-01-03', '2021-01-03', '1h', project_ouput=tmp_path, fname='jan.nc')
    jul = quiet(sg.get_solar_geom, df, '2021-07-04', '2021-07-04', '1h', project_ouput=tmp_path, fname='jul.nc')
    ratio = float(jan.SWtoa.max() / jan.mu0.max()) / float(jul.SWtoa.max() / jul.mu0.max())
    assert 1.06 < ratio < 1.075


def test_inputs_xy_or_lonlat_and_no_side_effect(points, ds_solar, tmp_path):
    from pyproj import Transformer
    sub = points.iloc[:3].copy()
    x, y = Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True).transform(sub.longitude, sub.latitude)
    df_xy = pd.DataFrame({'point_name': sub.point_name, 'x': x, 'y': y, 'elevation': sub.elevation})
    before = df_xy.copy()
    ds = quiet(sg.get_solar_geom, df_xy, '2021-12-20', '2021-12-23', '1h', sr_epsg='3857', project_ouput=tmp_path)
    pd.testing.assert_frame_equal(df_xy, before)  # input not modified
    np.testing.assert_allclose(ds.zenith.values, ds_solar.zenith.values[:3], atol=1e-8)


def test_solar_file_is_not_quantized(points, ds_solar, tmp_path):
    quiet(sg.get_solar_geom, points, '2021-12-20', '2021-12-23', '1h', project_ouput=tmp_path, fname='s.nc')
    back = xr.open_dataset(tmp_path / 's.nc', engine='h5netcdf')
    for v in ['zenith', 'mu0', 'SWtoa']:
        np.testing.assert_allclose(back[v].values, ds_solar[v].values, rtol=1e-6, atol=1e-6, err_msg=v)
    assert back.sunset.dtype == bool


def test_other_pvlib_method(points, ds_solar, tmp_path):
    ds = quiet(sg.get_solar_geom, points.iloc[:2], '2021-12-20', '2021-12-23', '1h', project_ouput=tmp_path,
               method_solar='ephemeris')
    np.testing.assert_allclose(ds.zenith.values, ds_solar.zenith.values[:2], atol=np.deg2rad(0.05))
