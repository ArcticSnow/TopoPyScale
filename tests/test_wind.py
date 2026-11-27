"""Wind options: blending with the ERA5 10 m wind, MicroMet terrain factors (Liston and Elder, 2006)."""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import synthetic_era5 as syn
from TopoPyScale import topo_scale_zarr as tz
from TopoPyScale import toposcale_core as core
from TopoPyScale import wind as wnd

G = 9.81


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


def terrain(u, v, slope_deg, aspect_deg, curvature=0., slope_scale=np.deg2rad(120.)):
    u, v = core.wind_terrain(np.array([[u]]), np.array([[v]]), np.deg2rad([slope_deg]), np.deg2rad([aspect_deg]),
                             np.array([curvature]), slope_scale)
    return core.wind_speed_direction(u, v)


def test_micromet_factors():
    # flat open ground: unchanged
    ws, wd = terrain(0., -5., 0., 0.)          # wind from the north
    np.testing.assert_allclose([ws[0, 0], wd[0, 0]], [5., 0.], atol=1e-12)
    # slope facing the wind (north, aspect 180 in the TopoPyScale convention): faster, the lee slower
    omega = np.deg2rad(30.) / np.deg2rad(120.)
    np.testing.assert_allclose(terrain(0., -5., 30., 180.)[0], 5 * (1 + 0.58 * omega))
    np.testing.assert_allclose(terrain(0., -5., 30., 0.)[0], 5 * (1 - 0.58 * omega))
    # ridge (curvature 0.5) faster, hollow slower
    np.testing.assert_allclose(terrain(0., -5., 0., 0., curvature=0.5)[0], 5 * 1.21)
    np.testing.assert_allclose(terrain(0., -5., 0., 0., curvature=-0.5)[0], 5 * 0.79)
    # wind across the slope: no change of speed nor direction; oblique: diverted (Ryan 1977)
    ws, wd = terrain(0., -5., 30., 90.)
    np.testing.assert_allclose([ws[0, 0], np.sin(wd[0, 0]), np.cos(wd[0, 0])], [5., 0., 1.], atol=1e-12)
    ws, wd = terrain(-5 * np.sin(np.pi / 4), -5 * np.cos(np.pi / 4), 30., 90.)   # from the NE on an east slope
    om = np.deg2rad(30.) * np.cos(np.pi / 4 - np.pi / 2) / np.deg2rad(120.)
    np.testing.assert_allclose(wd, np.pi / 4 - 0.5 * om * np.sin(2 * (np.pi / 2 - np.pi / 4)))


def test_curvature_of_hill_valley_and_plane():
    x = np.arange(-2000., 2001., 50.)
    xx, yy = np.meshgrid(x, -x)
    hill = 300 * np.exp(-(xx ** 2 + yy ** 2) / 600. ** 2)
    c = wnd.curvature(hill, 50., 500.)
    assert c[40, 40] > 0 and wnd.curvature(-hill, 50., 500.)[40, 40] < 0
    np.testing.assert_allclose(wnd.curvature(0.2 * xx + 0.1 * yy, 50., 500.)[10:-10, 10:-10], 0., atol=1e-12)


def dem_param(names=None):
    x = np.arange(0., 3001., 30.)
    y = np.arange(3000., -1., -30.)
    xx, yy = np.meshgrid(x, y)
    z = 1500 + 400 * np.exp(-((xx - 1500) ** 2 + (yy - 1500) ** 2) / 500. ** 2)
    gy, gx = np.gradient(z, 30.)
    ds = xr.Dataset({'elevation': (('y', 'x'), z), 'slope': (('y', 'x'), np.arctan(np.hypot(gx, gy)))},
                    coords={'x': x, 'y': y})
    if names is not None:
        ds['point_name'] = (('y', 'x'), np.where(xx < 1500, names[0], names[1]))
    return ds


def test_setup_scales_and_samples_the_curvature():
    df = pd.DataFrame({'point_name': ['top', 'foot'], 'x': [1500., 300.], 'y': [1500., 300.],
                       'slope': [0., 0.], 'aspect': [0., 0.]})
    out = wnd.setup(df, {'terrain': 'liston_elder'}, dem_param(), per_cluster=False)
    assert out['curvature'][0] > 0.4 and abs(out['curvature'][1]) < 0.05
    assert 'blend' not in out and np.allclose(df.wind_curvature, out['curvature'])
    clusters = wnd.setup(df.copy(), {'terrain': 'liston_elder'}, dem_param(['top', 'foot']), per_cluster=True)
    assert np.all(np.abs(clusters['curvature']) <= 0.5) and clusters['curvature'][0] < out['curvature'][0]
    assert wnd.setup(df, {}, None, False) is None
    np.testing.assert_allclose(wnd.setup(df, {'surface_blend': 0.3}, None, False)['blend'], 0.3)
    surface = {'weight': np.array([1.3, 0.4])}
    np.testing.assert_allclose(wnd.setup(df, {'surface_blend': 'surface_effect'}, None, False, surface)['blend'],
                               [1., 0.4])
    for bad, msg in [({'surface_blend': 1.5}, r'\[0, 1\]'), ({'terrain': 'helbig'}, 'not supported'),
                     ({'surface_blend': 'surface_effect'}, 'needs toposcale.surface_effect'),
                     ({'curvature': 1}, 'Unknown')]:
        with pytest.raises(ValueError, match=msg):
            wnd.setup(df, bad, None, False)


@pytest.fixture(scope='module')
def case():
    return syn.make_era5(start='2020-06-18', n_days=1)


def store(case, path, wind10=True):
    plev = case.plev.copy(deep=True)
    plev['z'] = plev.z * G
    surf = case.surf.rename({'z': 'z_surf'}).copy(deep=True)
    surf['z_surf'] = surf.z_surf * G
    if wind10:
        surf['u10'] = surf.t2m * 0 + 1.
        surf['v10'] = surf.t2m * 0 - 2.
    xr.merge([plev, surf]).astype('float32').to_zarr(path, mode='w', zarr_format=3)
    return path


def run(case, path, wd, name, wind):
    rows = [syn.make_point(case, name='001', elevation=2000.)]
    df = pd.DataFrame([vars(r) for r in rows])
    cd = tz.ClimateDownscaler(path, wd / 'down', df, syn.horizon_da(rows), None, syn.EPSG, case.start, case.end,
                              '1h', store_name=name, wind=wind)
    return quiet(cd.downscale).load().isel(point_name=0)


def test_downscaler_blends_with_the_10m_wind(case, tmp_path):
    path = store(case, tmp_path / 'w.zarr')
    ref = run(case, path, tmp_path, 'ref.zarr', None)
    full = run(case, path, tmp_path, 'full.zarr', {'blend': np.array([1.]), 'terrain': 'none'})
    half = run(case, path, tmp_path, 'half.zarr', {'blend': np.array([0.5]), 'terrain': 'none'})
    np.testing.assert_allclose(full.u, 1., atol=1e-5)
    np.testing.assert_allclose(full.v, -2., atol=1e-5)
    np.testing.assert_allclose(half.u, 0.5 * (ref.u + 1.), atol=1e-4)
    assert half.attrs['wind_surface_blend'] == '0.50 to 0.50' and ref.attrs['wind_terrain'] == 'none'
    with pytest.raises(ValueError, match='10 m wind'):
        run(case, store(case, tmp_path / 'no10.zarr', wind10=False), tmp_path, 'x.zarr',
            {'blend': np.array([1.]), 'terrain': 'none'})


def test_fetcher_adds_the_10m_wind_for_the_blend(tmp_path, monkeypatch):
    from munch import DefaultMunch
    from TopoPyScale import era5_store as es
    from TopoPyScale import fetch_era5 as fe
    seen = {}
    monkeypatch.setattr(fe, 'fetch_era5_to_zarr', lambda *a, **k: seen.update(k))
    config = DefaultMunch.fromDict({'project': {'start': '2020-01-30', 'end': '2020-02-02', 'climate': 'era5',
                                                'extent': {'latN': 46.6, 'latS': 46.1, 'lonW': 7.1, 'lonE': 7.6}},
                                    'climate': {'path': str(tmp_path), 'era5': {'timestep': '1h', 'plevels': [700]}},
                                    'toposcale': {'wind_downscaling': {'surface_blend': 0.5}}})
    with contextlib.redirect_stdout(io.StringIO()):
        fe.FetchERA5(config).go_fetch('all')
    assert seen['surf_variables'] == es.SURF_VARIABLES + [wnd.U10, wnd.V10]
