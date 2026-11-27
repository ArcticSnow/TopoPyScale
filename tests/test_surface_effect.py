"""Synthetic tests of the surface-effect correction (REDCAPP, DReaMIT) and of its calibration."""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr

import synthetic_era5 as syn
from TopoPyScale import surface_calib as sc
from TopoPyScale import surface_effect as se
from TopoPyScale import toposcale_core as core
from TopoPyScale import topo_scale_zarr as tz


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
#  correction
# ---------------------------------------------------------------------------------------------------------------------
def test_zero_weight_returns_inputs_and_humidity_stays_physical():
    rng = np.random.default_rng(1)
    t = rng.uniform(250, 300, (50, 4))
    p = rng.uniform(6e4, 9e4, (50, 4))
    q = core.q_from_dewpoint(t - rng.uniform(0.5, 15, t.shape), p)
    dT, dTd = rng.normal(-5, 4, t.shape), rng.normal(-3, 4, t.shape)
    t0, q0 = core.surface_correction(t, q, p, dT, dTd, np.zeros(4))
    np.testing.assert_allclose(t0, t)
    np.testing.assert_allclose(q0, q, rtol=1e-10)
    t1, q1 = core.surface_correction(t, q, p, dT, dTd, rng.uniform(0, 1.5, 4), beta=rng.normal(0, 1, (50, 1)))
    td1 = core.dewpoint_from_vp(core.vapor_pressure_from_q(q1, p))
    assert np.all(td1 <= t1 + 1e-9) and np.all(q1 > 0)


def test_full_weight_at_grid_elevation_gives_t2m():
    z = np.array([[3000., 1500., 100.]] * 2)[:, :, None]  # (time, level, point), levels by increasing pressure
    plev = {'z': z, 't': 290. - 0.0065 * z, 'q': np.full_like(z, 4e-3), 'u': 0 * z, 'v': 0 * z}
    surf = {'z_surf': np.full((2, 1), 2000. * core.G), 't2m': np.array([[270.], [280.]]),
            'd2m': np.array([[265.], [268.]])}
    dT, dTd = core.surface_signal(plev, surf, np.array([700., 850., 1000.]))
    out = core.interpolate_vertical(plev, np.array([2000.]), np.array([700., 850., 1000.]))
    t, q = core.surface_correction(out['t'], out['q'], out['p'], dT, dTd, np.ones(1))
    np.testing.assert_allclose(t, surf['t2m'])
    np.testing.assert_allclose(core.dewpoint_from_vp(core.vapor_pressure_from_q(q, out['p'])), surf['d2m'])


def test_published_weights():
    assert se.weight_dreamit(0., **se.DEFAULTS['dreamit']) == pytest.approx(0.449)
    assert se.weight_dreamit(1., **se.DEFAULTS['dreamit']) == pytest.approx(0.449 + np.exp(0.732) - 1)
    # flat terrain (no relief): REDCAPP weight is alpha
    assert se.weight_redcapp(0.3, 0., 5., **se.DEFAULTS['redcapp']) == pytest.approx(0.61)
    # rugged valley bottom: alpha H + beta V
    assert se.weight_redcapp(1., 1e5, 8., **se.DEFAULTS['redcapp']) == pytest.approx(0.61 + 1.56)
    b = se.beta_t(pd.date_range('2021-01-01', '2021-12-31', freq='1D'), beta_amp=1., t_star=0.5, beta_bias=2.)
    assert b.min() == pytest.approx(1., abs=1e-3) and b.max() == pytest.approx(3., abs=1e-3) and b[182] > 2.99


# ---------------------------------------------------------------------------------------------------------------------
#  terrain indices
# ---------------------------------------------------------------------------------------------------------------------
def write_dem(path, z, res=100.):
    ny, nx = z.shape
    da = xr.DataArray(z.astype('float32'), dims=('y', 'x'),
                      coords={'y': 5e6 - res * np.arange(ny), 'x': 5e5 + res * np.arange(nx)})
    da.rio.write_crs('EPSG:32632', inplace=True).rio.to_raster(path)
    return da


def test_hypsometric_position_relief_and_coverage(tmp_path):
    yy, xx = np.mgrid[0:201, 0:201]
    cone = 3000. - 10. * np.hypot(yy - 100, xx - 100)
    da = write_dem(tmp_path / 'cone.tif', cone)
    pts = pd.DataFrame({'x': [da.x[100].item(), da.x[100].item(), da.x[0].item()],
                        'y': [da.y[100].item(), da.y[170].item(), da.y[0].item()],
                        'elevation': [3000., 2300., cone[0, 0]]})
    idx = se.terrain_indices(pts, tmp_path / 'cone.tif', radius_km=5.)
    assert idx.hypso_pos[0] == 0.                       # summit
    assert 0.3 < idx.hypso_pos[1] < 0.7                 # mid slope
    assert idx.window_coverage[0] == pytest.approx(1., abs=0.01) and idx.window_coverage[2] < 0.3
    assert idx.relief[0] == pytest.approx(500., abs=1.)  # 50 cells of 10 m drop
    fl = write_dem(tmp_path / 'flat.tif', np.full((50, 50), 500.))
    pts_flat = pd.DataFrame({'x': [fl.x[25].item()], 'y': [fl.y[25].item()], 'elevation': [500.]})
    flat = se.terrain_indices(pts_flat, tmp_path / 'flat.tif', radius_km=1.)
    assert flat.relief[0] == 0 and flat.hypso_pos[0] == 0


def test_mrvbf_high_on_valley_floor_low_on_slopes(tmp_path):
    xx = np.arange(400)[None, :].repeat(300, axis=0)
    d = np.abs(xx - 200)
    valley = np.where(d < 60, 1000. + 0.2 * d, 1012. + 0.6 * (d - 60) * 30.)  # flat floor, 30 m DEM
    m, steps = se.mrvbf(valley, 30., slope_threshold=16.)
    assert steps >= 3
    assert np.nanmean(m[100:200, 180:220]) > 2. and np.nanmean(m[100:200, 300:350]) < 0.5


def test_setup_off_partial_window_and_weights(tmp_path):
    da = write_dem(tmp_path / 'dem.tif', 2000. + 300. * np.sin(np.arange(60)[None, :] / 6.) * np.ones((60, 1)))
    df = pd.DataFrame({'x': [da.x[30].item()], 'y': [da.y[30].item()], 'elevation': [1900.]})
    assert se.setup(df, None, tmp_path / 'dem.tif', 32632) is None
    assert se.setup(df, {'method': 'none', 'diagnostics': True}, tmp_path / 'dem.tif', 32632)['weight'] is None
    with pytest.raises(ValueError, match='allow_partial_window'):
        se.setup(df, {'method': 'dreamit'}, tmp_path / 'dem.tif', 32632)
    s = quiet(se.setup, df, {'method': 'redcapp', 'allow_partial_window': True}, tmp_path / 'dem.tif', 32632)
    assert s['weight'][0] == pytest.approx(df.w_surf[0]) and s['beta_coef'] is None
    assert {'hypso_pos', 'relief', 'mrvbf', 'window_coverage'} <= set(df.columns)


# ---------------------------------------------------------------------------------------------------------------------
#  downscaler
# ---------------------------------------------------------------------------------------------------------------------
def test_downscaler_off_unchanged_and_correction_applied(tmp_path):
    case = syn.make_era5(start='2020-01-10', n_days=2)
    night = np.cos(2 * np.pi * case.tvec.hour.values / 24)  # +1 at midnight
    case.surf['t2m'] = case.surf.t2m - (4. + 3. * night)[:, None, None]  # inversion at night
    rows = [syn.make_point(case, name=n, dx=dx, elevation=e) for n, dx, e in
            [('001', 800., 1300.), ('002', -2000., 2100.), ('003', 3000., 900.)]]
    ds_solar = quiet(syn.solar_geometry, case, rows, tmp_path)
    hor = syn.horizon_da(rows, 0.)
    df = pd.DataFrame([vars(r) for r in rows])
    store = syn.era5_store(case, tmp_path)

    def run(surface, name):
        cd = tz.ClimateDownscaler(store, tmp_path / 'down', df, hor, ds_solar, syn.EPSG, case.start, case.end, '1h',
                                  store_name=name, surface_effect=surface)
        return quiet(cd.downscale).load()

    off = run(None, 'off.zarr')
    zero = run({'weight': np.zeros(3), 'diagnostics': True}, 'zero.zarr')
    for v in ['t', 'q', 'LW', 'SW', 'p']:
        np.testing.assert_allclose(zero[v], off[v], rtol=2e-6, err_msg=v)
    np.testing.assert_allclose(zero.t_pl, off.t)
    assert float(zero.dT_surf.mean()) == pytest.approx(-4., abs=0.05)

    w = np.array([0., 0.5, 1.])
    on = run({'weight': w, 'beta_coef': {'beta_bias': 0.5}, 'diagnostics': True}, 'on.zarr')
    np.testing.assert_allclose(on.t - on.t_pl, w[:, None] * on.dT_surf + 0.5, atol=1e-4)
    assert float(on.LW.sel(point_name='003').mean()) < float(off.LW.sel(point_name='003').mean())
    np.testing.assert_allclose(on.sel(point_name='001').LW, off.sel(point_name='001').LW, rtol=0.01)


# ---------------------------------------------------------------------------------------------------------------------
#  calibration
# ---------------------------------------------------------------------------------------------------------------------
def synthetic_stations(method, coef, n_st=15, n_days=40, noise=1., seed=0):
    rng = np.random.default_rng(seed)
    names = [f's{i:02d}' for i in range(n_st)]
    idx = pd.DataFrame({'hypso_pos': np.linspace(0.05, 0.95, n_st), 'relief': rng.uniform(100, 3000, n_st),
                        'mrvbf': rng.uniform(0, 6, n_st)}, index=names)
    time = pd.date_range('2021-01-01', periods=24 * n_days, freq='1h')
    rows = []
    for s in names:
        diurnal = -4 - 4 * np.cos(2 * np.pi * time.hour.values / 24)
        synoptic = np.repeat(rng.normal(0, 3, n_days), 24)
        dT = diurnal + synoptic + rng.normal(0, 1, time.size)
        e = np.zeros(time.size)
        for k in range(1, time.size):  # AR(1) noise
            e[k] = 0.8 * e[k - 1] + rng.normal(0, noise * 0.6)
        d = pd.DataFrame({'point_name': s, 'time': time, 'dT': dT, 't_pl': 270., 'dTd': dT, 'yd': np.nan})
        d['y'] = sc.predict(method, coef, d, idx) + e
        rows.append(d)
    return pd.concat(rows, ignore_index=True), idx


@pytest.mark.parametrize('method,truth', [
    ('dreamit', {'alpha_intercept': 0.2, 'alpha_slope': 0.6, 'beta_amp': 0., 't_star': 0., 'beta_bias': 0.}),
    ('redcapp', {'alpha': 0.5, 'beta': 0.8, 'gamma': 600., 'beta_amp': 0., 't_star': 0., 'beta_bias': 0.})])
def test_parameter_recovery_cross_validation_and_file(method, truth, tmp_path):
    data, idx = synthetic_stations(method, truth)
    res = sc.calibrate(method, data, {10.: idx}, n_boot=30)
    c, ci = res['coefficients'], res['intervals_95']
    for k in ['alpha_intercept', 'alpha_slope'] if method == 'dreamit' else ['alpha', 'beta']:
        assert c[k] == pytest.approx(truth[k], abs=0.08), k
        assert ci[k][0] - 0.05 <= truth[k] <= ci[k][1] + 0.05, (k, ci[k])
    assert res['cv_by'] == 'station' and res['cv'].rmse.max() < 2.
    f = sc.save_coefficients(res, tmp_path / 'coef.yml')
    loaded = se.load_coefficients(method, f)
    assert loaded['radius_km'] == 10. and loaded['beta_bias'] == 0.
    with pytest.raises(ValueError, match='coefficients of'):
        se.load_coefficients('redcapp' if method == 'dreamit' else 'dreamit', f)


def test_seasonal_beta_and_station_weights():
    truth = {'alpha_intercept': 0.3, 'alpha_slope': 0.5, 'beta_amp': 1., 't_star': 0.1, 'beta_bias': 0.7}
    data, idx = synthetic_stations('dreamit', truth, n_st=6, n_days=365, noise=0.5)
    c = sc.fit('dreamit', data, idx, fit_beta='seasonal')
    assert c['beta_bias'] == pytest.approx(0.7, abs=0.1) and c['beta_amp'] == pytest.approx(1., abs=0.1)
    assert c['t_star'] == pytest.approx(0.1, abs=0.03)
    st = sc.fit_stations(data)
    w = se.weight_dreamit(idx.hypso_pos, **truth)
    np.testing.assert_allclose(st.w.astype(float), w, atol=0.1)


def test_pair_with_hour_ending_observations():
    time = pd.date_range('2021-01-01', periods=6, freq='1h')
    ds = xr.Dataset({v: (('point_name', 'time'), np.full((1, 6), x)) for v, x in
                     [('dT_surf', -3.), ('dTd_surf', -2.), ('q', 3e-3), ('p', 8e4)]},
                    coords={'point_name': ['a'], 'time': time})
    ds['t_pl'] = (('point_name', 'time'), 270. + np.arange(6.)[None])
    obs = pd.DataFrame({'t': 268. + np.arange(6.), 'td': 260.}, index=time)
    d = sc.pair(ds, {'a': obs})
    assert len(d) == 5 and np.allclose(d.y, -2 + 0.5)  # model averaged over the hour ending at the stamp
