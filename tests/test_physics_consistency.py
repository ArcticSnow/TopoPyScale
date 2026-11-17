"""
Physics-consistency tests for the TopoPyScale downscaling chain.

The pipeline tests run the real downscaling engine (topo_scale_zarr.py) on synthetic ERA5 fields with a known
answer (see synthetic_era5.py).

Tests marked `known_violation` document constraints the current code breaks. They are strict xfails: once the
underlying issue is fixed the test XPASSes, which fails the suite, as a reminder to remove the marker.

Run with:  pytest tests -q
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import synthetic_era5 as syn
from TopoPyScale import meteo_util as mu
from TopoPyScale import topo_checks as tc

def known_violation(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


def downscale(case, rows, workdir, horizon_deg=0., **kwargs):
    rows = rows if isinstance(rows, list) else [rows]
    workdir.mkdir(parents=True, exist_ok=True)
    ds_solar = quiet(syn.solar_geometry, case, rows, workdir)
    hor = syn.horizon_da(rows, horizon_deg)
    return quiet(syn.run_zarr, case, rows, ds_solar, hor, workdir, **kwargs), ds_solar


def statuses(rows):
    return {(r['check'], r['variable']): r['status'] for r in rows}


# ---------------------------------------------------------------------------------------------------------------------
#  synthetic cases
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def summer():
    tp = np.zeros(144)
    tp[30:40] = 0.8
    return syn.make_era5(start='2020-06-18', n_days=6, cloud=[1, .5, 1, .3, 1, .9], tp_mm=tp)


@pytest.fixture(scope='module')
def delta_tp():
    """1 mm/h on the central ERA5 node only: downscaled tp equals the interpolation weight of that node."""
    w = np.zeros((3, 3))
    w[1, 1] = 1.
    return syn.make_era5(start='2020-06-18', n_days=2, tp_mm=np.ones(48), tp_weights=w)


@pytest.fixture(scope='module')
def winter():
    return syn.make_era5(start='2020-01-15', n_days=3, T0_mean=268., tp_mm=np.full(72, 0.5))


@pytest.fixture(scope='module')
def cold_dry():
    return syn.make_era5(start='2020-01-15', n_days=2, T0_mean=245., T0_amp=2., rh=0.6)


@pytest.fixture(scope='module')
def saturated():
    return syn.make_era5(start='2020-06-18', n_days=2, rh=1.0)


# ---------------------------------------------------------------------------------------------------------------------
#  1. meteo_util functions
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['continuous', 'jennings2018_bivariate', 'jennings2018_trivariate'])
def test_partition_snow_closure_and_bounds(method):
    T = np.linspace(263.15, 283.15, 401)
    P = np.random.default_rng(0).uniform(0, 5, T.size)
    rain, snow = mu.partition_snow(P, T, np.full(T.size, 80.), np.full(T.size, 8e4), method=method)
    np.testing.assert_allclose(rain + snow, P, atol=1e-12)
    assert (snow >= -1e-12).all() and (rain >= -1e-12).all()


def test_partition_snow_continuous_monotonic():
    report = statuses(tc.check_snow_partition('continuous'))
    assert report[('partition_monotonic', 'continuous')] == 'pass'


@pytest.mark.parametrize('method', ['jennings2018_bivariate', 'jennings2018_trivariate'])
def test_jennings_partition_warm_and_cold(method):
    report = statuses(tc.check_snow_partition(method))
    assert report[('partition_warm_humid', method)] == 'pass'
    assert report[('partition_cold_humid', method)] == 'pass'


@pytest.mark.parametrize('method', ['jennings2018_bivariate', 'jennings2018_trivariate'])
def test_fsm_export_passes_rh_in_percent_to_jennings(method, tmp_path):
    """+5 degC and RH 95 %: Jennings gives < 1 % snow with RH in %, ~95 % snow if RH is passed as a fraction."""
    from TopoPyScale import topo_export as te
    n = 2000
    t = np.full(n, 278.15)
    p = np.full(n, 8e4)
    e = 0.95 * tc.esat_water(t)
    q = tc.EPS * e / (p - (1 - tc.EPS) * e)
    dims = ('point_name', 'time')
    ds = xr.Dataset({'t': (dims, t[None]), 'p': (dims, p[None]), 'q': (dims, q[None]), 'tp': (dims, np.ones((1, n))),
                     'SW': (dims, np.zeros((1, n))), 'LW': (dims, np.full((1, n), 300.)),
                     'ws': (dims, np.full((1, n), 2.))},
                    coords={'point_name': ['001'], 'time': pd.date_range('2020-01-01', periods=n, freq='1h')})
    quiet(te.to_fsm, ds, fname_format=str(tmp_path / 'FSM_pt_*.txt'), snow_partition_method=method)
    out = pd.read_csv(tmp_path / 'FSM_pt_0.txt', sep=r'\s+', header=None)
    assert out[6].sum() / (out[6].sum() + out[7].sum()) < 0.1


def test_q_2_rh_roundtrip():
    T, p = np.meshgrid(np.linspace(230, 310, 50), np.linspace(5e4, 1.05e5, 20))
    rh = np.full(T.shape, 0.63)
    e = rh * tc.esat_water(T)
    q = tc.EPS * e / (p - (1 - tc.EPS) * e)
    np.testing.assert_allclose(mu.q_2_rh(T.copy(), p, q), rh, rtol=1e-4)  # q_2_rh uses eps = 0.62197


def test_mixing_ratio_and_vapor_pressure_accuracy():
    q = np.linspace(1e-6, 0.03, 200)
    ds = xr.Dataset({'q': ('n', q), 'p': ('n', np.full(q.size, 8e4))})
    ds = mu.vapor_pressure(mu.mixing_ratio(ds, mu.var_era_plevel), mu.var_era_plevel)
    np.testing.assert_allclose(ds.w, q / (1 - q), rtol=5e-3)
    np.testing.assert_allclose(ds.vp, q * 8e4 / (tc.EPS + (1 - tc.EPS) * q), rtol=5e-3)


# ---------------------------------------------------------------------------------------------------------------------
#  2. downscaling: identity, interpolation, vertical structure
# ---------------------------------------------------------------------------------------------------------------------
def test_null_downscaling_reproduces_era5(summer, tmp_path):
    """Flat, unobstructed point at the ERA5 node and orography: output must equal the ERA5 surface forcing."""
    row = syn.make_point(summer, elevation=summer.z_surf)
    ds, _ = downscale(summer, row, tmp_path)
    ds = ds.isel(point_name=0)
    era = summer.surf.isel(latitude=1, longitude=1)
    np.testing.assert_allclose(ds.p, era.sp, rtol=1e-3)
    np.testing.assert_allclose(ds.t, era.t2m, atol=0.05)
    np.testing.assert_allclose(ds.tp, era.tp * 1000, atol=1e-4)
    np.testing.assert_allclose(ds.LW, era.strd / 3600, rtol=0.01)
    sw_day = ds.SW.resample(time='1D').sum()
    era_day = (era.ssrd / 3600).resample(time='1D').sum()
    np.testing.assert_allclose(sw_day, era_day, rtol=1e-3)  # daily shortwave energy conserved


@pytest.mark.parametrize('method', ['idw', 'linear'])
def test_point_on_era5_node_is_finite(summer, method, tmp_path):
    row = syn.make_point(summer, dx=0., dy=0.)
    ds, _ = downscale(summer, row, tmp_path, interp_method=method)
    for var in ['t', 'q', 'p', 'tp', 'SW', 'LW']:
        assert np.isfinite(ds[var]).all(), var


@pytest.mark.parametrize('method', ['idw', 'linear'])
def test_nearest_era5_node_dominates(delta_tp, method, tmp_path):
    row = syn.make_point(delta_tp, dx=500.)
    ds, _ = downscale(delta_tp, row, tmp_path, interp_method=method)
    assert float(ds.tp.mean()) > 0.9  # weight of the central node, in mm/h


def test_bilinear_weights_reproduce_linear_field():
    from types import SimpleNamespace
    from TopoPyScale.toposcale_core import horizontal_weights
    lats, lons = np.array([60.75, 60.5, 60.25]), np.array([7.25, 7.5, 7.75])
    field = 3 * lats[:, None] - 2 * lons[None, :]
    for lat, lon in [(60.5, 7.5), (60.61, 7.43), (60.3, 7.7)]:
        w = horizontal_weights(SimpleNamespace(lat=lat, lon=lon, x=0, y=0), lats, lons, None, None, method='linear')
        assert w.min() >= 0 and np.isclose(w.sum(), 1)
        assert np.isclose((w * field).sum(), 3 * lat - 2 * lon)


def test_precip_lapse_factor_symmetric_and_bounded():
    from TopoPyScale.toposcale_core import precip_lapse_factor
    dz = np.linspace(-6000, 6000, 241)
    f = precip_lapse_factor(dz, np.full(dz.size, 1), max_factor=5.)
    assert np.all(np.isfinite(f)) and f.min() >= 1 / 5 - 1e-12 and f.max() <= 5 + 1e-12
    assert np.all(np.diff(f) >= 0)  # more precipitation with elevation
    np.testing.assert_allclose(f * f[::-1], 1)  # f(-dz) = 1 / f(dz)


def test_vertical_profile_temperature_and_pressure(summer, tmp_path):
    z = summer.z_surf + 1500.
    row = syn.make_point(summer, elevation=z)
    ds, _ = downscale(summer, row, tmp_path)
    ds = ds.isel(point_name=0)
    np.testing.assert_allclose(ds.t, summer.T0 - summer.gamma * z, atol=0.05)
    np.testing.assert_allclose(ds.p, syn.p_of_z(summer.T0, z, summer.gamma, summer.p0), rtol=1e-3)


def test_no_supersaturation_from_vertical_interpolation(saturated, tmp_path):
    rows = [syn.make_point(saturated, name=f'00{i}', elevation=saturated.z_surf + dz)
            for i, dz in enumerate([300., 900., 1700.], start=1)]
    ds, _ = downscale(saturated, rows, tmp_path)
    report = statuses(tc.check_humidity(tc._prepare(ds)))
    assert report[('supersaturation', 'q')] == 'pass'


def test_specific_humidity_precision_in_cold_air(cold_dry, tmp_path):
    z = cold_dry.z_surf + 1000.
    row = syn.make_point(cold_dry, elevation=z)
    ds, _ = downscale(cold_dry, row, tmp_path)
    lev = cold_dry.plev.isel(latitude=1, longitude=1)
    # what the method intends: linear interpolation in z of the pressure-level q
    q_lin = [np.interp(z, zl[::-1], ql[::-1]) for zl, ql in zip(lev.z.values, lev.q.values)]
    np.testing.assert_allclose(ds.q.isel(point_name=0), q_lin, rtol=0.01)


def test_precip_lapse_factor_bounded(winter, tmp_path):
    row = syn.make_point(winter, elevation=winter.z_surf + 3000.)
    ds, _ = downscale(winter, row, tmp_path, precip_lapse=True)
    report = statuses(tc.check_precipitation(tc._prepare(ds), max_factor=10.))
    assert report[('tp_non_negative', 'tp')] == 'pass'
    assert report[('precip_factor_bounded', 'precip_lapse_rate')] == 'pass'


# ---------------------------------------------------------------------------------------------------------------------
#  3. radiation
# ---------------------------------------------------------------------------------------------------------------------
def test_direct_beam_transmissivity_increases_with_altitude(summer, tmp_path):
    rows = [syn.make_point(summer, name='001', elevation=summer.z_surf),
            syn.make_point(summer, name='002', elevation=summer.z_surf + 1500.)]
    ds, _ = downscale(summer, rows, tmp_path)
    low, high = ds.SW_direct.isel(point_name=0).sum(), ds.SW_direct.isel(point_name=1).sum()
    assert float(high / low) > 1.02


def test_shortwave_closure_beam_bound_and_night(summer, tmp_path):
    rows = [syn.make_point(summer, name='001', elevation=1500., slope_deg=40., aspect_deg=0., svf=0.85),
            syn.make_point(summer, name='002', elevation=1500., slope_deg=40., aspect_deg=90., svf=0.85),
            syn.make_point(summer, name='003', elevation=1500., slope_deg=60., aspect_deg=-120., svf=0.7)]
    ds, ds_solar = downscale(summer, rows, tmp_path)
    df = syn.centroids_frame(rows)
    ds = tc._prepare(ds)
    report = statuses(tc.check_shortwave(ds, tc._align_solar(ds, ds_solar), tc._terrain(df, ds.point_name.values)))
    for key in [('sw_closure', 'SW'), ('direct_with_sun_down', 'SW_direct'), ('sw_at_night', 'SW'),
                ('direct_beam_bound', 'SW_direct'), ('direct_in_terrain_shadow', 'SW_direct')]:
        assert report[key] == 'pass', key


def test_no_direct_beam_behind_horizon(summer, tmp_path):
    row = syn.make_point(summer, elevation=1500.)
    ds, ds_solar = downscale(summer, row, tmp_path, horizon_deg=25.)
    elev = np.rad2deg(ds_solar.elevation.sel(point_name=row.point_name).values)
    sw_dir = ds.SW_direct.isel(point_name=0).values
    assert (sw_dir[(elev > 0) & (elev < 25.)] < 1.).all()
    assert (sw_dir[elev > 30.] > 1.).any()  # not vacuous: the sun is above the horizon at noon


def test_shortwave_timing_matches_solar_geometry(summer, tmp_path):
    row = syn.make_point(summer, elevation=summer.z_surf)
    ds, ds_solar = downscale(summer, row, tmp_path)
    df = syn.centroids_frame([row])
    ds = tc._prepare(ds)
    rows = tc.check_shortwave_timing(ds, tc._align_solar(ds, ds_solar), tc._terrain(df, ds.point_name.values),
                                     min_days=3)
    assert rows[0]['status'] == 'pass', rows[0]['worst']


def test_longwave_emissivity_bounds(summer, tmp_path):
    rows = [syn.make_point(summer, name='001', elevation=1500., slope_deg=35., svf=0.75)]
    ds, _ = downscale(summer, rows, tmp_path, lw_terrain=True)
    report = statuses(tc.check_longwave(tc._prepare(ds), tc._terrain(syn.centroids_frame(rows), ['001'])))
    assert report[('lw_emissivity_upper', 'LW')] == 'pass'
    assert report[('lw_emissivity_lower', 'LW')] == 'pass'


# ---------------------------------------------------------------------------------------------------------------------
#  4. metadata, full report
# ---------------------------------------------------------------------------------------------------------------------
def test_units_attributes_match_values(summer, tmp_path):
    ds, _ = downscale(summer, syn.make_point(summer), tmp_path)
    report = tc.check_unit_attributes(tc._prepare(ds))
    assert all(r['status'] == 'pass' for r in report), [r for r in report if r['status'] != 'pass']


KNOWN_FAILING_CHECKS = set()


def test_full_report_has_no_unexpected_failures(summer, tmp_path):
    rows = [syn.make_point(summer, name='001', elevation=1200.),
            syn.make_point(summer, name='002', elevation=1900., slope_deg=30., aspect_deg=0., svf=0.9),
            syn.make_point(summer, name='003', elevation=2600., slope_deg=35., aspect_deg=-90., svf=0.8)]
    ds, ds_solar = downscale(summer, rows, tmp_path)
    report = tc.run_all_checks(ds, ds_solar=ds_solar, df_centroids=syn.centroids_frame(rows),
                               snow_partition_method=None)
    print(tc.summarize(report))
    failing = set(report.loc[report.status.isin(['error', 'warning']), 'check'])
    assert failing <= KNOWN_FAILING_CHECKS, failing - KNOWN_FAILING_CHECKS


# ---------------------------------------------------------------------------------------------------------------------
#  5. clustering and DEM conventions
# ---------------------------------------------------------------------------------------------------------------------
def test_aspect_and_horizon_conventions_match_solar_azimuth():
    """compute_dem_param calls gradient_d8 with dy < 0 (north-up raster): aspect must be 0 = south, +90 = east,
    the convention of solar_geom.py azimuth (180 - pvlib azimuth). Horizons must use the same azimuths."""
    from TopoPyScale import topo_param as tp
    dx, n = 30., 60
    i, j = np.mgrid[0:n, 0:n]  # i increases southward in a north-up raster
    s = np.tan(np.deg2rad(30))
    planes = {0.: 2000 - i * dx * s, 90.: 2000 - j * dx * s, -90.: 2000 + j * dx * s, 180.: 2000 + i * dx * s}
    for expected, dem in planes.items():
        _, asp = tp.gradient_d8(dem, dx, -dx)
        got = np.angle(np.exp(1j * np.deg2rad(np.median(asp))), deg=True)
        assert abs(np.angle(np.exp(1j * np.deg2rad(got - expected)), deg=True)) < 1, (expected, got)
    dem = np.full((n, n), 1000.)
    dem[45:, :] = 1600.  # ridge to the south
    x, y = dx * np.arange(n), -dx * np.arange(n)
    rays = tp.horizon_rays(dem, x, y, [x[30]], [y[30]], [0., 180.])[:, 0]
    sweep = [tp.horizon_sweep(dem, dx, -dx, a)[30, 30] for a in [0., 180.]]
    for h in [rays, sweep]:
        assert h[0] > 0.5 > h[1]


def test_cluster_centroid_aspect_is_circular_mean(tmp_path):
    pytest.importorskip('sklearn')
    import rasterio
    from munch import DefaultMunch
    from rasterio.transform import from_origin
    from types import SimpleNamespace
    from TopoPyScale import topo_param as tp
    from TopoPyScale.topoclass import Topoclass

    # cone-shaped mountain: every aspect is represented, south-facing pixels straddle aspect = 0 = 2pi
    n, dx = 121, 30.
    i, j = np.mgrid[0:n, 0:n]
    dem = (3000 - 0.6 * dx * np.hypot(i - n // 2, j - n // 2)).astype('float32')
    dem_file = tmp_path / 'dem.tif'
    with rasterio.open(dem_file, 'w', driver='GTiff', height=n, width=n, count=1, dtype='float32',
                       crs=f'EPSG:{syn.EPSG}', transform=from_origin(400000, 6710000, dx, dx)) as dst:
        dst.write(dem, 1)
    (tmp_path / 'outputs').mkdir()
    ds_param = quiet(tp.compute_dem_param, dem_file, project_directory=tmp_path)

    config = DefaultMunch.fromDict({'sampling': {'toposub': {
        'clustering_mask': None, 'clustering_groups': None, 'n_clusters': 8, 'random_seed': 2,
        'clustering_method': 'kmean',
        'clustering_features': {'x': 1, 'y': 1, 'elevation': 1, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1,
                                'svf': 1}}},
        'outputs': {'path': tmp_path / 'outputs', 'file': {'ds_param': 'ds_param.nc'}}})
    obj = SimpleNamespace(config=config, toposub=SimpleNamespace(ds_param=ds_param), plot=SimpleNamespace())
    quiet(Topoclass.extract_topo_cluster_param, obj)

    report = statuses(tc.check_centroid_aspect(obj.toposub.df_centroids))
    assert report[('centroid_aspect', 'aspect')] == 'pass'


def test_centroid_aspect_check_tolerates_dispersed_minibatch_clusters():
    """A spread cluster (short centre vector, mini-batch k-means) may differ by ~20 deg; the mean bug gives ~180."""
    def df(dev_deg, length):
        a = np.deg2rad(np.array([10., 10.]))
        return pd.DataFrame({'aspect': np.mod(a + np.deg2rad([0., dev_deg]), 2 * np.pi), 'slope': np.deg2rad([20., 20.]),
                             'aspect_cos': length * np.cos(a), 'aspect_sin': length * np.sin(a)})
    assert statuses(tc.check_centroid_aspect(df(20., 0.6)))[('centroid_aspect', 'aspect')] == 'pass'
    assert statuses(tc.check_centroid_aspect(df(20., 0.95)))[('centroid_aspect', 'aspect')] == 'error'
    assert statuses(tc.check_centroid_aspect(df(170., 0.6)))[('centroid_aspect', 'aspect')] == 'error'


# ---------------------------------------------------------------------------------------------------------------------
#  6. exports
# ---------------------------------------------------------------------------------------------------------------------
def _three_hourly(summer, tmp_path):
    ds, _ = downscale(summer, syn.make_point(summer), tmp_path)
    ds = ds.isel(time=slice(0, None, 3))
    ds['tp'] = ds.tp * 3  # what the downscaling writes for 3h data: mm per 3 h step
    ds['tp'] = ds.tp.where(ds.tp > 0, 0) + 0.3  # make sure precipitation occurs
    return ds


def test_fsm_export_conserves_precipitation_3h(summer, tmp_path):
    from TopoPyScale import topo_export as te
    ds = _three_hourly(summer, tmp_path)
    quiet(te.to_fsm, ds, fname_format=str(tmp_path / 'FSM_pt_*.txt'))
    out = pd.read_csv(tmp_path / 'FSM_pt_0.txt', sep=r'\s+', header=None)
    sf, rf = out[6].values, out[7].values  # kg m-2 s-1
    np.testing.assert_allclose((sf + rf).sum() * 3 * 3600, float(ds.tp.sum()), rtol=1e-3)


def test_snowpack_export_wind_direction_in_degrees(summer, tmp_path):
    from TopoPyScale import topo_export as te
    row = syn.make_point(summer)
    ds, _ = downscale(summer, row, tmp_path)
    quiet(te.to_snowpack, ds, fname_format=str(tmp_path / 'smet_pt_*.txt'))
    out = pd.read_csv(next(tmp_path.glob('smet_pt_*.txt')), sep=' ', header=None)
    assert pd.to_datetime(out[0], format='%Y-%m-%dT%H:%M:%S').notna().all()  # ISO time stamps, one column
    dw = out[4].values
    expected = np.rad2deg(ds.wd.isel(point_name=0).values)
    np.testing.assert_allclose(dw, expected, atol=0.5)


def test_topoclass_check_physics_writes_report(summer, tmp_path):
    from types import SimpleNamespace
    from munch import DefaultMunch
    from TopoPyScale.topoclass import Topoclass
    rows = [syn.make_point(summer, name='001', elevation=1200.),
            syn.make_point(summer, name='002', elevation=2000., slope_deg=30., svf=0.9)]
    ds, ds_solar = downscale(summer, rows, tmp_path)
    ds_param = xr.Dataset({'point_name': (('y', 'x'), np.array([['001', '001'], ['002', '002']]))},
                          coords={'y': [10., 0.], 'x': [0., 10.]})
    obj = SimpleNamespace(downscaled_pts=ds, ds_solar=ds_solar,
                          toposub=SimpleNamespace(df_centroids=syn.centroids_frame(rows), ds_param=ds_param),
                          config=DefaultMunch.fromDict({'outputs': {'path': tmp_path}}))
    report = quiet(Topoclass.check_physics, obj)
    assert (tmp_path / 'physics_checks.csv').is_file()
    assert {'dem_orientation', 'domain_precip_budget', 'centroid_aspect'} <= set(report.check)
    obj.downscaled_pts = ds.assign(tp=-ds.tp - 1)  # inject a violation
    with pytest.raises(tc.PhysicsConsistencyError):
        quiet(Topoclass.check_physics, obj, strict=True)
