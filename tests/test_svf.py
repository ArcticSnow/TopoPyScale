"""
Slope, aspect, horizons of every cell (line sweep) and sky view factor (topo_param), without topocalc.
"""
import contextlib
import io

import numpy as np
import pytest

from TopoPyScale import topo_param as tp

DX = 30.


def fractal(ny=50, nx=70, seed=0, relief=600.):
    rng = np.random.default_rng(seed)
    kx, ky = np.fft.fftfreq(nx)[None, :], np.fft.fftfreq(ny)[:, None]
    k = np.hypot(kx, ky)
    k[0, 0] = 1
    spec = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / k ** 1.5
    spec[0, 0] = 0
    z = np.real(np.fft.ifft2(spec))
    return 1000 + relief * (z - z.min()) / np.ptp(z)


def coords(z, north_up=True):
    ny, nx = z.shape
    x = 500000. + DX * np.arange(nx)
    y = 6700000. + DX * np.arange(ny)
    return x, (y[::-1] if north_up else y)


def svf_from_rays(z, n_angles=72):
    """Sky view factor integrated from exact straight rays at every cell (reference)."""
    x, y = coords(z)
    X, Y = np.meshgrid(x, y)
    angles = np.linspace(-180, 180, n_angles, endpoint=False)
    hor = tp.horizon_rays(z, x, y, X.ravel(), Y.ravel(), angles)
    slope = tp.gradient_d8(z, DX, DX)[0]
    aspect = tp._aspect_south(z, DX, -DX)
    terms = []
    for h, a in zip(hor, angles):
        hcos = np.sin(h.reshape(z.shape))
        sin2 = (1 - hcos) * (1 + hcos)
        v = np.cos(slope) * sin2 + np.sin(slope) * np.cos(np.deg2rad(a) - aspect) * (np.arccos(hcos) - np.sqrt(sin2) * hcos)
        terms.append(np.where(v > 0, v, 0))
    return np.mean(terms, axis=0)


def test_gradient_of_a_plane():
    ny, nx = 20, 30
    i, j = np.mgrid[0:ny, 0:nx]
    gx, gy = 0.2, -0.1                                   # gain per metre towards east and north
    z = 1000 + gx * DX * j + gy * (-DX) * i              # north-up: y decreases with row
    slope, aspect = tp.gradient_d8(z, DX, -DX)
    np.testing.assert_allclose(slope, np.arctan(np.hypot(gx, gy)), atol=1e-12)
    # facing the steepest descent (-gx east, -gy north): west-north-west, aspect from south positive east
    expected = np.rad2deg(np.arctan2(-gx, gy))
    np.testing.assert_allclose(np.angle(np.exp(1j * np.deg2rad(aspect - expected)), deg=True), 0, atol=1e-9)
    np.testing.assert_allclose(tp._aspect_south(z, DX, -DX), np.deg2rad(expected), atol=1e-12)


def test_sweep_exact_along_axes_and_diagonals():
    z = fractal()
    x, y = coords(z)
    X, Y = np.meshgrid(x, y)
    for az in np.arange(-180, 180, 45.):
        rays = tp.horizon_rays(z, x, y, X.ravel(), Y.ravel(), [az])[0].reshape(z.shape)
        np.testing.assert_allclose(tp.horizon_sweep(z, DX, -DX, az), rays, atol=1e-12, err_msg=f'azimuth {az}')


def test_sweep_close_to_rays_elsewhere():
    z = fractal()
    x, y = coords(z)
    X, Y = np.meshgrid(x, y)
    for az in [5., 30., 95., -112., 160.]:
        rays = tp.horizon_rays(z, x, y, X.ravel(), Y.ravel(), [az])[0].reshape(z.shape)
        d = np.rad2deg(np.abs(tp.horizon_sweep(z, DX, -DX, az) - rays))
        # exact within `near` cells; beyond, the two lines around the ray stand in for it
        assert np.median(d) < 1e-3 and np.percentile(d, 95) < 0.05 and d.max() < 5, (az, np.percentile(d, 95), d.max())
        far_only = np.rad2deg(np.abs(tp.horizon_sweep(z, DX, -DX, az, near=0) - rays))
        assert np.percentile(far_only, 95) > 10 * np.percentile(d, 95)


def test_sweep_independent_of_raster_orientation():
    z = fractal()
    for az in [0., 37., 95., -150.]:
        np.testing.assert_allclose(tp.horizon_sweep(z[::-1], DX, DX, az)[::-1], tp.horizon_sweep(z, DX, -DX, az),
                                   atol=1e-12)


def test_svf_flat_and_tilted_plane():
    np.testing.assert_allclose(tp.sky_view_factor(np.full((15, 20), 500.), DX, -DX), 1., atol=1e-12)
    ny, nx = 40, 40
    i, j = np.mgrid[0:ny, 0:nx]
    for s_deg, (gx, gy) in [(20., (1, 0)), (35., (0.6, -0.8)), (10., (-0.7071, 0.7071))]:
        g = np.tan(np.deg2rad(s_deg))
        z = 1000 + g * DX * (gx * j - gy * i)
        svf = tp.sky_view_factor(z, DX, -DX)
        # away from the upslope edge, the plane is its own horizon: svf = (1 + cos S) / 2
        np.testing.assert_allclose(svf[5:-5, 5:-5], (1 + np.cos(np.deg2rad(s_deg))) / 2, atol=2e-3)


def test_svf_matches_exact_rays():
    z = fractal(40, 55, seed=2)
    ref = svf_from_rays(z)
    d = np.abs(tp.sky_view_factor(z, DX, -DX) - ref)
    assert np.median(d) < 1e-4 and d.max() < 5e-3, (np.median(d), d.max())


def test_svf_processes_orientation_and_no_data():
    z = fractal(30, 40, seed=5)
    one = tp.sky_view_factor(z, DX, -DX, n_angles=24)
    np.testing.assert_array_equal(tp.sky_view_factor(z, DX, -DX, n_angles=24, n_processes=2), one)
    np.testing.assert_allclose(tp.sky_view_factor(z[::-1], DX, DX, n_angles=24)[::-1], one, atol=1e-12)
    zn = z.copy()
    zn[10:14, 20:25] = np.nan
    with np.errstate(invalid='ignore'):
        svf = tp.sky_view_factor(zn, DX, -DX, n_angles=24)
        hor = tp.horizon_sweep(zn, DX, -DX, 90.)
    assert np.isnan(svf[9:15, 19:26]).all() and np.isnan(hor[10:14, 20:25]).all()   # svf: no slope next to no data
    valid = np.isfinite(svf)
    assert valid.sum() > 0.9 * z.size and np.isfinite(hor[valid]).all()
    assert (svf[valid] <= 1 + 1e-12).all() and (svf[valid] > 0).all()
    # no data hides nothing: the cells just west of the gap see past it as they see past a pit
    assert (hor[10:14, 15:20] <= tp.horizon_sweep(z, DX, -DX, 90.)[10:14, 15:20] + 1e-12).all()


def test_compute_dem_param_and_horizon_maps(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    z = fractal(30, 40, seed=7)
    with rasterio.open(tmp_path / 'dem.tif', 'w', driver='GTiff', height=z.shape[0], width=z.shape[1], count=1,
                       dtype='float32', crs='EPSG:32632', transform=from_origin(500000., 6700000., DX, DX)) as dst:
        dst.write(z.astype('float32'), 1)
    (tmp_path / 'outputs').mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        ds = tp.compute_dem_param(tmp_path / 'dem.tif', project_directory=tmp_path, n_processes=2)
        maps = tp.compute_horizon(tmp_path / 'dem.tif', azimuth_inc=90, fname='h.nc', output_directory=tmp_path)
    zf = z.astype('float32').astype(float)
    svf = tp.sky_view_factor(zf, DX, -DX)
    # ds_param is stored packed on 10 bits by topo_export.to_netcdf: svf within half a quantization step
    assert np.abs(ds.svf.values - svf).max() < 1e-3
    assert maps.dims == ('azimuth', 'y', 'x') and (tmp_path / 'h.nc').is_file()
    np.testing.assert_allclose(maps.sel(azimuth=45.).values, tp.horizon_sweep(zf, DX, -DX, 45.), atol=1e-6)
