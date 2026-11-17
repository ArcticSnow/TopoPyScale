"""
Horizons at the points along straight rays (topo_param.horizon_rays / compute_horizon_points) and their use
through the pipeline (topo_utils.horizon_at_points, Topoclass.compute_horizon).
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TopoPyScale import topo_param as tp
from TopoPyScale import topo_utils as tu

AZ = np.arange(-175, 180, 10.)
DX = 30.


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


def grid(ny=40, nx=50, north_up=True):
    x = 500000. + DX * np.arange(nx)
    y = 6700000. + DX * np.arange(ny)
    return x, (y[::-1] if north_up else y)


def fractal(ny=60, nx=80, seed=0):
    rng = np.random.default_rng(seed)
    kx, ky = np.fft.fftfreq(nx)[None, :], np.fft.fftfreq(ny)[:, None]
    k = np.hypot(kx, ky)
    k[0, 0] = 1
    spec = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / k ** 1.5
    spec[0, 0] = 0
    z = np.real(np.fft.ifft2(spec))
    return 500 + 2000 * (z - z.min()) / np.ptp(z)


def interior_points(x, y, n=40, margin=2, seed=1):
    rng = np.random.default_rng(seed)
    px = rng.uniform(min(x) + margin * DX, max(x) - margin * DX, n)
    py = rng.uniform(min(y) + margin * DX, max(y) - margin * DX, n)
    return px, py


@pytest.mark.parametrize('north_up', [True, False])
def test_tilted_plane_exact(north_up):
    x, y = grid(north_up=north_up)
    X, Y = np.meshgrid(x, y)
    ge, gn = 0.3, -0.2                               # elevation gain per metre towards east and north
    z = 1000 + ge * (X - x.min()) + gn * (Y - y.min())
    px, py = interior_points(x, y)
    hor = tp.horizon_rays(z, x, y, px, py, AZ)
    a = np.deg2rad(AZ)[:, None]
    gain = ge * np.sin(a) + gn * (-np.cos(a))        # 0 = south, 90 = east
    np.testing.assert_allclose(hor, np.arctan(np.maximum(gain, 0)) * np.ones_like(px), atol=1e-12)


@pytest.mark.parametrize('az', [90., 95., 85.])
def test_wall(az):
    x, y = grid()
    z = np.zeros((y.size, x.size))
    cw, height = 35, 300.
    z[:, cw] = height                                # north-south wall
    c, r = 20, 20                                    # pixel centre west of the wall
    hor = tp.horizon_rays(z, x, y, [x[c]], [y[r]], [az, -90.])
    dist = (cw - c) * DX / np.sin(np.deg2rad(az))
    np.testing.assert_allclose(hor[0, 0], np.arctan(height / dist), atol=1e-12)
    assert hor[1, 0] == 0                            # nothing higher to the west


def exhaustive_axis_horizon(z, az):
    """Horizon of every cell along rows or columns, by checking every cell ahead (north-up raster, square cells)."""
    step = {0.: (1, 0), 180.: (-1, 0), 90.: (0, 1), -90.: (0, -1)}[az]
    ny, nx = z.shape
    out = np.zeros(z.shape)
    for r in range(ny):
        for c in range(nx):
            k = np.arange(1, max(ny, nx))
            rr, cc = r + k * step[0], c + k * step[1]
            ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
            if ok.any():
                out[r, c] = max(0., np.arctan(np.max((z[rr[ok], cc[ok]] - z[r, c]) / (k[ok] * DX))))
    return out


def test_grid_axes_match_exhaustive_search():
    z = fractal(30, 40)
    x, y = grid(*z.shape)
    X, Y = np.meshgrid(x, y)
    for az in [0., 90., -90., 180.]:
        ref = exhaustive_axis_horizon(z, az)
        hr = tp.horizon_rays(z, x, y, X.ravel(), Y.ravel(), [az])[0].reshape(z.shape)
        np.testing.assert_allclose(hr, ref, atol=1e-12, err_msg=f'azimuth {az}')


def test_pruning_and_threads_do_not_change_result():
    z = fractal()
    x, y = grid(*z.shape)
    px, py = interior_points(x, y, n=200, margin=0)
    ref = tp.horizon_rays(z, x, y, px, py, AZ, prune=False)
    np.testing.assert_array_equal(tp.horizon_rays(z, x, y, px, py, AZ), ref)
    np.testing.assert_array_equal(tp.horizon_rays(z, x, y, px, py, AZ, n_threads=3, max_rays=500), ref)


def test_no_streaks_at_skewed_azimuths():
    """
    Lines of sight that step by whole rows (as topocalc's) leave streaks at azimuths a few degrees off the grid
    axes: at azimuth 95, the cells just before each step, every 1/tan(5 deg) ~ 11.4 columns, read their nearest
    terrain from the adjacent row. East-west corrugations on a slope rising east make that error large and the
    natural column-to-column change small. Straight rays and the line sweep show no jump at those columns.
    """
    ny, nx = 80, 160
    x, y = grid(ny, nx)
    X, Y = np.meshgrid(x, y)
    z = 1000 + 100 * np.sin(2 * np.pi * np.arange(ny)[:, None] / 7) + 0.3 * DX * np.arange(nx)[None, :] + 0 * X
    offset = (np.arange(nx) * np.tan(np.deg2rad(5)) + 0.5).astype(int)    # columns where staircases step
    step = np.diff(offset)[:-15] != 0                                     # away from the east edge

    def ratio(h):
        jump = np.abs(np.diff(h, axis=1))[:, :-15].mean(axis=0)
        return jump[step].mean() / jump[~step].mean()

    # a staircase line of sight (integer row shifts) shows the streaks on this terrain
    stair = np.zeros(z.shape)
    for r in range(ny):
        for c in range(nx):
            k = np.arange(1, nx - c)
            rows = r - (offset[c + k] - offset[c])
            ok = (rows >= 0) & (rows < ny)
            d = k * DX / np.cos(np.deg2rad(5))
            stair[r, c] = max(0., np.arctan(np.max((z[rows[ok], c + k[ok]] - z[r, c]) / d[ok]))) if ok.any() else 0.
    assert ratio(stair) > 5
    assert ratio(tp.horizon_rays(z, x, y, X.ravel(), Y.ravel(), [95.])[0].reshape(z.shape)) < 1.5
    assert ratio(tp.horizon_sweep(z, DX, -DX, 95.)) < 1.5


def test_no_data_and_outside_points():
    x, y = grid()
    z = np.zeros((y.size, x.size))
    z[:, 30:] = np.nan                               # no data east of the point
    z[5:8, 10] = 50.
    hor = tp.horizon_rays(z, x, y, [x[20]], [y[20]], [90., -90.])
    assert hor[0, 0] == 0 and np.isfinite(hor).all()
    with pytest.raises(ValueError, match='outside the DEM'):
        tp.horizon_rays(z, x, y, [x[0] - 2 * DX], [y[0]], [90.])


def test_compute_horizon_points_and_lookup(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    z = fractal()
    x, y = grid(*z.shape)
    with rasterio.open(tmp_path / 'dem.tif', 'w', driver='GTiff', height=z.shape[0], width=z.shape[1], count=1,
                       dtype='float64', crs='EPSG:32632',
                       transform=from_origin(x[0] - DX / 2, y[0] + DX / 2, DX, DX)) as dst:
        dst.write(z, 1)
    px, py = interior_points(x, y, n=5)
    df = pd.DataFrame({'point_name': ['a', 'b', 'c', 'd', 'e'], 'x': px, 'y': py})
    da = quiet(tp.compute_horizon_points, tmp_path / 'dem.tif', df, 10, 'h.nc', tmp_path)
    assert da.dims == ('azimuth', 'point_name') and (tmp_path / 'h.nc').is_file()
    np.testing.assert_allclose(da.values, tp.horizon_rays(z, x, y, px, py, AZ), atol=1e-12)

    # lookup follows the rows of the dataframe, whatever their order
    shuffled = df.iloc[[3, 0, 4, 1, 2]]
    hp = tu.horizon_at_points(xr.open_dataarray(tmp_path / 'h.nc'), shuffled)
    np.testing.assert_array_equal(hp.values, da.sel(point_name=shuffled.point_name.values).values)
    assert tu.horizon_matches_points(da, shuffled)

    moved = df.assign(x=df.x + 100)
    assert not tu.horizon_matches_points(da, moved)
    with pytest.raises(ValueError, match='other points'):
        tu.horizon_at_points(da, moved)

    # full-grid horizons (e.g. from compute_horizon) are not horizons of the points
    gridded = xr.DataArray(np.zeros((AZ.size, y.size, x.size)), dims=['azimuth', 'y', 'x'],
                           coords={'azimuth': AZ, 'y': y, 'x': x})
    assert not tu.horizon_matches_points(gridded, df)


def test_topoclass_compute_horizon(tmp_path, monkeypatch, capsys):
    from test_config import make_project
    from TopoPyScale import topoclass as tc
    project, fconfig = make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    mp = quiet(tc.Topoclass, str(fconfig))
    quiet(mp.compute_dem_param)
    quiet(mp.extract_topo_cluster_param)
    quiet(mp.compute_horizon)
    df = mp.toposub.df_centroids
    da = mp.da_horizon
    assert da.dims == ('azimuth', 'point_name') and da.point_name.size == len(df)
    for az in da.azimuth.values:
        np.testing.assert_array_equal(df['hori_azi_' + str(az)].values,
                                      da.sel(azimuth=az, point_name=df.point_name.values).values)
    stored = pd.read_pickle(mp.config.outputs.path / mp.config.outputs.file.df_centroids)
    assert all('hori_azi_' + str(az) in stored for az in da.azimuth.values)

    # an existing file for these points is loaded, not recomputed
    def fail(*args, **kwargs):
        raise AssertionError('recomputed')
    monkeypatch.setattr(tc.tp, 'compute_horizon_points', fail)
    quiet(mp.compute_horizon)

    # points that moved: recomputed
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    mp.toposub.df_centroids['x'] += 10.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mp.compute_horizon()
    assert 'recomputing' in out.getvalue()
    assert tu.horizon_matches_points(mp.da_horizon, mp.toposub.df_centroids)

    # full-grid file of an earlier version: recomputed at the points
    fname = mp.config.outputs.path / mp.config.outputs.file.da_horizon
    ds_dem = tp.open_dem(mp.config.dem.filepath)
    old = xr.DataArray(np.full((AZ.size,) + ds_dem.elevation.shape, 0.1), dims=['azimuth', 'y', 'x'], name='horizon',
                       coords={'azimuth': AZ, 'y': ds_dem.y.values, 'x': ds_dem.x.values})
    old.to_dataset().to_netcdf(fname)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mp.compute_horizon()
    assert 'recomputing' in out.getvalue()
    assert mp.da_horizon.dims == ('azimuth', 'point_name')
