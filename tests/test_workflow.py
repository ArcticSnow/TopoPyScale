"""
Workflow helpers around the downscaling: DEM parameter file, cluster maps, FSM forcing and outputs.
"""
import contextlib
import io
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TopoPyScale import sim_fsm
from TopoPyScale import topo_export as te
from TopoPyScale import topo_param as tp


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


def write_tif(path, arr, dx=30.):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype='float32', crs='EPSG:32632', transform=from_origin(400000., 6710000., dx, dx)) as dst:
        dst.write(arr.astype('float32'), 1)


@pytest.fixture
def dem_project(tmp_path):
    ny, nx = 50, 70                          # not square: a transposed map would not even have the right shape
    i, j = np.mgrid[0:ny, 0:nx]
    rng = np.random.default_rng(1)
    dem = 2500 - 15 * np.hypot(i - 20, j - 45) + 40 * np.sin(j / 4.) + rng.normal(0, 5, (ny, nx))
    write_tif(tmp_path / 'dem.tif', dem)
    write_tif(tmp_path / 'mask.tif', (j < 50).astype(float))   # only the western part is clustered
    (tmp_path / 'outputs').mkdir()
    return tmp_path


def test_complete_ds_param_is_loaded_not_rewritten(dem_project):
    f = dem_project / 'outputs' / 'ds_param.nc'
    first = quiet(tp.compute_dem_param, dem_project / 'dem.tif', project_directory=dem_project)
    mtime = os.stat(f).st_mtime_ns
    again = quiet(tp.compute_dem_param, dem_project / 'dem.tif', project_directory=dem_project)
    assert os.stat(f).st_mtime_ns == mtime
    np.testing.assert_allclose(again.svf.transpose('y', 'x').values, first.svf.transpose('y', 'x').values, atol=1e-3)


def cluster(project, ds_param):
    pytest.importorskip('sklearn')
    from munch import DefaultMunch
    from TopoPyScale.topoclass import Topoclass
    config = DefaultMunch.fromDict({
        'project': {'directory': project},
        'sampling': {'toposub': {
            'clustering_mask': 'mask.tif', 'clustering_groups': None, 'n_clusters': 6, 'random_seed': 2,
            'clustering_method': 'kmean',
            'clustering_features': {'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1,
                                    'svf': 1}}},
        'outputs': {'path': project / 'outputs', 'file': {'ds_param': 'ds_param.nc'}}}, None)
    obj = SimpleNamespace(config=config, toposub=SimpleNamespace(ds_param=ds_param), plot=SimpleNamespace())
    quiet(Topoclass.extract_topo_cluster_param, obj)
    return obj.toposub


def test_cluster_maps_whatever_the_dimension_order(dem_project):
    """ds_param built from the DEM has dims (x, y), read from ds_param.nc (y, x): same mask, same cluster maps."""
    fresh = quiet(tp.compute_dem_param, dem_project / 'dem.tif', project_directory=dem_project)
    reloaded = quiet(tp.compute_dem_param, dem_project / 'dem.tif', project_directory=dem_project)
    maps = []
    for ds in (fresh, reloaded):
        sub = cluster(dem_project, ds)
        p = sub.ds_param.transpose('y', 'x')
        names = p.point_name.values
        inside = p.x.values[None, :] < p.x.values[50]
        assert (names[~np.broadcast_to(inside, names.shape)] == '-9999').all(), 'pixels outside the mask clustered'
        assert (names[np.broadcast_to(inside, names.shape)] != '-9999').all()
        # the centroid of each cluster is the mean of the pixels mapped to it
        cent = sub.df_centroids.set_index('point_name')
        for name, row in cent.iterrows():
            px = names == name
            assert px.any()
            assert abs(float(p.elevation.values[px].mean()) - row.elevation) < 1e-3 * abs(row.elevation)
            assert abs(float(p.x.values[np.nonzero(px)[1]].mean()) - row.x) < 1e-6 * abs(row.x)
        maps.append(names)
    np.testing.assert_array_equal(maps[0], maps[1])


def six_hourly_point(n=12):
    t = pd.date_range('2020-01-01', periods=n, freq='6h')
    ones = np.ones((1, n))
    return xr.Dataset({'t': (('point_name', 'time'), 268. * ones), 'p': (('point_name', 'time'), 8e4 * ones),
                       'q': (('point_name', 'time'), 2e-3 * ones), 'tp': (('point_name', 'time'), 3. * ones),
                       'SW': (('point_name', 'time'), 100. * ones), 'LW': (('point_name', 'time'), 250. * ones),
                       'ws': (('point_name', 'time'), 2. * ones)},
                      coords={'point_name': ['a'], 'time': t})


def test_fsm_forcing_time_step_and_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs').mkdir()
    quiet(te.to_fsm, six_hourly_point(), 'outputs/FSM_pt_*.txt')
    met = 'outputs/FSM_pt_0.txt'
    forcing = pd.read_csv(met, sep=r'\s+', header=None)
    np.testing.assert_allclose(forcing[6] + forcing[7], 3. / 21600)       # 3 mm per 6 h step, in kg m-2 s-1
    assert sim_fsm.infer_met_timestep(met) == 21600
    sim_fsm.fsm_nlst(31, met, 4, dt_seconds=21600)
    nlst = (tmp_path / 'fsm_sims' / 'nlst_FSM_pt_0.txt').read_text()
    assert '  dt = 21600\n' in nlst and 'Nave = 4' in nlst
    # FSM output: no header, year month day hour (float) and 6 variables
    out = tmp_path / 'fsm_sims' / 'sim_FSM_pt_0.txt'
    out.write_text('2020 1 1 23.000 0.8 0.0 0.12 30.0 268.1 271.0\n2020 1 2 23.000 0.79 0.0 0.15 35.0 267.9 271.0\n')
    df = sim_fsm.read_pt_fsm(str(out))
    assert list(df.columns) == ['albedo', 'runoff', 'snd', 'swe', 't_surface', 't_soil']
    assert df.index[1] == pd.Timestamp('2020-01-02 23:00') and df.snd.iloc[1] == 0.15


@pytest.mark.skipif(not os.environ.get('FSM_EXEC'), reason='set FSM_EXEC to an FSM executable to run FSM')
def test_fsm_runs_with_six_hourly_forcing(tmp_path, monkeypatch):
    """Same snow with the 6-hourly forcing as with the hourly one when both carry the same precipitation."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs').mkdir()
    exe = shutil.copy(os.environ['FSM_EXEC'], tmp_path / 'FSM')
    ds6 = six_hourly_point(4 * 20)
    ds1 = ds6.resample(time='1h').ffill().reindex(time=pd.date_range('2020-01-01', periods=24 * 20, freq='1h'),
                                                      method='ffill')
    ds1['tp'] = ds1.tp / 6
    snow = {}
    for tag, ds, nave in (('h6', ds6, 4), ('h1', ds1, 24)):
        quiet(te.to_fsm, ds, f'outputs/{tag}_pt_*.txt')
        quiet(sim_fsm.fsm_sim_parallel, fsm_input=f'outputs/{tag}_pt_*.txt', fsm_nave=nave, fsm_exec=str(exe), n_core=1)
        snow[tag] = sim_fsm.read_pt_fsm(f'fsm_sims/sim_{tag}_pt_0.txt').swe
    assert snow['h1'].iloc[-1] > 30                       # 12 mm per day for 20 days at -5 C: snow accumulates
    # daily means of the SWE: the 6-hourly steps add their snow in 4 lumps, hence an offset of ~1.5 mm
    np.testing.assert_allclose(snow["h6"].values, snow["h1"].values, atol=2.)
