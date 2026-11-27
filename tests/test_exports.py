"""
Exports of the downscaled points (topo_export): one file per point, from blocks of points read once and written by
one or several processes. The files must not depend on the number of processes, and hold the values of the store.
"""
import contextlib
import filecmp
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from munch import DefaultMunch
from zarr.codecs import BloscCodec

from TopoPyScale import meteo_util as mu
from TopoPyScale import topo_export as te
from TopoPyScale import topo_utils as tu

N_PTS, N_T = 25, 72


@pytest.fixture(scope='module')
def store(tmp_path_factory):
    """Downscaled store with the layout of topo_scale_zarr: chunks of 10 points, string point names."""
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp('down') / 'down.zarr'
    time = pd.date_range('2020-03-01', periods=N_T, freq='1h')
    hours = np.arange(N_T)
    elev = rng.uniform(1000, 3000, N_PTS)
    sw = np.clip(800 * np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0, None)[None] * rng.uniform(0.3, 1, (N_PTS, 1))
    data = {'t': 270 + 5 * np.sin(2 * np.pi * hours / 24)[None] + rng.normal(0, 2, (N_PTS, N_T)),
            'p': 101325 * np.exp(-elev / 8000)[:, None] + rng.normal(0, 100, (N_PTS, N_T)),
            'q': np.clip(0.003 + 0.001 * rng.normal(size=(N_PTS, N_T)), 2e-4, None),
            'u': rng.normal(0, 3, (N_PTS, N_T)), 'v': rng.normal(0, 3, (N_PTS, N_T)),
            'SW': sw, 'SW_direct': 0.7 * sw, 'SW_diffuse': 0.3 * sw, 'LW': 250 + rng.normal(0, 20, (N_PTS, N_T)),
            'tp': np.clip(rng.gamma(0.3, 1.0, (N_PTS, N_T)) - 0.2, 0, None)}
    data['ws'] = np.hypot(data['u'], data['v'])
    data['wd'] = np.mod(np.arctan2(-data['u'], -data['v']), 2 * np.pi)
    ds = xr.Dataset({k: (('point_name', 'time'), v.astype('float32')) for k, v in data.items()},
                    coords={'point_name': [f'{i:03d}' for i in range(N_PTS)], 'time': time,
                            'elevation': ('point_name', elev), 'x': ('point_name', rng.uniform(0, 1e4, N_PTS)),
                            'y': ('point_name', rng.uniform(0, 1e4, N_PTS)),
                            'lat': ('point_name', rng.uniform(46, 47, N_PTS)),
                            'lon': ('point_name', rng.uniform(9, 10, N_PTS)),
                            'slope': ('point_name', rng.uniform(0, 0.7, N_PTS)),
                            'aspect': ('point_name', rng.uniform(0, 6.28, N_PTS)),
                            'svf': ('point_name', rng.uniform(0.7, 1, N_PTS))})
    comp = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)
    ds.to_zarr(path, zarr_format=3,
               encoding={v: {'compressors': comp, 'chunks': (10, N_T)} for v in data})
    return path


@pytest.fixture
def ds(store):
    return tu.open_downscaled_store(store)


def df_pts(ds):
    return pd.DataFrame({'x': ds.x.values, 'y': ds.y.values, 'elevation': ds.elevation.values,
                         'slope': ds.slope.values, 'aspect': ds.aspect.values, 'svf': ds.svf.values,
                         'latitude': ds.lat.values, 'longitude': ds.lon.values, 'lat': ds.lat.values,
                         'lon': ds.lon.values})


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


EXPORTS = {
    'fsm': lambda ds, d, n: te.to_fsm(ds, d / 'FSM_pt_*.txt', n_jobs=n),
    'tc': lambda ds, d, n: te.to_TC(ds, d / 'TC_*.csv', n_jobs=n),
    'micromet': lambda ds, d, n: te.to_micromet_single_station(ds, df_pts(ds), d / 'SM_*.csv', headers=True,
                                                               n_jobs=n),
    'snowpack': lambda ds, d, n: te.to_snowpack(ds, d / 'smet_*.smet', n_jobs=n),
    'geotop': lambda ds, d, n: te.to_geotop(ds, d / 'geotop_*.txt', n_jobs=n),
    'crocus': lambda ds, d, n: te.to_crocus(ds, df_pts(ds), d / 'CROCUS_pt_*.nc', n_jobs=n),
    'cryogrid': lambda ds, d, n: te.to_cryogrid(ds, df_pts(ds), 'CG_*.nc', path=d, n_jobs=n),
}


@pytest.mark.parametrize('name', list(EXPORTS))
def test_processes_do_not_change_the_files(ds, tmp_path, name):
    out = {}
    for n in (1, 2):
        d = tmp_path / f'n{n}'
        d.mkdir()
        files = quiet(EXPORTS[name], ds, d, n)
        assert len(files) == N_PTS and sorted(f.name for f in d.iterdir()) == sorted(map(lambda f: f.split('/')[-1],
                                                                                         files))
        out[n] = d
    for f in sorted(out[1].iterdir()):
        other = out[2] / f.name
        if f.suffix == '.nc':
            a, b = xr.load_dataset(f), xr.load_dataset(other)
            a.attrs.pop('date_created', None)
            b.attrs.pop('date_created', None)
            assert a.identical(b), f.name
        else:
            assert filecmp.cmp(f, other, shallow=False), f.name


def test_fsm_file_holds_the_store_values(ds, tmp_path):
    files = quiet(te.to_fsm, ds, tmp_path / 'FSM_pt_*.txt', n_jobs=2)
    assert [f.split('/')[-1] for f in files[:2]] == ['FSM_pt_00.txt', 'FSM_pt_01.txt']
    pt = ds.isel(point_name=13).load()
    df = pd.read_csv(files[13], sep=' ', header=None)
    times = pd.DatetimeIndex(pt.time.values)
    np.testing.assert_array_equal(df[[0, 1, 2, 3]].values, np.c_[times.year, times.month, times.day, times.hour])
    rh = mu.q_2_rh(pt.t.values, pt.p.values, pt.q.values) * 100
    np.testing.assert_allclose(df[4], pt.SW, rtol=1e-6)
    np.testing.assert_allclose(df[8], np.round(pt.t.values, 2), rtol=1e-6)
    np.testing.assert_allclose(df[9], rh, rtol=1e-5)
    rain, snow = mu.partition_snow(pt.tp.values, pt.t.values, rh, pt.p.values, method='continuous')
    np.testing.assert_allclose(df[6] + df[7], (rain + snow) / 3600, rtol=1e-5, atol=1e-12)


def test_crocus_one_point_per_file(ds, tmp_path):
    pts = df_pts(ds)
    files = quiet(te.to_crocus, ds, pts, tmp_path / 'CROCUS_pt_*.nc')
    assert [f.split('/')[-1] for f in files[:2]] == ['CROCUS_pt_00.nc', 'CROCUS_pt_01.nc']   # digits of '024'
    fo = xr.load_dataset(files[7])
    assert fo.Tair.dims == ('time', 'Number_of_points') and fo.sizes['Number_of_points'] == 1
    assert fo.FRC_TIME_STP.dims == ()
    np.testing.assert_allclose(fo.ZS.values, [pts.elevation.iloc[7]])
    np.testing.assert_allclose(fo.Tair.values[:, 0], ds.t.isel(point_name=7).values, rtol=1e-6)


def test_micromet_station_ids_and_header(ds, tmp_path):
    files = quiet(te.to_micromet_single_station, ds, df_pts(ds), tmp_path / 'SM_*.csv', headers=True, n_jobs=2)
    lines = open(files[4]).read().splitlines()
    assert lines[0].startswith('year   mo   dy') and lines[1].startswith('(yyyy)') and lines[2] == ''
    assert lines[3].split(' ')[4] == '4'   # stn_id: the index of the point


def test_blocks_follow_the_chunks(ds):
    assert te._point_blocks(ds, ['t'], 1) == [slice(0, 10), slice(10, 20), slice(20, 25)]
    # more processes: blocks of at least one chunk
    assert te._point_blocks(ds, ['t'], 8) == [slice(0, 10), slice(10, 20), slice(20, 25)]
    mem = ds.load()
    assert te._point_blocks(mem, ['t'], 2) == [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16),
                                               slice(16, 20), slice(20, 24), slice(24, 25)]


def test_blocks_stay_within_the_memory_cap(ds):
    nt = ds.sizes['time']
    mb = 3 * nt * 2 * 8 / 1e6   # 3 points of 2 variables
    blocks = te._point_blocks(ds, ['t', 'SW'], 1, max_block_mb=mb)
    assert max(b.stop - b.start for b in blocks) == 3
    assert [i for b in blocks for i in range(b.start, b.stop)] == list(range(ds.sizes['point_name']))


def test_fsm2oshd_met_files(ds, tmp_path):
    rng = np.random.default_rng(1)
    x, y = np.arange(0, 1e4 + 1, 500.), np.arange(1e4, -1, -500.)
    grid = {v: (('y', 'x'), rng.uniform(0, 1, (y.size, x.size)) * s)
            for v, s in [('isfor', 1), ('forcov', 1), ('svf_for', 100), ('CC5', 100), ('CC50', 100), ('CH5', 2000),
                         ('LAI5', 300)]}
    mp = SimpleNamespace(toposub=SimpleNamespace(ds_param=xr.Dataset(grid, coords={'x': x, 'y': y}),
                                                 df_centroids=df_pts(ds)),
                         config=DefaultMunch.fromDict({'dem': {'epsg': 2056}, 'sampling': {'method': 'points'}}),
                         downscaled_pts=ds)
    tau = xr.Dataset({'for_tau': (('point_name', 'time'), rng.uniform(0, 100, (N_PTS, N_T)))},
                     coords={'point_name': ds.point_name.values, 'time': ds.time.values})
    for n in (1, 2):
        quiet(te.to_fsm2oshd, mp, tau, simulation_path=tmp_path / f'n{n}', n_jobs=n)
    for f in sorted((tmp_path / 'n1').glob('*_met_*.txt')):
        assert filecmp.cmp(f, tmp_path / 'n2' / f.name, shallow=False)
    met = pd.read_csv(tmp_path / 'n1' / 'fsm__met_003.txt', sep=' ', header=None)
    np.testing.assert_allclose(met[14], np.round(tau.for_tau.isel(point_name=3).values, 4) / 100)
    assert len(list((tmp_path / 'n1').glob('fsm__forest_*.nam'))) == N_PTS
    with pytest.raises(ValueError, match='svf_for'):
        quiet(te.to_fsm2oshd, mp, 'canopy', simulation_path=tmp_path / 'bad')
