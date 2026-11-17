"""Tests of the zarr-based workflow: ERA5 netcdf -> zarr store -> vectorized downscaling -> zarr store of points."""
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import synthetic_era5 as syn
from TopoPyScale import topo_utils as tu
from TopoPyScale import fetch_era5 as fe

VARS = ['t', 'q', 'u', 'v', 'p', 'ws', 'wd', 'tp', 'LW', 'SW', 'SW_direct', 'SW_diffuse']


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


@pytest.fixture(scope='module')
def case():
    tp = np.zeros(72)
    tp[20:30] = 0.6
    return syn.make_era5(start='2020-06-18', n_days=3, cloud=[1, .5, .9], tp_mm=tp)


@pytest.fixture(scope='module')
def rows(case):
    return [syn.make_point(case, name='001', dx=800., elevation=1300.),
            syn.make_point(case, name='002', dx=-3000., dy=2500., elevation=2100., slope_deg=30., aspect_deg=20.,
                           svf=0.9),
            syn.make_point(case, name='003', dx=5000., dy=-4000., elevation=2700., slope_deg=45., aspect_deg=-100.,
                           svf=0.75)]


@pytest.fixture(scope='module')
def inputs(case, rows, tmp_path_factory):
    wd = tmp_path_factory.mktemp('inputs')
    return wd, quiet(syn.solar_geometry, case, rows, wd), syn.horizon_da(rows, 5.)


def test_blocks_and_threads_give_identical_results(case, rows, inputs, tmp_path):
    wd, ds_solar, hor = inputs
    ref = quiet(syn.run_zarr, case, rows, ds_solar, hor, wd, lw_terrain=True, precip_lapse=True, store_name='a.zarr')
    for kw in [dict(point_block=1, time_block=24, store_name='b.zarr'),
               dict(point_block=2, time_block=7, n_core=3, store_name='c.zarr')]:
        other = quiet(syn.run_zarr, case, rows, ds_solar, hor, wd, lw_terrain=True, precip_lapse=True, **kw)
        for v in VARS:
            np.testing.assert_array_equal(ref[v].values, other[v].values, err_msg=f'{v} {kw}')


def test_output_store_layout(case, rows, inputs):
    wd, ds_solar, hor = inputs
    ds = quiet(syn.run_zarr, case, rows, ds_solar, hor, wd, time_block=24, store_name='layout.zarr')
    assert ds[VARS[0]].dims == ('point_name', 'time')
    assert list(ds.point_name.values) == ['001', '002', '003']
    assert ds.p.attrs['units'] == 'Pa' and ds.wd.attrs['units'] == 'rad'
    assert {'elevation', 'slope', 'aspect', 'svf', 'lat', 'lon'} <= set(ds.coords)
    raw = xr.open_zarr(wd / 'downscaled' / 'layout.zarr')
    assert raw.t.encoding['chunks'] == (3, 24) and raw.t.encoding['shards'] == (3, 24)
    assert ds.t.dtype == np.float32


def test_point_at_domain_edge_raises(case, inputs):
    wd, ds_solar, hor = inputs
    edge = syn.make_point(case, name='001', dy=27000.)  # nearest node is the northern row of the 3x3 grid
    with pytest.raises(ValueError, match='edge of the climate data domain'):
        quiet(syn.run_zarr, case, [edge], ds_solar, syn.horizon_da([edge]), wd, store_name='edge.zarr')


@pytest.mark.parametrize('cds_beta', [False, True])
def test_netcdf_to_zarr(case, cds_beta, tmp_path):
    folder = syn.write_era5_netcdf(case, tmp_path / 'yearly', n_parts=3, cds_beta=cds_beta)
    store = quiet(fe.netcdf_to_zarr, folder, tmp_path / 'ERA5.zarr', time_chunk=30)
    ds = xr.open_zarr(store)
    ref = xr.open_zarr(syn.era5_store(case, tmp_path))
    assert ds.sizes['time'] == case.tvec.size
    assert np.all(np.diff(ds.level.values) > 0)
    for v in ['z', 't', 'q', 'u', 'v', 'z_surf', 'sp', 't2m', 'd2m', 'tp', 'ssrd', 'strd']:
        np.testing.assert_allclose(ds[v].transpose(*ref[v].dims).values, ref[v].values, rtol=1e-6, err_msg=v)


def test_read_downscaled(case, rows, inputs):
    wd, ds_solar, hor = inputs
    quiet(syn.run_zarr, case, rows, ds_solar, hor, wd, store_name='read.zarr')
    ds = tu.read_downscaled(wd / 'downscaled' / 'read.zarr')
    assert ds.point_name.dtype.kind == 'U' and ds.sizes == {'point_name': 3, 'time': case.tvec.size}
    with pytest.raises(ValueError, match='not a zarr store'):
        tu.read_downscaled(str(wd / 'down_pt_*.nc'))


def test_topoclass_downscales_from_netcdf_through_zarr(case, rows, inputs, tmp_path):
    """Topoclass converts the yearly netcdf files to ERA5.zarr when missing, then writes down.zarr."""
    from munch import DefaultMunch
    from TopoPyScale.topoclass import Topoclass
    _, ds_solar, hor = inputs
    climate = tmp_path / 'climate'
    syn.write_era5_netcdf(case, climate / 'yearly', n_parts=2)
    obj = object.__new__(Topoclass)
    obj.config = DefaultMunch.fromDict({
        'project': {'climate': 'era5', 'start': case.start, 'end': case.end,
                    'parallelization': {'setting': {'multicore': {'CPU_cores': 2}}}},
        'climate': {'path': climate, 'era5': {'timestep': '1h'}, 'precip_lapse_rate': True},
        'dem': {'epsg': syn.EPSG},
        'toposcale': {'interpolation_method': 'idw', 'LW_terrain_contribution': True},
        'outputs': {'file': {}}})
    obj.toposub = SimpleNamespace(df_centroids=pd.DataFrame([vars(r) for r in rows]))
    obj.da_horizon, obj.ds_solar = hor, ds_solar
    ds = quiet(obj._downscale_zarr, tmp_path / 'downscaled')
    assert (climate / 'ERA5.zarr').exists() and (tmp_path / 'downscaled' / 'down.zarr').exists()
    ref = quiet(syn.run_zarr, case, rows, ds_solar, hor, tmp_path / 'ref', lw_terrain=True, precip_lapse=True)
    for v in VARS:
        np.testing.assert_allclose(ds[v].values, ref[v].values, rtol=1e-6, atol=1e-6, err_msg=v)


def test_climate_files_found_beside_an_empty_yearly_folder(case, tmp_path):
    """Files placed directly in the climate folder (as in the examples), with the empty yearly/ of Topoclass."""
    from munch import DefaultMunch
    from TopoPyScale.topoclass import Topoclass
    climate = tmp_path / 'climate'
    syn.write_era5_netcdf(case, climate, n_parts=1)
    (climate / 'yearly').mkdir()
    obj = object.__new__(Topoclass)
    obj.config = DefaultMunch.fromDict({'project': {'climate': 'era5', 'start': case.start, 'end': case.end},
                                        'climate': {'path': climate, 'era5': {}}})
    store = quiet(obj._climate_store)
    assert xr.open_zarr(store).sizes['time'] == case.tvec.size


def test_topoclass_rebuilds_store_not_covering_period(case, rows, inputs, tmp_path):
    from munch import DefaultMunch
    from TopoPyScale.topoclass import Topoclass
    climate = tmp_path / 'climate'
    syn.write_era5_netcdf(case, climate / 'yearly', n_parts=3)
    # stale store built from the first part only
    (tmp_path / 'part').mkdir()
    for kind in ['PLEV', 'SURF']:
        (tmp_path / 'part' / f'{kind}_00.nc').write_bytes((climate / 'yearly' / f'{kind}_00.nc').read_bytes())
    quiet(fe.netcdf_to_zarr, tmp_path / 'part', climate / 'ERA5.zarr')
    obj = object.__new__(Topoclass)
    obj.config = DefaultMunch.fromDict({'project': {'climate': 'era5', 'start': case.start, 'end': case.end},
                                        'climate': {'path': climate, 'era5': {}}})
    store = quiet(obj._climate_store)
    assert xr.open_zarr(store).sizes['time'] == case.tvec.size
