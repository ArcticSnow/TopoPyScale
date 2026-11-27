"""The solar geometry computed block by block in the downscaler equals the one precomputed by get_solar_geom."""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import synthetic_era5 as syn
from TopoPyScale import solar_geom as sg
from TopoPyScale import topo_scale_zarr as tz

VARS = ['t', 'q', 'u', 'v', 'p', 'ws', 'wd', 'tp', 'LW', 'SW', 'SW_direct', 'SW_diffuse', 'cos_illumination']


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


@pytest.fixture(scope='module')
def case():
    return syn.make_era5(start='2020-06-18', n_days=3, cloud=[1, .5, .9])


@pytest.fixture(scope='module')
def rows(case):
    return [syn.make_point(case, name='001', dx=800., elevation=1300.),
            syn.make_point(case, name='002', dx=-3000., dy=2500., elevation=2100., slope_deg=30., aspect_deg=20.,
                           svf=0.9),
            syn.make_point(case, name='003', dx=5000., dy=-4000., elevation=2700., slope_deg=45., aspect_deg=-100.,
                           svf=0.75)]


def downscale(case, rows, ds_solar, wd, store, **kw):
    df = pd.DataFrame([vars(r) for r in rows])
    cd = tz.ClimateDownscaler(syn.era5_store(case, wd), wd / 'down', df, syn.horizon_da(rows, 20.), ds_solar,
                              syn.EPSG, case.start, case.end, '1h', lw_terrain_flag=True, store_name=store, **kw)
    return quiet(cd.downscale).load()


def test_block_solar_equals_precomputed_file(case, rows, tmp_path):
    ds_solar = quiet(syn.solar_geometry, case, rows, tmp_path)
    ref = downscale(case, rows, ds_solar, tmp_path, 'ref.zarr')
    for kw in [dict(), dict(point_block=1, time_block=7, store_name='blk.zarr')]:
        kw.setdefault('store_name', 'none.zarr')
        out = downscale(case, rows, None, tmp_path, kw.pop('store_name'), **kw)
        for v in VARS:
            np.testing.assert_allclose(out[v].values, ref[v].values, rtol=0, atol=1e-4, err_msg=f'{v} {kw}')
    assert out.attrs['accumulation_window'] == '1h' and out.attrs['solar_substeps'] == 6
    assert out.attrs['solar_geometry_time_offset'] == ref.attrs['solar_geometry_time_offset']


def test_solar_block_slices_match_full_period(rows, case):
    df = pd.DataFrame([vars(r) for r in rows])
    lon, lat, elev = sg.point_coordinates(df, syn.EPSG)
    terms = sg.window_terms(pd.date_range(case.start, periods=48, freq='1h'), '1h', 6)
    full = sg.solar_block(terms, lon, lat, elev)
    part = sg.solar_block(terms, lon[1:], lat[1:], elev[1:], 10, 30)
    for v in full:
        np.testing.assert_array_equal(part[v], full[v][10:30, 1:])


def test_point_coordinates_needs_an_epsg_for_x_y():
    df = pd.DataFrame({'x': [0.], 'y': [0.], 'elevation': [0.]})
    with pytest.raises(ValueError, match='EPSG'):
        sg.point_coordinates(df)
