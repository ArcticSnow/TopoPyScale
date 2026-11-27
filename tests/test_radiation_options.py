"""Options of the radiation: direct/diffuse split from ERA5 fdir."""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import synthetic_era5 as syn
from TopoPyScale import topo_scale_zarr as tz
from TopoPyScale import toposcale_core as core

G = 9.81


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fun(*args, **kwargs)


def solar_noon(n=1):
    z = np.full((n, 1), np.deg2rad(40.))
    return {'mu0': np.cos(z), 'SWtoa': 1000. * np.cos(z), 'zenith': z, 'azimuth': np.zeros((n, 1)),
            'elevation': np.pi / 2 - z, 'sunset': np.zeros((n, 1), bool)}


def test_fdir_gives_the_split_on_flat_open_ground():
    sol = solar_noon()
    ssrd, fdir = np.array([[500. * 3600]]), np.array([[350. * 3600]])
    sp = np.array([[85000.]])
    out = core.shortwave(ssrd, sp, sp, sol, np.zeros(1), np.zeros(1), np.ones(1), np.zeros((1, 1)), fdir=fdir)
    np.testing.assert_allclose(out['SW_direct'], 350.)
    np.testing.assert_allclose(out['SW_diffuse'], 150.)
    # bounded by the global radiation
    out = core.shortwave(ssrd, sp, sp, sol, np.zeros(1), np.zeros(1), np.ones(1), np.zeros((1, 1)),
                         fdir=np.array([[700. * 3600]]))
    np.testing.assert_allclose(out['SW_direct'], 500.)
    np.testing.assert_allclose(out['SW_diffuse'], 0.)


@pytest.fixture(scope='module')
def case():
    return syn.make_era5(start='2020-06-18', n_days=2, cloud=[1, .5])


def store(case, path, fdir_fraction=None):
    plev = case.plev.copy(deep=True)
    plev['z'] = plev.z * G
    surf = case.surf.rename({'z': 'z_surf'}).copy(deep=True)
    surf['z_surf'] = surf.z_surf * G
    if fdir_fraction is not None:
        surf['fdir'] = surf.ssrd * fdir_fraction
    xr.merge([plev, surf]).astype('float32').to_zarr(path, mode='w', zarr_format=3)
    return path


def run(case, path, rows, wd, name, **kw):
    df = pd.DataFrame([vars(r) for r in rows])
    cd = tz.ClimateDownscaler(path, wd / 'down', df, syn.horizon_da(rows), None, syn.EPSG, case.start, case.end,
                              '1h', store_name=name, **kw)
    return quiet(cd.downscale).load()


def test_era5_split_uses_fdir(case, tmp_path):
    rows = [syn.make_point(case, name='001', elevation=case.z_surf),
            syn.make_point(case, name='002', dx=2000., elevation=2200., slope_deg=35., aspect_deg=0., svf=0.85)]
    path = store(case, tmp_path / 'era5.zarr', fdir_fraction=0.6)
    emp = run(case, path, rows, tmp_path, 'emp.zarr')
    era = run(case, path, rows, tmp_path, 'era.zarr', direct_diffuse='era5')
    assert era.attrs['direct_diffuse'] == 'era5' and emp.attrs['direct_diffuse'] == 'empirical'
    # flat point at the ERA5 surface: direct = 60 % of the global radiation
    day = era.SW.isel(point_name=0) > 50
    frac = (era.SW_direct / era.SW).isel(point_name=0).where(day)
    np.testing.assert_allclose(frac.dropna('time').values, 0.6, atol=0.02)
    np.testing.assert_allclose(era.SW.isel(point_name=0).values, emp.SW.isel(point_name=0).values, rtol=0.02,
                               atol=2.)
    # the slope differs: the split changes what the terrain corrections act on
    assert np.abs(era.SW.isel(point_name=1) - emp.SW.isel(point_name=1)).max() > 5.


def test_era5_split_needs_fdir(case, tmp_path):
    rows = [syn.make_point(case, name='001')]
    path = store(case, tmp_path / 'nofdir.zarr')
    with pytest.raises(ValueError, match='direct_diffuse: era5'):
        run(case, path, rows, tmp_path, 'x.zarr', direct_diffuse='era5')
    with pytest.raises(ValueError, match='not supported'):
        run(case, path, rows, tmp_path, 'x.zarr', direct_diffuse='ecmwf')
