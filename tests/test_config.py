"""
Configuration handling of Topoclass: paths, time step and clustering group weights, as documented in
doc/docs/03_configurationFile.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

rasterio = pytest.importorskip('rasterio')
from rasterio.transform import from_origin

from TopoPyScale import topoclass as tc

EPSG = 32632
NX = NY = 24
RES = 30.


def write_tif(path, data, dtype='float32'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1,
                       dtype=dtype, crs=f'EPSG:{EPSG}', transform=from_origin(420000., 6720000., RES, RES)) as dst:
        dst.write(data.astype(dtype), 1)


def dem_array():
    y, x = np.mgrid[0:NY, 0:NX]
    return 1200. + 8. * x + 5. * y + 30. * np.sin(x / 3.) * np.cos(y / 4.)


def make_project(tmp_path, climate_path='inputs/climate/', dem_path=None, timestep='1H', toposub=None):
    project = tmp_path / 'project'
    dem_dir = project / (dem_path or 'inputs/dem')
    write_tif(dem_dir / 'dem.tif', dem_array())
    config = {
        'project': {'name': 'test', 'directory': str(project), 'start': '2020-01-01', 'end': '2020-01-02',
                    'climate': 'era5', 'split': {'IO': False},
                    'parallelization': {'setting': {'multicore': {'CPU_cores': 1}}}},
        'climate': {'era5': {'path': str(climate_path), 'product': 'reanalysis', 'timestep': timestep,
                             'plevels': [700, 850, 1000], 'download_threads': 1, 'data_repository': 'cds'},
                    'precip_lapse_rate': False},
        'dem': {'file': 'dem.tif', 'epsg': EPSG, 'horizon_increments': 45},
        'sampling': {'method': 'toposub',
                     'toposub': {'clustering_method': 'kmean', 'n_clusters': 6, 'random_seed': 2,
                                 'clustering_features': {'x': 1, 'y': 1, 'elevation': 1, 'slope': 1}}},
        'toposcale': {'interpolation_method': 'idw', 'LW_terrain_contribution': True},
        'outputs': {'variables': 'all',
                    'file': {'clean_outputs': False, 'clean_FSM': False, 'df_centroids': 'df_centroids.pck',
                             'ds_param': 'ds_param.nc', 'ds_solar': 'ds_solar.nc', 'da_horizon': 'da_horizon.nc'}},
        'clean_up': {'rm_tmp_dirs': True},
    }
    if dem_path is not None:
        config['dem']['path'] = dem_path
    if toposub:
        config['sampling']['toposub'].update(toposub)
    project.mkdir(parents=True, exist_ok=True)
    fconfig = project / 'config.yml'
    fconfig.write_text(yaml.safe_dump(config))
    return project, fconfig


def test_relative_dem_path_and_timestep_case(tmp_path, monkeypatch):
    project, fconfig = make_project(tmp_path, dem_path='my_dem', timestep='1H')
    monkeypatch.chdir(tmp_path)
    mp = tc.Topoclass(str(fconfig))
    assert mp.config.dem.filepath == project / 'my_dem' / 'dem.tif'
    assert mp.config.climate.era5.timestep == '1h'
    assert set(mp.config.project.extent) == {'latN', 'latS', 'lonW', 'lonE'}


def test_absolute_climate_path_has_no_tmp_and_cleans_up(tmp_path, monkeypatch):
    shared = tmp_path / 'shared_era5'
    project, fconfig = make_project(tmp_path, climate_path=shared)
    monkeypatch.chdir(tmp_path)
    mp = tc.Topoclass(str(fconfig))
    assert mp.config.climate.path == shared
    assert not (shared / 'tmp').exists()
    assert mp.config.outputs.tmp_path.exists()
    mp._remove_tmp_dirs()  # used to raise TypeError on the missing climate tmp path
    assert not mp.config.outputs.tmp_path.exists()


@pytest.mark.parametrize('column', ['weight', 'weights'])
def test_group_weights(tmp_path, monkeypatch, column):
    groups = np.where(np.arange(NX)[None, :] < NX // 2, 1, 2) * np.ones((NY, 1))
    project, fconfig = make_project(tmp_path, toposub={'clustering_groups': 'inputs/dem/groups.tif',
                                                       'clustering_group_weights': 'inputs/dem/gw.csv'})
    write_tif(project / 'inputs/dem/groups.tif', groups)
    # 0.7 + 0.3 is not exactly 1 in floating point
    pd.DataFrame({'group': [1, 2], column: [0.7, 0.3]}).to_csv(project / 'inputs/dem/gw.csv', index=False)
    monkeypatch.chdir(tmp_path)
    mp = tc.Topoclass(str(fconfig))
    mp.compute_dem_param()
    mp.extract_topo_cluster_param()
    labels = mp.toposub.ds_param.cluster_labels.values
    n1 = len(np.unique(labels[groups == 1]))
    n2 = len(np.unique(labels[groups == 2]))
    assert (n1, n2) == (int(np.ceil(6 * 0.7)), int(np.ceil(6 * 0.3)))


def test_groups_without_weights_file(tmp_path, monkeypatch):
    groups = np.where(np.arange(NX)[None, :] < NX // 4, 1, 2) * np.ones((NY, 1))
    project, fconfig = make_project(tmp_path, toposub={'clustering_groups': 'inputs/dem/groups.tif'})
    write_tif(project / 'inputs/dem/groups.tif', groups)
    monkeypatch.chdir(tmp_path)
    mp = tc.Topoclass(str(fconfig))
    mp.compute_dem_param()
    mp.extract_topo_cluster_param()  # used to raise TypeError when clustering_group_weights is not set
    labels = mp.toposub.ds_param.cluster_labels.values
    # clusters proportional to the group areas: 25 % and 75 % of 6
    assert len(np.unique(labels[groups == 1])) == 2
    assert len(np.unique(labels[groups == 2])) == 5


@pytest.mark.parametrize('variables, expected', [(None, None), ('all', None), ('missing', None),
                                                 (['t', 'tp'], ['t', 'tp'])])
def test_output_variables(tmp_path, monkeypatch, variables, expected):
    project, fconfig = make_project(tmp_path)
    config = yaml.safe_load(fconfig.read_text())
    if variables == 'missing':
        del config['outputs']['variables']
    else:
        config['outputs']['variables'] = variables
    fconfig.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    mp = tc.Topoclass(str(fconfig))

    captured = {}

    class FakeDownscaler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def downscale(self, n_core):
            return None

    monkeypatch.setattr(tc.tz, 'ClimateDownscaler', FakeDownscaler)
    monkeypatch.setattr(mp, '_climate_store', lambda: None)
    mp._downscale_zarr(mp.config.outputs.downscaled)
    # a missing key reads as {} with DefaultMunch: it must mean all variables, not none
    assert captured['varout'] == expected
    assert captured['point_block'] == 100
