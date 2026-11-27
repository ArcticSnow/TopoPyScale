"""Maps of downscaled points (topo_plot)."""
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from TopoPyScale import topo_plot as tpl  # noqa: E402


def test_map_of_points_carrying_their_coordinates():
    """The zarr store gives each point its x, y and elevation: they must not clash with the grid of ds_param."""
    y, x = np.arange(5.)[::-1] * 100, np.arange(4.) * 100
    labels = np.array([['00', '01', '01', '02']] * 5)
    ds_param = xr.Dataset({'point_name': (('y', 'x'), labels), 'elevation': (('y', 'x'), np.random.rand(5, 4))},
                          coords={'y': y, 'x': x})
    ds = xr.Dataset({'t': (('point_name', 'time'), np.array([[1., 1.], [2., 2.], [3., 3.]]))},
                    coords={'point_name': ['00', '01', '02'], 'time': [0, 1],
                            'x': ('point_name', [0., 150., 300.]), 'y': ('point_name', [200., 200., 200.]),
                            'elevation': ('point_name', [1000., 1100., 1200.])})
    grid = tpl._on_map(ds.t.isel(time=0), ds_param)
    np.testing.assert_array_equal(grid.values, np.array([[1., 2., 2., 3.]] * 5))
    tpl.map_variable(ds.t.mean('time').to_dataset(), ds_param)
    tpl.map_variable(ds, ds_param, time_step=1, var='t', hillshade=False)
    plt.close('all')


def test_cluster_maps_with_point_names_as_labels(tmp_path):
    """ds_param stores the cluster labels as point names ('00', '01'... and '-9999' outside the mask)."""
    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin
    from TopoPyScale import topo_sub as ts
    labels = np.array([['00', '01', '-9999'], ['01', '02', '02']])
    ds_param = xr.Dataset({'cluster_labels': (('y', 'x'), labels), 'elevation': (('y', 'x'), np.random.rand(2, 3))},
                          coords={'y': [100., 0.], 'x': [0., 100., 200.]})
    ts.plot_center_clusters(None, ds_param, pd.DataFrame({'x': [0., 100.], 'y': [0., 100.]}))
    plt.close('all')
    dem = tmp_path / 'dem.tif'
    with rasterio.open(dem, 'w', driver='GTiff', height=2, width=3, count=1, dtype='float32', crs='EPSG:32632',
                       transform=from_origin(0, 150, 100, 100)) as dst:
        dst.write(np.ones((1, 2, 3), 'float32'))
    out = ts.write_landform(dem, ds_param.stack(cell=('y', 'x')).to_dataframe(), out_dir=str(tmp_path))
    with rasterio.open(out) as src:
        np.testing.assert_array_equal(src.read(1), [[0, 1, -1], [1, 2, 2]])
