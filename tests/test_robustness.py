"""Failures that must be visible: missing values kept through files, unfinished functions refusing to run."""
import numpy as np
import pytest
import xarray as xr

from TopoPyScale import topo_export as te


def test_packed_netcdf_keeps_missing_values(tmp_path):
    """ds_param is saved packed as int16: masked or no-data cells must come back as NaN, not as a value."""
    elev = np.array([[1000., np.nan], [2000., 3000.]])
    ds = xr.Dataset({'elevation': (('y', 'x'), elev), 'cluster_labels': (('y', 'x'), np.array([[0, 1], [1, 2]]))},
                    coords={'x': [0., 30.], 'y': [30., 0.]})
    te.to_netcdf(ds, fname=tmp_path / 'p.nc')
    back = xr.load_dataset(tmp_path / 'p.nc')
    assert np.isnan(back.elevation.values[0, 1])
    np.testing.assert_allclose(back.elevation.values[~np.isnan(elev)], elev[~np.isnan(elev)], atol=2.)
    np.testing.assert_array_equal(back.cluster_labels.values, ds.cluster_labels.values)


def test_unfinished_and_invalid_calls_raise(tmp_path):
    with pytest.raises(NotImplementedError, match='to_fsm2oshd'):
        te.to_fsm2(None, None, None, None)
    ds = xr.Dataset({'t': (('point_name', 'time'), np.zeros((1, 2)))},
                    coords={'point_name': ['0'], 'time': np.array(['2020-01-01T00', '2020-01-01T01'],
                                                                  dtype='datetime64[ns]')})
    with pytest.raises(ValueError, match='da_label'):
        te.to_cryogrid(ds, None, path=tmp_path, label_map=True)
