"""
S. Filhol, 2025

TopoSCALE downscaling from an ERA5 zarr store to a zarr store of downscaled points.

The ERA5 store (pressure levels z, t, q, u, v and surface fields z_surf, sp, t2m, d2m, tp, ssrd, strd) is read in
blocks of time steps and points. For each block, only the ERA5 nodes surrounding the points of the block are read,
and all points are downscaled at once with the array functions of toposcale_core.py. Results are written to one zarr
store with dimensions (point_name, time), chunked by point and by time block.

Create the ERA5 store from yearly netcdf files with fetch_era5.netcdf_to_zarr().
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import dask.array as da
from pyproj import Transformer

try:
    from zarr.codecs import BloscCodec
except ImportError:
    from zarr.codecs._blosc import BloscCodec

from TopoPyScale import toposcale_core as core
from TopoPyScale import topo_utils as tu
from TopoPyScale.toposcale_core import PRECIP_LAPSE_MAX_FACTOR

PLEV_VARS = ['z', 't', 'q', 'u', 'v']
SURF_VARS = ['z_surf', 'sp', 't2m', 'd2m', 'tp', 'ssrd', 'strd']
SOLAR_VARS = ['mu0', 'SWtoa', 'zenith', 'azimuth', 'elevation', 'sunset']

varout_default = ['t', 'u', 'v', 'q', 'p', 'tp', 'precip_lapse_rate', 'wd', 'ws', 'w', 'vp', 'cse',
                  'LW', 'SW', 'SW_direct', 'SW_diffuse', 'cos_illumination']

VAR_ATTRS = {
    't': {'units': 'K', 'long_name': 'Temperature', 'standard_name': 'air_temperature'},
    'q': {'units': 'kg kg**-1', 'long_name': 'Specific humidity', 'standard_name': 'specific_humidity'},
    'u': {'units': 'm s**-1', 'long_name': 'U component of wind', 'standard_name': 'eastward_wind'},
    'v': {'units': 'm s**-1', 'long_name': 'V component of wind', 'standard_name': 'northward_wind'},
    'p': {'units': 'Pa', 'long_name': 'Surface air pressure', 'standard_name': 'surface_air_pressure'},
    'ws': {'units': 'm s**-1', 'long_name': 'Wind speed', 'standard_name': 'wind_speed'},
    'wd': {'units': 'rad', 'long_name': 'Wind direction (from), clockwise from north',
           'standard_name': 'wind_from_direction'},
    'precip_lapse_rate': {'units': '1', 'long_name': 'Precipitation elevation adjustment factor (already applied to tp)'},
    'w': {'units': 'kg kg**-1', 'long_name': 'Mixing ratio', 'standard_name': 'humidity_mixing_ratio'},
    'vp': {'units': 'Pa', 'long_name': 'Vapor pressure', 'standard_name': 'water_vapor_partial_pressure_in_air'},
    'cse': {'units': '1', 'long_name': 'Clear sky emissivity'},
    'LW': {'units': 'W m**-2', 'long_name': 'Surface longwave radiation downwards',
           'standard_name': 'surface_downwelling_longwave_flux_in_air'},
    'SW': {'units': 'W m**-2', 'long_name': 'Surface solar radiation downwards',
           'standard_name': 'surface_downwelling_shortwave_flux_in_air'},
    'SW_direct': {'units': 'W m**-2', 'long_name': 'Surface direct solar radiation downwards'},
    'SW_diffuse': {'units': 'W m**-2', 'long_name': 'Surface diffuse solar radiation downwards'},
    'cos_illumination': {'units': '1', 'long_name': 'Cosine of the solar illumination angle on the slope'},
}


def tp_attrs(tstep_h):
    return {'units': 'mm hr**-1' if tstep_h == 1 else f'mm ({tstep_h} hr)**-1',
            'long_name': f'Precipitation over the {tstep_h} h time step ending at the time stamp',
            'standard_name': 'precipitation_amount'}


class ClimateDownscaler:
    """
    Downscale an ERA5 zarr store to a list of points.

    Args:
        era5_zarr_path (str): path to the ERA5 zarr store (see fetch_era5.netcdf_to_zarr)
        output_path (str): directory of the output store
        df_centroids (dataframe): points with point_name, x, y, lat, lon, elevation, slope, aspect, svf
        da_horizon (dataarray): horizon angles [rad] of the points (azimuth, point_name), from
            topo_param.compute_horizon_points, or of the DEM grid (azimuth, y, x), taken at the nearest pixel
        ds_solar (dataset): solar geometry (point_name, time), from solar_geom.get_solar_geom
        target_EPSG (int): EPSG code of x, y
        start_date, end_date (str): period to downscale (end_date included)
        tstep (str): '1h', '3h' or '6h'
        varout (list): variables to store, default varout_default
        interp_method (str): 'idw' or 'linear'
        lw_terrain_flag (bool): add the longwave emitted by the surrounding terrain
        precip_lapse_rate_flag (bool): apply the precipitation elevation factor
        store_name (str): name of the output zarr store, default 'down.zarr'
        precip_lapse_rate_max_factor (float): bound of the precipitation elevation factor
        point_block (int): number of points processed together, also the shard size of the output store along points
        time_block (int): number of time steps processed together, also the chunk size of the output store along time.
            Default: one year
        max_box_mb (float): the ERA5 nodes around all points are read once per time block if they fit in this
            memory (MB), otherwise once per point block
    """

    def __init__(self,
                 era5_zarr_path,
                 output_path,
                 df_centroids,
                 da_horizon,
                 ds_solar,
                 target_EPSG,
                 start_date,
                 end_date,
                 tstep,
                 varout=None,
                 interp_method='idw',
                 lw_terrain_flag=True,
                 precip_lapse_rate_flag=False,
                 store_name='down.zarr',
                 precip_lapse_rate_max_factor=PRECIP_LAPSE_MAX_FACTOR,
                 point_block=100,
                 time_block=None,
                 max_box_mb=2000):

        store_name = store_name or 'down.zarr'
        if not str(store_name).endswith('.zarr'):
            raise ValueError('store_name must end with .zarr')

        self.ERA = xr.open_zarr(str(era5_zarr_path))
        missing = [v for v in PLEV_VARS + SURF_VARS if v not in self.ERA]
        if missing:
            raise ValueError(f'Variables {missing} missing from the ERA5 store {era5_zarr_path}')
        self.ERA = self.ERA.sortby('level')  # level=-1 must be the lowest in altitude
        self.level_hpa = self.ERA.level.values.astype(float)

        tstep_dict = {'1h': 1, '3h': 3, '6h': 6}
        self.tstep_h = tstep_dict.get(tstep)
        if self.tstep_h is None:
            raise ValueError(f'tstep {tstep} not supported, use 1h, 3h or 6h')
        self.tvec = pd.date_range(start_date, pd.to_datetime(end_date) + pd.to_timedelta('1D'), freq=tstep,
                                  inclusive='left')
        self.era_time_index = self._time_index(pd.DatetimeIndex(self.ERA.time.values), 'ERA5 store')

        self.df = df_centroids.reset_index(drop=True).copy()
        self.df['point_name'] = self.df.point_name.astype(str)
        transformer = Transformer.from_crs('epsg:4326', f'epsg:{target_EPSG}', always_xy=True)
        self.ilat, self.ilon, self.weights = core.neighbours(self.df, self.ERA.latitude.values,
                                                              self.ERA.longitude.values, transformer, interp_method)
        self.points = {k: self.df[k].values.astype(float) for k in ['elevation', 'slope', 'aspect', 'svf']}

        # horizon profile of each point, sorted by azimuth
        hp = tu.horizon_at_points(da_horizon, self.df).sortby('azimuth')
        self.horizon_azimuth = hp.azimuth.values
        self.horizon_profiles = np.asarray(hp.transpose('point', 'azimuth').values, dtype=float)

        sol = ds_solar.assign_coords(point_name=tu.as_str(ds_solar.point_name.values))
        absent = sorted(set(self.df.point_name) - set(sol.point_name.values))
        if absent:
            raise ValueError(f'Points {absent[:5]} missing from ds_solar')
        self.ds_solar = sol
        self.solar_point_index = pd.Index(sol.point_name.values).get_indexer(self.df.point_name)
        self.solar_time_index = self._time_index(pd.DatetimeIndex(sol.time.values), 'ds_solar')
        self._solar_lock = threading.Lock()  # HDF5 files are not safe to read from several threads

        self.lw_terrain_flag = lw_terrain_flag
        self.precip_lapse_rate_flag = precip_lapse_rate_flag
        self.precip_lapse_rate_max_factor = precip_lapse_rate_max_factor or PRECIP_LAPSE_MAX_FACTOR
        self.varout = list(varout) if varout not in (None, 'all') else list(varout_default)
        self.point_block = int(point_block)
        self.time_block = int(time_block) if time_block else int(round(365 * 24 / self.tstep_h))
        self.time_block = min(self.time_block, self.tvec.size)
        self.output_path = Path(output_path)
        self.store = self.output_path / store_name

        self.max_box_mb = max_box_mb
        # contiguous blocks of points: each block fills whole shards of the output store
        npts = len(self.df)
        self.point_block = min(self.point_block, npts)
        self.point_blocks = [(s, min(s + self.point_block, npts)) for s in range(0, npts, self.point_block)]
        self.time_blocks = [(s, min(s + self.time_block, self.tvec.size))
                            for s in range(0, self.tvec.size, self.time_block)]

    def _time_index(self, available, name):
        idx = available.get_indexer(self.tvec)
        if (idx < 0).any():
            miss = self.tvec[idx < 0]
            raise ValueError(f'{miss.size} time steps missing from the {name} (first: {miss[0]}). '
                             f'Available: {available.min()} to {available.max()}')
        return idx

    # -----------------------------------------------------------------------------------------------------------------
    def setup_output_store(self):
        """Create the output store: coordinates, attributes and empty variables chunked (1 point, time_block)."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        npts, nt = len(self.df), self.tvec.size
        coords = {'point_name': self.df.point_name.values.astype(str), 'time': self.tvec}
        for c in ['point_ind', 'x', 'y', 'lat', 'lon', 'elevation', 'slope', 'aspect', 'svf']:
            if c in self.df:
                coords[c] = ('point_name', self.df[c].values)
        data = {v: (('point_name', 'time'), da.zeros((npts, nt), dtype='float32',
                                                      chunks=(self.point_block, self.time_block)))
                for v in self.varout}
        ds = xr.Dataset(data, coords=coords)
        for v in self.varout:
            ds[v].attrs = tp_attrs(self.tstep_h) if v == 'tp' else VAR_ATTRS.get(v, {})
        ver = tu.get_versionning()
        ds.attrs = {'title': 'Downscaled timeseries with TopoPyScale',
                    'created with': 'TopoPyScale, see more at https://topopyscale.readthedocs.io',
                    'package_version': ver.get('package_version'), 'git_commit': ver.get('git_commit'),
                    'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                    'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'solar_geometry_time_offset': self.ds_solar.attrs.get('solar_geometry_time_offset', 'unknown')}
        comp = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)
        # chunks of up to 10 points x time block, grouped in shards of point_block points: each block writes whole
        # shards (few files, few writes) and reading the time series of one point only reads a small chunk
        chunk_p = max(d for d in range(1, min(10, self.point_block) + 1) if self.point_block % d == 0)
        encoding = {v: {'compressors': comp, 'chunks': (chunk_p, self.time_block),
                        'shards': (self.point_block, self.time_block)} for v in self.varout}
        ds.to_zarr(self.store, mode='w', compute=False, zarr_format=3, encoding=encoding)

    def _read_solar(self, p0, p1, t0, t1):
        tsl = self.solar_time_index[t0:t1]
        with self._solar_lock:
            sol = self.ds_solar[SOLAR_VARS].isel(point_name=self.solar_point_index[p0:p1], time=tsl).load()
        out = {v: np.asarray(sol[v].transpose('time', 'point_name').values) for v in SOLAR_VARS}
        out['sunset'] = out['sunset'].astype(bool)
        for v in SOLAR_VARS[:-1]:
            out[v] = out[v].astype(float)
        return out

    def _box(self, p0, p1):
        """Slices of the ERA5 nodes needed by points p0:p1."""
        ilat, ilon = self.ilat[p0:p1], self.ilon[p0:p1]
        return slice(ilat.min(), ilat.max() + 1), slice(ilon.min(), ilon.max() + 1)

    def _read_era(self, t0, t1, box):
        """Read the ERA5 variables for time steps t0:t1 on a box of nodes, as float64 numpy arrays."""
        tidx = self.era_time_index[t0:t1]
        tsel = slice(tidx[0], tidx[-1] + 1) if np.all(np.diff(tidx) == 1) else tidx
        sub = self.ERA[PLEV_VARS + SURF_VARS].isel(time=tsel, latitude=box[0], longitude=box[1])
        era = {v: np.asarray(sub[v].transpose('time', 'level', 'latitude', 'longitude').values, dtype=float)
               for v in PLEV_VARS}
        era.update({v: np.asarray(sub[v].transpose('time', 'latitude', 'longitude').values, dtype=float)
                    for v in SURF_VARS})
        return era, box

    def _box_mb(self, box, nt):
        cells = (box[0].stop - box[0].start) * (box[1].stop - box[1].start)
        return nt * cells * (len(PLEV_VARS) * self.level_hpa.size + len(SURF_VARS)) * 8 / 1e6

    def process_block(self, p0, p1, t0, t1, era=None):
        """
        Downscale points p0:p1 (positions in df_centroids) for time steps t0:t1 and write them to the store.

        Args:
            era (tuple): optional (arrays, box) already read by _read_era() on a box containing the points
        """
        if era is None:
            era = self._read_era(t0, t1, self._box(p0, p1))
        fields, box = era
        W = core.weight_matrix(self.ilat[p0:p1] - box[0].start, self.ilon[p0:p1] - box[1].start,
                               self.weights[p0:p1], box[0].stop - box[0].start, box[1].stop - box[1].start)
        plev = {v: core.apply_weights(fields[v], W) for v in PLEV_VARS}
        surf = {v: core.apply_weights(fields[v], W) for v in SURF_VARS}
        solar = self._read_solar(p0, p1, t0, t1)
        shade = core.horizon_at_sun(self.horizon_profiles[p0:p1], self.horizon_azimuth, solar['azimuth']) > \
            solar['elevation']
        points = {k: v[p0:p1] for k, v in self.points.items()}
        out = core.downscale_block(plev, surf, self.level_hpa, self.tvec[t0:t1].month.values, points, solar, shade,
                                   self.tstep_h, self.lw_terrain_flag, self.precip_lapse_rate_flag,
                                   self.precip_lapse_rate_max_factor)
        group = zarr.open_group(str(self.store), mode='r+')
        for v in self.varout:
            group[v][p0:p1, t0:t1] = out[v].T.astype('float32')
        return (p1 - p0) * (t1 - t0)

    def downscale(self, n_core=1):
        """
        Downscale all points and write them to the output store.

        Args:
            n_core (int): number of threads processing point blocks in parallel. Blocks write disjoint shards.

        Returns:
            dataset: the output store opened with xarray
        """
        start = time.time()
        self.setup_output_store()
        print(f'---> Downscaling {len(self.df)} points x {self.tvec.size} time steps '
              f'({len(self.point_blocks)} point blocks x {len(self.time_blocks)} time blocks)')
        full_box = self._box(0, len(self.df))
        with zarr.config.set({'array.write_empty_chunks': True}), \
                ThreadPoolExecutor(max_workers=max(1, int(n_core or 1))) as pool:
            for t0, t1 in self.time_blocks:
                era = self._read_era(t0, t1, full_box) if self._box_mb(full_box, t1 - t0) <= self.max_box_mb \
                    else None
                list(pool.map(lambda pb: self.process_block(pb[0], pb[1], t0, t1, era), self.point_blocks))
        elapsed = time.time() - start
        print(f'---> Downscaling finished in {elapsed:.1f} s ({elapsed / len(self.df):.3f} s per point)')
        return tu.open_downscaled_store(self.store)
