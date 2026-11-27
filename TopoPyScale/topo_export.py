"""
Functions to export the downscaled points to formats compatible with existing models (e.g. CROCUS, Cryogrid, Snowmodel, ...)

S. Filhol, December 2021

TODO;
- SPHY forcing (grids)
"""
import csv
import datetime as dt
import pdb
import sys, os

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from TopoPyScale import meteo_util as mu
from TopoPyScale import topo_utils as tu
from TopoPyScale import topo_param as tp

from pathlib import Path


def _tstep_seconds(ds):
    """Time step of a dataset in seconds (tp is in mm per time step)."""
    times = pd.DatetimeIndex(ds.time.values)
    return (times[1] - times[0]).total_seconds()


# ---------------------------------------------------------------------------------------------------------------
# One file per point: blocks of points read once, files written by parallel processes
# ---------------------------------------------------------------------------------------------------------------
def point_file_names(fname_format, n_points, n_digits):
    """File name of each point: the point index, zero-padded to n_digits, in place of '*' in fname_format."""
    parts = str(fname_format).split('*')
    return [parts[0] + str(i).zfill(n_digits) + parts[1] for i in range(n_points)]


def _point_blocks(ds, variables, n_jobs, max_points=100):
    """
    Slices of consecutive points, aligned on the chunks of the store along point_name, of about max_points points
    and at least 4 per process.
    """
    n = ds.sizes['point_name']
    target = max(1, min(max_points, -(-n // (4 * n_jobs))))
    da_ = ds[variables[0]]
    sizes = da_.chunks[da_.dims.index('point_name')] if da_.chunks is not None else (1,) * n
    blocks, start = [], 0
    for end in np.cumsum(sizes):
        if end - start >= target or end == n:
            blocks.append(slice(start, int(end)))
            start = int(end)
    return blocks


def _write_block(ds, variables, write, fnames, info, extra, options):
    """Read a block of points once, then write the file of each point (runs in a worker process)."""
    data = ds[variables].load()
    extra = None if extra is None else extra.load()
    for i, fname in enumerate(fnames):
        write(fname, data.isel(point_name=i), info[i], None if extra is None else extra.isel(point_name=i),
              **options)
    return len(fnames)


def export_points(ds, variables, write, fnames, info=None, extra=None, options=None, n_jobs=1, label='files'):
    """
    Write one file per point with write(fname, ds_pt, info, extra_pt, **options).

    The points are processed by blocks of consecutive points aligned on the chunks of the zarr store: each block is
    read once, instead of one read per point and variable, then the file of each of its points is written. With
    n_jobs > 1 the blocks are shared between n_jobs processes (joblib loky workers, which need no main guard), each
    reading its blocks itself, as formatting the files holds the Python lock.

    Args:
        ds (dataset): downscaled points (point_name, time), lazy (zarr store) or in memory
        variables (list): variables read by write
        write (callable): module-level function writing the file of one point from ds_pt (one point, in memory)
        fnames (list): file of each point, in the order of ds.point_name
        info (list): object given to write for each point (e.g. a row of df_pts). Default None
        extra (dataset): dataset with a point_name dimension, selected like ds and given to write for each point
        options (dict): keyword arguments of write
        n_jobs (int): number of processes. 1: in this process
        label (str): what is written, for the progress messages

    Returns:
        list: the files written
    """
    n = ds.sizes['point_name']
    fnames = [str(f) for f in fnames]
    if len(fnames) != n:
        raise ValueError(f'{len(fnames)} file names for {n} points')
    info = [None] * n if info is None else list(info)
    n_jobs = max(1, int(n_jobs or 1))
    blocks = _point_blocks(ds, variables, n_jobs)
    names = ds.point_name.values

    def args(b):
        return (ds.isel(point_name=b), variables, write, fnames[b], info[b],
                None if extra is None else extra.sel(point_name=names[b]), options or {})

    if n_jobs == 1 or len(blocks) == 1:
        done = (_write_block(*args(b)) for b in blocks)
    else:
        done = Parallel(n_jobs=min(n_jobs, len(blocks)), backend='loky', return_as='generator_unordered')(
            delayed(_write_block)(*args(b)) for b in blocks)
    written = 0
    for k in done:
        written += k
        print(f'---> {written}/{n} {label} written')
    return fnames


def compute_scaling_and_offset(da, n=16):
    """
    Compute offset and scale factor for int conversion

    Args:
        da (dataarray): of a given variable
        n (int): number of digits to account for
    """
    vmin = float(da.min().values)
    vmax = float(da.max().values)

    # stretch/compress data to the available packed range
    scale_factor = (vmax - vmin) / (2 ** n - 1)
    if not np.isfinite(scale_factor) or scale_factor == 0:
        scale_factor = 1.  # constant (or empty) variable: packed value 0 decodes to vmin
        return scale_factor, vmin
    # translate the range to be symmetric about zero
    add_offset = vmin + 2 ** (n - 1) * scale_factor

    return scale_factor, add_offset


def to_netcdf(ds, fname='output.nc', variables=None, complevel=9, n_bits=10):
    """
    Generic function to save a datatset to one single compressed netcdf file. Float variables are packed as int16.

    Args:
        fname (str): name of export file
        variables (list str): list of variable to export. Default exports all variables
        complevel (int): Compression level. 1-9
        n_bits (int): number of bits used to pack float variables (at most 15). Resolution = range / (2**n_bits - 1)
    """

    encod_dict = {}
    if variables is None:
        variables = list(ds.keys())

    for var in variables:
        scale_factor, add_offset = compute_scaling_and_offset(ds[var], n=min(int(n_bits), 15))
        if str(ds[var].dtype)[:3] == 'int' or var == 'cluster_labels' or var == 'point_name':
            encod_dict.update({var: {
                'dtype': ds[var].dtype}})
        else:
            encod_dict.update({var: {"zlib": True,
                                     "complevel": complevel,
                                     'dtype': 'int16',
                                     'scale_factor': scale_factor,
                                     'add_offset': add_offset}})
    ds[variables].to_netcdf(fname, encoding=encod_dict, engine='h5netcdf')

    print(f'---> File {fname} saved')


# MuSA forcing variables (TopoPyScale names) in the units MuSA expects, so that MuSA's
# forcing_multiplier and forcing_offset can stay at their defaults (1 and 0)
MUSA_VARS = {'SW': {'units': 'W m-2', 'long_name': 'Surface incoming shortwave radiation'},
             'LW': {'units': 'W m-2', 'long_name': 'Surface incoming longwave radiation'},
             'precip': {'units': 'kg m-2 s-1', 'long_name': 'Total precipitation rate'},
             't': {'units': 'K', 'long_name': 'Near surface air temperature'},
             'RH': {'units': '%', 'long_name': 'Near surface relative humidity'},
             'ws': {'units': 'm s-1', 'long_name': 'Wind speed'},
             'p': {'units': 'Pa', 'long_name': 'Surface air pressure'}}


def _musa_write_day(fname, times, data, idx, dims, coords, attrs, chunk=32):
    """
    Write one daily MuSA forcing file. Output cell [i, j] gets the timeseries of point idx[i, j].

    Args:
        fname (Path): file to write
        times (DatetimeIndex): timesteps of that day
        data (dict): variable -> array (time, point), with one extra NaN column for cells without a point
        idx (array): 2D index into the point axis of data
        dims (tuple): names of the two spatial dimensions
        coords (tuple): coordinate values of the two spatial dimensions
        attrs (dict): global attributes
        chunk (int): netcdf chunk size along the spatial dimensions
    """
    fo = xr.Dataset({var: (('time',) + dims, arr[:, idx], MUSA_VARS[var]) for var, arr in data.items()},
                    coords={'time': times, dims[0]: coords[0], dims[1]: coords[1]},
                    attrs=attrs)
    chunks = (len(times), min(chunk, idx.shape[0]), min(chunk, idx.shape[1]))
    encod_dict = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32', 'chunksizes': chunks} for var in data}
    # write under another name first: MuSA reads every *.nc of the folder
    fo.to_netcdf(f'{fname}.tmp', encoding=encod_dict)
    os.replace(f'{fname}.tmp', fname)


def to_musa(ds,
            df_pts,
            ds_param=None,
            path=None,
            layout='grid',
            bbox=None,
            stride=1,
            cell_mask=None,
            start=None,
            end=None,
            n_cores=4,
            climate_dataset_name='ERA5',
            project_authors='S. Filhol'
            ):
    """
    Function to export downscaled timeseries as forcing for MuSA (https://github.com/ealonsogzl/MuSA).

    TopoPyScale downscales to a set of points: the TopoSUB cluster centroids, or points given by the user, one
    timeseries each (ds). MuSA reads its forcing from netcdf files on a 2D grid of cells, and runs one simulation
    per cell where mask.nc is 1. This function writes the points onto such a grid, in one of two layouts.

    Layouts:
        - 'grid' (default): fully distributed, one MuSA cell per DEM cell. Variables var(time, northing, easting)
          on the DEM grid of ds_param, where northing and easting are the DEM y and x coordinates (DEM CRS). Every
          DEM cell gets the timeseries of its TopoSUB cluster (ds_param.point_name). Cells without a cluster
          (point_name '-9999'), outside bbox or outside cell_mask get NaN forcing and a NaN mask, so MuSA skips
          them. MuSA saves the results of the cell at northing[i], easting[j] of mask.nc in cell_<i>_<j>.pkl.blp.
          Needs the toposub sampling method.
        - 'points': one MuSA cell per point (e.g. per TopoSUB cluster). Variables var(time, y, x) with
          y = point id and x = [0], i.e. a column of N cells. The point id is point_name as an integer: 0..N-1 for
          toposub, 1..N for point sampling (0..N-1 in name order when the names are not numbers). MuSA saves the
          results of the i-th point (0-based, sorted by point id) in cell_<i>_0.pkl.blp; with toposub, i is the
          cluster number.

    Both layouts hold the same values: a DEM cell in 'grid' has exactly the timeseries of its cluster in
    'points'. With the same forcing and perturbations and without observations, MuSA therefore gives every cell
    of a cluster the same result. Use 'points' for open-loop runs (N simulations; map the results back to the DEM
    with ds_param.point_name), and 'grid' to assimilate observations that differ between DEM cells.

    Saved in path:
        - forcing/YYYY-MM-DD.nc: one file per day (MuSA nc_forcing_path). MuSA reads every *.nc in this folder, so
          the function stops if other netcdf files are in it, including days left from a longer export.
        - mask.nc: mask = 1 for the cells to simulate, NaN to skip them (MuSA nc_maks_path). MuSA needs NaN rather
          than 0 for the cells to skip. Set cells to NaN to run a subset.
        - DEM.nc ('grid'): DEM(northing, easting), elevation in m (MuSA dem_path, nc_dem_varname = "DEM")
        - points_meta.csv ('points'): one row per point with musa_cell (the result file name), musa_y (point id),
          musa_x (0), point_name, and x, y, lon, lat, elevation, slope, aspect, svf of the point from df_pts

    Variables, in the units MuSA expects so that its forcing_multiplier and forcing_offset keep their defaults
    (1 and 0); float32, zlib-compressed, NaN where a cell has no forcing:
        - SW, LW: incoming shortwave and longwave radiation (W m-2), from SW and LW
        - t: air temperature (K)
        - RH: relative humidity (%), 100 * meteo_util.q_2_rh(t, p, q)
        - ws: wind speed (m s-1)
        - p: surface air pressure (Pa)
        - precip: precipitation rate (kg m-2 s-1), tp (mm per timestep) divided by the timestep in seconds

    MuSA config.py settings for these files:
        nc_forcing_path = "<path>/forcing/"
        nc_maks_path = "<path>/mask.nc"
        dem_path = "<path>/DEM.nc" and nc_dem_varname = "DEM" (layout 'grid', if MuSA needs the DEM)
        forcing_var_names = {"SW_var_name": "SW", "LW_var_name": "LW", "Precip_var_name": "precip",
                             "Press_var_name": "p", "RH_var_name": "RH", "Temp_var_name": "t", "Wind_var_name": "ws"}
        forcing_dim_names = {"lat_forz_var_name": "northing" ('grid') or "y" ('points'),
                             "lon_forz_var_name": "easting" ('grid') or "x" ('points'), "time_forz_var_name": "time"}
        date_ini, date_end, dt: the values returned by this function
    MuSA also needs observation files on the same grid as mask.nc (nc_obs_path, obs_var_names, dates_obs), even
    for an open-loop run. They are not written here.

    Args:
        ds (dataset): downscaled points with dims (point_name, time)
        df_pts (dataframe): points or cluster centroids (df_centroids)
        ds_param (dataset): DEM parameters with the TopoSUB map point_name. Required for layout 'grid'
        path (str or Path): output folder. Default: outputs/musa_<layout>
        layout (str): 'grid' (default) or 'points', see above
        bbox (list): [xmin, xmax, ymin, ymax] in DEM coordinates, to crop the DEM for layout 'grid'
        stride (int): keep every stride-th DEM cell for layout 'grid'
        cell_mask (array): boolean (y, x) array on the DEM grid of the cells to simulate for layout 'grid'
        start (str): first timestep to export. Default: first timestep of ds
        end (str): last timestep to export. Default: last timestep of ds
        n_cores (int): number of cores to write the daily files
        climate_dataset_name (str): name of the original climate dataset
        project_authors (str): authors of the dataset

    Returns:
        dict: layout, dims, grid shape, number of cells to simulate (n_cells), number of daily files, and
            date_ini, date_end and dt for MuSA's config.py
    """
    if layout not in ['grid', 'points']:
        raise ValueError(f"layout must be 'grid' or 'points', not {layout}")
    path = Path(f'outputs/musa_{layout}' if path is None else path)
    ver_dict = tu.get_versionning()

    ds = ds[['SW', 'LW', 't', 'ws', 'p', 'tp', 'q']].sel(time=slice(start, end)).transpose('time', 'point_name').load()
    point_names = ds.point_name.values.astype(str)
    numeric = all(name.isdigit() for name in point_names)
    # sort by point id, so that y increases with the MuSA cell index
    ds = ds.isel(point_name=np.argsort([int(name) for name in point_names] if numeric else point_names, kind='stable'))
    point_names = ds.point_name.values.astype(str)
    n_pts = point_names.size
    times = pd.DatetimeIndex(ds.time.values)
    tstep = (times[1] - times[0]).total_seconds()
    fo = {var: ds[var].values for var in ['SW', 'LW', 't', 'ws', 'p']}
    fo['precip'] = ds.tp.values / tstep  # mm per timestep -> kg m-2 s-1
    fo['RH'] = mu.q_2_rh(ds.t.values, ds.p.values, ds.q.values) * 100
    # one extra NaN column for the DEM cells without a point (index n_pts)
    data = {var: np.concatenate([fo[var], np.full((times.size, 1), np.nan)], axis=1).astype(np.float32)
            for var in MUSA_VARS}

    # MuSA reads every *.nc in nc_forcing_path, so stop if other files are in the way
    days = times.normalize()
    fnames = [day.strftime('%Y-%m-%d.nc') for day in days.unique()]
    (path / 'forcing').mkdir(parents=True, exist_ok=True)
    foreign = sorted(f.name for f in (path / 'forcing').glob('*.nc') if f.name not in fnames)
    if foreign:
        raise ValueError(f'MuSA would also read {foreign} from {path / "forcing"}. Remove them or use another path')

    if layout == 'points':
        dims = ('y', 'x')
        idx = np.arange(n_pts)[:, None]
        point_ids = np.array([int(name) for name in point_names]) if numeric else np.arange(n_pts)
        coords = (point_ids, np.array([0]))
        df_meta = pd.DataFrame({'musa_cell': [f'cell_{i}_0' for i in range(n_pts)], 'musa_y': point_ids,
                                'musa_x': 0, 'point_name': point_names})
        if df_pts is not None:
            cols = [col for col in ['x', 'y', 'lon', 'lat', 'elevation', 'slope', 'aspect', 'svf'] if col in df_pts]
            df_meta = df_meta.merge(df_pts[cols].assign(point_name=df_pts.point_name.astype(str)),
                                    on='point_name', how='left')
        df_meta.to_csv(path / 'points_meta.csv', index=False)
    else:
        if ds_param is None or 'point_name' not in ds_param:
            raise ValueError("layout 'grid' needs ds_param with the TopoSUB map point_name (sampling method "
                             "toposub). Use layout='points' otherwise")
        # TopoSUB map as index into the points of ds (-1 for cells without a cluster, point_name '-9999')
        lut = {name: i for i, name in enumerate(point_names)}
        names = ds_param.point_name.values.astype(str)
        dom = xr.Dataset({'idx': (('y', 'x'), pd.Series(names.ravel()).map(lut).fillna(-1).astype(int)
                                  .to_numpy().reshape(names.shape)),
                          'elevation': (('y', 'x'), ds_param.elevation.values)},
                         coords={'y': ds_param.y.values, 'x': ds_param.x.values})
        if cell_mask is not None:
            dom['idx'] = dom.idx.where(np.asarray(cell_mask, bool), -1)
        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
            dom = dom.sel(x=slice(xmin, xmax), y=slice(ymax, ymin) if dom.y[0] > dom.y[-1] else slice(ymin, ymax))
        dom = dom.isel(y=slice(None, None, stride), x=slice(None, None, stride))

        dims = ('northing', 'easting')
        idx = dom.idx.values
        coords = (dom.y.values, dom.x.values)
        xr.Dataset({'DEM': (dims, dom.elevation.values.astype(np.float32), {'units': 'm'})},
                   coords={dims[0]: coords[0], dims[1]: coords[1]}).to_netcdf(
            path / 'DEM.nc', encoding={'DEM': {'_FillValue': None}})
    if not (idx >= 0).any():
        raise ValueError('No cell to simulate in the selected domain')

    # MuSA skips cells where the mask is NaN and simulates cells where it is 1
    mask = np.where(idx >= 0, 1, np.nan).astype(np.float32)
    xr.Dataset({'mask': (dims, mask, {'long_name': 'MuSA mask: 1 = simulate, NaN = skip'})},
               coords={dims[0]: coords[0], dims[1]: coords[1]}).to_netcdf(
        path / 'mask.nc', encoding={'mask': {'_FillValue': None}})

    attrs = {'title': 'Forcing for the MuSA assimilation toolbox',
             'source': 'Data from {} downscaled with TopoPyScale'.format(climate_dataset_name),
             'creator_name': 'Dataset created by {}'.format(project_authors),
             'package_version': ver_dict.get('package_version'),
             'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
             'git_commit': ver_dict.get('git_commit'),
             'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
             'musa_layout': layout}
    gidx = np.where(idx >= 0, idx, n_pts)
    fun_param = [(path / 'forcing' / fname, times[days == day], {var: arr[days == day] for var, arr in data.items()},
                  gidx, dims, coords, attrs) for fname, day in zip(fnames, days.unique())]
    tu.multicore_pooling(_musa_write_day, fun_param, n_cores)

    meta = {'layout': layout, 'dims': dims, 'shape': idx.shape, 'n_cells': int((idx >= 0).sum()),
            'n_files': len(fnames), 'date_ini': times[0].strftime('%Y-%m-%d %H:%M'),
            'date_end': times[-1].strftime('%Y-%m-%d %H:%M'), 'dt': int(tstep)}
    print(f'---> MuSA forcing saved in {path}: {len(fnames)} daily files, grid {idx.shape} {dims}, '
          f'{meta["n_cells"]} cells to simulate, {meta["date_ini"]} to {meta["date_end"]}, dt = {meta["dt"]} s')
    return meta


def _write_cryogrid(fname, ds_pt, info, extra_pt, tstep, snow_partition_method, climate_dataset_name,
                    project_author, ver_dict):
    fo = xr.Dataset()
    fo['time'] = ds_pt.time
    fo['Tair'] = ('time', ds_pt.t.values - 273.15)
    fo['q'] = ('time', ds_pt.q.values)
    fo['wind'] = ('time', ds_pt.ws.values)
    fo['Sin'] = ('time', ds_pt.SW.values)
    fo['Lin'] = ('time', ds_pt.LW.values)
    fo['p'] = ('time', ds_pt.p.values)
    rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
    rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh * 100, ds_pt.p.values,
                                   method=snow_partition_method)
    steps_per_day = 86400 / tstep
    fo['rainfall'], fo['snowfall'] = ('time', rain * steps_per_day), ('time', snow * steps_per_day)  # mm per time step to mm/day

    fo.Tair.attrs = {'units': '°C', 'standard_name': 'Tair', 'long_name': 'Near Surface Air Temperature',
                     '_FillValue': -9999.0}
    fo.q.attrs = {'units': 'kg/kg', 'standard_name': 'q', 'long_name': 'Near Surface Specific Humidity',
                  '_FillValue': -9999.0}
    fo.wind.attrs = {'units': 'm/s', 'standard_name': 'wind', 'long_name': 'Wind Speed', '_FillValue': -9999.0}
    fo.rainfall.attrs = {'units': 'mm/d', 'standard_name': 'rainfall', 'long_name': 'Rainfall Rate',
                         '_FillValue': -9999.0}
    fo.snowfall.attrs = {'units': 'mm/d', 'standard_name': 'snowfall', 'long_name': 'Snowfall Rate',
                         '_FillValue': -9999.0}
    fo.Sin.attrs = {'units': 'W/m^2', 'standard_name': 'Sin',
                    'long_name': 'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999.0}
    fo.Lin.attrs = {'units': 'W/m^2', 'standard_name': 'Lin', 'long_name': 'Surface Incident Longtwave Radiation',
                    '_FillValue': -9999.0}
    fo.p.attrs = {'units': 'Pa', 'standard_name': 'p', 'long_name': 'Surface Pressure', '_FillValue': -9999.0}

    fo.attrs = {'title': 'Forcing for Cryogrid Community model',
                'source': 'Data from {} downscaled with TopoPyScale'.format(climate_dataset_name),
                'creator_name': 'Dataset created by {}'.format(project_author),
                'package_version': ver_dict.get('package_version'),
                'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                'git_commit': ver_dict.get('git_commit'),
                'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    encod_dict = {}
    for var in list(fo.keys()):
        scale_factor, add_offset = compute_scaling_and_offset(fo[var], n=10)
        encod_dict.update({var: {"zlib": True,
                                 "complevel": 9,
                                 'dtype': 'int16',
                                 'scale_factor': scale_factor,
                                 'add_offset': add_offset}})
    fo['latitude'] = info['latitude']
    fo['longitude'] = info['longitude']
    fo['elevation'] = info['elevation']
    fo.latitude.attrs = {'units': 'deg', 'standard_name': 'latitude'}
    fo.longitude.attrs = {'units': 'deg', 'standard_name': 'longitude'}
    fo.elevation.attrs = {'units': 'm', 'standard_name': 'elevation'}
    fo.to_netcdf(fname, encoding=encod_dict)


def to_cryogrid(ds,
                df_pts,
                fname_format='Cryogrid_pt_*.nc',
                path='outputs',
                label_map=False,
                da_label=None,
                snow_partition_method='continuous',
                climate_dataset_name='ERA5',
                project_author='S. Filhol',
                n_jobs=1):
    """
    Function to export TopoPyScale downscaled dataset in a netcdf format compatible for Cryogrid-community model.

    Args:
        ds (dataset): downscaled points (Topoclass.downscaled_pts)
        df_pts (dataframe): with metadata of all point of downscaling
        fname_format (str): file name for output
        path (str): path where to export files
        label_map (bool): export cluster_label map
        da_label (dataarray):(optional) dataarray containing label value for each pixel of the DEM
        n_jobs (int): number of processes writing the files (see export_points)
    
    """
    # Add logic to save maps of cluster labels in case of usage of topo_sub

    ver_dict = tu.get_versionning()
    if label_map:
        if da_label is None:
            print('ERROR: no cluster label (da_label) provided')
            sys.exit()
        else:
            da_label.to_netcdf(str(Path(path) / 'cluster_labels_map.nc'))

    n_digits = len(str(len(ds.point_name))) + 1
    fnames = [str(Path(path) / f) for f in point_file_names(fname_format, ds.sizes['point_name'], n_digits)]
    info = [{'latitude': df_pts.latitude.iloc[i], 'longitude': df_pts.longitude.iloc[i],
             'elevation': df_pts.elevation.iloc[i]} for i in range(len(fnames))]
    options = {'tstep': _tstep_seconds(ds), 'snow_partition_method': snow_partition_method,
               'climate_dataset_name': climate_dataset_name, 'project_author': project_author, 'ver_dict': ver_dict}
    return export_points(ds, ['t', 'q', 'ws', 'SW', 'LW', 'p', 'tp'], _write_cryogrid, fnames, info=info,
                         options=options, n_jobs=n_jobs, label='CryoGrid files')


def to_fsm2(ds_down,
                fsm_param,
                simulation_path='fsm_sim',
                fname_format='fsm_',
                namelist_options=None,
                n_digits=None,
                snow_partition_method='continuous',
                cluster_method=True,
                epsg_ds_param=2056,
                temperature_correction=0,
                forest_param_scaler={'hcan':100, 'lai':100}):
    """
    Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd).
    FSM2oshd includes canopy structures processes
    one simulation consists of 2 driving file:
       - met.txt with variables:
           year, month, day, hour, SWdir, SWdif, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt
       - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10

    Args:
        ds_down:  Downscaled weather variable dataset
        fsm_param:  terrain and canopy parameter dataset
        df_centroids:  cluster centroids statistics (terrain + canopy)
        ds_tvt (dataset, int, float, or str):  transmisivity. Can be a dataset, a constant or 'svf_for'
        simulation_path (str): 'fsm_sim'
        fname_format (str):'fsm_'
        namelist_options (dict): {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]}
        n_digits (int): Number of digits (for filename system)
        snow_partition_method (str): method for snow partitioning. Default: 'continuous'
        cluster_method (bool): boolean to be True is using cluster appraoch
        epsg_ds_param (int): epsg code of ds_parma: example: 2056

    """

    print(" \n##### WARNING ######\n")
    print(" Method is not fully implemented")
    print(" \n#####         ######\n")


    def write_fsm2_compilation_nc(file_compil_nc='compil_nc.sh',
                                    ALBEDO=2, 
                                    CANMOD=2,  
                                    CANRAD=2,  
                                    CONDCT=1,
                                    DENSITY=2,
                                    EXCHNG=1,
                                    HYDROL=2,
                                    SGRAIN=2, 
                                    SNFRAC=2,
                                    DRIV1D=1):       
        compil_part1 = f"""
#!/bin/bash

FC=gfortran
cd src

cat > OPTS.h << EOF
/* Process options                                  : Possible values */
#define ALBEDO {ALBEDO}   /* snow albedo                   : 1, 2            */
#define CANMOD {CANMOD}   /* forest canopy layers          : 1, 2            */
#define CANRAD {CANRAD}   /* canopy radiative properties   : 1, 2            */
#define CONDCT {CONDCT}   /* snow thermal conductivity     : 0, 1            */
#define DENSTY {DENSITY}  /* snow density                  : 0, 1, 2         */
#define EXCHNG {EXCHNG}   /* turbulent exchange            : 0, 1            */
#define HYDROL {HYDROL}   /* snow hydraulics               : 0, 1, 2         */
#define SGRAIN {SGRAIN}   /* snow grain growth             : 1, 2            */
#define SNFRAC {SNFRAC}   /* snow cover fraction           : 1, 2            */
/* Driving data options                             : Possible values */
#define DRIV1D {DRIV1D}   /* 1D driving data format        : 1, 2            */
#define SETPAR 1   /* parameter inputs              : 0, 1            */
#define SWPART 0   /* SW radiation partition        : 0, 1            */
#define ZOFFST 1   /* measurement height offset     : 0, 1            */
/* Output options                                   : Possible values */
#define PROFNC 1   /* netCDF output                 : 0, 1            */
EOF

"""

        compil_part2 = """
files=(FSM2_MODULES.F90 FSM2.F90 FSM2_DRIVE.F90 FSM2_MAP.F90          \
       FSM2_PARAMS.F90 FSM2_PREPNC.F90 FSM2_TIMESTEP.F90              \
       FSM2_WRITENC.F90                                               \
       CANOPY.F90 INTERCEPT.F90 LUDCMP.F90 PSIMH.F90 QSAT.F90         \
       SNOW.F90 SOIL.F90 SOLARPOS.F90 SRFEBAL.F90 SWRAD.F90           \
       THERMAL.F90 TRIDIAG.F90)
> FSM2_temp.f90
for file in ${files[*]}
do
  $FC -cpp -E -P -o out $file
  cat --squeeze-blank out >> FSM2_temp.f90
done

$FC -O3 -c -I/usr/lib64/gfortran/modules FSM2_temp.f90
$FC -o FSM2 FSM2_temp.o -L/usr/lib64 -lnetcdff
mv FSM2 ../FSM2

rm out
rm *.mod
rm *.o
rm FSM2_temp.f90
cd ..
"""

        with open(file_compil_nc, "w") as file:
                file.write(compil_part1 + compil_part2)



    def write_fsm2_namelist(row,
                                pt_ind,
                                n_digits,
                                fname_format='fsm_sim/fsm_',
                                namelist_param=None,
                                scaler={ 'hcan':100, 'lai':100, 'svf_for':1}):
        # Function to write namelist file (.nam) for each point where to run FSM.

        file_namelist = str(fname_format) + f'_{mode}_' + str(pt_ind).zfill(n_digits) + '.nam'
        file_met = str(fname_format) + '_met_' + str(pt_ind).zfill(n_digits) + '.txt'
        file_output = str(fname_format) + f'_outputs_{mode}_' + str(pt_ind).zfill(n_digits) + '.nc'

        # populate namelist parameters
        max_sd = namelist_param.get('max_sd')
        z_snow = namelist_param.get('z_snow')
        z_soil = namelist_param.get('z_soil')
        z_tair = namelist_param.get('z_tair')
        z_wind = namelist_param.get('z_wind')
        soil_moist_init = namelist_param.get('soil_moist_init')
        soil_temp_init = namelist_param.get('soil_temp_init')
        latitude = namelist_param.get('latitude')


        # Values compatible with 'forest' mode
        albedo_snowfree = 0.2
        hcan = row.hcan/scaler.get('hcan')
        lai = row.lai5/scaler.get('lai')
        svf_for = row.lai5/scaler.get('svf_for')


        if os.path.exists(file_met):
            nlst = f"""
&params
  asmn = {namelist_param.get('asmn')}
  asmx = {namelist_param.get('asmx')}
  eta0 = {namelist_param.get('eta0')}
  hfsn = {namelist_param.get('hfsn')}
  kfix = {namelist_param.get('kfix')}
  rcld = {namelist_param.get('rcld')}
  rfix = {namelist_param.get('rfix')}
  rgr0 = {namelist_param.get('rgr0')}
  rhof = {namelist_param.get('rhof')}
  rhow = {namelist_param.get('rhow')}
  rmlt = {namelist_param.get('rmlt')}
  Salb = {namelist_param.get('Salb')}
  snda = {namelist_param.get('snda')}
  Talb = {namelist_param.get('Talb')}
  tcld = {namelist_param.get('tcld')}
  tmlt = {namelist_param.get('tmlt')}
  trho = {namelist_param.get('trho')}
  Wirr = {namelist_param.get('Wirr')}
  z0sn = {namelist_param.get('z0sn')}
  acn0 = {namelist_param.get('acn0')}
  acns = {namelist_param.get('acns')}
  avg0 = {namelist_param.get('avg0')}
  avgs = {namelist_param.get('avgs')}
  cvai = {namelist_param.get('cvai')}
  gsnf = {namelist_param.get('gsnf')}
  hbas = {namelist_param.get('hbas')}
  kext = {namelist_param.get('kext')}
  rveg = {namelist_param.get('rveg')}
  svai = {namelist_param.get('svai')}
  tunl = {namelist_param.get('tunl')}
  wcan = {namelist_param.get('wcan')}
  fcly = {namelist_param.get('fcly')}
  fsnd = {namelist_param.get('fsnd')}
  gsat = {namelist_param.get('gsat')}
  z0sf = {namelist_param.get('z0sf')}

/
&gridpnts
  Ncols = 1
  Nrows = 1
  Nsmax = {len(z_snow)}
  Nsoil = {len(z_soil)}

/
&gridlevs
  Dzsnow = {', '.join(str(i) for i in z_snow)}          ! Minimum snow layer thicknesses (m)
  Dzsoil = {', '.join(str(i) for i in z_soil)}          ! Soil layer thicknesses (m)
  fvg1 = 0.5                                            ! Fraction of vegetation in upper layer (CANMOD=2)
  zsub = 1.5                                            ! Subcanopy wind speed diagnostic height
/
&drive
  met_file = {file_met}
  dt = 3600                     ! Timestep
  zT = {z_tair}                 ! Temperature and humidity measurement height
  zU = {z_wind}                 ! Wind speed measurement height
  lat = {latitude}
  noon = {solar_noon}                     ! Time of solar noon (for SWPART=1)
/
&veg
  ab0 = 0.2
  fsky = {svf_for}
  vegh = {hcan}
  VAI =  {lai}
/
&initial
  fsat = {soil_moist_init}
  Tprf = {soil_temp_init}
/
&outputs
  runid = {path_to_store}
  dump_file = {file_output}
/
  """

            with open(file_namelist, "w") as nlst_file:
                nlst_file.write(nlst)
        else:
            print(f'ERROR: met_file: {file_met} not found')
            return


    def write_fsm2_met(ds_pt,
                           pt_name,
                           pt_ind,
                           n_digits,
                           fname_format='fsm_sim/fsm_',
                           temperature_correction=0,
                           DRIV1D=1):
        """
        Function to write meteorological forcing for FSM2


        for DRIV1D=1   (FSM format)

            year month  day   hour  SW  LW  Sf  Rf  Ta  RH   Ua    Ps    
            (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa)

        for DRIV1D=2    (ESM-SnowMIP format)

            year month  day   hour  SW  LW  Sf  Rf  Ta  Qa RH   Ua    Ps    
            (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (kg/kg) (RH 0-100) (m/s) (Pa)

        """
        
        # for storage optimization tvt is stored in percent.


        foutput = str(fname_format) + '_met_' + str(pt_ind).zfill(n_digits) + '.txt'
        df = pd.DataFrame()
        df['year'] = pd.to_datetime(ds_pt.time.values).year
        df['month']  = pd.to_datetime(ds_pt.time.values).month
        df['day']  = pd.to_datetime(ds_pt.time.values).day
        df['hr']  = pd.to_datetime(ds_pt.time.values).hour
        df['SWdir'] = np.round(ds_pt.SW_direct.values,2)              # change to direct SW
        df['SWdif'] = np.round(ds_pt.SW_diffuse.values,2)                # change to diffuse SW
        df['LW'] = np.round(ds_pt.LW.values,2)
        rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
        rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh * 100, ds_pt.p.values, method=snow_partition_method)
        df['snowfall'] = np.round(snow, 5)
        df['rainfall'] = np.round(rain, 5)
        df['Tair'] = np.round(ds_pt.t.values, 2) + temperature_correction
        if DRIV1D == 2:
            df['Qa'] = np.round(ds_pt.q.values, 2)
        df['RH'] = np.round(rh * 100,2)
        df['speed'] = np.round(ds_pt.ws.values,2)
        df['p'] = np.round(ds_pt.p.values,2)

        df.to_csv(foutput, index=False, header=False, sep=' ')
        print(f'---> Met file {foutput} saved')


    # 0. Add print statements about the need of data currently computed extrenally to TopoPyScale.
    # 1. Convert Clare's canopy output to FSM2oshd parametrization
    # 2. Extract cluster canopy and forest cover weights from ds_param_canopy
    # 3. loop through clusters and write met_file and namelist files
    # - 2 sets of FSM run, one with forest, another one without. Need to combine both output into a final value for the cluster centroid

    # ----- unpack and overwrite namelist_options ----
    # default namelist_param values
    namelist_param = { 'max_sd':4,
                       'z_snow':[0.1, 0.2, 0.4],
                       'z_soil':[0.1, 0.2, 0.4, 0.8],
                       'z_tair':2,
                       'z_wind':10,
                       'latitude':42,
                       'soil_moist_init':0.5,
                       'soil_temp_init':285,
                       'fsat':0.5,
                       'Tprf':285,
                       'asmn':0.1,
                       'asmx':0.4,
                       'eta0':0.21,
                       'hfsn':0.6,
                       'kfix':3.6e4,
                       'rcld':300,
                       'rfix':300,
                       'rgr0':5e-5,
                       'rhof':100,
                       'rhow':300,
                       'rmlt':500,
                       'Salb':10,
                       'snda':2.8e-6,
                       'Talb':-2,
                       'tcld':1000,
                       'tmlt':100,
                       'trho':200,
                       'Wirr':0.03,
                       'z0sn':0.01,
                       'acn0':0.1,
                       'acns':0.4,
                       'avg0':0.21,
                       'avgs':0.6,
                       'cvai':3.6e4,
                       'gsnf':0.01,
                       'hbas':2,
                       'kext':0.5,
                       'rveg':20,
                       'svai':4.4,
                       'tunl':240,
                       'wcan':2.5,
                       'fcly':0.3,
                       'fsnd':0.6,
                       'gsat':0.01,
                       'z0sf':0.1
                       }

    if namelist_options is not None:
        namelist_param.update(namelist_options)

    if n_digits is None:
        n_digits = len(str(len(ds_down.point_name))) + 1

    if cluster_method:
        # extract FSM forest parameters for each clusters
        # Aggregate forest parameters only to fores area

        dx = np.abs(np.diff(fsm_param.x)[0])
        dy = np.abs(np.diff(fsm_param.y)[0])
        df_forest = tu.ds_to_indexed_dataframe(fsm_param.drop(['point_name', 'cluster_labels'])).groupby('point_ind').mean()
        df_forest['cluster_total_area'] = fsm_param.elevation.groupby(fsm_param.point_ind).count().values * dx * dy
        df_forest['proportion_with_forest'] = fsm_param.isfor.groupby(fsm_param.point_ind).mean().values
        df_forest['cluster_domain_size'] = np.sqrt(df_forest.cluster_total_area)
        #df_forest['cluster_domain_size'] = np.sqrt(fsm_param.drop('point_name').groupby(fsm_param.point_name).count().to_dataframe().LAI5)*dx
        df_forest['forest_cover'] = fsm_param.forcov.groupby(fsm_param.point_ind).mean().values
        df_forest.forest_cover.loc[df_forest.proportion_with_forest<0.01] = 0
        df_forest['lon'], df_forest['lat'] = tp.convert_epsg_pts(df_forest.x, df_forest.y, epsg_ds_param, 4326)

    else:
        pass

    p = Path(simulation_path)
    # rename variable columns to match namelist functino varnames
    new_name = {'LAI5':'lai', 'svf_for':'vfhp', 'CH5':'hcan'}
    df_forest = df_forest.rename(columns=new_name)
    print(df_forest)
 
    # ----- Loop through all points-------
    # NOTE: eventually this for loop could be parallelized to several cores -----
    for pt_ind, pt_name in enumerate(ds_down.point_name.values):

        ds_pt = ds_down.sel(point_name=pt_name).copy()

        row_forest = df_forest.iloc[pt_ind]
        write_fsm2_compilation_nc(file_compil_nc='compil_nc.sh',
                                    ALBEDO=2, 
                                    CANMOD=2,  
                                    CANRAD=2,  
                                    CONDCT=1,
                                    DENSITY=2,
                                    EXCHNG=1,
                                    HYDROL=2,
                                    SGRAIN=2, 
                                    SNFRAC=2,
                                    DRIV1D=1)
        write_fsm2_met(ds_pt,
                           n_digits=n_digits,
                           pt_name=pt_name,
                           pt_ind=pt_ind,
                           fname_format=p/fname_format,
                           temperature_correction=temperature_correction)
        write_fsm2_namelist(row_forest,
                                pt_ind=pt_ind,
                                n_digits=n_digits,
                                fname_format=p/fname_format,
                                mode='forest',
                                namelist_param=namelist_param,
                                scaler=forest_param_scaler) # write forest namelist

        if cluster_method:
            row_open = df_forest.iloc[pt_ind]
            write_fsm2oshd_namelist(row_open,
                                    pt_ind=pt_ind,
                                    n_digits=n_digits,
                                    fname_format=p/fname_format,
                                    mode='open',
                                    namelist_param=namelist_param) # write open namelist

        ds_pt = None
        # [ ] add logic to computed weighted average outputs based on forest cover fraction per point.

    df_forest.to_pickle(p / 'df_forest.pckl')
    return



def _write_fsm2oshd_met(fname, ds_pt, tvt, ds_tvt_pt, snow_partition_method, temperature_correction=0):
    """
    Function to write meteorological forcing for FSM

    Format of the text file is:
        2021 9 1 6 61 45.01 206.7 0 0 275.02 74.08 0.29 74829 0 0.5
        2021 9 1 7 207.9 85.9 210.3 0 0 275.84 66.92 0.39 74864 0 0.5

    year month  day   hour  SWdir   SWdiff  LW  Sf  Rf     Ta  RH   Ua    Ps    Sf24 Tvt
    (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa) (mm) (-)

    """
    
    # for storage optimization tvt is stored in percent.


    df = pd.DataFrame()
    df['year'] = pd.to_datetime(ds_pt.time.values).year
    df['month']  = pd.to_datetime(ds_pt.time.values).month
    df['day']  = pd.to_datetime(ds_pt.time.values).day
    df['hr']  = pd.to_datetime(ds_pt.time.values).hour
    df['SWdir'] = np.round(ds_pt.SW_direct.values,2)              # change to direct SW
    df['SWdif'] = np.round(ds_pt.SW_diffuse.values,2)                # change to diffuse SW
    df['LW'] = np.round(ds_pt.LW.values,2)

    rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)   #convert pressure Bar to Pascal
    rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh * 100, ds_pt.p.values, method=snow_partition_method)
    df['snowfall'] = np.round(snow, 5)
    df['rainfall'] = np.round(rain, 5)
    df['Tair'] = np.round(ds_pt.t.values, 2) + temperature_correction
    df['RH'] = np.round(rh * 100,2)
    df['speed'] = np.round(ds_pt.ws.values,2)
    df['p'] = np.round(ds_pt.p.values,2)

    arr = df.snowfall.rolling(24).sum()  # hardcoded sum over last 24h [mm]
    arr.loc[np.isnan(arr)] = 0
    df['sf24'] = np.round(arr,3)
    #ds_pt['t_iter'] = ds_pt.time.dt.month*10000 + ds_pt.time.dt.day*100 + ds_pt.time.dt.hour
    ds_tvt_pt = tvt if ds_tvt_pt is None else ds_tvt_pt
    if type(ds_tvt_pt) in [int, np.int32, np.int64, float, np.float64, np.float16, np.float32]:
        if ds_tvt_pt>1:
            scale_tvt = 100
        else:
            scale_tvt = 1
        df['tvt'] = ds_tvt_pt / scale_tvt
    else:
        if ds_tvt_pt.for_tau.max()>10:
            scale_tvt = 100
        else:
            scale_tvt = 1
        df['tvt'] = np.round(ds_tvt_pt.for_tau.values, 4)/scale_tvt

    df.to_csv(fname, index=False, header=False, sep=' ')


def to_fsm2oshd(mp,
                ds_tvt,
                simulation_path='fsm_sim',
                fname_format='fsm_',
                namelist_options=None,
                snow_partition_method='continuous',
                temperature_correction=0,
                forest_param_scaler={'vfhp':100, 'fveg':100, 'fves':100, 'hcan':100, 'lai5':100},
                n_jobs=None):
    """
    Function to generate forcing files for FSM2oshd (https://github.com/oshd-slf/FSM2oshd).
    FSM2oshd includes canopy structures processes
    one simulation consists of 2 driving file:
       - met.txt with variables:
           year, month, day, hour, SWdir, SWdif, LW, Sf, Rf, Ta, RH, Ua, Ps, Sf24h, Tvt
       - param.nam with canopy and model constants. See https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/runner.py#L10

    Args:
        mp:  TopoClass object
        df_centroids:  cluster centroids statistics (terrain + canopy)
        ds_tvt (dataset, int, float, or str):  transmisivity. Can be a dataset, a constant or 'svf_for'
        simulation_path (str): 'fsm_sim'
        fname_format (str):'fsm_'
        namelist_options (dict): {'precip_multiplier':1, 'max_sd':4,'z_snow':[0.1, 0.2, 0.4], 'z_soil':[0.1, 0.2, 0.4, 0.8]}
        n_digits (int): Number of digits (for filename system)
        snow_partition_method (str): method for snow partitioning. Default: 'continuous'
        cluster_method (bool): boolean to be True is using cluster appraoch
        epsg_ds_param (int): epsg code of ds_param: example: 2056
        n_jobs (int): number of processes writing the met files (see export_points). Default: the CPU_cores of
            the project configuration

    """
    fsm_param = mp.toposub.ds_param
    epsg_ds_param = mp.config.dem.epsg
    if mp.config.sampling.method == 'toposub':
        sampling_method = 'clusters'
    else:
        sampling_method = mp.config.sampling.method

    def write_fsm2oshd_namelist(row,
                                pt_ind,
                                n_digits,
                                fname_format='fsm_sim/fsm_',
                                mode='forest',
                                namelist_param=None,
                                modconf=None,
                                scaler={'vfhp':100, 'fveg':100, 'fves':100, 'hcan':100, 'lai5':100}):
        # Function to write namelist file (.nam) for each point where to run FSM.

        file_namelist = str(fname_format) + f'_{mode}_' + str(pt_ind).zfill(n_digits) + '.nam'
        file_met = str(fname_format) + '_met_' + str(pt_ind).zfill(n_digits) + '.txt'
        file_output = str(fname_format) + f'_outputs_{mode}_' + str(pt_ind).zfill(n_digits) + '.txt'

        if modconf is None:
            modconf = {
                'albedo':2,
                'condct':1,
                'density':3,
                'hydrol':2
            }

        # populate namelist parameters
        precip_multi = namelist_param.get('precip_multi')
        max_sd = namelist_param.get('max_sd')
        z_snow = namelist_param.get('z_snow')
        z_soil = namelist_param.get('z_soil')
        z_tair = namelist_param.get('z_tair')
        z_wind = namelist_param.get('z_wind')
        diag_var_outputs = namelist_param.get('diag_var_outputs')
        state_var_outputs = namelist_param.get('state_var_outputs')


        if mode == 'forest':
            # Values compatible with 'forest' mode
            canmod = 1
            turbulent_exchange = 2
            z_offset = 1
            fveg = row.fveg/scaler.get('fveg')
            hcan = row.hcan/scaler.get('hcan')
            lai5 = row.lai5/scaler.get('lai5')
            vfhp = row.vfhp/scaler.get('vfhp')
            fves = row.fves/scaler.get('fves')

        elif mode == 'open':
            # default values for 'open' mode
            canmod = 0
            turbulent_exchange = 1
            z_offset, fveg, fves, hcan, lai5, vfhp= 0, 0, 0, 0, 0, 1
        else:
            raise ValueError('mode is not available. Must be open or forest')
        if os.path.exists(file_met):
            nlst = f"""
&nam_grid
  NNx = 1,                          ! Grid x-indexing. Not relevant with this implementation
  NNy = 1,                          ! Grid y-indexing. Not relevant with this implementation
  NNsmax = {len(z_snow)},           ! Number snow layers. Default 3-layers
  NNsoil = {len(z_soil)},           ! Number soil layers. Default 4-layers
/
&nam_layers
  DDzsnow = {', '.join(str(i) for i in z_snow)},    ! Minimum thickness to define a new snow layer
  DDzsoil = {', '.join(str(i) for i in z_soil)},    ! soil layer thickness
/
&nam_driving
  zzT = {z_tair},                   ! Height of temperature forcing (2 or 10m)
  zzU = {z_wind},                   ! Height of wind forcing (2 or 10m)
  met_file = '{file_met}',            ! name of met file associatied to this namelist
  out_file = '{file_output}',         ! name of output file for this FSM2oshd simulation
/
&nam_modconf
  NALBEDO = {modconf.get('albedo')},           ! albedo parametrization (special one for oshd)
  NCANMOD = {canmod},               ! canopy is switch ON/OFF. must sync to CTILE param (open vs. forest).
  NCONDCT = {modconf.get('condct')},           ! thermal conductivity (from original FSM)
  NDENSTY = {modconf.get('density')},          ! densification module 
  NEXCHNG = {turbulent_exchange},   ! option for turbulent heat exchange (need in sync  with in forest or in open)
  NHYDROL = {modconf.get('hydrol')},           ! water transport through snowpack
  NSNFRAC = 3,                      ! fractional snow cover (consider pathciness during melt or not). relevant for large scale run
  NRADSBG = 0,                      ! ignore
  NZOFFST = {z_offset},             ! 0 for above ground, 1 for above canopy. Z-offset for temperature and wind forcing. in sync with 'open' or 'forest'
  NOSHDTN = 0,                      ! switch if compensate for elevation precipitation lapse rate. Relevant for swiss operational model. Turn on if systematic bias of not enough snow in high elevation
  LHN_ON  = .FALSE.,                ! irrelevant for none operational setup. keep to false
  LFOR_HN = .FALSE.,                ! irrelevant for none operational setup. keep to false
/
&nam_modpert
  LZ0PERT = .FALSE.,                ! switch for perturbation run. Not available, in development...
/
&nam_modtile
  CTILE = '{mode}',               !  open or forest mode to run FSM
  rtthresh = 0.1,                   ! threshold of forest cover fraction at which to run forest mode. relevant when run in a grid setup. Not relevant here.
/
&nam_results                        ! https://github.com/oshd-slf/FSM2oshd/blob/048e824fb1077b3a38cc24c0172ee3533475a868/src/core/MODULES.F90#L80
   CLIST_DIAG_RESULTS = {', '.join(f"'{i}'" for i in diag_var_outputs)},   ! need all?   
   CLIST_STATE_RESULTS = {', '.join(f"'{i}'" for i in state_var_outputs)}, ! try if can be deleted
/
&nam_location
  fsky_terr = {np.round(row.svf,3)},            ! terrain svf 
  slopemu = {np.round(row.slope,3)},            ! slope in rad
  xi = 0,                           ! to be ignored. relevant coarse scale run. see Nora's paper
  Ld = {np.round(row.cluster_domain_size,3)},              ! grid cell size in meters (used in snow fractional cover) linked to Nora's paper
  lat = {np.round(row.lat,3)},             ! DD.DDD
  lon = {np.round(row.lon,3)},            ! DD.DDD
  dem = {np.round(row.elevation,0)},            ! elevation
  pmultf = {precip_multi},          ! precip multiplier default 1
  fveg = {np.round(fveg,3)},                    ! local canopy cover fraction (set to 0 in open)
  hcan = {np.round(hcan,3)},                    ! canopy height (meters) (set to 0 in open)
  lai = {np.round(lai5,2)},                      ! Leaf area index  (set to 0 in open)
  vfhp = {np.round(vfhp,3)},                    ! sky view fraction of canopy and terrain(set to 1 in open case)
  fves = {np.round(fves,3)},                ! canopy cover fraction (larger area)
/
  """

            with open(file_namelist, "w") as nlst_file:
                nlst_file.write(nlst)
        else:
            print(f'ERROR: met_file: {file_met} not found')
            return




    # 0. Add print statements about the need of data currently computed extrenally to TopoPyScale.
    # 1. Convert Clare's canopy output to FSM2oshd parametrization
    # 2. Extract cluster canopy and forest cover weights from ds_param_canopy
    # 3. loop through clusters and write met_file and namelist files
    # - 2 sets of FSM run, one with forest, another one without. Need to combine both output into a final value for the cluster centroid

    # ----- unpack and overwrite namelist_options ----
    # default namelist_param values
    namelist_param = { 'precip_multi':1,
                       'max_sd':4,
                       'z_snow':[0.1, 0.2, 0.4],
                       'z_soil':[0.1, 0.2, 0.4, 0.8],
                       'z_tair':2,
                       'z_wind':10,
                       'diag_var_outputs' : ['rotc', 'hsnt', 'swet', 'slqt',  'romc', 'sbsc'],
                       'state_var_outputs':['Tsrf', 'Sveg', 'Ds']}
    if namelist_options is not None:
        namelist_param.update(namelist_options)

    n_digits = len(str(len(mp.downscaled_pts.point_name))) + 1

    if sampling_method == 'clusters':
        # extract FSM forest parameters for each clusters
        # Aggregate forest parameters only to fores area

        dx = np.abs(np.diff(fsm_param.x)[0])
        dy = np.abs(np.diff(fsm_param.y)[0])
        df_forest = tu.ds_to_indexed_dataframe(fsm_param.drop(['point_name', 'cluster_labels'])).groupby('point_ind').mean()
        df_forest['cluster_total_area'] = fsm_param.elevation.groupby(fsm_param.point_ind).count().values * dx * dy
        df_forest['proportion_with_forest'] = fsm_param.isfor.groupby(fsm_param.point_ind).mean().values
        df_forest['cluster_domain_size'] = np.sqrt(df_forest.cluster_total_area)
        #df_forest['cluster_domain_size'] = np.sqrt(fsm_param.drop('point_name').groupby(fsm_param.point_name).count().to_dataframe().LAI5)*dx
        df_forest['forest_cover'] = fsm_param.forcov.groupby(fsm_param.point_ind).mean().values
        df_forest.forest_cover.loc[df_forest.proportion_with_forest<0.01] = 0
        df_forest['lon'], df_forest['lat'] = tp.convert_epsg_pts(df_forest.x, df_forest.y, epsg_ds_param, 4326)

    elif sampling_method == 'points':
        df_forest = mp.toposub.df_centroids.copy()

        xs = xr.DataArray(df_forest.x.values, dims='pts')
        ys = xr.DataArray(df_forest.y.values, dims='pts')

        # sampling forest parameters from ds_param
        df_forest[['isfor', 'forcov', 'svf_for', 'CC5', 'CC50', 'CH5', 'LAI5']] = mp.toposub.ds_param.sel(x=xs, y=ys, method='nearest').to_dataframe()[['isfor', 'forcov', 'svf_for', 'CC5', 'CC50', 'CH5', 'LAI5']]
        pixel_size = np.abs(mp.toposub.ds_param.x.diff(dim='x').mean().values) 
        df_forest['cluster_domain_size'] = pixel_size

    else:
        raise ValueError("Sampling method not yet supported. Avail: clusters, points")

    p = Path(simulation_path)
    # rename variable columns to match namelist functino varnames
    new_name = {'LAI5':'lai5', 'svf_for':'vfhp', 'CC5':'fveg', 'CC50':'fves', 'CH5':'hcan'}
    df_forest = df_forest.rename(columns=new_name)
 
    p.mkdir(parents=True, exist_ok=True)
    # ----- met files: blocks of points read once, written by n_jobs processes -----
    # [ ] Add checking of NaNs in ds_tvt. If NaN present stop process and send ERROR message
    ds = mp.downscaled_pts
    n_pts = ds.sizes['point_name']
    fnames = [str(p / fname_format) + '_met_' + str(i).zfill(n_digits) + '.txt' for i in range(n_pts)]
    tvt, extra = [None] * n_pts, None
    if type(ds_tvt) in [int, float, np.float64, np.float16, np.float32]:
        print('Warning: tvt is constant')
        tvt = [ds_tvt] * n_pts
    elif type(ds_tvt) in [str]:
        if ds_tvt != 'svf_for':
            raise ValueError(f"ds_tvt must be a number, 'svf_for' or a dataset with for_tau, not {ds_tvt}")
        print('Warning: tvt is constant')
        tvt = [df_forest.vfhp.iloc[i] for i in range(n_pts)]
    else:
        extra = ds_tvt
    if n_jobs is None:
        try:
            n_jobs = mp.config.project.parallelization.setting.multicore.CPU_cores
        except AttributeError:
            n_jobs = 1
    export_points(ds, ['SW_direct', 'SW_diffuse', 'LW', 't', 'p', 'q', 'tp', 'ws'], _write_fsm2oshd_met, fnames,
                  info=tvt, extra=extra, n_jobs=n_jobs or 1, label='FSM2oshd met files',
                  options={'snow_partition_method': snow_partition_method,
                           'temperature_correction': temperature_correction})

    # ----- namelists -----
    for pt_ind in range(n_pts):
        row_forest = df_forest.iloc[pt_ind]
        write_fsm2oshd_namelist(row_forest,
                                pt_ind=pt_ind,
                                n_digits=n_digits,
                                fname_format=p/fname_format,
                                mode='forest',
                                namelist_param=namelist_param,
                                scaler=forest_param_scaler) # write forest namelist

        if sampling_method=='clusters':
            row_open = df_forest.iloc[pt_ind]
            write_fsm2oshd_namelist(row_open,
                                    pt_ind=pt_ind,
                                    n_digits=n_digits,
                                    fname_format=p/fname_format,
                                    mode='open',
                                    namelist_param=namelist_param) # write open namelist
        # [ ] add logic to computed weighted average outputs based on forest cover fraction per point.

    df_forest.to_pickle(p / 'df_forest.pckl')
    return


def _write_fsm(fname, ds_pt, info, extra_pt, tstep, snow_partition_method):
    df = pd.DataFrame()
    df['year'] = pd.to_datetime(ds_pt.time.values).year
    df['month'] = pd.to_datetime(ds_pt.time.values).month
    df['day'] = pd.to_datetime(ds_pt.time.values).day
    df['hr'] = pd.to_datetime(ds_pt.time.values).hour
    df['SW'] = ds_pt.SW.values
    df['LW'] = ds_pt.LW.values
    rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
    rain, snow = mu.partition_snow(ds_pt.tp.values, ds_pt.t.values, rh * 100, ds_pt.p.values,
                                   method=snow_partition_method)
    df['snowfall'] = snow / tstep  # mm per time step to kg m-2 s-1
    df['rainfall'] = rain / tstep
    df['Tair'] = np.round(ds_pt.t.values, 2)
    df['RH'] = rh * 100
    df['speed'] = ds_pt.ws.values
    df['p'] = ds_pt.p.values
    df.to_csv(fname, index=False, header=False, sep=' ')


def to_fsm(ds, fname_format='FSM_pt_*.tx', snow_partition_method='continuous', n_digits=None, n_jobs=1):
    """
    Function to export data for FSM.

    Args:
        ds (dataset): downscaled_pts,
        fname_format (str pattern): output format of filename
        snow_partition_method (str): snow/rain partitioning method: default 'jennings2018_trivariate'
        n_digits (int): digits of the point index in the file names. Default: those of the number of points
        n_jobs (int): number of processes writing the files (see export_points)

    format is a text file with the following columns
    year month  day   hour  SW      LW      Sf         Rf     Ta  RH   Ua    Ps
    (yyyy) (mm) (dd) (hh)  (W/m2) (W/m2) (kg/m2/s) (kg/m2/s) (K) (RH 0-100) (m/s) (Pa)

    See README.md file from FSM source code for further details

    TODO: 
    - Check unit DONE jf
    - Check format is compatible with compiled model DONE jf
    - ensure ds.point_name.values always
    """
    if n_digits is None:
        n_digits = len(str(len(ds.point_name)))
    fnames = point_file_names(fname_format, ds.sizes['point_name'], n_digits)
    return export_points(ds, ['SW', 'LW', 't', 'p', 'q', 'tp', 'ws'], _write_fsm, fnames,
                         options={'tstep': _tstep_seconds(ds), 'snow_partition_method': snow_partition_method},
                         n_jobs=n_jobs, label='FSM files')


def _write_tc(fname, ds_pt, info, extra_pt):
    df = pd.DataFrame()
    df['datetime'] = pd.to_datetime(ds_pt.time.values)
    df['TA'] = np.round(ds_pt.t.values, 2)  # air temperature
    rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
    df['RH'] = rh * 100  # relative humidity
    df['PP'] = ds_pt.tp.values  # precipitation
    df['PRESS'] = ds_pt.p.values  # pressure
    df['SWIN'] = ds_pt.SW.values  # shortwave
    df['LWIN'] = ds_pt.LW.values  # longwave
    df['WS'] = ds_pt.ws.values  # wind speed
    df['WDIR'] = ds_pt.wd.values * (180 / np.pi)  # wind direction

    # convert units
    df.TA = df.TA - 273.15  # to °C (from Kelvin)
    df.PRESS = df.PRESS / 100  # to hPa (from Pascal)
    df.to_csv(fname, index=False, header=True, sep=',')


def to_TC(ds, fname_format='pt_*.tx', n_jobs=1):
    """
    Function to export data for the T&C model.
    https://hyd.ifu.ethz.ch/research-data-models/t-c.html

    Args:
        ds (dataset): downscaled_pts,
        fname_format (str pattern): output format of filename
        n_jobs (int): number of processes writing the files (see export_points)

    format is a text file with the following columns
    datetime    TA  P       PRESS RH  SWIN    LWIN    WS   WDIR
    (datetime) (°C) (mm/h)  (Pa)  (%) (W/m2)  (W/m2)  (m/s) (°)

    """

    n_digits = len(str(len(ds.point_name))) + 1
    fnames = point_file_names(fname_format, ds.sizes['point_name'], n_digits)
    return export_points(ds, ['t', 'p', 'q', 'tp', 'SW', 'LW', 'ws', 'wd'], _write_tc, fnames, n_jobs=n_jobs,
                         label='T&C files')


MICROMET_HEADER = ('year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     '
                   'precip\n(yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   '
                   '(deg)    (mm/dt)\n')


def _write_micromet(fname, ds_pt, info, extra_pt, na_values, headers):
    df = pd.DataFrame()
    df['year'] = pd.to_datetime(ds_pt.time.values).year
    df['mo'] = pd.to_datetime(ds_pt.time.values).month
    df['dy'] = pd.to_datetime(ds_pt.time.values).day
    df['hr'] = pd.to_datetime(ds_pt.time.values).hour
    df['stn_id'] = info['stn_id']
    df['easting'] = np.round(info['x'], 2)
    df['northing'] = np.round(info['y'], 2)
    df['elevation'] = np.round(info['elevation'], 2)
    df['Tair'] = np.round(ds_pt.t.values - 273.15, 2)
    df['RH'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values) * 100
    df['speed'] = ds_pt.ws.values
    df['dir'] = np.round(np.rad2deg(ds_pt.wd.values), 1)
    df['precip'] = ds_pt.tp.values
    with open(fname, 'w') as f:
        if headers:
            f.write(MICROMET_HEADER + '\n')
        df.to_csv(f, index=False, header=False, sep=' ', na_rep=na_values)


def to_micromet_single_station(ds,
                               df_pts,
                               fname_format='Snowmodel_pt_*.csv',
                               na_values=-9999,
                               headers=False,
                               n_jobs=1):
    """
    Function to export TopoScale output in the format for Listn's Snowmodel (using Micromet). One CSV file per point_name

    Args:
        ds (dataset): TopoPyScale xarray dataset, downscaled product
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_name is inserted where * is
        na_values (int): na_value default
        headers (bool): add headers to file
        n_jobs (int): number of processes writing the files (see export_points)


    Example of the compatible format for micromet (headers should be removed to run the model:

     year   mo   dy    hr     stn_id  easting  northing  elevation   Tair     RH     speed    dir     precip
    (yyyy) (mm) (dd) (hh.hh) (number)   (m)       (m)      (m)        (C)    (%)     (m/s)   (deg)    (mm/dt)

     2002   10    1   12.00    101   426340.0  4411238.0  3598.0     0.92    57.77     4.80   238.29 -9999.00
     2002   10    2   12.00    101   426340.0  4411238.0  3598.0    -3.02    78.77 -9999.00 -9999.00 -9999.00
     2002   10    8   12.00    101   426340.0  4411238.0  3598.0    -5.02    88.77 -9999.00 -9999.00 -9999.00
    """

    n_digits = len(str(len(ds.point_name))) + 1
    fnames = point_file_names(fname_format, ds.sizes['point_name'], n_digits)
    # stn_id is the index of the point, its position in df_pts
    info = [{'stn_id': i, 'x': df_pts.x.iloc[i], 'y': df_pts.y.iloc[i], 'elevation': df_pts.elevation.iloc[i]}
            for i in range(len(fnames))]
    return export_points(ds, ['t', 'p', 'q', 'ws', 'wd', 'tp'], _write_micromet, fnames, info=info,
                         options={'na_values': na_values, 'headers': headers}, n_jobs=n_jobs,
                         label='SnowModel files')


def _write_crocus(fname, ds_pt, row, extra_pt, tstep, scale_precip, snow_partition_method, climate_dataset_name,
                  project_author, ver_dict):
    df = pd.DataFrame()
    df['time'] = ds_pt.time.values
    df['Tair'] = ds_pt.t.values
    df['DIR_SWdown'] = ds_pt.SW.values
    df['LWdown'] = ds_pt.LW.values
    df['PSurf'] = ds_pt.p.values
    df['HUMREL'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values) * 100
    df['xwind'] = ds_pt.u.values
    df['ywind'] = ds_pt.v.values
    df['precip'] = ds_pt.tp.values / tstep * scale_precip  # mm per time step to mm/s
    rh = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
    df['Rainf'], df['Snowf'] = mu.partition_snow(df.precip, ds_pt.t.values, rh * 100, ds_pt.p.values,
                                                 method=snow_partition_method)

    # Derive variables: Q- humidity, WD - wind direction (deg), and WS
    df['Qair'] = ds_pt.q.values
    df['Wind'], df['Wind_DIR'] = ds_pt.ws.values, np.rad2deg(ds_pt.wd.values)
    df['NEB'] = np.zeros(df.shape[0])
    df['CO2air'] = np.zeros(df.shape[0])
    df['SCA_SWdown'] = np.zeros(df.shape[0])

    df.drop(columns=['precip', 'xwind', 'ywind'], inplace=True)
    ds_pt.close()

    # df = df.loc[(df.time>=pd.Timestamp(start)) & (df.time<pd.Timestamp(end)+pd.Timedelta('1D'))]

    # Build netcdf forcing file for CROCUS
    fo = xr.Dataset.from_dataframe(df.set_index('time'))
    fo = df.set_index(['time']).to_xarray()
    fo.Tair.attrs = {'units': 'K', 'standard_name': 'Tair', 'long_name': 'Near Surface Air Temperature',
                     '_FillValue': -9999999.0}
    fo.Qair.attrs = {'units': 'kg/kg', 'standard_name': 'Qair', 'long_name': 'Near Surface Specific Humidity',
                     '_FillValue': -9999999.0}
    fo.HUMREL.attrs = {'units': '%', 'standard_name': 'HUMREL', 'long_name': 'Relative Humidity',
                       '_FillValue': -9999999.0}
    fo.Wind.attrs = {'units': 'm/s', 'standard_name': 'Wind', 'long_name': 'Wind Speed', '_FillValue': -9999999.0}
    fo.Wind_DIR.attrs = {'units': 'deg', 'standard_name': 'Wind_DIR', 'long_name': 'Wind Direction',
                         '_FillValue': -9999999.0}
    fo.Rainf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Rainf', 'long_name': 'Rainfall Rate',
                      '_FillValue': -9999999.0}
    fo.Snowf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Snowf', 'long_name': 'Snowfall Rate',
                      '_FillValue': -9999999.0}
    fo.DIR_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'DIR_SWdown',
                           'long_name': 'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
    fo.LWdown.attrs = {'units': 'W/m2', 'standard_name': 'LWdown',
                       'long_name': 'Surface Incident Longtwave Radiation', '_FillValue': -9999999.0}
    fo.PSurf.attrs = {'units': 'Pa', 'standard_name': 'PSurf', 'long_name': 'Surface Pressure',
                      '_FillValue': -9999999.0}
    fo.SCA_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'SCA_SWdown',
                           'long_name': 'Surface Incident Diffuse Shortwave Radiation', '_FillValue': -9999999.0}
    fo.CO2air.attrs = {'units': 'kg/m3', 'standard_name': 'CO2air', 'long_name': 'Near Surface CO2 Concentration',
                       '_FillValue': -9999999.0}
    fo.NEB.attrs = {'units': 'between 0 and 1', 'standard_name': 'NEB', 'long_name': 'Nebulosity',
                    '_FillValue': -9999999.0}

    fo.attrs = {'title': 'Forcing for SURFEX CROCUS',
                'source': 'data from {} downscaled with TopoPyScale ready for CROCUS'.format(climate_dataset_name),
                'creator_name': 'Dataset created by {}'.format(project_author),
                'package_version': ver_dict.get('package_version'),
                'git_commit': ver_dict.get('git_commit'),
                'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

    fo['ZS'] = np.float64(row.elevation)
    fo['aspect'] = np.float64(np.rad2deg(row.aspect))
    fo['slope'] = np.float64(np.rad2deg(row.slope))
    fo['massif_number'] = np.float64(0)
    fo['LAT'] = row.latitude
    fo['LON'] = row.longitude
    fo.LAT.attrs = {'units': 'degrees_north', 'standard_name': 'LAT', 'long_name': 'latitude',
                    '_FillValue': -9999999.0}
    fo.LON.attrs = {'units': 'degrees_east', 'standard_name': 'LON', 'long_name': 'longitude',
                    '_FillValue': -9999999.0}
    fo['ZREF'] = np.float64(2)
    fo['UREF'] = np.float64(10)
    fo.ZREF.attrs = {'units': 'm', 'standard_name': 'ZREF', 'long_name': 'Reference_Height',
                     '_FillValue': -9999999.0}
    fo.UREF.attrs = {'units': 'm', 'standard_name': 'UREF', 'long_name': 'Reference_Height_for_Wind',
                     '_FillValue': -9999999.0}
    fo.aspect.attrs = {'units': 'degree from north', 'standard_name': 'aspect', 'long_name': 'slope aspect',
                       '_FillValue': -9999999.0}
    fo.slope.attrs = {'units': 'degrees from horizontal', 'standard_name': 'slope', 'long_name': 'slope angle',
                      '_FillValue': -9999999.0}
    fo.ZS.attrs = {'units': 'm', 'standard_name': 'ZS', 'long_name': 'altitude', '_FillValue': -9999999.0}
    fo.massif_number.attrs = {'units': '', 'standard_name': 'massif_number',
                              'long_name': 'SAFRAN massif number in SOPRANO world', '_FillValue': -9999999.0}
    fo.time.attrs = {'standard_name': 'time', 'long_name': 'time', '_FillValue': -9999999.0}
    fo['FRC_TIME_STP'] = np.float64(3600)
    fo.FRC_TIME_STP.attrs = {'units': 's', 'standard_name': 'FRC_TIME_STP', 'long_name': 'Forcing_Time_Step',
                             '_FillValue': -9999999.0}
    fo['FORCE_TIME_STEP'] = np.float64(3600)
    fo.FORCE_TIME_STEP.attrs = {'units': 's', 'standard_name': 'FORCE_TIME_STEP', 'long_name': 'Forcing_Time_Step',
                                '_FillValue': -9999999.0}

    # one point: time-dependent and point variables on (time, Number_of_points), as SURFEX reads them
    scalars = ['FRC_TIME_STP', 'FORCE_TIME_STEP']
    fo = xr.merge([fo.drop_vars(scalars).expand_dims('Number_of_points'), fo[scalars]], combine_attrs='override')
    fo = fo.transpose()

    fo.to_netcdf(fname, mode='w', format='NETCDF3_CLASSIC', unlimited_dims={'time': True},
                 encoding={'time': {'dtype': 'float64', 'calendar': 'standard'},
                           'Tair': {'dtype': 'float64'},
                           'Qair': {'dtype': 'float64'},
                           'Snowf': {'dtype': 'float64'},
                           'Rainf': {'dtype': 'float64'},
                           'DIR_SWdown': {'dtype': 'float64'},
                           'LWdown': {'dtype': 'float64'},
                           'PSurf': {'dtype': 'float64'},
                           'HUMREL': {'dtype': 'float64'},
                           'Wind': {'dtype': 'float64'},
                           'Wind_DIR': {'dtype': 'float64'},
                           'CO2air': {'dtype': 'float64'},
                           'NEB': {'dtype': 'float64'},
                           'SCA_SWdown': {'dtype': 'float64'},
                           'ZREF': {'dtype': 'float64'},
                           'UREF': {'dtype': 'float64'}})


def to_crocus(ds,
              df_pts,
              fname_format='CROCUS_pt_*.nc',
              scale_precip=1,
              climate_dataset_name='ERA5',
              project_author='S. Filhol',
              snow_partition_method='continuous',
              n_jobs=1):
    """
    Functiont to export toposcale output to CROCUS netcdf format. Generates one file per point_name

    Args:
        ds (dataset): Toposcale downscaled dataset.
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_name is inserted where * is
        scale_precip (float): scaling factor to apply on precipitation. Default is 1
        climate_dataset_name (str): name of original climate dataset. Default 'ERA5',
        project_author (str): name of project author(s)
        snow_partition_method (str): snow/rain partitioning method: default 'jennings2018_trivariate'
        n_jobs (int): number of processes writing the files (see export_points)
    """
    # one file per point, named by the index of the point, with the digits of the largest point number
    names = ds.point_name.values
    numeric = all(str(n).isdigit() for n in names)
    n_digits = len(str(max(int(n) for n in names))) if numeric else len(str(len(names)))
    fnames = point_file_names(fname_format, len(names), n_digits)
    options = {'tstep': _tstep_seconds(ds), 'scale_precip': scale_precip, 'snow_partition_method': snow_partition_method,
               'climate_dataset_name': climate_dataset_name, 'project_author': project_author,
               'ver_dict': tu.get_versionning()}
    return export_points(ds, ['t', 'SW', 'LW', 'p', 'q', 'u', 'v', 'tp', 'ws', 'wd'], _write_crocus, fnames,
                         info=[df_pts.iloc[i] for i in range(len(names))], options=options, n_jobs=n_jobs,
                         label='CROCUS files')


def to_surfex(ds,
              df_pts,
              fname_format='FORCING*.nc',
              scale_precip=1,
              climate_dataset_name='ERA5',
              project_author='S. Filhol',
              snow_partition_method='continuous',
              year_start='2023-08-01 00:00:00',
              year_end='2024-06-30 23:00:00'):
    """
    Function to export toposcale output to Surfex forcing netcdf format. Generates one file per year

    Args:
        ds (dataset): Toposcale downscaled dataset.
        df_pts (dataframe): with point list info (x,y,elevation,slope,aspect,svf,...)
        fname_format (str): filename format. point_name is inserted where * is
        scale_precip (float): scaling factor to apply on precipitation. Default is 1
        climate_dataset_name (str): name of original climate dataset. Default 'ERA5',
        project_author (str): name of project author(s)
        snow_partition_method (str): snow/rain partitioning method: default 'jennings2018_trivariate'
        year_start (str): start of year for yearly Surfex forcing

    ToDo:
    - [x] split to yearly files,

    """
    # create one file per point_name
    n_digits = len(str(ds.point_name.max().values))
    ver_dict = tu.get_versionning()

    tvec_start, tvec_end = tu.time_vecs(start_date=year_start, end_date=year_end)
    
    for year_id, start in enumerate(tvec_start):
        fo = ds.sel(time=slice(start, tvec_end[year_id])).copy()
        start_str = start.strftime('%Y%m%d%H')
        end_str = tvec_end[year_id].strftime('%Y%m%d%H')
        foutput = f"{str(fname_format).split('*')[0]}_{start_str}_{end_str}.nc"
        fo = fo.rename_dims({'point_name':'Number_of_points'})
        fo = fo.rename({'t':'Tair', 
                              'SW':'DIR_SWdown', 
                              'LW':'LWdown',
                              'p':'PSurf',
                              'q':'Qair',
                              'ws':'Wind'})
        fo['Wind_DIR'] = fo.wd * 180 / np.pi
        fo['HUMREL'] = (('Number_of_points','time'), mu.q_2_rh(fo.Tair.values, fo.PSurf.values, fo.Qair.values) * 100)
        fo['precip'] = (('Number_of_points','time'), fo.tp.values / _tstep_seconds(ds) * scale_precip)  # mm per time step to mm/s
        rh = mu.q_2_rh(fo.Tair.values, fo.PSurf.values, fo.Qair.values)
        Rainf, Snowf = mu.partition_snow(fo.precip.values, fo.Tair.values, rh * 100, fo.PSurf.values,
                                                     method=snow_partition_method)
        fo['Rainf'] = (('Number_of_points','time'), Rainf)
        fo['Snowf'] = (('Number_of_points','time'), Snowf)
                
        fo['NEB'] = (('Number_of_points','time'), np.zeros(fo.Tair.shape))
        fo['CO2air'] = (('Number_of_points','time'), np.zeros(fo.Tair.shape))
        fo['SCA_SWdown'] = (('Number_of_points','time'), np.zeros(fo.Tair.shape))

        fo = fo.drop_vars(['precip', 'u', 'v', 'w', 'wd', 'vp', 'cse','SW_direct', 'cos_illumination', 'SW_diffuse', 'precip_lapse_rate', 'tp' ])

        fo.Tair.attrs = {'units': 'K', 'standard_name': 'Tair', 'long_name': 'Near Surface Air Temperature',
                         '_FillValue': -9999999.0}
        fo.Qair.attrs = {'units': 'kg/kg', 'standard_name': 'Qair', 'long_name': 'Near Surface Specific Humidity',
                         '_FillValue': -9999999.0}
        fo.HUMREL.attrs = {'units': '%', 'standard_name': 'HUMREL', 'long_name': 'Relative Humidity',
                           '_FillValue': -9999999.0}
        fo.Wind.attrs = {'units': 'm/s', 'standard_name': 'Wind', 'long_name': 'Wind Speed', '_FillValue': -9999999.0}
        fo.Wind_DIR.attrs = {'units': 'deg', 'standard_name': 'Wind_DIR', 'long_name': 'Wind Direction',
                             '_FillValue': -9999999.0}
        fo.Rainf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Rainf', 'long_name': 'Rainfall Rate',
                          '_FillValue': -9999999.0}
        fo.Snowf.attrs = {'units': 'kg/m2/s', 'standard_name': 'Snowf', 'long_name': 'Snowfall Rate',
                          '_FillValue': -9999999.0}
        fo.DIR_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'DIR_SWdown',
                               'long_name': 'Surface Incident Direct Shortwave Radiation', '_FillValue': -9999999.0}
        fo.LWdown.attrs = {'units': 'W/m2', 'standard_name': 'LWdown',
                           'long_name': 'Surface Incident Longtwave Radiation', '_FillValue': -9999999.0}
        fo.PSurf.attrs = {'units': 'Pa', 'standard_name': 'PSurf', 'long_name': 'Surface Pressure',
                          '_FillValue': -9999999.0}
        fo.SCA_SWdown.attrs = {'units': 'W/m2', 'standard_name': 'SCA_SWdown',
                               'long_name': 'Surface Incident Diffuse Shortwave Radiation', '_FillValue': -9999999.0}
        fo.CO2air.attrs = {'units': 'kg/m3', 'standard_name': 'CO2air', 'long_name': 'Near Surface CO2 Concentration',
                           '_FillValue': -9999999.0}
        fo.NEB.attrs = {'units': 'between 0 and 1', 'standard_name': 'NEB', 'long_name': 'Nebulosity',
                        '_FillValue': -9999999.0}

        fo.attrs = {'title': 'Forcing for SURFEX CROCUS',
                    'source': 'data from {} downscaled with TopoPyScale ready for CROCUS'.format(climate_dataset_name),
                    'creator_name': 'Dataset created by {}'.format(project_author),
                    'package_version': ver_dict.get('package_version'),
                    'git_commit': ver_dict.get('git_commit'),
                    'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                    'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

        ds_pts = xr.Dataset(df_pts).rename_dims({'dim_0':'Number_of_points'})
        fo['ZS'] = (('Number_of_points'), np.float64(ds_pts.elevation.sel(Number_of_points=fo.Number_of_points).values))
        fo['aspect'] = (('Number_of_points'), np.float64(np.rad2deg(ds_pts.aspect.sel(Number_of_points=fo.Number_of_points).values)))
        fo['slope'] = (('Number_of_points'), np.float64(np.rad2deg(ds_pts.slope.sel(Number_of_points=fo.Number_of_points).values)))
        fo['massif_number'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(0))
        fo['LAT'] = (('Number_of_points'), ds_pts.latitude.sel(Number_of_points=fo.Number_of_points).values)
        fo['LON'] = (('Number_of_points'), ds_pts.longitude.sel(Number_of_points=fo.Number_of_points).values)
        ds_pts=None
        fo.LAT.attrs = {'units': 'degrees_north', 'standard_name': 'LAT', 'long_name': 'latitude',
                        '_FillValue': -9999999.0}
        fo.LON.attrs = {'units': 'degrees_east', 'standard_name': 'LON', 'long_name': 'longitude',
                        '_FillValue': -9999999.0}
        fo['ZREF'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(2))
        fo['UREF'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(10))
        fo.ZREF.attrs = {'units': 'm', 'standard_name': 'ZREF', 'long_name': 'Reference_Height',
                         '_FillValue': -9999999.0}
        fo.UREF.attrs = {'units': 'm', 'standard_name': 'UREF', 'long_name': 'Reference_Height_for_Wind',
                         '_FillValue': -9999999.0}
        fo.aspect.attrs = {'units': 'degree from north', 'standard_name': 'aspect', 'long_name': 'slope aspect',
                           '_FillValue': -9999999.0}
        fo.slope.attrs = {'units': 'degrees from horizontal', 'standard_name': 'slope', 'long_name': 'slope angle',
                          '_FillValue': -9999999.0}
        fo.ZS.attrs = {'units': 'm', 'standard_name': 'ZS', 'long_name': 'altitude', '_FillValue': -9999999.0}
        fo.massif_number.attrs = {'units': '', 'standard_name': 'massif_number',
                                  'long_name': 'SAFRAN massif number in SOPRANO world', '_FillValue': -9999999.0}
        fo.time.attrs = {'standard_name': 'time', 'long_name': 'time', '_FillValue': -9999999.0}
        fo['FRC_TIME_STP'] = ((), np.float64(3600))
        fo.FRC_TIME_STP.attrs = {'units': 's', 'standard_name': 'FRC_TIME_STP', 'long_name': 'Forcing_Time_Step',
                                 '_FillValue': -9999999.0}
        #fo['FORCE_TIME_STEP'] = (('Number_of_points'), np.zeros(fo.ZS.shape) + np.float64(3600))
        #fo.FORCE_TIME_STEP.attrs = {'units': 's', 'standard_name': 'FORCE_TIME_STEP', 'long_name': 'Forcing_Time_Step',
        #                            '_FillValue': -9999999.0}

        fo = fo.transpose()

        fo.to_netcdf(foutput, mode='w', format='NETCDF3_CLASSIC', unlimited_dims={'time': True},
                     encoding={'time': {'dtype': 'float64', 'calendar': 'standard'},
                               'Tair': {'dtype': 'float64'},
                               'Qair': {'dtype': 'float64'},
                               'Snowf': {'dtype': 'float64'},
                               'Rainf': {'dtype': 'float64'},
                               'DIR_SWdown': {'dtype': 'float64'},
                               'LWdown': {'dtype': 'float64'},
                               'PSurf': {'dtype': 'float64'},
                               'HUMREL': {'dtype': 'float64'},
                               'Wind': {'dtype': 'float64'},
                               'Wind_DIR': {'dtype': 'float64'},
                               'CO2air': {'dtype': 'float64'},
                               'NEB': {'dtype': 'float64'},
                               'SCA_SWdown': {'dtype': 'float64'},
                               'ZREF': {'dtype': 'float64'},
                               'UREF': {'dtype': 'float64'}})
        print('---> File {} saved'.format(foutput))
        # clear memory
        fo = None














def _write_snowpack(fname, ds_pt, info, extra_pt):
    df = pd.DataFrame()
    df['timestamp'] = pd.DatetimeIndex(ds_pt.time.values).strftime('%Y-%m-%dT%H:%M:%S')  # SMET ISO format
    df['Tair'] = np.round(ds_pt.t.values, 2)
    df['RH'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values)
    df['VW'] = ds_pt.ws.values
    df['DW'] = np.rad2deg(ds_pt.wd.values)  # SMET expects degrees
    df['SW'] = ds_pt.SW.values
    df['LW'] = ds_pt.LW.values
    df['PSUM'] = ds_pt.tp.values
    df.to_csv(fname, index=False, header=False, sep=' ')


def to_snowpack(ds, fname_format='smet_pt_*.tx', n_jobs=1):
    """
    Function to export data for snowpack model as smet.
    https://models.slf.ch/docserver/meteoio/SMET_specifications.pdf

    Args:
        ds (dataset): downscaled_pts,
        fname_format (str pattern): output format of filename
        n_jobs (int): number of processes writing the files (see export_points)

    format is a text file with the following columns (coulumns can vary)

    SMET 1.1 ASCII
    [HEADER]
    station_id       = meteoc1
    station_name     = WFJ2
    latitude         = 46.829650
    longitude        = 9.809328
    altitude         = 2539.0
    easting          = 780851.861845
    northing         = 189232.420554
    epsg             = 21781
    nodata           = -999
    tz               = 0
    plot_unit        = time K - m/s ° W/m2 W/m2 kg/m2 -
    plot_description = time air_temperature relative_humidity wind_velocity wind_direction incoming_short_wave_radiation incoming_long_wave_radiation water_equivalent_precipitation_sum -
    plot_color       = 0x000000 0x8324A4 0x50CBDB 0x297E24 0x64DD78 0xF9CA25 0xD99521 0x2431A4 0xA0A0A0
    plot_min         = -999 253.15 0 0 0 0 150 0 -999
    plot_max         = -999 283.15 1 30 360 1400 400 20 -999
    fields           = timestamp TA RH VW DW ISWR ILWR PSUM PINT
    [DATA]
    2014-09-01T00:00:00   271.44   1.000    5.9   342     67    304  0.000    0.495
    2014-09-01T01:00:00   271.50   0.973    6.7   343    128    300  0.166    0.663
    2014-09-01T02:00:00   271.46   0.968    7.4   343    244    286  0.197    0.788
    2014-09-01T03:00:00   271.41   0.975    7.6   345    432    273  0.156    0.626
    2014-09-01T04:00:00   271.57   0.959    7.8   347    639    249  0.115    0.462
    2014-09-01T05:00:00   271.52   0.965    8.2   350    632    261  0.081    0.323

    """

    n_digits = len(str(len(ds.point_name))) + 1
    fnames = point_file_names(fname_format, ds.sizes['point_name'], n_digits)
    return export_points(ds, ['t', 'p', 'q', 'ws', 'wd', 'SW', 'LW', 'tp'], _write_snowpack, fnames, n_jobs=n_jobs,
                         label='SNOWPACK files')


GEOTOP_HEADER = 'Date, AirT, RelHum, WindS, WindDr, Swglob, Lwin, Iprec'


def _write_geotop(fname, ds_pt, info, extra_pt):
    df = pd.DataFrame()
    dt = pd.to_datetime(ds_pt.time.values)
    df['Date'] = dt.strftime("%d/%m/%Y %H:%M")
    df['AirT'] = np.round(ds_pt.t.values - 273.15, 2)
    df['RelHum'] = mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values) * 100
    df['WindS'] = ds_pt.ws.values
    df['WIndDr'] = ds_pt.wd.values * 180 / np.pi
    df['Swglob'] = ds_pt.SW.values
    df['Lwin'] = ds_pt.LW.values
    df['Iprec'] = ds_pt.tp.values
    with open(fname, 'w') as f:
        f.write(GEOTOP_HEADER + '\n')
        df.to_csv(f, index=False, header=False, sep=',', quoting=csv.QUOTE_NONE)


def to_geotop(ds, fname_format='geotop_pt_*.txt', n_jobs=1):
    """
    Function to export data for snowpack model as smet.
    https://models.slf.ch/docserver/meteoio/SMET_specifications.pdf

    Args:
        ds (dataset): downscaled_pts,
        fname_format (str pattern): output format of filename
        n_jobs (int): number of processes writing the files (see export_points)


    Date format = DD/MM/YYYY hh:mm
    Air temperature = Degree Celsius
    Relative Humidity = %
    Radiations components such SWin, SWout, LWin = W/m2
    Precipitation = mm/hr
    Wind speed = m/s
    Wind direction = degree
    Air pressure = mbar

    The format of the file is in *.txt format.



    format is a text file with the following columns (coulumns can vary)

    Date,AirT,WindS,WindDr,RelHum,Swglob,Swout,Lwin,Lwout,Iprec,AirPressure
    01/09/2015 00:00,1.5,0.7,165,59,0,0,223.9,315.5,0,581.02897
    01/09/2015 01:00,1.2,0.6,270,59,0,0,261.8,319,0,581.02897
    01/09/2015 02:00,0.8,0.5,152,61,0,0,229.9,312.2,0,581.02897
    01/09/2015 03:00,0.8,0.5,270,60,0,0,226.1,310.8,0,581.02897
    01/09/2015 04:00,0.3,0.8,68,60,0,0,215.3,309.6,0,581.02897
    01/09/2015 05:00,0.2,0.6,270,62,0,0,230.2,309.1,0,581.02897
    01/09/2015 06:00,0.3,0.6,35,62,0,0,222.8,306.7,0,581.02897
    01/09/2015 07:00,-0.2,0.3,270,65,0,0,210,305.5,0,581.02897
    01/09/2015 08:00,0,0.6,52,59,114,23,218.3,312.3,0,581.02897
    01/09/2015 09:00,1.9,1.5,176,57,173,35,220.2,322.8,0,581.02897
    01/09/2015 10:00,3.4,1.9,183,47,331,67,245.8,372.6,0,581.02897


    """

    n_digits = len(str(len(ds.point_name))) + 1
    fnames = point_file_names(fname_format, ds.sizes['point_name'], n_digits)
    return export_points(ds, ['t', 'p', 'q', 'ws', 'wd', 'SW', 'LW', 'tp'], _write_geotop, fnames, n_jobs=n_jobs,
                         label='GEOtop files')
