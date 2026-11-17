"""
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- an improvement could be to first copmute horizons, and then SVF to avoid computing horizon twice
"""

import rasterio
from pyproj import Transformer
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import xarray as xr
from topocalc import gradient
from topocalc import viewf
from topocalc import horizon
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
import shutil
from TopoPyScale import topo_export as te
from pathlib import Path



def convert_epsg_pts(xs,ys, epsg_src=4326, epsg_tgt=3844):
    """
    Simple function to convert a list fo poitn from one projection to another oen using PyProj

    Args:
        xs (array): 1D array with X-coordinate expressed in the source EPSG
        ys (array): 1D array with Y-coordinate expressed in the source EPSG
        epsg_src (int): source projection EPSG code
        epsg_tgt (int): target projection EPSG code

    Returns: 
        array: Xs 1D arrays of the point coordinates expressed in the target projection
        array: Ys 1D arrays of the point coordinates expressed in the target projection
    """
    print('Convert coordinates from EPSG:{} to EPSG:{}'.format(epsg_src, epsg_tgt))
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:{}".format(epsg_tgt), always_xy=True)
    Xs, Ys = trans.transform(xs, ys)
    return Xs, Ys


def get_extent_latlon(dem_file, epsg_src):
    """
    Function to extract DEM extent in Lat/Lon

    Args:
        dem_file (str): path to DEM file (GeoTiFF)
        epsg_src (int): EPSG projection code

    Returns: 
        dict: extent in lat/lon, {latN, latS, lonW, lonE}
    """
    with rasterio.open(dem_file) as rf:
        xs, ys = [rf.bounds.left, rf.bounds.right], [rf.bounds.bottom, rf.bounds.top]
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:4326", always_xy=True)
    lons, lats = trans.transform(xs, ys)
    extent = {'latN': lats[1],
              'latS': lats[0],
              'lonW': lons[0],
              'lonE': lons[1]}
    return extent


def extract_pts_param(df_pts, ds_param, method='nearest'):
    """
    Function to sample DEM parameters for a list point. This is used as an alternative the the TopoSub method, to perform downscaling at selected locations (x,y)
    WARNING: the projection and coordiante system of the EDM and point coordinates MUST be the same!

    Args:
        df_pts (dataframe): list of points coordinates with coordiantes in (x,y).
        ds_param (dataset): dem parameters
        method (str): sampling method. Supported 'nearest', 'linear' interpolation, 'idw' interpolation (inverse-distance weighted)

    Returns: 
        dataframe: df_pts updated with new columns ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']
    """
    print('\n---> Extracting DEM parameters for the given list of point coordinates')
    # delete columns in case they already exist
    df_pts = df_pts.drop(['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf'], errors='ignore')
    # create columns, filled with 0
    df_pts[['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = 0.0

    if method == 'nearest':
        for row in df_pts.itertuples():
            d_mini = ds_param.sel(x=row.x, y=row.y, method='nearest')
            df_pts.loc[row.Index, ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = np.array((d_mini.elevation.values,
                                                                                                   d_mini.slope.values,
                                                                                                   d_mini.aspect.values,
                                                                                                   d_mini.aspect_cos,
                                                                                                   d_mini.aspect_sin,
                                                                                                   d_mini.svf.values)).astype('float64')
    elif method == 'idw' or method == 'linear':
        for row in df_pts.itertuples():
            ind_lat = np.abs(ds_param.y-row.y).argmin()
            ind_lon = np.abs(ds_param.x-row.x).argmin()
            ds_param_pt = ds_param.isel(y=[ind_lat-1, ind_lat, ind_lat+1], x=[ind_lon-1, ind_lon, ind_lon+1])
            Xs, Ys = np.meshgrid(ds_param_pt.x.values, ds_param_pt.y.values)

            if method == 'idw':
                dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
                idw = 1/(dist**2)
                weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1

            if method == 'linear':
                dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
                weights = dist / np.sum(dist)
            da_idw = xr.DataArray(data=weights,
                                  coords={
                                    "y": ds_param_pt.y.values,
                                    "x": ds_param_pt.x.values,
                              },
                              dims=["y", "x"]
                              )
            dw = xr.Dataset.weighted(ds_param_pt, da_idw)
            d_mini = dw.sum(['x', 'y'], keep_attrs=True)
            df_pts.loc[row.Index, ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = np.array((d_mini.elevation.values,
                                                                                                   d_mini.slope.values,
                                                                                                   d_mini.aspect.values,
                                                                                                   d_mini.aspect_cos,
                                                                                                   d_mini.aspect_sin,
                                                                                                 d_mini.svf.values))
    else:
        raise ValueError('ERROR: Method not implemented. Only nearest, linear or idw available')
    return df_pts


def compute_dem_complexity(elev, slope, weights={'slope': 0.5, 'elev_norm':0.3, 'curvature':0.2}):
    """
    Function to compute topographic complexity index. This index is an arbitray index considering 

    Args:
        elev (array): ndarray of elevations 
        slope (array): ndarray of slopes 
        weights (dict): dictionnary of the relative weights of slope vs normalized elevation vs curvature. Sum must be equal to 1

    Returns:
        array: complexity index 
    """
    elev_norm = (elev - np.min(elev)) / (np.max(elev) - np.min(elev))
    elev_smooth = gaussian_filter(elev, sigma=1)

    if sum(weights.values()) != 1:
        raise ValueError('Sum of weights must be equal to 1')

    curvature = np.abs(laplace(elev_smooth))
    complexity = (
            weights.get('slope') * (slope / np.max(slope)) +
            weights.get('elev_norm') * elev_norm +
            weights.get('curvature') * (curvature / np.max(curvature))
        )**1.5

    return complexity


def open_dem(dem_file):
    """
    Function to quickly open a DEM raster using xarray and removing the band dimension, combersome when dealing with DEM

    Args:
        dem_file (str): path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM)

    Returns:  
        dataset: x, y, elevation
    """
    tmp = xr.open_dataset(dem_file, engine='rasterio')
    ds = tmp.band_data.isel(band=0).to_dataset()
    ds = ds.rename({"band_data": 'elevation'})
    return ds



def compute_dem_param(dem_file, fname='ds_param.nc', project_directory=Path('./'), output_folder='outputs'):
    """
    Function to compute and derive DEM parameters: slope, aspect, sky view factor

    Args:
        dem_file (str): path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM)

    Returns:  
        dataset: x, y, elev, slope, aspect, svf

    """
    pdir = project_directory
    file_ds = pdir / output_folder / fname
    if file_ds.is_file():
        print(f'\n---> Dataset {fname} found.')
        try:
            ds = xr.open_dataset(file_ds)
            # Test if we can actually read the elevation data
            _ = ds.elevation.values
        except (RuntimeError, OSError) as e:
            if "NetCDF: HDF error" in str(e) or "filter returned failure" in str(e):
                print(f'\n---> Dataset {fname} corrupted (HDF/compression error). Trying h5netcdf backend...')
                try:
                    ds = xr.open_dataset(file_ds, engine='h5netcdf')
                    _ = ds.elevation.values
                except Exception:
                    print(f'\n---> h5netcdf backend failed. Regenerating dataset from DEM...')
                    if Path(dem_file).is_file():
                        ds = open_dem(dem_file)
                    else:
                        raise ValueError(f'ERROR: Dataset corrupted and no DEM available to regenerate')
            else:
                raise e

    else:
        if Path(dem_file).is_file():
            print(f'\n---> No {fname} Dataset found. DEM {dem_file} available.')
            ds = open_dem(dem_file)

        else:
            raise ValueError(f'ERROR: No DEM or dataset available')

    var_in = list(ds.variables.keys())
    print('\n---> Extracting DEM parameters (slope, aspect, svf)')
    dx = ds.x.diff('x').median().values
    dy = ds.y.diff('y').median().values
    
    # Safely access elevation data with error handling
    try:
        dem_arr = ds.elevation.values
    except (RuntimeError, OSError) as e:
        if "NetCDF: HDF error" in str(e) or "filter returned failure" in str(e):
            print(f'---> Error reading elevation data: {str(e)}')
            print('---> Attempting to load elevation data with different method...')
            try:
                # Try loading the data chunk by chunk or using compute()
                dem_arr = ds.elevation.load().values
            except Exception:
                # If all else fails, regenerate from DEM file
                print('---> All methods failed. Regenerating from original DEM...')
                if Path(dem_file).is_file():
                    ds_new = open_dem(dem_file)
                    dem_arr = ds_new.elevation.values
                    ds = ds_new
                    var_in = list(ds.variables.keys())
                else:
                    raise ValueError(f'ERROR: Cannot read elevation data and no DEM file available')
        else:
            raise e

    if ('slope' not in var_in) or ('aspect' not in var_in):
        print('Computing slope and aspect ...')
        slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
        ds['slope'] = (["y", "x"], slope)
        ds['aspect'] = (["y", "x"], np.deg2rad(aspect))
        if 'aspect_cos' not in var_in:
            ds['aspect_cos'] = (["y", "x"], np.cos(np.deg2rad(aspect)))
        if 'aspect_sin' not in var_in:
            ds['aspect_sin'] = (["y", "x"], np.sin(np.deg2rad(aspect)))

    if 'svf' not in var_in:
        print('Computing svf ...')
        svf = viewf.viewf(np.double(dem_arr), dx)[0]
        ds['svf'] = (["y", "x"], svf)

    ds.attrs = dict(description="DEM input parameters to TopoSub",
                   author="TopoPyScale, https://github.com/ArcticSnow/TopoPyScale")
    ds.x.attrs = {'units': 'm'}
    ds.y.attrs = {'units': 'm'}
    ds.elevation.attrs = {'units': 'm'}
    ds.slope.attrs = {'units': 'rad'}
    ds.aspect.attrs = {'units': 'rad'}
    ds.aspect_cos.attrs = {'units': 'cosinus'}
    ds.aspect_sin.attrs = {'units': 'sinus'}
    ds.svf.attrs = {'units': 'ratio', 'standard_name': 'svf', 'long_name': 'Sky view factor'}

    if file_ds.is_file():
        te.to_netcdf(ds, fname=pdir / output_folder / 'tmp' / fname)
        ds = None
        shutil.move(pdir / output_folder / 'tmp' /fname, file_ds)
        ds = xr.open_dataset(file_ds)
    else:
        te.to_netcdf(ds, fname=file_ds)

    return ds


def compute_horizon(dem_file, azimuth_inc=30, num_threads=None, fname='da_horizon.nc', output_directory=Path('./outputs')):
    """
    Horizon angles of every pixel of the DEM, with topocalc (e.g. for maps).

    topocalc follows lines of sight that are staircases stepping at the same columns (or rows) for all
    pixels. For azimuths a few degrees off the grid axes, the pixels just before a step read their nearest
    terrain from the adjacent row, which leaves streaks in the horizon maps. Topoclass uses
    compute_horizon_points() instead, which traces a straight ray from each point.

    Args:
        dem_file (str): path and filename of the dem
        azimuth_inc (int): angle increment to compute horizons at, in Degrees [0-359]
        num_threads (int): number of threads to parallize on

    Returns: 
        dataarray: all horizon angles for x,y,azimuth coordinates
         
    """
    print('\n---> Computing horizons with {} degree increments'.format(azimuth_inc))
    ds = open_dem(dem_file)
    dx = ds.x.diff('x').median().values

    azimuth = np.arange(-180 + azimuth_inc / 2, 180, azimuth_inc) # center the azimuth in middle of the bin
    arr_val = np.empty((azimuth.shape[0], ds.elevation.shape[0], ds.elevation.shape[1]))

    if num_threads is None:
        pool = ThreadPool(mproc.cpu_count() - 2)
    else:
        pool = ThreadPool(num_threads)

    elev = []
    dxs = []
    for azi in azimuth:
        elev.append(ds.elevation.values)
        dxs.append(dx)

    arr = pool.starmap(horizon.horizon, zip(list(azimuth),
                                                   elev,
                                                   dxs))
    pool.close()
    pool.join()

    for i, a in enumerate(arr):
        arr_val[i,:,:] = a

    da = xr.DataArray(data=np.pi/2 - np.arccos(arr_val),
                      coords={
                          "y": ds.y.values,
                          "x": ds.x.values,
                          "azimuth": azimuth
                      },
                      dims=["azimuth", "y", "x"],
                      name='horizon',
                      attrs={
                          'long_name': 'Horizon angles',
                          'units':'rad'
                      }
                      )
    #te.to_netcdf(da.to_dataset(), fname=output_directory / fname)
    da.to_dataset().to_netcdf(output_directory / fname)
    return da




def _bilinear(z, r, c):
    """Bilinear interpolation of the 2D array z at fractional row and column indices within the grid."""
    ny, nx = z.shape
    i = np.clip(np.floor(r).astype(np.intp), 0, ny - 2)
    j = np.clip(np.floor(c).astype(np.intp), 0, nx - 2)
    wr, wc = r - i, c - j
    return ((1 - wr) * ((1 - wc) * z[i, j] + wc * z[i, j + 1])
            + wr * ((1 - wc) * z[i + 1, j] + wc * z[i + 1, j + 1]))


def horizon_rays(elevation, x, y, px, py, azimuths, max_rays=2_000_000, n_threads=1, prune=True):
    """
    Horizon angles of points along straight rays, for a list of azimuths.

    From each point, a ray is followed in each direction one cell at a time along the major grid axis, the
    DEM being interpolated bilinearly at each sample. The horizon is the steepest elevation angle to a
    sample, or 0 if no sample is higher than the point (as topocalc). The observer is on the DEM surface,
    interpolated at the point. Rays stop at the edge of the DEM, or as soon as no sample further away can
    be steeper than the horizon found ((max DEM elevation - point elevation) / distance), which does not
    change the result. No-data cells (NaN) never form a horizon.

    Args:
        elevation (2D array): DEM (y, x)
        x, y (1D arrays): coordinates of the DEM columns and rows (regular grid, either orientation)
        px, py (1D arrays): coordinates of the points, in the projection of the DEM
        azimuths (1D array): directions [deg]: 0 = south, 90 = east, -90 = west, +-180 = north
        max_rays (int): number of rays (azimuths x points) in memory at once
        n_threads (int): number of threads tracing blocks of points. The result does not depend on it
        prune (bool): stop the rays when no further sample can be steeper. Only for testing

    Returns:
        array: horizon elevation angles [rad], (azimuth, point)
    """
    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError('elevation must be a 2D array of at least 2 x 2 cells')
    ny, nx = z.shape
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = x[1] - x[0], y[1] - y[0]
    if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dy)):
        raise ValueError('the DEM must be a regular grid')

    c = (np.asarray(px, dtype=float) - x[0]) / dx
    r = (np.asarray(py, dtype=float) - y[0]) / dy
    outside = (c < -0.5) | (c > nx - 0.5) | (r < -0.5) | (r > ny - 0.5)
    if outside.any():
        raise ValueError(f'{outside.sum()} points are outside the DEM, e.g. x={np.asarray(px)[outside][0]}, '
                         f'y={np.asarray(py)[outside][0]}')
    c, r = np.clip(c, 0, nx - 1), np.clip(r, 0, ny - 1)

    # elevation of the observers: bilinear, or the nearest cell next to no data
    z0 = _bilinear(z, r, c)
    near = np.isnan(z0)
    z0[near] = z[np.rint(r[near]).astype(np.intp), np.rint(c[near]).astype(np.intp)]
    if np.isnan(z0).any():
        print(f'-> WARNING: {np.isnan(z0).sum()} points on no-data cells of the DEM get a horizon of 0')

    zmax = np.nanmax(z)
    zf = np.where(np.isnan(z), np.nanmin(z) - 1e4, z)    # no data never forms a horizon

    az = np.deg2rad(np.asarray(azimuths, dtype=float))
    ux, uy = np.sin(az), -np.cos(az)                      # map direction (east, north)
    ux[np.abs(ux) < 1e-12], uy[np.abs(uy) < 1e-12] = 0., 0.  # exact grid axes (sin(pi) is not 0)
    ds = 1. / np.maximum(np.abs(ux / dx), np.abs(uy / dy))  # one cell along the major axis per step
    dc, dr = ux * ds / dx, uy * ds / dy

    n_az, n_pt = az.size, r.size
    n_threads = max(1, int(n_threads or 1))

    def trace(pts):
        ia = np.repeat(np.arange(n_az), pts.size)
        ip = np.tile(pts, n_az)
        r0, c0, zz0 = r[ip], c[ip], z0[ip]
        rdr, rdc, rds = dr[ia], dc[ia], ds[ia]
        out = np.zeros(ip.size)
        live = np.arange(ip.size)
        b = np.zeros(ip.size)
        s = 0
        while live.size:
            s += 1
            rr, cc = r0 + s * rdr, c0 + s * rdc
            inside = (rr >= -1e-9) & (rr <= ny - 1 + 1e-9) & (cc >= -1e-9) & (cc <= nx - 1 + 1e-9)
            dist = s * rds
            zs = _bilinear(zf, rr, cc)   # indices are clipped: samples outside are masked below
            b = np.fmax(b, np.where(inside, (zs - zz0) / dist, -np.inf))
            keep = inside & ((zmax - zz0) / dist > b) if prune else inside
            if not keep.all():
                out[live[~keep]] = b[~keep]
                live = live[keep]
                r0, c0, zz0, rdr, rdc, rds, b = (a[keep] for a in (r0, c0, zz0, rdr, rdc, rds, b))
        return ia, ip, out

    # blocks of points, small enough to bound memory and to keep the threads busy
    block = max(1, min(max_rays // (max(n_az, 1) * n_threads), -(-n_pt // n_threads)))
    blocks = [np.arange(p0, min(p0 + block, n_pt)) for p0 in range(0, n_pt, block)]
    best = np.zeros((n_az, n_pt))
    if n_threads > 1 and len(blocks) > 1:
        with ThreadPool(n_threads) as pool:
            results = pool.map(trace, blocks)
    else:
        results = map(trace, blocks)
    for ia, ip, out in results:
        best[ia, ip] = out
    return np.arctan(best)


def compute_horizon_points(dem_file, df_points, azimuth_inc=10, fname='da_horizon.nc',
                           output_directory=Path('./outputs'), n_threads=1):
    """
    Horizon angles of a list of points, along straight rays from each point (see horizon_rays).

    Args:
        dem_file (str): path and filename of the dem
        df_points (dataframe): points with columns x, y (projection of the DEM) and point_name
        azimuth_inc (int): angle increment of the azimuths [deg], centred in their bins as compute_horizon()
        fname (str): filename of the netcdf file written in output_directory. None to skip writing
        output_directory (Path): output directory
        n_threads (int): number of threads

    Returns:
        dataarray: horizon angles [rad], (azimuth, point_name), with the point coordinates x, y
    """
    df_points = df_points.reset_index(drop=True)
    print(f'\n---> Computing horizons of {len(df_points)} points with {azimuth_inc} degree increments')
    ds = open_dem(dem_file)
    azimuth = np.arange(-180 + azimuth_inc / 2, 180, azimuth_inc)  # center the azimuth in middle of the bin
    hor = horizon_rays(ds.elevation.values, ds.x.values, ds.y.values, df_points.x.values, df_points.y.values,
                       azimuth, n_threads=n_threads)
    names = df_points.point_name if 'point_name' in df_points else df_points.index
    names = np.array([str(v) for v in names])
    if np.unique(names).size != names.size:
        raise ValueError('point names must be unique')
    da = xr.DataArray(hor, dims=['azimuth', 'point_name'], name='horizon',
                      coords={'azimuth': azimuth, 'point_name': names,
                              'x': ('point_name', df_points.x.values.astype(float)),
                              'y': ('point_name', df_points.y.values.astype(float))},
                      attrs={'long_name': 'Horizon angles', 'units': 'rad',
                             'method': 'straight rays from each point, bilinear interpolation of the DEM',
                             'azimuth_convention': 'deg, 0 = south, 90 = east, +-180 = north'})
    da.azimuth.attrs = {'units': 'degree', 'long_name': 'centre of the azimuth bin'}
    if fname is not None:
        da.to_dataset().to_netcdf(Path(output_directory) / fname)
    return da
