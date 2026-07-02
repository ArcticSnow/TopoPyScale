"""
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- an improvement could be to first compute horizons, and then SVF to avoid computing horizon twice
"""

import rasterio
from pyproj import Transformer
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import xarray as xr
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
import shutil
from TopoPyScale import topo_export as te
from pathlib import Path

# Conditional imports for terrain computation backends
try:
    import horayzon
    HORAYZON_AVAILABLE = True
    print("---> HORAYZON available - using high-performance terrain computations")
except ImportError:
    HORAYZON_AVAILABLE = False
    try:
        from topocalc import gradient
        from topocalc import viewf
        from topocalc import horizon
        TOPOCALC_AVAILABLE = True
        print("---> Using topocalc for terrain computations")
    except ImportError:
        TOPOCALC_AVAILABLE = False
        raise ImportError("Neither HORAYZON nor topocalc available. Install one of them:\n"
                         "  HORAYZON: conda install -c conda-forge embree tbb-devel && pip install git+https://github.com/ChristianSteger/HORAYZON.git\n"
                         "  topocalc: pip install git+https://github.com/ArcticSnow/topocalc")

def _compute_slope_aspect_horayzon(dem_arr, dx, dy):
    """
    Compute slope and aspect using HORAYZON backend.
    
    HORAYZON uses meteorological aspect convention (0° = North, clockwise)
    while topocalc uses mathematical convention (0° = East, counter-clockwise).
    We convert HORAYZON output to match topocalc convention for compatibility.
    """
    # Create coordinates for HORAYZON
    ny, nx = dem_arr.shape
    x_1d = np.arange(nx, dtype=np.float64) * dx
    y_1d = np.arange(ny, dtype=np.float64) * dy
    
    # HORAYZON expects (longitude, latitude) order for geographic data
    # For projected data, we use (x, y) directly
    slope_rad, aspect_rad = horayzon.topo_param.slope_aspect_numba(
        elevation=dem_arr.astype(np.float64),
        x_axis=x_1d,
        y_axis=y_1d,
        pixel_res=dx
    )
    
    # Convert HORAYZON aspect convention to topocalc convention
    # HORAYZON: 0° = North, clockwise (meteorological)
    # topocalc: 0° = East, counter-clockwise (mathematical)
    # Conversion: topocalc_aspect = 90° - horayzon_aspect (in degrees)
    aspect_deg_horayzon = np.rad2deg(aspect_rad)
    aspect_deg_topocalc = (90.0 - aspect_deg_horayzon) % 360.0
    
    slope_deg = np.rad2deg(slope_rad)
    
    return slope_deg, aspect_deg_topocalc

def _compute_slope_aspect_topocalc(dem_arr, dx, dy):
    """Compute slope and aspect using topocalc backend."""
    return gradient.gradient_d8(dem_arr, dx, dy)

def _compute_svf_horayzon(dem_arr, dx, azimuth_inc=5):
    """
    Compute sky view factor using HORAYZON backend with optimized parameters.
    
    HORAYZON's SVF is more accurate than topocalc and supports parallelization.
    """
    ny, nx = dem_arr.shape
    x_1d = np.arange(nx, dtype=np.float64) * dx
    y_1d = np.arange(ny, dtype=np.float64) * dy
    
    # HORAYZON SVF computation with high accuracy
    svf = horayzon.topo_param.sky_view_factor(
        elevation=dem_arr.astype(np.float64),
        x_axis=x_1d,
        y_axis=y_1d,
        azimuth_increment=azimuth_inc,  # Higher resolution than default
        dist_search=np.inf,  # Search entire domain
        ray_org_elev=0.1  # Ray origin height above surface
    )
    
    return svf

def _compute_svf_topocalc(dem_arr, dx):
    """Compute sky view factor using topocalc backend."""
    return viewf.viewf(np.double(dem_arr), dx)[0]

def _compute_horizon_horayzon(dem_arr, dx, azimuth, num_threads=None):
    """
    Compute horizon angles using HORAYZON backend.
    
    HORAYZON provides superior performance and handles large DEMs efficiently.
    """
    ny, nx = dem_arr.shape  
    x_1d = np.arange(nx, dtype=np.float64) * dx
    y_1d = np.arange(ny, dtype=np.float64) * dy
    
    # Set number of threads (HORAYZON handles this automatically if None)
    if num_threads is None:
        num_threads = mproc.cpu_count()
    
    # HORAYZON uses degrees for azimuth input
    azimuth_deg = np.degrees(azimuth) if np.max(azimuth) <= 2*np.pi else azimuth
    
    # Compute horizon elevation angles for all azimuths at once
    horizon_angles = horayzon.horizon.horizon_gridded(
        elevation=dem_arr.astype(np.float64),
        x_axis=x_1d,
        y_axis=y_1d, 
        azimuth=azimuth_deg,
        dist_search=np.inf,
        ray_org_elev=0.1,
        geom='planar',  # Use planar geometry for projected coordinates
        hori_acc=0.25,  # High accuracy 
        num_threads=num_threads
    )
    
    # HORAYZON returns elevation angles, convert to match topocalc convention
    # topocalc uses: horizon_angle = π/2 - arccos(horizon_cos_elevation)
    # HORAYZON returns elevation angles directly
    horizon_cos_elevation = np.cos(np.deg2rad(horizon_angles))
    
    return horizon_cos_elevation

def _compute_horizon_topocalc(dem_arr, dx, azimuth_single):
    """Compute horizon for a single azimuth using topocalc backend."""
    return horizon.horizon(azimuth_single, dem_arr, dx)



def check_terrain_backend():
    """
    Check which terrain computation backend is available and provide installation help.
    
    Returns:
        str: 'horayzon', 'topocalc', or 'none'
    """
    if HORAYZON_AVAILABLE:
        try:
            print("✅ HORAYZON backend available")
            print(f"   Version: {horayzon.__version__}")
            print("   Features: High-performance ray-tracing, parallel computation")
            return 'horayzon'
        except:
            print("⚠️ HORAYZON imported but version unavailable")
            return 'horayzon'
    elif TOPOCALC_AVAILABLE:
        print("✅ topocalc backend available")
        print("   Features: Standard terrain computation")
        print("   Recommendation: Install HORAYZON for better performance")
        print("   Install HORAYZON: conda install -c conda-forge embree tbb-devel")
        print("                     pip install git+https://github.com/ChristianSteger/HORAYZON.git")
        return 'topocalc'
    else:
        print("❌ No terrain computation backend available")
        print("   Install topocalc: pip install git+https://github.com/ArcticSnow/topocalc")
        print("   Or install HORAYZON: conda install -c conda-forge embree tbb-devel")
        print("                         pip install git+https://github.com/ChristianSteger/HORAYZON.git") 
        return 'none'

def get_terrain_performance_info():
    """
    Get performance information about available terrain backends.
    
    Returns:
        dict: Performance characteristics of available backends
    """
    info = {
        'backend': check_terrain_backend(),
        'parallel_horizon': HORAYZON_AVAILABLE,
        'parallel_svf': HORAYZON_AVAILABLE,
        'high_accuracy': HORAYZON_AVAILABLE,
        'memory_efficient': HORAYZON_AVAILABLE
    }
    
    if HORAYZON_AVAILABLE:
        info.update({
            'expected_speedup_horizon': '5-20x (depending on domain size)',
            'expected_speedup_svf': '2-10x',
            'max_dem_size': 'Limited by RAM (efficient for large DEMs)',
            'aspect_convention': 'Meteorological (0°=North)',
            'conversion_applied': True
        })
    else:
        info.update({
            'expected_speedup_horizon': '1x (baseline)',
            'expected_speedup_svf': '1x (baseline)',
            'max_dem_size': 'Limited by single-core performance',
            'aspect_convention': 'Mathematical (0°=East)', 
            'conversion_applied': False
        })
    
    return info

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
        if HORAYZON_AVAILABLE:
            slope, aspect = _compute_slope_aspect_horayzon(dem_arr, dx, dy)
        else:
            slope, aspect = _compute_slope_aspect_topocalc(dem_arr, dx, dy)
        
        ds['slope'] = (["y", "x"], np.deg2rad(slope))
        ds['aspect'] = (["y", "x"], np.deg2rad(aspect))
        if 'aspect_cos' not in var_in:
            ds['aspect_cos'] = (["y", "x"], np.cos(np.deg2rad(aspect)))
        if 'aspect_sin' not in var_in:
            ds['aspect_sin'] = (["y", "x"], np.sin(np.deg2rad(aspect)))

    if 'svf' not in var_in:
        print('Computing svf ...')
        if HORAYZON_AVAILABLE:
            svf = _compute_svf_horayzon(dem_arr, dx, azimuth_inc=5)
        else:
            svf = _compute_svf_topocalc(dem_arr, dx)
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


def compute_horizon(dem_file, azimuth_inc=30, num_threads=None, fname='da_horizon.nc',
                    output_directory=Path('./outputs'), format='netcdf'):
    """
    Function to compute horizon angles using the best available backend (HORAYZON preferred).

    Args:
        dem_file (str): path and filename of the dem
        azimuth_inc (int): angle increment to compute horizons at, in Degrees [0-359]
        num_threads (int): number of threads to parallelize on
        fname (str): output filename
        output_directory (Path): output directory
        format (str): 'netcdf' or 'zarr' — zarr provides faster point lookups and
                       parallel reads during downscaling

    Returns:
        dataarray: all horizon angles for x,y,azimuth coordinates

    """
    print(f'\n---> Computing horizons with {azimuth_inc} degree increments')
    ds = open_dem(dem_file)
    dx = ds.x.diff('x').median().values
    dy = ds.y.diff('y').median().values

    azimuth = np.arange(-180 + azimuth_inc / 2, 180, azimuth_inc)  # center the azimuth in middle of the bin

    if HORAYZON_AVAILABLE:
        print('---> Using HORAYZON for high-performance horizon computation')
        
        # HORAYZON can compute all azimuths at once - much faster!
        horizon_cos_elev = _compute_horizon_horayzon(
            ds.elevation.values, dx, azimuth, num_threads
        )
        
        # Convert back to horizon elevation angles
        # HORAYZON gives cos of elevation angles, convert to elevation angles
        arr_val = np.pi/2 - np.arccos(np.clip(horizon_cos_elev, -1, 1))
        
    else:
        print('---> Using topocalc for horizon computation (consider installing HORAYZON for better performance)')
        
        # Fallback to original topocalc method
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

        arr = pool.starmap(_compute_horizon_topocalc, 
                          zip(list(azimuth), elev, dxs))
        pool.close()
        pool.join()

        for i, a in enumerate(arr):
            arr_val[i, :, :] = np.pi/2 - np.arccos(a)

    da = xr.DataArray(data=arr_val,
                      coords={
                          "y": ds.y.values,
                          "x": ds.x.values,
                          "azimuth": azimuth
                      },
                      dims=["azimuth", "y", "x"],
                      name='horizon',
                      attrs={
                          'long_name': 'Horizon angles',
                          'units': 'radian',
                          'backend': 'HORAYZON' if HORAYZON_AVAILABLE else 'topocalc'
                      }
                      )
    
    ds = da.to_dataset()
    ds.attrs = {
        'description': 'Horizon angles for TopoPyScale',
        'azimuth_inc': str(azimuth_inc),
        'backend': 'HORAYZON' if HORAYZON_AVAILABLE else 'topocalc',
        'format': format
    }

    output_path = output_directory / fname
    output_directory.mkdir(parents=True, exist_ok=True)

    if format.lower() == 'zarr':
        if not fname.endswith('.zarr'):
            output_path = output_path.with_suffix('.zarr')
        print(f'---> Saving horizon to zarr: {output_path}')
        try:
            from zarr.codecs import BloscCodec
        except ImportError:
            from zarr import Blosc as BloscCodec

        n_az = da.azimuth.size
        ny = da.y.size
        nx = da.x.size
        az_chunk = max(1, n_az)
        y_chunk = max(64, min(512, ny))
        x_chunk = max(64, min(512, nx))
        chunks = (az_chunk, y_chunk, x_chunk)

        encoding = {
            'horizon': {
                'compressor': BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle', blocksize=0),
                'chunks': chunks
            }
        }
        ds = ds.chunk({'azimuth': az_chunk, 'y': y_chunk, 'x': x_chunk})
        ds.to_zarr(str(output_path), mode='w', encoding=encoding, zarr_format=3, consolidated=True)
    else:
        print(f'---> Saving horizon to netcdf: {output_path}')
        ds.to_netcdf(output_path)

    return da


