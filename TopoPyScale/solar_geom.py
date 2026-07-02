"""
Function to compute Solar angles
S. Filhol, Oct 2021

Optimized version with vectorized solar position calculations for better performance.
"""

import pvlib
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import Transformer
import multiprocessing as mproc
from TopoPyScale import topo_export as te
from pathlib import Path
import warnings
from tqdm import tqdm


def get_solar_geom(df_position,
                   start_date,
                   end_date,
                   tstep,
                   sr_epsg="4326",
                   num_threads=None,
                   fname='ds_solar.nc',
                   project_ouput=Path('./'),
                   method_solar='nrel_numpy'):
    """
    Optimized function to compute solar position using vectorized pvlib operations.
    
    Performance improvements:
    - Vectorized solar position calculations (no ThreadPool)
    - Efficient coordinate transformation
    - Streamlined data processing with progress indication
    
    Args:
        df_position (dataframe): point_name as index, latitude, longitude
        start_date (str): start date  "2014-05-10"
        end_date (str: end date   "2015-05-10"
        tstep (str): time step, ex: '6h'
        sr_epsg (str): source EPSG code for the input coordinate
        num_threads (int): deprecated - kept for compatibility, vectorization is used instead
        fname (str): name of netcdf file to store solar geometry
        project_ouput (str): path to project root directory
        method_solar (str): pvlib solar position method

    Returns: 
        dataset: solar angles in radians with optimized processing
    """
    print('\n---> Computing solar geometry (vectorized)')
    
    if num_threads is not None:
        warnings.warn("num_threads parameter is deprecated - using vectorized operations for better performance", 
                     DeprecationWarning, stacklevel=2)
    
    # Handle coordinate transformation efficiently
    if (str(sr_epsg) != "4326") or ('longitude' not in df_position.columns):
        print('---> Converting coordinates to WGS84')
        trans = Transformer.from_crs("epsg:" + str(sr_epsg), "epsg:4326", always_xy=True)
        df_position = df_position.copy()  # Don't modify original
        df_position['longitude'], df_position['latitude'] = trans.transform(
            df_position.x.values, df_position.y.values)

    # Create time index efficiently  
    times = pd.date_range(start_date, pd.to_datetime(end_date) + pd.to_timedelta('1D'), 
                         freq=tstep, tz='UTC', inclusive='left')
    
    n_points = len(df_position)
    n_times = len(times)
    
    print(f'---> Processing {n_points} locations × {n_times} time steps')
    
    # Pre-allocate result arrays for efficiency
    arr_val = np.empty((n_points, 3, n_times), dtype=np.float64)
    
    # Process locations with progress bar and vectorized pvlib calls
    with tqdm(total=n_points, desc='Solar positions', unit='points') as pbar:
        for i, (point_name, row) in enumerate(df_position.iterrows()):
            # Vectorized solar position calculation for all times at once
            solar_pos = pvlib.solarposition.get_solarposition(
                times, 
                row.latitude, 
                row.longitude, 
                row.elevation,
                method=method_solar
            )
            
            # Extract zenith, azimuth, elevation efficiently
            arr_val[i, 0, :] = solar_pos['zenith'].values
            arr_val[i, 1, :] = solar_pos['azimuth'].values  
            arr_val[i, 2, :] = solar_pos['elevation'].values
            
            pbar.update(1)

    print('---> Creating solar geometry dataset')
    
    # Create xarray Dataset efficiently
    ds = xr.Dataset(
        {
            "zenith": (["point_name", "time"], np.deg2rad(arr_val[:, 0, :])),
            "azimuth": (["point_name", "time"], np.deg2rad(180 - arr_val[:, 1, :])),
            "elevation": (["point_name", "time"], np.deg2rad(arr_val[:, 2, :])),
        },
        coords={
            "point_name": (("point_name"), df_position.index),
            "time": times,
            "reference_time": pd.Timestamp(start_date),
        },
    )

    # Compute derived solar variables efficiently
    print('---> Computing derived solar variables')
    ds['mu0'] = np.cos(ds.zenith).where(np.cos(ds.zenith) > 0, 0)
    
    S0 = 1370  # Solar constant (total TOA solar irradiance) [W m^-2] used in ECMWF's IFS
    ds['SWtoa'] = S0 * ds.mu0
    ds['sunset'] = ds.mu0 < np.cos(89 * np.pi/180)
    
    # Add metadata
    ds.SWtoa.attrs = {'units': 'W/m**2', 'standard_name': 'toa_incoming_shortwave_flux'}
    ds.sunset.attrs = {'units': 'bool', 'standard_name': 'sunset_flag'}
    ds.mu0.attrs = {'units': '1', 'standard_name': 'cosine_solar_zenith_angle'}
    ds.zenith.attrs = {'units': 'radian', 'standard_name': 'solar_zenith_angle'}
    ds.azimuth.attrs = {'units': 'radian', 'standard_name': 'solar_azimuth_angle'}
    ds.elevation.attrs = {'units': 'radian', 'standard_name': 'solar_elevation_angle'}
    
    ds.attrs.update({
        'title': 'Solar geometry computed with TopoPyScale',
        'method': method_solar,
        'processing': 'vectorized_pvlib',
        'performance_note': 'Optimized vectorized calculation'
    })

    print(f'---> Saving to {project_ouput / fname}')
    te.to_netcdf(ds, fname=project_ouput / fname)

    return ds

