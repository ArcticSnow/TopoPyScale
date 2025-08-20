import zarr
import dask.array as da
from dask.distributed import Client
import numpy as np
from typing import Callable, Tuple, Union, Dict
from datetime import datetime
import os
from pathlib import Path


class ZarrParallelProcessor:
    def __init__(self, 
                 era5_zarr_path,
                 output_zarr_path, 
                 array_path = None):
        """
        Initialize the parallel processor for zarr datasets.
        
        Args:
            input_zarr_path (str): Path to the input zarr dataset
            output_zarr_path (str): Path to store the computed results
            array_path (str): Path to the specific array within the zarr group (if dataset is a group)
        """
        self.era5_zarr_path = era5_zarr_path
        self.output_zarr_path = output_zarr_path
        self.dera = zarr.open(era5_zarr_path, mode='r')
        
        # Create output zarr store
        self.setup_output_store()

    def setup_output_store(self):
        """Setup the output Zarr store with proper structure"""
        # Create output directory if it doesn't exist
        Path(self.output_zarr_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create the output zarr store
        self.output_store = zarr.open(self.output_zarr_path, mode='w')
        
        # Add metadata
        self.output_store.attrs['created_at'] = datetime.utcnow().isoformat()
        self.output_store.attrs['created_by'] = 'Author Name'
        self.output_store.attrs['source_dataset'] = self.input_zarr_path

    def create_subset_indices(self, row)
        """Create slice objects for subsetting."""

        ind_lat = np.abs(ds_.latitude.values - row.lat).argmin()
        ind_lon = np.abs(ds_.longitude.values - row.lon).argmin()

        ilatitude = [ind_lat - 1, ind_lat, ind_lat + 1]
        ilongitude = [ind_lon - 1, ind_lon, ind_lon + 1]
                
        return (
            {'latitude':ilatitude,
             'longitude':ilongitude}  )



    def pt_downscale_radiations(row, subset, down_pt, ds_solar, horizon_da, meta):
    	pt_id = row.point_name
	    print(f'Downscaling LW, SW for point: {pt_id}')

	    # ======== Longwave downward radiation ===============
	    x1, x2 = 0.43, 5.7
	    sbc = 5.67e-8

	    down_pt = mu.mixing_ratio(down_pt, mu.var_era_plevel)
	    down_pt = mu.vapor_pressure(down_pt, mu.var_era_plevel)
	    surf_interp = mu.mixing_ratio(surf_interp, mu.var_era_surf)
	    surf_interp = mu.vapor_pressure(surf_interp, mu.var_era_surf)

	    # ========= Compute clear sky emissivity ===============
	    down_pt['cse'] = 0.23 + x1 * (down_pt.vp / down_pt.t) ** (1 / x2)
	    surf_interp['cse'] = 0.23 + x1 * (surf_interp.vp / surf_interp.t2m) ** (1 / x2)
	    # Calculate the "cloud" emissivity, UNIT OF STRD (Jjoel_testJun2023/m2)

	    tstep_seconds = pd.Timedelta(f"{meta.get('tstep')}H").seconds

	    surf_interp['cle'] = (surf_interp.strd / tstep_seconds) / (sbc * surf_interp.t2m ** 4) - \
	                         surf_interp['cse']
	    # Use the former cloud emissivity to compute the all sky emissivity at subgrid.
	    surf_interp['aef'] = down_pt['cse'] + surf_interp['cle']
	    if meta.get('lw_terrain_flag'):
	        down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4 + \
	                        0.5 * (1 + np.cos(row.slope)) * (1 - row.svf) * 0.99 * 5.67e-8 * (273.15 ** 4)
	    else:
	        down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4

	    kt = surf_interp.ssrd * 0
	    sunset = ds_solar.sunset.astype(bool).compute()
	    mu0 = ds_solar.mu0
	    SWtoa = ds_solar.SWtoa

	    # pdb.set_trace()
	    kt[~sunset] = (surf_interp.ssrd[~sunset] / tstep_seconds) / SWtoa[~sunset]  # clearness index
	    kt[kt < 0] = 0
	    kt[kt > 1] = 1
	    kd = 0.952 - 1.041 * np.exp(-1 * np.exp(2.3 - 4.702 * kt))  # Diffuse index

	    surf_interp['SW'] = surf_interp.ssrd / tstep_seconds
	    surf_interp['SW'][surf_interp['SW'] < 0] = 0
	    surf_interp['SW_diffuse'] = kd * surf_interp.SW
	    down_pt['SW_diffuse'] = row.svf * surf_interp.SW_diffuse

	    surf_interp['SW_direct'] = surf_interp.SW - surf_interp.SW_diffuse
	    # scale direct solar radiation using Beer's law (see Aalstad 2019, Appendix A)
	    ka = surf_interp.ssrd * 0
	    # pdb.set_trace()
	    ka[~sunset] = (g * mu0[~sunset] / down_pt.p) * np.log(SWtoa[~sunset] / surf_interp.SW_direct[~sunset])
	    # Illumination angle
	    down_pt['cos_illumination_tmp'] = mu0 * np.cos(row.slope) + np.sin(ds_solar.zenith) * \
	                                      np.sin(row.slope) * np.cos(ds_solar.azimuth - row.aspect)
	    down_pt['cos_illumination'] = down_pt.cos_illumination_tmp * (
	            down_pt.cos_illumination_tmp > 0)  # remove selfdowing ccuring when |Solar.azi - aspect| > 90
	    down_pt = down_pt.drop(['cos_illumination_tmp'])
	    illumination_mask = down_pt['cos_illumination'] < 0
	    illumination_mask = illumination_mask.compute()
	    down_pt['cos_illumination'][illumination_mask] = 0

	    # Binary shadow masks.
	    horizon = horizon_da.sel(x=row.x, y=row.y, azimuth=np.rad2deg(ds_solar.azimuth), method='nearest')
	    shade = (horizon > ds_solar.elevation)
	    down_pt['SW_direct_tmp'] = down_pt.t * 0
	    down_pt['SW_direct_tmp'][~sunset] = SWtoa[~sunset] * np.exp(
	        -ka[~sunset] * down_pt.p[~sunset] / (g * mu0[~sunset]))
	    down_pt['SW_direct'] = down_pt.t * 0
	    down_pt['SW_direct'][~sunset] = down_pt.SW_direct_tmp[~sunset] * (
	            down_pt.cos_illumination[~sunset] / mu0[~sunset]) * (1 - shade)
	    down_pt['SW'] = down_pt.SW_diffuse + down_pt.SW_direct

	    # currently drop azimuth and level as they are coords. Could be passed to variables instead.
	    # round(5) required to sufficiently represent specific humidty, q (eg typical value 0.00078)
	    down_pt = down_pt.drop(['level']).round(5)

	    # adding metadata
	    down_pt.LW.attrs = {'units': 'W m**-2', 'long_name': 'Surface longwave radiation downwards',
	                        'standard_name': 'longwave_radiation_downward'}
	    down_pt.cse.attrs = {'units': 'xxx', 'standard_name': 'Clear sky emissivity'}
	    down_pt = down_pt.drop(['SW_direct_tmp'])
	    down_pt.SW.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar radiation downwards',
	                        'standard_name': 'shortwave_radiation_downward'}
	    down_pt.SW_diffuse.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar diffuse radiation downwards',
	                                'standard_name': 'shortwave_diffuse_radiation_downward'}
	    ver_dict = tu.get_versionning()
	    down_pt.attrs = {'title': 'Downscaled timeseries with TopoPyScale',
	                     'created with': 'TopoPyScale, see more at https://topopyscale.readthedocs.io',
	                      'package_version':ver_dict.get('package_version'),
	                      'git_commit': ver_dict.get('git_commit'),
	                     'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
	                     'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}


    def downscale_atmo(row, subset, meta):
    	pt_id = row.point_name
	    print(f'Downscaling t,q,p,tp,ws, wd for point: {pt_id}')

	    # ====== Horizontal interpolation ====================
	    interp_method = meta.get('interp_method')

	    # convert gridcells coordinates from WGS84 to DEM projection
	    lons, lats = np.meshgrid(subset.longitude.values, subset.latitude.values)
	    trans = Transformer.from_crs("epsg:4326", "epsg:" + str(meta.get('target_epsg')), always_xy=True)
	    Xs, Ys = trans.transform(lons.flatten(), lats.flatten())
	    Xs = Xs.reshape(lons.shape)
	    Ys = Ys.reshape(lons.shape)

	    # ========= Converting z from [m**2 s**-2] to [m] asl =======
	    subset.z.values = subset.z.values / g
	    subset.z.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of plevel'}

	    subset.z_surf.values = subset.z_surf.values / g  # convert geopotential height to elevation (in m), normalizing by g
	    subset.z_surf.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of ERA5 surface'}

	    # ========== Horizontal interpolation ==================
	    dist = np.sqrt((row.x - Xs) ** 2 + (row.y - Ys) ** 2)
	    if interp_method == 'idw':
	        idw = 1 / (dist ** 2)
	        weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1
	    elif interp_method == 'linear':
	        weights = dist / np.sum(dist)
	    else:
	        sys.exit('ERROR: interpolation method not available')

	    # create a dataArray of weights to then propagate through the dataset
	    da_idw = xr.DataArray(data=weights,
	                              coords={
	                                  "latitude": subset.latitude.values,
	                                  "longitude": subset.longitude.values,
	                              },
	                              dims=["latitude", "longitude"]
	                              )
	    # convert to float64 consequence of CDS-BETA change to dtype in netcdf download
	    #subset = subset.astype('float64')
	    #da_idw = da_idw.astype('float64')

	    dw = xr.Dataset.weighted(subset, da_idw)
	    subset_interp = dw.sum(['longitude', 'latitude'],
	                             keep_attrs=True)  # compute horizontal inverse weighted horizontal interpolation
	    


	    # ============ Extract specific humidity (q) for both dataset ============
	    subset_interp = mu.dewT_2_q_magnus(subset_interp, mu.var_era_surf)
	    subset_interp = mu.t_rh_2_dewT(subset_interp, mu.var_era_plevel)

	    down_pt = xr.Dataset(coords={
	            'time': subset_interp.time,
	            #'point_name': pd.to_numeric(pt_id, errors='ignore')
	            'point_name': pt_id
	    })

	    if (row.elevation < subset_interp.z.isel(level=-1)).sum():
	        pt_elev_diff = np.round(np.min(row.elevation - subset_interp.z.isel(level=-1).values), 0)
	        print(f"---> WARNING: Point {pt_id} is {pt_elev_diff} m lower than the {subset_interp.isel(level=-1).level.data} hPa geopotential\n=> "
	                  "Values sampled from Psurf and lowest Plevel. No vertical interpolation")
	        
	        ind_z_top = (subset_interp.where(subset_interp.z > row.elevation).z - row.elevation).argmin('level')
	        top = subset_interp.isel(level=ind_z_top)

	        down_pt['t'] = top['t']
	        down_pt['u'] = top.u
	        down_pt['v'] = top.v
	        down_pt['q'] = top.q
	        down_pt['p'] = top.level * (10 ** 2) * np.exp(
	                -(row.elevation - top.z) / (0.5 * (top.t + down_pt.t) * R / g))  # Pressure in bar

	    else:
	        # ========== Vertical interpolation at the DEM surface z  ===============
	        ind_z_bot = (subset_interp.where(subset_interp.z < row.elevation).z - row.elevation).argmax('level')

	        # In case upper pressure level elevation is not high enough, stop process and print error
	        try:
	            ind_z_top = (subset_interp.where(subset_interp.z > row.elevation).z - row.elevation).argmin('level')
	        except:
	            raise ValueError(
	                f'ERROR: Upper pressure level {subset_interp.level.min().values} hPa geopotential is lower than cluster mean elevation {row.elevation} {subset_interp.z}')
	               

	        top = subset_interp.isel(level=ind_z_top)
	        bot = subset_interp.isel(level=ind_z_bot)

	        # Preparing interpolation weights for linear interpolation =======================
	        dist = np.array([np.abs(bot.z - row.elevation).values, np.abs(top.z - row.elevation).values])
	        weights = dist / np.sum(dist, axis=0)  # [x] removed idw interp for vertical - Aug 20 2025

	        # ============ Creating a dataset containing the downscaled timeseries ========================
	        down_pt['t'] = bot.t * weights[1] + top.t * weights[0]
	        down_pt['u'] = bot.u * weights[1] + top.u * weights[0]
	        down_pt['v'] = bot.v * weights[1] + top.v * weights[0]
	        down_pt['q'] = bot.q * weights[1] + top.q * weights[0]
	        down_pt['p'] = top.level * (10 ** 2) * np.exp(
	                -(row.elevation - top.z) / (0.5 * (top.t + down_pt.t) * R / g))  # Pressure in bar

	        # ======= logic  to compute ws, wd without loading data in memory, and maintaining the power of dask
	    down_pt['month'] = ('time', down_pt.time.dt.month.data)
	    print("precip lapse rate = " + str(meta.get('precip_lapse_rate_flag')))
	    if meta.get('precip_lapse_rate_flag'):
	        monthly_coeffs = xr.Dataset(
	                {
	                    'coef': (['month'], [0.35, 0.35, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35])
	                },
	                coords={'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
	            )
	        down_pt['precip_lapse_rate'] = (1 + monthly_coeffs.coef.sel(month=down_pt.month.values).data * (
	                    row.elevation - subset_interp.z_surf) * 1e-3) / \
	                                           (1 - monthly_coeffs.coef.sel(month=down_pt.month.values).data * (
	                                                   row.elevation - subset_interp.z_surf) * 1e-3)
	    else:
	        down_pt['precip_lapse_rate'] = down_pt.t * 0 + 1

	    down_pt['tp'] = down_pt.precip_lapse_rate * subset_interp.tp * 1 / meta.get('tstep') * 10 ** 3  # Convert to mm/hr
	    down_pt['theta'] = np.arctan2(-down_pt.u, -down_pt.v)
	    down_pt['theta_neg'] = (down_pt.theta < 0) * (down_pt.theta + 2 * np.pi)
	    down_pt['theta_pos'] = (down_pt.theta >= 0) * down_pt.theta
	    down_pt = down_pt.drop('theta')
	    down_pt['wd'] = (down_pt.theta_pos + down_pt.theta_neg)  # direction in Rad
	    down_pt['ws'] = np.sqrt(down_pt.u ** 2 + down_pt.v ** 2)
	    down_pt = down_pt.drop(['theta_pos', 'theta_neg', 'month'])

	    return down_pt




    
    def process_and_store_subset(self, 
                               subset_indices: Tuple[slice, slice, slice],
                               row,
                               computation_func: Callable,
                               subset_id: int) -> str:
        """
        Process a subset and store results directly to disk.
        
        Args:
            subset_indices: Tuple of slice objects for indexing
            computation_func: Function to apply to the subset
            subset_id: Identifier for this subset
            
        Returns:
            Path to the stored results
        """
        try:
            # Get the subset
            subset = self.dera.isel(latitude=subset_indices.get('latitude'), longitude=subset_indices.get('longitude'))

            
            result = downscale_atmo(subset)

            # Compute the result
            result = computation_func(subset)
            
            # Create a group for this subset
            subset_group = self.output_store.create_group(f'subset_{subset_id}')
            
            # Store the result
            if isinstance(result, dict):
                # If result is a dictionary, store each key-value pair
                for key, value in result.items():
                    if isinstance(value, (da.Array, np.ndarray)):
                        subset_group.create_dataset(
                            key,
                            data=value,
                            chunks=True,  # Let zarr choose optimal chunking
                            compression='lz4'
                        )
                    else:
                        subset_group.attrs[key] = value
            else:
                # If result is an array, store it directly
                subset_group.create_dataset(
                    'result',
                    data=result,
                    chunks=True,
                    compression='lz4'
                )
            
            # Store metadata
            subset_group.attrs['processed_at'] = datetime.utcnow().isoformat()
            subset_group.attrs['x_range'] = f"{subset_indices[0].start}:{subset_indices[0].stop}"
            subset_group.attrs['y_range'] = f"{subset_indices[1].start}:{subset_indices[1].stop}"
            subset_group.attrs['time_range'] = f"{subset_indices[2].start}:{subset_indices[2].stop}"
            
            return f"{self.output_zarr_path}/subset_{subset_id}"
            
        except Exception as e:
            logger.error(f"Error processing subset {subset_id}: {str(e)}")
            raise

    def parallel_process_multiple_subsets(self,
                                        subset_list: list[Tuple[Tuple[int, int], 
                                                              Tuple[int, int], 
                                                              Tuple[int, int]]],
                                        computation_func: Callable,
                                        n_workers: int = None) -> list:
        """Process multiple subsets in parallel and store results to disk."""
        client_kwargs = {}
        if n_workers is not None:
            client_kwargs['n_workers'] = n_workers
            
        with Client(**client_kwargs) as client:
            logger.info(f"Dask client started with {len(client.scheduler_info()['workers'])} workers")
            
            futures = []
            for i, subset_ranges in enumerate(subset_list):
                x_range, y_range, time_range = subset_ranges
                indices = self.create_subset_indices(x_range, y_range, time_range)
                future = client.submit(
                    self.process_and_store_subset,
                    indices,
                    computation_func,
                    i
                )
                futures.append(future)
            
            return client.gather(futures)
