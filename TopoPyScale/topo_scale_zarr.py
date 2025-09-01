"""
S. Filhol, 2025

Methods to execute downscaling using TopoScale method.

WARNING: Dask worker parameters must be fine tune to your machine (local, cluster or whaterver). If workers go to "pause" because of memory limit, the processing stops.

TODO:
- [x] implement another method using multiprocessing
- [ ] make Dask worker parameters flexible. 
- [ ] try pushing data to a Zarr store for outputs. Currently going to individual netcdf files
- [ ] See how to potentially automate setting Dask worker config based on the number of core, and the memory size of one job.
- [ ] incorporate method into TopoClass
"""


from zarr.codecs import BloscCodec
from dask.distributed import LocalCluster, Client
import numpy as np
from datetime import datetime
import os
from pathlib import Path

from pyproj import Transformer
from TopoPyScale import meteo_util as mu
from TopoPyScale import topo_utils as tu
import pandas as pd
import xarray as xr
import numpy as np 
import dask.array as da

import pdb
import time


varout_default = ['t',
         'u',
         'v',
         'q',
         'p',
         'tp',
         'wd',
         'ws',
         'w',
         'LW',
         'SW_diffuse',
         'SW_direct',
         'SW']



class ClimateDownscaler:
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
                 varout=varout_default,
                 interp_method='idw',
                 lw_terrain_flag=True,
                 precip_lapse_rate_flag=False,
                 file_pattern='down_pt*.nc',
                 store_name=None):
        """
        Initialize the parallel processor for zarr datasets.
        
        Args:
            input_zarr_path (str): Path to the input zarr dataset
            output_path (str): Path to store the computed results
            array_path (str): Path to the specific array within the zarr group (if dataset is a group)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.ERA = xr.open_zarr(str(era5_zarr_path))
        self.df_centroids = df_centroids
        self.ds_solar = ds_solar
        self.da_horizon = da_horizon
        self.varout = varout

        self.start_time = time.time()
        self.n_digits = len(str(self.df_centroids.index.max()))
        self.ds_solar = ds_solar
        self.da_horizon = da_horizon
        self.start_date = start_date
        self.end_date = end_date
        self.tvec = pd.date_range(start_date, pd.to_datetime(end_date) + pd.to_timedelta('1D'), freq=tstep, inclusive='left')

        tstep_dict = {'1H': 1, '3H': 3, '6H': 6}
        #n_digits = len(str(self.df_centroids.cluster_labels.max()))
        self.meta = {'interp_method': interp_method,
                                  'lw_terrain_flag': lw_terrain_flag,
                                  'tstep': tstep_dict.get(tstep),
                                  'n_digits': self.n_digits,
                                  'file_pattern': file_pattern,
                                  'target_epsg':target_EPSG,
                                  'precip_lapse_rate_flag':precip_lapse_rate_flag,
                                  'output_directory':output_path,
                                  'lw_terrain_flag':lw_terrain_flag}

        
        # Logic to setup output name
        # Create output zarr store if specified output to be zarr store
        if (store_name is not None):
            if store_name[-5:] == '.zarr':
                self.output_format = 'zarr'
                self.store_name = store_name
                self.setup_output_store()
            else:
                raise ValueError("Store_name must be xxx.zarr")

            if file_pattern is not None:
                raise ValueError("It is only possible to save results to netcdf or a zarr store, not both simultaneously")

        elif file_pattern is not None:
            if file_pattern[-4:] == '*.nc':
                self.output_format = 'netcdf'
            else:
                raise ValueError("file_pattern must finish with *.nc")

            if store_name is not None:
                raise ValueError("It is only possible to save results to netcdf or a zarr store, not both simultaneously")


    def setup_output_store(self):
        """Setup the output Zarr store with proper structure"""

        # Create output directory if it doesn't exist
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy dataset
        shape=(self.df_centroids.shape[0],len(self.tvec))
        dv = {}
        for var in self.varout:
            dv.update({var:(('point_ind', 'time'), da.zeros(shape))})
        ds = xr.Dataset(
            data_vars = dv,
            coords={
                'time':self.tvec,
                'point_ind':self.df_centroids.point_ind.values,
                'reference_time':self.tvec[0]
            })

        # Store dataset to Zarr
        comp = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)
        encoder = dict(zip(self.varout, [{'compressors': comp}]*len(self.varout)))
        ds = ds.chunk({'time':self.ERA.chunks.get('time')[0], 'point_ind':1})
        ds.to_zarr(self.output_path / self.store_name, mode='w', encoding=encoder, zarr_format=3)


    def create_subset_indices(self, row):
        """Create slice objects for subsetting."""

        ind_lat = np.abs(self.ERA.latitude.values - row.lat).argmin()
        ind_lon = np.abs(self.ERA.longitude.values - row.lon).argmin()

        ilatitude = np.array([ind_lat - 1, ind_lat, ind_lat + 1]).astype(int)
        ilongitude = np.array([ind_lon - 1, ind_lon, ind_lon + 1]).astype(int)
                
        return (
            {'centroid': row,
            'point_id':row.point_ind,
            'point_name':row.point_name,
            'latitude':ilatitude,
             'longitude':ilongitude,
             'tvec':self.tvec})


    def downscale_atmo(self, row, subset, meta, ds_solar, da_horizon):
        """
        Downscaling method following the toposcale method introduced by J. Fiddes et al (2012)
        """

        # Physical constants
        g = 9.81  # Acceleration of gravity [ms^-1]
        R = 287.05  # Gas constant for dry air [JK^-1kg^-1]

        pt_id = row.point_ind
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


        #subset_interp['rh'] = mu.q_2_rh(subset_interp.t, subset_interp.tp, subset_interp.q)
        subset_interp = mu.t_rh_2_dewT(subset_interp, mu.var_era_plevel)

        subset_interp = subset_interp.persist()

        down_pt = xr.Dataset(coords={
                'time': subset_interp.time,
                'point_ind': pt_id})

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


        #============================================================
        #           Radiation corrections
        #============================================================
        # ======== Longwave downward radiation ===============
        x1, x2 = 0.43, 5.7
        sbc = 5.67e-8

        down_pt = mu.mixing_ratio(down_pt, mu.var_era_plevel)
        down_pt = mu.vapor_pressure(down_pt, mu.var_era_plevel)
        subset_interp = mu.mixing_ratio(subset_interp, mu.var_era_surf)
        subset_interp = mu.vapor_pressure(subset_interp, mu.var_era_surf)

        # ========= Compute clear sky emissivity ===============
        down_pt['cse'] = 0.23 + x1 * (down_pt.vp / down_pt.t) ** (1 / x2)
        subset_interp['cse'] = 0.23 + x1 * (subset_interp.vp / subset_interp.t2m) ** (1 / x2)
        # Calculate the "cloud" emissivity, UNIT OF STRD (Jjoel_testJun2023/m2)

        tstep_seconds = pd.Timedelta(f"{meta.get('tstep')}h").seconds

        subset_interp['cle'] = (subset_interp.strd / tstep_seconds) / (sbc * subset_interp.t2m ** 4) - \
                             subset_interp['cse']
        # Use the former cloud emissivity to compute the all sky emissivity at subgrid.
        subset_interp['aef'] = down_pt['cse'] + subset_interp['cle']
        if meta.get('lw_terrain_flag'):
            down_pt['LW'] = row.svf * subset_interp['aef'] * sbc * down_pt.t ** 4 + \
                            0.5 * (1 + np.cos(row.slope)) * (1 - row.svf) * 0.99 * 5.67e-8 * (273.15 ** 4)
        else:
            down_pt['LW'] = row.svf * subset_interp['aef'] * sbc * down_pt.t ** 4

        kt = subset_interp.ssrd * 0
        sunset = ds_solar.sunset.astype(bool).compute()
        mu0 = ds_solar.mu0
        SWtoa = ds_solar.SWtoa

        kt[~sunset] = (subset_interp.ssrd[~sunset] / tstep_seconds) / SWtoa[~sunset]  # clearness index
        
        kt[(kt < 0).values] = 0
        kt[(kt > 1).values] = 1
        kd = 0.952 - 1.041 * np.exp(-1 * np.exp(2.3 - 4.702 * kt))  # Diffuse index

        subset_interp['SW'] = subset_interp.ssrd / tstep_seconds
        subset_interp['SW'][(subset_interp['SW'] < 0).values] = 0
        subset_interp['SW_diffuse'] = kd * subset_interp.SW
        down_pt['SW_diffuse'] = row.svf * subset_interp.SW_diffuse

        subset_interp['SW_direct'] = subset_interp.SW - subset_interp.SW_diffuse
        # scale direct solar radiation using Beer's law (see Aalstad 2019, Appendix A)
        ka = subset_interp.ssrd * 0
        # pdb.set_trace()
        ka[~sunset] = (g * mu0[~sunset] / down_pt.p) * np.log(SWtoa[~sunset] / subset_interp.SW_direct[~sunset])
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
        horizon = da_horizon.sel(x=row.x, y=row.y, azimuth=np.rad2deg(ds_solar.azimuth), method='nearest')
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
        down_pt = down_pt.drop(['level'])
        down_pt = down_pt.drop(['precip_lapse_rate'])
        # adding metadata
        down_pt.t.attrs = {'units': 'K', 'long_name': 'Temperature', 'standard_name': 'air_temperature'}
        down_pt.q.attrs = {'units': 'kg kg**-1', 'long_name': 'Specific humidity', 'standard_name': 'specific_humidity'}
        down_pt.u.attrs = {'units': 'm s**-1', 'long_name': 'U component of wind', 'standard_name': 'eastward_wind'}
        down_pt.v.attrs = {'units': 'm s**-1', 'long_name': 'V component of wind', 'standard_name': 'northward wind'}
        down_pt.p.attrs = {'units': 'bar', 'long_name': 'Pression atmospheric', 'standard_name': 'pression_atmospheric'}

        down_pt.ws.attrs = {'units': 'm s**-1', 'long_name': 'Wind speed', 'standard_name': 'wind_speed'}
        down_pt.wd.attrs = {'units': 'deg', 'long_name': 'Wind direction', 'standard_name': 'wind_direction'}
        down_pt.tp.attrs = {'units': 'mm hr**-1', 'long_name': 'Precipitation', 'standard_name': 'precipitation'}
        #down_pt.precip_lapse_rate.attrs = {'units': 'mm hr**-1',
        #                                       'long_name': 'Precipitation after lapse-rate correction',
        #                                       'standard_name': 'precipitation_after_lapse-rate_correction'}

        down_pt.LW.attrs = {'units': 'W m**-2', 'long_name': 'Surface longwave radiation downwards',
                            'standard_name': 'longwave_radiation_downward'}
        down_pt.cse.attrs = {'units': 'xxx', 'standard_name': 'Clear sky emissivity'}
        down_pt = down_pt.drop(['SW_direct_tmp'])
        down_pt.SW.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar radiation downwards',
                            'standard_name': 'shortwave_radiation_downward'}
        down_pt.SW_diffuse.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar diffuse radiation downwards',
                                    'standard_name': 'shortwave_diffuse_radiation_downward'}
        return down_pt


    def process_and_store_subset(self, subset_indices):

        try:
            # Get the subset
            subset = self.ERA.sel(time=subset_indices.get('tvec')).isel(latitude=subset_indices.get('latitude'), 
                                longitude=subset_indices.get('longitude'))
            result = self.downscale_atmo(row=subset_indices.get('centroid'), 
                                        subset=subset, 
                                        meta=self.meta, 
                                        ds_solar=self.ds_solar.sel(point_name=subset_indices.get('point_name')), 
                                        da_horizon=self.da_horizon)
            ver_dict = tu.get_versionning()
            result.attrs = {'title': 'Downscaled timeseries with TopoPyScale',
                             'created with': 'TopoPyScale, see more at https://topopyscale.readthedocs.io',
                              'package_version':ver_dict.get('package_version'),
                              'git_commit': ver_dict.get('git_commit'),
                             'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                             'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            result = result.drop_vars(['point_name', 'reference_time'])

            if self.output_format.lower() == 'netcdf':
            
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in result.data_vars}
                result.to_netcdf(self.output_path / self.meta.get('file_pattern').replace("*", str(subset_indices.get('point_id'))),
                          engine='h5netcdf', encoding=encoding, mode='a')

            elif self.output_format.lower() == 'zarr':
                vars = list(result.keys())
                comp = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)
                encoder = dict(zip(vars, [{'compressors': comp}]*len(vars)))
                result = result.expand_dims(dim='point_ind', axis=0)
                result = result.chunk({'time':self.ERA.chunks.get('time')[0], 'point_ind':1})
                #if not (self.output_path / self.store_name).exists():
                #    result.to_zarr(self.output_path / self.store_name, mode='w', encoding=encoder, zarr_format=3)

                #else:
                result[self.varout].to_zarr(self.output_path / self.store_name, mode='a', zarr_format=3, region='auto')
            else:
                raise ValueError('Only netcdf or zarr available')
            print(f"---> Point {subset_indices.get('point_id')} stored to {self.output_format.lower()}")

            
        except Exception as e:
            raise ValueError(f"Error processing subset")

    def multicore_parallel_process_multiple_subsets(self, n_core=4):
        start_time = time.time()
        indices_list = []
        n = self.df_centroids.shape[0]
        for _, row in self.df_centroids.iterrows():
            indices_list.append(self.create_subset_indices(row))

        fun_param = zip(indices_list)  # construct here the tuple that goes into the pooling for arguments
        tu.multicore_pooling(self.process_and_store_subset, fun_param, n_core)
        print('---> Downscaling finished in {}s'.format(np.round(time.time() - start_time), 1))


    def dask_parallel_process_multiple_subsets(self, dask_worker={'n_workers':4, 'threads_per_worker':1, 'memory_target_fraction':0.95, 'memory_limit':'1.5GB'}):
        """Process multiple subsets in parallel and store results to disk."""
        
        start_time = time.time()
        cluster = LocalCluster(processes=True, **dask_worker)

        with Client(cluster) as client:
            print(f"Dask client started with {len(client.scheduler_info()['workers'])} workers")
            
            futures = []
            for _, row in self.df_centroids.iterrows():
                indices = self.create_subset_indices(row)
                future = client.submit(
                    self.process_and_store_subset,
                    indices
                )
                futures.append(future)
            
            print('---> Downscaling finished in {}s'.format(np.round(time.time() - start_time), 1))
            return client.gather(futures)
        

    def downscale_parallel(self, parallel_method='multicore', n_core=4, dask_worker={'n_workers':4, 'threads_per_worker':1, 'memory_target_fraction':0.95, 'memory_limit':'1.5GB'}):
        
        if parallel_method == 'multicore':
            self.multicore_parallel_process_multiple_subsets(n_core)
        elif parallel_method == 'dask':
            self.dask_parallel_process_multiple_subsets(dask_worker)
        else:
            raise ValueError('Method not available')
