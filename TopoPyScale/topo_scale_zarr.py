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


try:
    from zarr.codecs import BloscCodec
except ImportError:
    # For newer versions of zarr
    try:
        from zarr.codecs._blosc import BloscCodec
    except ImportError:
        # Fallback for older versions or different structure
        from zarr import Blosc as BloscCodec
from dask.distributed import LocalCluster, Client
import numpy as np
from datetime import datetime
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

MONTHLY_COEFFS = xr.Dataset(
    {
        'coef': (['month'], [0.35, 0.35, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35])
    },
    coords={'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
)



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

        tstep_dict = {'1h': 1, '3h': 3, '6h': 6}
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


    def _preload_worker_data(self):
        """
        Load shared datasets into memory before multiprocessing Pool creation.

        On Linux with fork(), child processes share read-only memory pages
        from the parent. Pre-loading means each worker accesses the same
        in-memory data via copy-on-write, avoiding per-worker dask I/O.
        """
        import resource
        try:
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            soft, hard = rlimit
            if soft < 4096:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
        except Exception:
            pass

        if hasattr(self.da_horizon.data, 'compute'):
            print("---> Pre-loading da_horizon into shared memory ... ", end='', flush=True)
            self.da_horizon = self.da_horizon.load()
            print("done")
        if hasattr(self.ds_solar.data, 'compute'):
            print("---> Pre-loading ds_solar into shared memory ... ", end='', flush=True)
            self.ds_solar = self.ds_solar.load()
            print("done")
        
        # Pre-compute reusable pyproj transformer to avoid repeated CRS parsing
        self._transformer = Transformer.from_crs(
            "epsg:4326", f"epsg:{self.meta.get('target_epsg')}", always_xy=True
        )
        
        # Pre-compute monthly coefficient data (already done via module constant, no-op)
        MONTHLY_COEFFS.load()  # Ensure it's computed if lazy

    def setup_output_store(self):
        """Setup the output Zarr store with optimal chunking for multiprocessing"""
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Optimal chunking strategy for concurrent writes
        n_points = len(self.df_centroids)  
        n_time = len(self.tvec)
        
        # Time chunking: monthly (744h) or smaller for short periods
        time_chunk = min(744, max(24, n_time // 6))
        # Point chunking: small batches to minimize lock contention
        point_chunk = min(20, max(1, n_points // 10))
        
        shape = (n_points, n_time)
        chunks = (point_chunk, time_chunk)
        
        print(f"---> Creating zarr store: shape {shape}, chunks {chunks}")
        
        # Create data variables with float32 (sufficient precision, 50% memory saving)
        dv = {}
        for var in self.varout:
            dv.update({var: (('point_ind', 'time'), 
                            da.zeros(shape, chunks=chunks, dtype='float32'))})
        
        ds = xr.Dataset(
            data_vars=dv,
            coords={
                'time': self.tvec,
                'point_ind': self.df_centroids.index.values,
                'reference_time': self.tvec[0]
            }
        )

        # Optimized compression: zstd level 3 (fast + good compression)
        comp = BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle', blocksize=0)
        encoder = {var: {'compressor': comp} for var in self.varout}
        
        # Create zarr store with consolidate_metadata for faster opens
        ds.to_zarr(
            self.output_path / self.store_name, 
            mode='w', 
            encoding=encoder, 
            zarr_format=3,
            consolidated=True
        )
        
        print(f"---> Zarr store created at {self.output_path / self.store_name}")

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


    def _downscale_atmo_optimized(self, row, subset, meta, ds_solar, da_horizon):
        """
        Memory and speed optimized version of downscale_atmo for multiprocessing.
        
        Key optimizations:
        - Reuse pre-computed transformer
        - Vectorized vertical interpolation
        - Reduced intermediate object creation
        - Batch computations where possible
        """
        # Physical constants
        g = 9.81
        R = 287.05
        pt_id = row.point_ind

        # ====== Horizontal interpolation (optimized) ==================
        # Use pre-computed transformer to avoid CRS parsing overhead
        lons, lats = np.meshgrid(subset.longitude.values, subset.latitude.values)
        Xs, Ys = self._transformer.transform(lons.flatten(), lats.flatten())
        Xs = Xs.reshape(lons.shape)
        Ys = Ys.reshape(lons.shape)

        # Convert geopotential to meters in-place
        subset.z.values /= g
        subset.z_surf.values /= g

        # Horizontal interpolation weights (vectorized distance calculation)
        dist_sq = (row.x - Xs) ** 2 + (row.y - Ys) ** 2
        if meta.get('interp_method') == 'idw':
            idw = 1.0 / dist_sq
            weights = idw / np.sum(idw)
        elif meta.get('interp_method') == 'linear':
            dist = np.sqrt(dist_sq)
            weights = dist / np.sum(dist) 
        else:
            raise ValueError('ERROR: interpolation method not available')

        da_idw = xr.DataArray(
            weights,
            coords={"latitude": subset.latitude.values, "longitude": subset.longitude.values},
            dims=["latitude", "longitude"]
        )
        
        # Horizontal interpolation
        dw = xr.Dataset.weighted(subset, da_idw)
        subset_interp = dw.sum(['longitude', 'latitude'], keep_attrs=True)

        # Meteorology conversions
        subset_interp = mu.dewT_2_q_magnus(subset_interp, mu.var_era_surf)
        subset_interp = mu.t_rh_2_dewT(subset_interp, mu.var_era_plevel)
        subset_interp = subset_interp.persist()

        # ====== Vertical interpolation (optimized) ===============
        # Pre-allocate output variables in a dictionary (avoid repeated Dataset expansion)
        n_time = len(subset_interp.time)
        data_vars = {}
        
        # Check elevation bounds once
        z_min = subset_interp.z.isel(level=-1)
        below_surface = (row.elevation < z_min).any()
        
        if below_surface:
            # Use surface + lowest level (no interpolation needed)
            ind_z_top = (subset_interp.z - row.elevation).where(
                subset_interp.z > row.elevation
            ).argmin('level')
            top = subset_interp.isel(level=ind_z_top)
            
            data_vars.update({
                't': top.t,
                'u': top.u, 
                'v': top.v,
                'q': top.q,
                'p': top.level * 100 * np.exp(-(row.elevation - top.z) / (0.5 * (top.t + top.t) * R / g))
            })
        else:
            # Vectorized vertical interpolation
            z_diff = subset_interp.z - row.elevation
            
            # Find bounding levels for all times at once
            ind_z_bot = z_diff.where(z_diff < 0).argmax('level')
            ind_z_top = z_diff.where(z_diff > 0).argmin('level')
            
            top = subset_interp.isel(level=ind_z_top)
            bot = subset_interp.isel(level=ind_z_bot)
            
            # Linear interpolation weights (vectorized)
            z_dist = np.stack([np.abs(bot.z - row.elevation).values, 
                              np.abs(top.z - row.elevation).values])
            weights = z_dist / np.sum(z_dist, axis=0)
            
            # Apply interpolation to all variables at once
            data_vars.update({
                't': bot.t * weights[1] + top.t * weights[0],
                'u': bot.u * weights[1] + top.u * weights[0],
                'v': bot.v * weights[1] + top.v * weights[0], 
                'q': bot.q * weights[1] + top.q * weights[0],
                'p': top.level * 100 * np.exp(-(row.elevation - top.z) / (0.5 * (top.t + data_vars['t']) * R / g))
            })

        # ====== Wind and precipitation (vectorized) ===============
        month_data = subset_interp.time.dt.month
        if meta.get('precip_lapse_rate_flag'):
            monthly_coef = MONTHLY_COEFFS.coef.sel(month=month_data).values
            elev_diff = (row.elevation - subset_interp.z_surf) * 1e-3
            precip_lapse = (1 + monthly_coef * elev_diff) / (1 - monthly_coef * elev_diff)
        else:
            precip_lapse = xr.ones_like(subset_interp.time)
        
        # Wind calculations (vectorized)
        theta = np.arctan2(-data_vars['u'], -data_vars['v'])
        wd = np.where(theta < 0, theta + 2*np.pi, theta)
        ws = np.sqrt(data_vars['u']**2 + data_vars['v']**2)
        
        data_vars.update({
            'tp': precip_lapse * subset_interp.tp / meta.get('tstep') * 1000,
            'wd': wd,
            'ws': ws
        })

        # ====== Radiation (optimized) ===============
        # Longwave 
        x1, x2 = 0.43, 5.7
        sbc = 5.67e-8

        # Batch meteorology calculations
        temp_vars = xr.Dataset(data_vars)
        temp_vars = mu.mixing_ratio(temp_vars, mu.var_era_plevel)
        temp_vars = mu.vapor_pressure(temp_vars, mu.var_era_plevel)
        subset_interp = mu.mixing_ratio(subset_interp, mu.var_era_surf)
        subset_interp = mu.vapor_pressure(subset_interp, mu.var_era_surf)

        # Clear sky emissivity
        cse_point = 0.23 + x1 * (temp_vars.vp / temp_vars.t) ** (1 / x2)
        cse_grid = 0.23 + x1 * (subset_interp.vp / subset_interp.t2m) ** (1 / x2)

        tstep_seconds = pd.Timedelta(f"{meta.get('tstep')}h").seconds
        cle = (subset_interp.strd / tstep_seconds) / (sbc * subset_interp.t2m ** 4) - cse_grid
        aef = cse_point + cle

        if meta.get('lw_terrain_flag'):
            data_vars['LW'] = (row.svf * aef * sbc * temp_vars.t ** 4 + 
                              0.5 * (1 + np.cos(row.slope)) * (1 - row.svf) * 0.99 * 5.67e-8 * (273.15 ** 4))
        else:
            data_vars['LW'] = row.svf * aef * sbc * temp_vars.t ** 4

        # Shortwave (pre-computed horizon + vectorized operations)
        sunset = ds_solar.sunset.astype(bool).compute()
        mu0 = ds_solar.mu0
        SWtoa = ds_solar.SWtoa
        
        # Clearness and diffuse calculations
        kt = xr.zeros_like(subset_interp.ssrd)
        kt = kt.where(sunset, (subset_interp.ssrd / tstep_seconds) / SWtoa)
        kt = np.clip(kt, 0, 1)
        kd = 0.952 - 1.041 * np.exp(-np.exp(2.3 - 4.702 * kt))

        sw_total = subset_interp.ssrd / tstep_seconds
        sw_total = np.maximum(sw_total, 0)
        sw_diffuse = kd * sw_total
        sw_direct_grid = sw_total - sw_diffuse

        # Direct SW with Beer's law and shading (optimized)
        ka = xr.zeros_like(subset_interp.ssrd)
        valid = ~sunset & (sw_direct_grid > 0) & (SWtoa > 0)
        ka = ka.where(~valid, (g * mu0 / temp_vars.p) * np.log(SWtoa / sw_direct_grid))

        # Illumination (vectorized)
        cos_illum = (mu0 * np.cos(row.slope) + 
                    np.sin(ds_solar.zenith) * np.sin(row.slope) * 
                    np.cos(ds_solar.azimuth - row.aspect))
        cos_illum = np.maximum(cos_illum, 0)

        # Horizon shading (pre-optimized)
        horizon_slice = da_horizon.sel(x=row.x, y=row.y, method='nearest')
        az_deg = np.rad2deg(ds_solar.azimuth.values)
        az_inds = np.abs(horizon_slice.azimuth.values - az_deg[:, np.newaxis]).argmin(axis=1)
        horizon = horizon_slice.values[az_inds]
        shade = horizon > ds_solar.elevation.values

        # Final SW direct calculation
        sw_direct_toa = xr.zeros_like(sw_total)
        sw_direct_toa = sw_direct_toa.where(sunset, SWtoa * np.exp(-ka * temp_vars.p / (g * mu0)))
        
        data_vars.update({
            'SW_diffuse': row.svf * sw_diffuse,
            'SW_direct': sw_direct_toa.where(sunset, sw_direct_toa * (cos_illum / mu0) * (1 - shade)),
            'cse': cse_point
        })
        data_vars['SW'] = data_vars['SW_diffuse'] + data_vars['SW_direct']

        # Create optimized output dataset
        down_pt = xr.Dataset(
            data_vars=data_vars,
            coords={'time': subset_interp.time, 'point_ind': pt_id}
        )

        # Add metadata (minimal set)
        down_pt.t.attrs = {'units': 'K', 'standard_name': 'air_temperature'}
        down_pt.p.attrs = {'units': 'Pa', 'standard_name': 'air_pressure'} 
        down_pt.tp.attrs = {'units': 'mm hr**-1', 'standard_name': 'precipitation_flux'}
        down_pt.ws.attrs = {'units': 'm s**-1', 'standard_name': 'wind_speed'}
        down_pt.LW.attrs = {'units': 'W m**-2', 'standard_name': 'surface_downwelling_longwave_flux'}
        down_pt.SW.attrs = {'units': 'W m**-2', 'standard_name': 'surface_downwelling_shortwave_flux'}

        return down_pt

    def downscale_atmo(self, row, subset, meta, ds_solar, da_horizon):
        """
        Downscaling method following the toposcale method introduced by J. Fiddes et al (2012)
        """
        # Physical constants
        g = 9.81  # Acceleration of gravity [ms^-1]
        R = 287.05  # Gas constant for dry air [JK^-1kg^-1]
        pt_id = row.point_ind
        t_sections = {}
        t0 = time.perf_counter()
        print(f'Downscaling t,q,p,tp,ws,wd for point: {pt_id}')

        # ====== Horizontal interpolation ====================
        t_start = time.perf_counter()
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
        dw = xr.Dataset.weighted(subset, da_idw)
        subset_interp = dw.sum(['longitude', 'latitude'],
                                 keep_attrs=True)  # compute horizontal inverse weighted horizontal interpolation

        # ============ Extract specific humidity (q) for both dataset ============
        subset_interp = mu.dewT_2_q_magnus(subset_interp, mu.var_era_surf)
        subset_interp = mu.t_rh_2_dewT(subset_interp, mu.var_era_plevel)
        t_sections['horiz_interp'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        subset_interp = subset_interp.persist()
        t_sections['persist'] = time.perf_counter() - t_start

        down_pt = xr.Dataset(coords={
                'time': subset_interp.time,
                'point_ind': pt_id})

        t_start = time.perf_counter()
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
                    f'Upper pressure level {subset_interp.level.min().values} hPa geopotential is lower than cluster mean elevation {row.elevation} {subset_interp.z}')


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
        t_sections['vert_interp'] = time.perf_counter() - t_start

            # ======= logic  to compute ws, wd without loading data in memory, and maintaining the power of dask
        t_start = time.perf_counter()
        down_pt['month'] = ('time', down_pt.time.dt.month.data)
        print("precip lapse rate = " + str(meta.get('precip_lapse_rate_flag')))
        if meta.get('precip_lapse_rate_flag'):
            monthly_coef = MONTHLY_COEFFS.coef.sel(month=down_pt.month.values).data
            elev_diff = (row.elevation - subset_interp.z_surf) * 1e-3
            down_pt['precip_lapse_rate'] = (1 + monthly_coef * elev_diff) / (1 - monthly_coef * elev_diff)
        else:
            down_pt['precip_lapse_rate'] = down_pt.t * 0 + 1

        down_pt['tp'] = down_pt.precip_lapse_rate * subset_interp.tp * 1 / meta.get('tstep') * 10 ** 3  # Convert to mm/hr
        down_pt['theta'] = np.arctan2(-down_pt.u, -down_pt.v)
        down_pt['theta_neg'] = (down_pt.theta < 0) * (down_pt.theta + 2 * np.pi)
        down_pt['theta_pos'] = (down_pt.theta >= 0) * down_pt.theta
        down_pt = down_pt.drop_vars('theta')
        down_pt['wd'] = (down_pt.theta_pos + down_pt.theta_neg)  # direction in Rad
        down_pt['ws'] = np.sqrt(down_pt.u ** 2 + down_pt.v ** 2)
        down_pt = down_pt.drop_vars(['theta_pos', 'theta_neg', 'month'])
        t_sections['wind_precip'] = time.perf_counter() - t_start

        #============================================================
        #           Radiation corrections
        #============================================================
        print(f"Downscaling SW,LW for point: {pt_id}")

        t_start = time.perf_counter()
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
        t_sections['lw_radiation'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        kt = subset_interp.ssrd * 0
        sunset = ds_solar.sunset.astype(bool).compute()
        t_sections['sw_sync_sunset'] = time.perf_counter() - t_start
        mu0 = ds_solar.mu0
        SWtoa = ds_solar.SWtoa

        t_start = time.perf_counter()
        try:
            kt[~sunset] = (subset_interp.ssrd[~sunset] / tstep_seconds) / SWtoa[~sunset]  # clearness index
        except:
            ValueError("ds_solar is probably not mathcing the timerange of the downscaling job.")
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
        down_pt = down_pt.drop_vars(['cos_illumination_tmp'])
        illumination_mask = down_pt['cos_illumination'] < 0
        illumination_mask = illumination_mask.compute()
        down_pt['cos_illumination'][illumination_mask] = 0

        # Binary shadow masks.
        horizon_slice = da_horizon.sel(x=row.x, y=row.y, method='nearest').load()
        az_deg = np.rad2deg(ds_solar.azimuth.values)
        az_inds = np.abs(horizon_slice.azimuth.values - az_deg[:, np.newaxis]).argmin(axis=1)
        horizon = xr.DataArray(horizon_slice.values[az_inds], dims='time', coords={'time': ds_solar.time})
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
        down_pt = down_pt.drop_vars(['level', 'precip_lapse_rate'])
        t_sections['sw_radiation'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
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
        down_pt = down_pt.drop_vars(['SW_direct_tmp'])
        down_pt.SW.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar radiation downwards',
                            'standard_name': 'shortwave_radiation_downward'}
        down_pt.SW_diffuse.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar diffuse radiation downwards',
                                    'standard_name': 'shortwave_diffuse_radiation_downward'}
        t_sections['finalize'] = time.perf_counter() - t_start

        t_sections['total'] = time.perf_counter() - t0
        down_pt.attrs['timing_sections'] = str(t_sections)
        down_pt.attrs['timing_horiz_interp'] = t_sections['horiz_interp']
        down_pt.attrs['timing_persist'] = t_sections['persist']
        down_pt.attrs['timing_vert_interp'] = t_sections['vert_interp']
        down_pt.attrs['timing_wind_precip'] = t_sections['wind_precip']
        down_pt.attrs['timing_lw_radiation'] = t_sections['lw_radiation']
        down_pt.attrs['timing_sw_sync_sunset'] = t_sections['sw_sync_sunset']
        down_pt.attrs['timing_sw_radiation'] = t_sections['sw_radiation']
        down_pt.attrs['timing_finalize'] = t_sections['finalize']
        down_pt.attrs['timing_total'] = t_sections['total']
        return down_pt


    def _process_preloaded_subset(self, task):
        """
        Optimized worker function that uses pre-loaded data to minimize I/O.
        
        Parameters
        ----------
        task : dict
            Contains pre-extracted subset, centroid, and ds_solar data.
        """
        try:
            result = self._downscale_atmo_optimized(
                row=task['centroid'],
                subset=task['subset'], 
                meta=self.meta,
                ds_solar=task['ds_solar'],
                da_horizon=self.da_horizon
            )
            
            # Minimal versioning info (reduce redundancy)
            ver_dict = tu.get_versionning()
            result.attrs.update({
                'package_version': ver_dict.get('package_version'),
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Remove timing attrs to save memory (keep only in profile mode)
            timing_keys = [k for k in result.attrs.keys() if k.startswith('timing_')]
            for k in timing_keys:
                del result.attrs[k]
            
            result = result.drop_vars(['point_name', 'reference_time'], errors='ignore')

            if self.output_format.lower() == 'netcdf':
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in result.data_vars}
                result.to_netcdf(
                    self.output_path / self.meta.get('file_pattern').replace("*", str(task['point_id'])),
                    engine='h5netcdf', encoding=encoding, mode='w'
                )
            elif self.output_format.lower() == 'zarr':
                # Optimized zarr write with explicit region (avoids locks)  
                result = self._prepare_zarr_output(result, task['point_id'])
                self._write_zarr_region(result, task['point_id'])
            else:
                raise ValueError('Only netcdf or zarr available')
            
            result = None  # Explicit cleanup
            print(f"---> Point {task['point_id']} stored to {self.output_format.lower()}")
            
        except Exception as e:
            print(f"ERROR processing point {task['point_id']}: {e}")
            raise ValueError(f"Error processing point {task['point_id']}")

    def _prepare_zarr_output(self, result, point_id):
        """Prepare result for optimized zarr writing."""
        # Find the point index in df_centroids
        point_idx = self.df_centroids.index.get_loc(point_id)
        
        # Ensure result has point_ind dimension for zarr alignment  
        if 'point_ind' not in result.dims:
            result = result.expand_dims('point_ind', axis=0)
            result = result.assign_coords(point_ind=[point_id])
        
        # Rechunk to match zarr store chunking for efficient writes
        zarr_chunks = {'time': 744, 'point_ind': 1}  # Match setup_output_store
        result = result.chunk(zarr_chunks)
        
        return result

    def _write_zarr_region(self, result, point_id):
        """Write result to specific zarr region to avoid lock contention."""
        # Calculate explicit region for this point
        point_idx = self.df_centroids.index.get_loc(point_id) 
        time_slice = slice(0, len(self.tvec))
        point_slice = slice(point_idx, point_idx + 1)
        
        # Explicit region avoids zarr metadata locks
        region = {'point_ind': point_slice, 'time': time_slice}
        
        try:
            # Write only the requested variables with explicit region
            result[self.varout].to_zarr(
                self.output_path / self.store_name,
                mode='r+',  # Read-write mode (store must exist)
                region=region,
                zarr_format=3
            )
        except Exception as e:
            # Fallback to append mode if region write fails
            print(f"Region write failed for point {point_id}, using append mode: {e}")
            result = result.expand_dims('point_ind', axis=0)
            result[self.varout].to_zarr(
                self.output_path / self.store_name,
                mode='a',
                zarr_format=3,
                region='auto'
            )

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
                result = None

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
                result=None
            else:
                raise ValueError('Only netcdf or zarr available')
            print(f"---> Point {subset_indices.get('point_id')} stored to {self.output_format.lower()}")

            
        except Exception as e:
            raise ValueError("Error processing subset")

    def profile_atmo(self, n_points=3, profile_path=None):
        """
        Profile the core downscale_atmo algorithm serially (no Pool).

        Runs cProfile + per-section wall-clock timing on the first n_points.
        Prints a timing breakdown and saves .prof file for snakeviz.

        Parameters
        ----------
        n_points : int
            Number of centroid points to profile (serial, in main process).
        profile_path : str or Path, optional
            Where to save the cProfile stats (default: output_path/profile_atmo.prof).
        """
        import cProfile
        import io
        import pstats
        import tracemalloc

        n = min(n_points, len(self.df_centroids))
        print(f"\n{'='*70}")
        print(f"  Profiling downscale_atmo on {n} point(s) (serial, no multiprocessing)")
        print(f"{'='*70}\n")

        profiler = cProfile.Profile()
        tracemalloc.start()

        # Take a memory baseline before any computation
        _baseline = tracemalloc.take_snapshot()

        wall_times = {}
        profiler.enable()
        for i in range(n):
            row = self.df_centroids.iloc[i]
            idx = self.create_subset_indices(row)
            subset = self.ERA.sel(time=idx['tvec']).isel(
                latitude=idx['latitude'], longitude=idx['longitude'])
            ds_solar_pt = self.ds_solar.sel(point_name=idx['point_name'])

            t0 = time.perf_counter()
            result = self.downscale_atmo(
                row, subset, self.meta, ds_solar_pt, self.da_horizon)
            elapsed = time.perf_counter() - t0
            wall_times[f"pt_{row.point_ind}"] = elapsed

            if i == 0:
                sections = result.attrs
        profiler.disable()

        post_snap = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # ── cProfile report ──
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats(40)
        print(s.getvalue())

        # ── Per-section timing breakdown ──
        section_keys = [
            'timing_horiz_interp', 'timing_persist', 'timing_vert_interp',
            'timing_wind_precip', 'timing_lw_radiation', 'timing_sw_sync_sunset',
            'timing_sw_radiation', 'timing_finalize', 'timing_total',
        ]
        label_width = 22
        print(f"\n{'Section timing breakdown (pt 0):':-^70}")
        print(f"  {'Section':<{label_width}} {'Time (s)':>10}  {'%':>6}")
        print(f"  {'-' * (label_width + 18)}")
        for key in section_keys:
            label = key.replace('timing_', '')
            val = sections.get(key, 0)
            pct = val / sections.get('timing_total', 1) * 100
            print(f"  {label:<{label_width}} {val:10.3f}  {pct:5.1f}%")
        print(f"  {'-' * (label_width + 18)}")
        tot = sections.get('timing_total', sum(wall_times.values()))
        print(f"  {'all_pts_wall':<{label_width}} {sum(wall_times.values()):10.3f}")
        if n > 1:
            print(f"  {'per_pt_avg':<{label_width}} {sum(wall_times.values()) / n:10.3f}")

        # ── Memory report ──
        top_stats = post_snap.compare_to(_baseline, 'lineno', cumulative=True)
        print(f"\n{'Top memory allocations (cumulative diff from baseline):':-^70}")
        allocs = [s for s in top_stats if 'topo_scale_zarr' in str(s.traceback)]
        if not allocs:
            allocs = top_stats  # fallback to top-level
        for stat in allocs[:10]:
            size_mb = stat.size_diff / (1024 * 1024)
            print(f"  {stat.size_diff:>12,d} B  ({size_mb:7.2f} MB)  — {stat.count_diff} allocs")
            for line in stat.traceback.format()[:3]:
                print(f"     {line.strip()}")

        # ── Save ──
        if profile_path is None:
            profile_path = Path(self.output_path) / 'profile_atmo.prof'
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        ps.dump_stats(str(profile_path))
        print(f"\n  cProfile stats saved to: {profile_path}")
        print(f"  Visualize with:        snakeviz {profile_path}\n")

    def multicore_parallel_process_multiple_subsets(self, n_core=4, chunk_size=None):
        """
        Optimized multiprocessing with ERA5 subset pre-loading and chunked processing.
        
        Parameters
        ---------- 
        n_core : int
            Number of CPU cores to use.
        chunk_size : int, optional
            Process points in chunks to limit memory usage. If None, uses
            adaptive chunking based on available cores and point count.
        """
        start_time = time.time()

        self._preload_worker_data()

        # Adaptive chunk sizing to balance memory and CPU efficiency
        n_points = len(self.df_centroids)
        if chunk_size is None:
            # Rule of thumb: ~50 points per core to balance parallelism and memory
            chunk_size = max(50, min(200, n_points // n_core))
        
        print(f"---> Processing {n_points} points in chunks of {chunk_size}")

        # Pre-batch ERA5 subsets to reduce per-worker zarr I/O
        print("---> Pre-extracting unique ERA5 subsets ... ", end='', flush=True)
        batched_subsets = {}
        unique_grids = set()
        
        for _, row in self.df_centroids.iterrows():
            idx = self.create_subset_indices(row)
            key = (tuple(idx['latitude']), tuple(idx['longitude']))
            unique_grids.add((key, tuple(idx['tvec'][[0, -1]])))  # Store first/last time for chunking
        
        # Load unique ERA5 subsets in batches to manage memory
        for i, (spatial_key, time_range) in enumerate(unique_grids):
            lat_idx, lon_idx = spatial_key
            start_time_chunk, end_time_chunk = time_range
            
            subset = self.ERA.sel(time=slice(start_time_chunk, end_time_chunk)).isel(
                latitude=list(lat_idx), longitude=list(lon_idx)
            )
            # Load and persist for worker sharing
            batched_subsets[spatial_key] = subset.load()
            
        print(f"done ({len(batched_subsets)} unique grids)")

        # Process points in chunks
        total_processed = 0
        for chunk_start in range(0, n_points, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_points)
            chunk_df = self.df_centroids.iloc[chunk_start:chunk_end]
            
            print(f"---> Processing chunk {chunk_start//chunk_size + 1}/{(n_points-1)//chunk_size + 1} "
                  f"(points {chunk_start}-{chunk_end-1})")
            
            # Create tasks for this chunk
            tasks = []
            for _, row in chunk_df.iterrows():
                idx = self.create_subset_indices(row)
                spatial_key = (tuple(idx['latitude']), tuple(idx['longitude']))
                subset = batched_subsets[spatial_key].sel(time=idx['tvec'])
                ds_solar_pt = self.ds_solar.sel(point_name=idx['point_name'])
                
                task = {
                    'centroid': row,
                    'subset': subset,
                    'ds_solar': ds_solar_pt,
                    'point_id': idx['point_id'],
                    'point_name': idx['point_name']
                }
                tasks.append(task)

            # Process chunk with multiprocessing
            fun_param = ((task,) for task in tasks)
            tu.multicore_pooling(self._process_preloaded_subset, fun_param, n_core)
            
            total_processed += len(chunk_df)
            print(f"    Completed {total_processed}/{n_points} points")

        elapsed = np.round(time.time() - start_time, 0)
        print(f'---> Downscaling finished in {elapsed}s  -- or -- {elapsed/n_points:.1f} s/pts')


    def dask_parallel_process_multiple_subsets(self, dask_worker={'n_workers':4, 'threads_per_worker':1, 'memory_target_fraction':0.95, 'memory_limit':'1.5GB'}):
        """Process multiple subsets in parallel and store results to disk."""

        start_time = time.time()
        self._preload_worker_data()
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
        

    def downscale_parallel(self, parallel_method='multicore', n_core=4, 
                          dask_worker={'n_workers':4, 'threads_per_worker':1, 'memory_target_fraction':0.95, 'memory_limit':'1.5GB'},
                          output_format='zarr', chunk_size=None):
        """
        Run parallel downscaling with optimized zarr I/O (recommended).
        
        Parameters
        ----------
        parallel_method : str 
            'multicore' (recommended) or 'dask'
        n_core : int
            Number of CPU cores for multicore method
        dask_worker : dict  
            Dask worker configuration  
        output_format : str
            'zarr' (recommended, fast concurrent writes) or 'netcdf' (many files)
        chunk_size : int, optional
            Points per processing chunk (for memory management)
        """
        # Force zarr output for best performance
        if output_format == 'zarr' and not hasattr(self, 'store_name'):
            raise ValueError("zarr output requires store_name parameter in constructor")
            
        original_format = self.output_format
        self.output_format = output_format
        
        try:
            if parallel_method == 'multicore':
                if chunk_size is not None:
                    self.multicore_parallel_process_multiple_subsets(n_core, chunk_size)
                else:
                    self.multicore_parallel_process_multiple_subsets(n_core)
            elif parallel_method == 'dask':
                self.dask_parallel_process_multiple_subsets(dask_worker)
            else:
                raise ValueError('Method not available')
        finally:
            # Restore original format
            self.output_format = original_format
