"""
Toposcale functionalities

S. Filhol, Oct 2021

======= Organization of input data to Toposcale ======
dem_description (nb_points)
    X
    Y,
    elev,
    slope,
    aspect,
    svf,
    x_ERA,
    y_ERA,
    elev_ERA,

horizon_angles (nb_points * bins)
solar_geom (nb_points * time * 2)
surface_climate_data (spatial + time + var)
plevels_climate_data (spatial + time + var + plevel)


======= Dataset naming convention:  =========
ds_surf => dataset direct from ERA5 single level surface
ds_plev => dataset direct from ERA5 pressure levels

ds_surf_pt => 3*3 grid ERA5 single surface level around a given point (lat,lon)
ds_plev_pt => 3*3 grid ERA5 pressure level around a given point (lat,lon)

plev_interp => horizontal interpolation of the 3*3 grid at the point (lat,lon) for the pressure levels
surf_interp => horizontal interpolation of the 3*3 grid at the point (lat,lon) for the single surface level

top => closest pressure level above the point (lat,lon,elev) for each timesep
bot => closest pressure level below the point (lat,lon,elev) for each timesep

down_pt => downscaled data time series (t, u, v, q, LW, SW, tp)

TODO:
- check that transformation from r,t to q for plevels is working fine
- add metadata to all newly created variables in datasets ds_psurf, ds_plevel, down_pt
- upscale method for all points in df_centroids:
    - method (1) simply using a for loop over each row
    - method (2) concatenate 3*3 grid along an additional dimension to process all points in one go.

"""
import pdb

import pandas as pd
import xarray as xr
from pyproj import Transformer
import numpy as np
import sys, time
from TopoPyScale import meteo_util as mu
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
import os

# Physical constants
g = 9.81    #  Acceleration of gravity [ms^-1]
R = 287.05  #  Gas constant for dry air [JK^-1kg^-1]

def clear_files(path):
    # Clear outputs/tmp/ folder
    for dirpath, dirnames, filenames in os.walk(path):
        # Remove regular files, ignore directories
        for filename in filenames:
            os.unlink(os.path.join(dirpath, filename))
        print(f'{path} cleaned')

def downscale_climate(project_directory,
                      df_centroids,
                      horizon_da,
                      ds_solar,
                      target_EPSG,
                      start_date,
                      end_date,
                      interp_method='idw',
                      lw_terrain_flag=True,
                      tstep='1H',
                      file_pattern='down_pt*.nc'):
    """
    Function to perform downscaling of climate variables (t,q,u,v,tp,SW,LW) based on Toposcale logic

    Args:
        project_directory: path to project root directory
        df_centroids (dataframe): containing a list of point for which to downscale (includes all terrain data)
        horizon_da (dataarray): horizon angles for a list of azimuth
        target_EPSG (int): EPSG code of the DEM
        interp_method (str): interpolation method for horizontal interp. 'idw' or 'linear'
        lw_terrain_flag (bool): flag to compute contribution of surrounding terrain to LW or ignore
        tstep (str): timestep of the input data, default = 1H
        file_pattern (str): filename pattern for storing downscaled points, default = 'down_pt_*.nc'

    Returns:
        dataset: downscaled data organized with time, point_id, lat, long
    """
    print('\n---> Downscaling climate to list of points using TopoScale')

    clear_files(f'{project_directory}outputs/tmp')

    start_time = time.time()
    tstep_dict = {'1H': 1, '3H': 3, '6H': 6}
    # =========== Open dataset with Dask =================
    tvec = pd.date_range(start_date, pd.to_datetime(end_date) + pd.to_timedelta('1D'), freq=tstep, closed='left')
    ds_plev = xr.open_mfdataset(project_directory + 'inputs/climate/PLEV*.nc', parallel=True).sel(time=tvec.values)
    ds_surf = xr.open_mfdataset(project_directory + 'inputs/climate/SURF*.nc', parallel=True).sel(time=tvec.values)

    # this block handles the expver dimension that is in downloaded ERA5 data if data is ERA5/ERA5T mix. If only ERA5 or
    # only ERA5T it is not present. ERA5T data can be present in the timewindow T-5days to T -3months, where T is today.
    # https://code.mpimet.mpg.de/boards/1/topics/8961
    # https://confluence.ecmwf.int/display/CUSF/ERA5+CDS+requests+which+return+a+mixture+of+ERA5+and+ERA5T+data
    try:
        # in case of there being an expver dimension and it has two or more values, select first value
        expverN = ds_plev["expver"].values[0]
        # create new datset when this dimoension only has a single value
        ds_plev = ds_plev.sel(expver=expverN)
        # finally this drops the coordinate
        ds_plev = ds_plev.drop("expver")
    except:
        print("No ERA5T  PRESSURE data present with additional dimension <expver>")

    try:
        # in case of there being an expver dimension and it has two or more values, select first value
        expverN = ds_surf["expver"].values[0]
        # create new datset when this dimoension only has a single value
        ds_surf = ds_surf.sel(expver=expverN)
        # finally this drops the coordinate
        ds_surf = ds_surf.drop("expver")
    except:
        print("No ERA5T SURFACE data present with additional dimension <expver>")

    # ============ Convert lat lon to projected coordinates ==================
    trans = Transformer.from_crs("epsg:4326", "epsg:" + str(target_EPSG), always_xy=True)
    nxv,  nyv = np.meshgrid(ds_surf.longitude.values, ds_surf.latitude.values)
    nlons, nlats = trans.transform(nxv, nyv)
    ds_surf = ds_surf.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})
    ds_plev = ds_plev.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})

    # ============ Loop over each point ======================================
    # Loop over each points (lat,lon) for which to downscale climate variable using Toposcale method


    dataset = []
    dpt_list = []
    dpt_paths = []
    surf_paths  = []
    surf_list = []

    n_digits = len(str(df_centroids.index.max()))

    for i, row in df_centroids.iterrows():
        pt_id = np.int(row.point_id)
        print('Downscaling t,q,u,v,tp,p for point: {} out of {}'.format(pt_id+1, df_centroids.index.max()+1))
        # =========== Extract the 3*3 cells centered on a given point ============
        ind_lat = np.abs(ds_surf.latitude-row.y).argmin()
        ind_lon = np.abs(ds_surf.longitude-row.x).argmin()
        ds_surf_pt = ds_surf.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])
        ds_plev_pt = ds_plev.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])

        # ====== Horizontal interpolation ====================
        interp_method = 'idw'
        Xs, Ys = np.meshgrid(ds_plev_pt.longitude.values, ds_plev_pt.latitude.values)
        dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
        if interp_method == 'idw':
            idw = 1/(dist**2)
            weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1
        elif interp_method == 'linear':
            weights = dist / np.sum(dist)
        else:
            sys.exit('ERROR: interpolation method not available')

        # create a dataArray of weights to then propagate through the dataset
        da_idw = xr.DataArray(data = weights,
                              coords={
                                  "latitude": ds_plev_pt.latitude.values,
                                  "longitude": ds_plev_pt.longitude.values,
                              },
                              dims=["latitude", "longitude"]
                              )
        dw = xr.Dataset.weighted(ds_plev_pt, da_idw)
        plev_interp = dw.sum(['longitude', 'latitude'], keep_attrs=True)    # compute horizontal inverse weighted horizontal interpolation
        dww = xr.Dataset.weighted(ds_surf_pt, da_idw)
        surf_interp = dww.sum(['longitude', 'latitude'], keep_attrs=True)    # compute horizontal inverse weighted horizontal interpolation

        # ========= Converting z from [m**2 s**-2] to [m] asl =======
        plev_interp.z.values = plev_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
        plev_interp.z.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of plevel'}

        surf_interp.z.values = surf_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
        surf_interp.z.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of ERA5 surface'}

        # ============ Extract specific humidity (q) for both dataset ============
        surf_interp = mu.dewT_2_q_magnus(surf_interp, mu.var_era_surf)
        plev_interp = mu.t_rh_2_dewT(plev_interp, mu.var_era_plevel)

        down_pt = xr.Dataset(coords={
                'time': plev_interp.time,
                'point_id': pt_id
            })

        if (row.elevation < plev_interp.z.isel(level=-1)).sum():
            print("---> WARNING: Point {} is {} m lower than the 1000hPa geopotential\n=> "
                  "Values sampled from Psurf and lowest Plevel. No vertical interpolation".
                  format(i, np.round(np.min(row.elevation - plev_interp.z.isel(level=-1).values),0)))
            ind_z_top = (plev_interp.where(plev_interp.z > row.elevation).z - row.elevation).argmin('level')
            top = plev_interp.isel(level=ind_z_top)

            down_pt['t'] = top['t']
            down_pt['u'] = top.u
            down_pt['v'] = top.v
            down_pt['q'] = top.q
            down_pt['tp'] = surf_interp.tp  * 1 / tstep_dict.get(tstep) * 10**3 # Convert to mm/hr
            down_pt['p'] = top.level*(10**2) * np.exp(-(row.elevation-top.z) / (0.5 * (top.t + down_pt.t) * R / g))  # Pressure in bar

        else:
            # ========== Vertical interpolation at the DEM surface z  ===============
            ind_z_bot = (plev_interp.where(plev_interp.z < row.elevation).z - row.elevation).argmax('level')
            ind_z_top = (plev_interp.where(plev_interp.z > row.elevation).z - row.elevation).argmin('level')
            top = plev_interp.isel(level=ind_z_top)
            bot = plev_interp.isel(level=ind_z_bot)

            # Preparing interpolation weights for linear interpolation =======================
            dist = np.array([np.abs(bot.z - row.elevation).values, np.abs(top.z - row.elevation).values])
            #idw = 1/dist**2
            #weights = idw / np.sum(idw, axis=0)
            weights = dist / np.sum(dist, axis=0)

            # ============ Creating a dataset containing the downscaled timeseries ========================
            down_pt['t'] = bot.t * weights[1] + top.t * weights[0]
            down_pt['u'] = bot.u * weights[1] + top.u * weights[0]
            down_pt['v'] = bot.v * weights[1] + top.v * weights[0]
            down_pt['q'] = bot.q * weights[1] + top.q * weights[0]
            down_pt['p'] = top.level*(10**2) * np.exp(-(row.elevation-top.z) / (0.5 * (top.t + down_pt.t) * R / g))  # Pressure in bar
            down_pt['tp'] = surf_interp.tp  * 1 / tstep_dict.get(tstep) * 10**3 # Convert to mm/hr

        # ======= logic  to compute ws, wd without loading data in memory, and maintaining the power of dask
        down_pt['theta'] = np.arctan2(-down_pt.u, -down_pt.v)
        down_pt['theta_neg'] = (down_pt.theta < 0)*(down_pt.theta + 2*np.pi)
        down_pt['theta_pos'] = (down_pt.theta >= 0)*down_pt.theta
        down_pt = down_pt.drop('theta')
        down_pt['wd'] = (down_pt.theta_pos + down_pt.theta_neg)  # direction in Rad
        down_pt['ws'] = np.sqrt(down_pt.u ** 2 + down_pt.v**2)
        down_pt = down_pt.drop(['theta_pos', 'theta_neg'])


        dpt_list.append(down_pt)
        dpt_paths.append(project_directory + 'outputs/tmp/down_pt_{}.nc'.format(str(pt_id).zfill(n_digits)))
        surf_list.append(surf_interp)
        surf_paths.append(project_directory + 'outputs/tmp/surf_interp_{}.nc'.format(str(pt_id).zfill(n_digits)))

        down_pt = None
        surf_interp = None

    print('---> Storing to outputs/tmp/')
    xr.save_mfdataset(dpt_list, dpt_paths, engine='h5netcdf')
    dpt_list = None
    dpt_paths = None
    xr.save_mfdataset(surf_list, surf_paths, engine='h5netcdf')
    surf_list = None
    surf_paths = None

        
    ds_list = []
    path_list = []
    for i, row in df_centroids.iterrows():
        pt_id = np.int(row.point_id)
        print('Downscaling LW, SW for point: {} out of {}'.format(pt_id+1,
                                                                  df_centroids.point_id.max()+1))

        down_pt = xr.open_dataset('outputs/tmp/down_pt_{}.nc'.format(str(pt_id).zfill(n_digits)), chunks='auto', engine='h5netcdf')
        surf_interp = xr.open_dataset('outputs/tmp/surf_interp_{}.nc'.format(str(pt_id).zfill(n_digits)), chunks='auto', engine='h5netcdf')


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
        # Calculate the "cloud" emissivity, UNIT OF STRD (J/m2)
        surf_interp['cle'] = (surf_interp.strd/pd.Timedelta('1H').seconds) / (sbc * surf_interp.t2m**4) - surf_interp['cse']
        # Use the former cloud emissivity to compute the all sky emissivity at subgrid.
        surf_interp['aef'] = down_pt['cse'] + surf_interp['cle']
        if lw_terrain_flag:
            down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4 + \
                            0.5 * (1 + np.cos(row.slope)) * (1 - row.svf) * 0.99 * 5.67e-8 * (273.15**4)
        else:
            down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4

        kt = surf_interp.ssrd * 0
        sunset = ds_solar.sel(point_id=pt_id).sunset.astype(bool)
        mu0 = ds_solar.sel(point_id=pt_id).mu0
        SWtoa = ds_solar.sel(point_id=pt_id).SWtoa


        #pdb.set_trace()
        kt[~sunset] = (surf_interp.ssrd[~sunset]/pd.Timedelta('1H').seconds) / SWtoa[~sunset]     # clearness index
        kt[kt < 0] = 0
        kt[kt > 1] = 1
        kd = 0.952 - 1.041 * np.exp(-1 * np.exp(2.3 - 4.702 * kt))    # Diffuse index

        surf_interp['SW'] = surf_interp.ssrd/pd.Timedelta('1H').seconds
        surf_interp['SW'][surf_interp['SW'] < 0] = 0
        surf_interp['SW_diffuse'] = kd * surf_interp.SW
        down_pt['SW_diffuse'] = row.svf * surf_interp.SW_diffuse

        surf_interp['SW_direct'] = surf_interp.SW - surf_interp.SW_diffuse
        # scale direct solar radiation using Beer's law (see Aalstad 2019, Appendix A)
        ka = surf_interp.ssrd * 0
        #pdb.set_trace()
        ka[~sunset] = (g * mu0[~sunset]/down_pt.p)*np.log(SWtoa[~sunset]/surf_interp.SW_direct[~sunset])
        # Illumination angle
        down_pt['cos_illumination_tmp'] = mu0 * np.cos(row.slope) + np.sin(ds_solar.sel(point_id=pt_id).zenith) *\
                                          np.sin(row.slope) * np.cos(ds_solar.sel(point_id=pt_id).azimuth - row.aspect)
        down_pt['cos_illumination'] = down_pt.cos_illumination_tmp * (down_pt.cos_illumination_tmp > 0)  # remove selfdowing ccuring when |Solar.azi - aspect| > 90
        down_pt = down_pt.drop(['cos_illumination_tmp'])
        down_pt['cos_illumination'][down_pt['cos_illumination'] < 0 ] =0

        # Binary shadow masks.
        horizon = horizon_da.sel(x=row.x, y=row.y, azimuth=np.rad2deg(ds_solar.azimuth.isel(point_id=pt_id)), method='nearest')
        shade = (horizon > ds_solar.sel(point_id=pt_id).elevation)
        down_pt['SW_direct_tmp'] = down_pt.t * 0
        down_pt['SW_direct_tmp'][~sunset] = SWtoa[~sunset] * np.exp(-ka[~sunset] * down_pt.p[~sunset] / (g * mu0[~sunset]))
        down_pt['SW_direct'] = down_pt.t * 0
        down_pt['SW_direct'][~sunset] = down_pt.SW_direct_tmp[~sunset] * (down_pt.cos_illumination[~sunset] / mu0[~sunset]) * (1 - shade)
        down_pt['SW'] = down_pt.SW_diffuse + down_pt.SW_direct

        # currently drop azimuth and level as they are coords. Could be passed to variables instead.
        # round(5) required to sufficiently represent specific humidty, q (eg typical value 0.00078)
        down_pt = down_pt.drop(['level']).round(5)

        # adding metadata
        down_pt.LW.attrs = {'units': 'W/m**2', 'standard_name': 'Longwave radiations downward'}
        down_pt.cse.attrs = {'units': 'xxx', 'standard_name': 'Clear sky emissivity'}
        down_pt = down_pt.drop(['SW_direct_tmp'])
        down_pt.SW.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave radiations downward'}
        down_pt.SW_diffuse.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave diffuse radiations downward'}

        ds_list.append(down_pt)

        num = str(pt_id).zfill(n_digits)
        path_list.append(f'{project_directory}outputs/downscaled/{file_pattern.split("*")[0]}{num}{file_pattern.split("*")[1]}')

        down_pt = None
        surf_interp = None
    xr.save_mfdataset(ds_list, path_list, engine='h5netcdf')
    ds_list = None
    path_list = None

    clear_files(f'{project_directory}outputs/tmp')

    # print timer to console
    print('---> Downscaling finished in {}s'.format(np.round(time.time()-start_time), 1))


def read_downscaled(path='outputs/down_pt*.nc'):
    """
    Function to read downscaled points saved into netcdf files into one single dataset using Dask

    Args:
        path (str): path and pattern to use into xr.open_mfdataset

    Returns:
        dataset: merged dataset readily to use and loaded in chuncks via Dask
    """
    down_pts = xr.open_mfdataset(path, concat_dim='point_id', combine='nested', parallel=True)
    return down_pts











