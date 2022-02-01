'''
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

'''
import pdb

import pandas as pd
import xarray as xr
from pyproj import Transformer
import numpy as np
import sys, time
from TopoPyScale import meteo_util as mu

# to get from Topo_class
#df_centroids = mp.toposub.df_centroids
#solar_ds = mp.solar_ds
#horizon_da = mp.horizon_da
#target_EPSG = mp.config.dem_epsg
# Physical constants
g = 9.81    #  Acceleration of gravity [ms^-1]
R = 287.05  #  Gas constant for dry air [JK^-1kg^-1]


def downscale_climate(path_forcing, df_centroids, solar_ds, horizon_da, target_EPSG, start_date, end_date, lw_terrain_flag=True, tstep='1H'):
    '''
    Function to perform downscaling of climate variables (t,q,u,v,tp,SW,LW) based on Toposcale logic

    :param path_forcing: path to forcing data [SURF*, PLEV*]
    :param df_centroids: pandas dataframe containing a list of point for which to downscale (includes all terrain data)
    :param solar_ds: xarray dataset containing solar geometry: sun zenith, azimuth, elevation
    :param horizon_da: xarray dataarray containing the horizon angles for a list of azimuth
    :param target_EPSG: int, EPSG code of the DEM
    :param lw_terrain_flag: boolean, flag to compute contribution of surrounding terrain to LW or ignore
    :param tstep: timestep of the input data, default = 1H
    :return: xarray dataset containing downscaled data organized with time, point_id, lat, long
    '''
    print('\n---> Downscaling climate to list of points using TopoScale')
    start_time = time.time()
    tstep_dict = {'1H': 1, '3H': 3, '6H': 6}
    # =========== Open dataset with Dask =================
    ds_plev = xr.open_mfdataset(path_forcing + 'PLEV*.nc', engine='netcdf4', parallel=True).sel(time=slice(start_date, end_date))
    ds_surf = xr.open_mfdataset(path_forcing + 'SURF*.nc', engine='netcdf4', parallel=True).sel(time=slice(start_date, end_date))

    # ============ Convert lat lon to projected coordinates ==================
    trans = Transformer.from_crs("epsg:4326", "epsg:" + str(target_EPSG), always_xy=True)
    nxv,  nyv = np.meshgrid(ds_surf.longitude.values, ds_surf.latitude.values)
    nlons, nlats = trans.transform(nxv, nyv)
    ds_surf = ds_surf.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})
    ds_plev = ds_plev.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})

    # ============ Extract specific humidity (q) for both dataset ============
    ds_surf = mu.dewT_2_q_magnus(ds_surf, mu.var_era_surf)
    ds_plev = mu.t_rh_2_dewT(ds_plev, mu.var_era_plevel)

    # ============ Loop over each point ======================================
    # Loop over each points (lat,lon) for which to downscale climate variable using Toposcale method
    dataset = []
    for i, row in df_centroids.iterrows():
        print('Downscaling point: {} out of {}'.format(row.name, df_centroids.index.max()))
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
        dw = xr.core.weighted.DatasetWeighted(ds_plev_pt, da_idw)
        plev_interp = dw.sum(['longitude', 'latitude'], keep_attrs=True)    # compute horizontal inverse weighted horizontal interpolation
        dw = xr.core.weighted.DatasetWeighted(ds_surf_pt, da_idw)
        surf_interp = dw.sum(['longitude', 'latitude'], keep_attrs=True)    # compute horizontal inverse weighted horizontal interpolation

        # ========= Converting z from [m**2 s**-2] to [m] asl =======
        plev_interp.z.values = plev_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
        plev_interp.z.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of plevel'}

        surf_interp.z.values = surf_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
        surf_interp.z.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of ERA5 surface'}

        if (row.elevation < plev_interp.z.isel(level=-1)).sum():
            print("---> WARNING: Point {} is {} m lower than the 1000hPa geopotential\n=> "
                  "Values sampled from Psurf and lowest Plevel. No vertical interpolation".
                  format(i, np.round(np.min(row.elevation - plev_interp.z.isel(level=-1).values),0)))
            ind_z_top = (plev_interp.where(plev_interp.z > row.elevation).z - row.elevation).argmin('level')
            top = plev_interp.isel(level=ind_z_top)
            down_pt = (top['t']).to_dataset()
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
            down_pt = (bot.t * weights[0] + top.t * weights[1]).to_dataset()
            down_pt['u'] = bot.u * weights[0] + top.u * weights[1]
            down_pt['v'] = bot.v * weights[0] + top.v * weights[1]
            down_pt['q'] = bot.q * weights[0] + top.q * weights[1]
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

        down_pt.cse.attrs = {'units': 'xxx', 'standard_name': 'Clear sky emissivity'}
        surf_interp.cse.attrs = {'units': 'xxx', 'standard_name': 'Clear sky emissivity'}
        surf_interp.cle.attrs = {'units': 'xxx', 'standard_name': 'Cloud emissivity'}
        surf_interp.aef.attrs = {'units': 'xxx', 'standard_name': 'All sky emissivity'}
        down_pt.LW.attrs = {'units': 'W/m**2', 'standard_name': 'Longwave radiations downward'}

        mu0 = np.cos(solar_ds.sel(point_id=row.name).zenith) * (np.cos(solar_ds.sel(point_id=row.name).zenith)>0)
        S0 = 1370 # Solar constat (total TOA solar irradiance) [Wm^-2] used in ECMWF's IFS
        solar_ds['SWtoa'] = S0 * mu0
        solar_ds['sunset'] = mu0 < np.cos(89*np.pi/180)
        solar_ds.SWtoa.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave radiations downward top of the atmosphere'}
        solar_ds.sunset.attrs = {'units': 'bool', 'standard_name': 'Sunset'}

        kt = surf_interp.ssrd * 0
        kt[~solar_ds.sunset] = (surf_interp.ssrd[~solar_ds.sunset]/pd.Timedelta('1H').seconds) / \
                               solar_ds.sel(point_id=row.name).SWtoa[~solar_ds.sunset]     # clearness index
        kd = np.max(0.952 - 1.041 * np.exp(-1 * np.exp(2.3 - 4.702 * kt)), 0)    # Diffuse index

        surf_interp['SW'] = surf_interp.ssrd/pd.Timedelta('1H').seconds
        surf_interp['SW_diffuse'] = kd * surf_interp.SW
        down_pt['SW_diffuse'] = row.svf * surf_interp.SW_diffuse
        surf_interp.SW.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave radiations downward'}
        surf_interp.SW_diffuse.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave diffuse radiations downward'}
        down_pt.SW_diffuse.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave diffuse radiations downward'}
        surf_interp['SW_direct'] = surf_interp.SW - surf_interp.SW_diffuse
        surf_interp.SW_direct.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave direct radiations downward'}

        # scale direct solar radiation using Beer's law (see Aalstad 2019, Appendix A)
        ka = surf_interp.ssrd * 0
        ka[~solar_ds.sunset] = (g*mu0[~solar_ds.sunset]/down_pt.p)*np.log(solar_ds.sel(point_id=row.name).SWtoa[~solar_ds.sunset]/surf_interp.SW_direct[~solar_ds.sunset])
        # Illumination angle
        down_pt['cos_illumination_tmp'] = mu0 * np.cos(row.slope) + np.sin(solar_ds.sel(point_id=row.name).zenith) *\
                                          np.sin(row.slope) * np.cos(solar_ds.sel(point_id=row.name).azimuth - row.aspect)
        down_pt['cos_illumination'] = down_pt.cos_illumination_tmp * (down_pt.cos_illumination_tmp < 0)  # remove selfdowing ccuring when |Solar.azi - aspect| > 90
        down_pt = down_pt.drop(['cos_illumination_tmp'])

        # Binary shadow masks.
        horizon = horizon_da.sel(x=row.x, y=row.y, azimuth=np.rad2deg(solar_ds.azimuth.isel(point_id=row.name)), method='nearest')
        shade = (horizon > solar_ds.sel(point_id=row.name).elevation)
        down_pt['SW_direct_tmp'] = down_pt.t * 0
        down_pt['SW_direct_tmp'][~solar_ds.sunset] = solar_ds.sel(point_id=row.name).SWtoa[~solar_ds.sunset] * np.exp(-ka[~solar_ds.sunset] * down_pt.p[~solar_ds.sunset] / (g*mu0[~solar_ds.sunset]))
        down_pt['SW_direct'] = down_pt.t * 0
        down_pt['SW_direct'][~solar_ds.sunset] = down_pt.SW_direct_tmp[~solar_ds.sunset] * (down_pt.cos_illumination[~solar_ds.sunset]/mu0[~solar_ds.sunset])*(1-shade)
        down_pt['SW'] = down_pt.SW_diffuse + down_pt.SW_direct
        down_pt = down_pt.drop(['SW_direct_tmp'])
        down_pt.SW_direct.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave direct radiations downward'}
        down_pt.SW.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave radiations downward'}

        # currently drop azimuth and level as they are coords. Could be passed to variables instead.
        down_pt = down_pt.drop(['level']).round(2)
        dataset.append(down_pt)


    down_pts = xr.concat(dataset, dim='point_id')
    # print timer to console
    print('---> Downscaling finished in {}s'.format(np.round(time.time()-start_time), 1))
    return down_pts




