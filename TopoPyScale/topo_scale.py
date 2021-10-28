'''
Toposcale functionalities
S. Filhol, Oct 2021

dem_descriton (nb_points)
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

NOTE:
surface and plevels, open with xr.open_mfdataset()


Dataset naming convention:
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
import dask.dataframe.methods
import pandas as pd
import xarray as xr
from pyproj import Transformer
import numpy as np
import sys
from TopoPyScale import meteo_util as mu

# =========== Open dataset with Dask =================
path_forcing = '/home/arcticsnow/Desktop/topoTest/inputs/forcings/'
ds_plev = xr.open_mfdataset(path_forcing + 'PLEV*')
ds_surf = xr.open_mfdataset(path_forcing + 'SURF*')

# ============ Convert lat lon to projected coordinates ==================
trans = Transformer.from_crs("epsg:4326", "epsg:3844", always_xy=True)
nxv,  nyv = np.meshgrid(ds_surf.longitude.values, ds_surf.latitude.values)
nlons, nlats = trans.transform(nxv, nyv)
ds_surf = ds_surf.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})
ds_plev = ds_plev.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})

# ============ Extract specific humidity (q) for both dataset ============
ds_surf = mu.dewT_2_q_magnus(ds_surf, mu.var_era_surf)
ds_plev = mu.t_rh_2_dewT(ds_plev, mu.var_era_plevel)



#selct the 9 era grid cell around a given point
row = df_centroids.iloc[0]

# =========== Extract the 3*3 cells centered on a given point ============
ind_lat = np.abs(ds_surf.latitude-row.y).argmin()
ind_lon = np.abs(ds_surf.longitude-row.x).argmin()
ds_surf_pt = ds_surf.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])
ds_plev_pt = ds_plev.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])

## ====== Horizontal interpolation ====================
interp_method = 'idw'
Xs, Ys = np.meshgrid(ds_plev_pt.longitude.values, ds_plev_pt.latitude.values)
if interp_method == 'idw':
    dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2 )
    idw = 1/(dist**2)
    weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1
elif interp_method == 'linear':
    dist = np.array([np.abs(row.x - Xs).values, np.abs(row.y - Ys).values])
    weights = dist / np.sum(dist, axis=0)
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
plev_interp.z.attrs['units'] = 'm'  # modify z unit
plev_interp.z.attrs['standard_name'] = 'Elevation'  # modify z unit
plev_interp.z.attrs['Long_name'] = 'Elevation of plevel'  # modify z unit

surf_interp.z.values = surf_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
surf_interp.z.attrs['units'] = 'm'  # modify z unit
surf_interp.z.attrs['standard_name'] = 'Elevation'  # modify z unit
surf_interp.z.attrs['Long_name'] = 'Elevation of ERA5 surface'  # modify z unit

# ========== Vertical interpolation at the DEM surface z  ===============

ind_z_bot = (plev_interp.where(plev_interp.z < row.elev).z - row.elev).argmin('level')
ind_z_top = (plev_interp.where(plev_interp.z > row.elev).z - row.elev).argmin('level')
top = plev_interp.isel(level=ind_z_top)
bot = plev_interp.isel(level=ind_z_bot)

# Preparing interpolation weights for linear interpolation =======================
dist = np.array([np.abs(bot.z - row.elev).values, np.abs(top.z - row.elev).values])
#idw = 1/dist**2
#weights = idw / np.sum(idw, axis=0)
weights = dist / np.sum(dist, axis=0)

# ============ Creating a dataset containing the downscaled timeseries ========================
down_pt = (bot.t * weights[0] + top.t * weights[1]).to_dataset()
down_pt['u'] = bot.u * weights[0] + top.u * weights[1]
down_pt['v'] = bot.v * weights[0] + top.v * weights[1]
down_pt['q'] = bot.q * weights[0] + top.q * weights[1]

# ======= logic  to compute ws, wd without loading data in memory, and maintaining the power of dask
down_pt['theta'] = np.arctan2(-down_pt.u, -down_pt.v)
down_pt['theta_neg'] = (down_pt.theta < 0)*(down_pt.theta + 2*np.pi)
down_pt['theta_pos'] = (down_pt.theta >= 0)*down_pt.theta
down_pt.drop('theta')
down_pt['wd'] = (down_pt.theta_pos + down_pt.theta_neg)  # direction in Rad
down_pt['ws'] = np.sqrt(down_pt.u ** 2 + down_pt.v**2)
down_pt.drop(['theta_pos', 'theta_neg'])

# Physical constants
g = 9.81    #  Acceleration of gravity [ms^-1]
R = 287.05  #  Gas constant for dry air [JK^-1kg^-1]
down_pt['p'] = top.level*(10**-3) * np.exp(-(row.elev-top.z) / (0.5 * (top.t + down_pt.t) * R / g) )  # Pressure in bar
down_pt['tp'] = surf_interp.tp

# ======== Longwave downward radiation ===============
x1, x2 =0.43, 5.7
sbc = 5.67e-8

down_pt = mu.mixing_ratio(down_pt, mu.var_era_plevel)
down_pt = mu.vapor_pressure(down_pt, mu.var_era_plevel)

surf_interp = mu.mixing_ratio(surf_interp, mu.var_era_surf)
surf_interp = mu.vapor_pressure(surf_interp, mu.var_era_surf)

# Compute clear sky emissivity
down_pt['cse'] = 0.23 + x1 * (down_pt.vp / down_pt.t) ** (1 / x2)
surf_interp['cse'] = 0.23 + x1 * (surf_interp.vp / surf_interp.t2m) ** (1 / x2)
# Calculate the "cloud" emissivity, CHECK UNIT OF SSRD (J/m2) or (W/m2)
surf_interp['cle'] = surf_interp.ssrd / (sbc * surf_interp.t2m^4) - surf_interp['cse']
# Use the former cloud emissivity to compute the all sky emissivity at subgrid.
surf_interp['aef'] = down_pt['cse'] + surf_interp['cle']
down_pt['ssrd'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4

# need to call solar_ds

def compute_SWtoa_irradiance(solar_ds):
    '''

    :param solar_ds:
    :return:
    '''
    solar_ds['zenith_avg'] = solar_ds.zenith.rolling(time=1).mean()
    solar_ds['azimuth_avg'] = solar_ds.azimuth.rolling(time=1).mean()
    mu0 = np.cos(solar_ds.zenith_avg*np.pi/180) * (np.cos(solar_ds.zenith_avg*np.pi/180)>0)
    S0=1370 # Solar constat (total TOA solar irradiance) [Wm^-2] used in ECMWF's IFS
    solar_ds['SWtoa'] = S0 * mu0
    solar_ds['sunset'] = mu0 < np.cos(89*np.pi/180)

    return solar_ds

solar_ds = compute_SWtoa_irradiance(solar_ds)
kt = (surf_interp.ssrd/pd.Timedelta(tstep).seconds) / solar_ds.SWtoa     # clearness index
kd = np.max(0.952 - 1.041 * np.exp(-1 * np.exp(2.3 - 4.702 * kt)), 0)    # Diffuse index

'''
- compute distance to the 9 neighbors. compute weights based on distance (1/d^2)  see line 188, and normalize that sum(w)=1 (line 202)
- loop on time:
1. x Plevel for tstep, and reorganize array in (plevel, index)  (line 259) Zpl=geopentaitl hehght in m  
2. x line 262: miultiple by wiegths to get z of pressure levels above target point
3. x look for plevel right below and above the surface
4. x Ignore surface levels for vertical interpolation at the moment

do inverse distance weighting for each variable, then sample from that during next step
5. x Z1 = plvel heigh below dem surf. z2= plevel above dem surf (line 292)
6. x T(z_dem) = T(z1)+((T(z2)-T(z1))/(z2-z1))*(z_dem-z1)   for each function
x rpeat for U, V, and compute Ws, Wd
for humidity, compute vapor pressure (magnus, line 338, Tdsl=dew point temp as surf) in matlab, copmute vpsl for each grid, and then idw interpolate
and pressure (get Z) and compute corresponding barometric pressure (see line 347-356)

# from surface ERA5
## longwave:
vpc=idw*vpsl # idw for vpsl
compute vapor pressure at z_dem (line 362-363) pout=downscaled pressure, qout= downscaled specific humi

cef/cec = clear sky emissivity  at Zde/Z_era_idw(line 368)

aec = all sky emissvity @ Z_era_idw
then follow logic until line 385 for LW

## shortwave:
copmute average soalr zenith and Az for timestep (line 393-398)
S0 = solar constant
SWc = idw interpolated shartwave at each era grid cells
Compute 2 sunset (one for direct radiastion below horizon, one for diffuse astronomical twilight)
if neg values in SWc, then replace by 0
if sunset_diffuse:
    SWout = 0
else:


line 437: psl= idw interp surface pressure at z_era_idw
slp = slope
szen = sun zenith

precip = simple idw




