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

TODO:
- check that transformation from r,t to q for plevels is working fine
- add metadata to q variable in datasets ds_psurf, and ds_plevel

'''
import dask.dataframe.methods
import xarray as xr
from pyproj import Transformer
import numpy as np
import sys

# Physical constants
g = 9.81    #  Acceleration of gravity [ms^-1]
R = 287.05  #  Gas constant for dry air [JK^-1kg^-1]
eps0=0.622  # Ratio of molecular weight of water and dry air [-]
S0=1370    # Solar constat (total TOA solar irradiance) [Wm^-2] used in ECMWF's IFS



def T_RH_2_d2m(ds):
    ds['d2m'] = 243.04*(np.log(ds.r/100)+((17.625*ds.t)/(243.04+ds.t)))/(17.625-np.log(ds.r/100)-((17.625*ds.t)/(243.04+ds.t)))
    return ds

def d2m_2_q_magnus(ds, plevel=False):
    '''
    A version of the Magnus formula with the AERK parameters. AERK from Alduchov and Eskridge (1996)
    Note, e=Magnus(tdc) and es=Magnus(tc)
    :param d2m: dew temperature
    :return:
    '''
    A1, B1, C1 =17.625, 243.04, 610.94
    vpsl = C1 * np.exp(A1 * (ds.d2m - 273.15) / (B1 + (ds.d2m - 273.15)))
    if plevel:
        wsl = eps0 * vpsl / (ds.level - vpsl)
    else:
        wsl = eps0 * vpsl / (ds.sp - vpsl)
    ds['qsl'] = wsl/(1+wsl)
    return ds


def wp2e(w,p):
    # Vapor pressure from mixing ratio based on 2nd order Taylor series expansion
    eps0=0.622  # Ratio of molecular weight of water and dry air [-]
    return 0.5 * p *(-1 + np.sqrt(1 + 4 * w / eps0))

def q2w(q):
    return 0.5 * (1 - np.sqrt(1 - 4 * q))

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
ds_surf = d2m_2_q_magnus(ds_surf)
ds_plev = d2m_2_q_magnus(T_RH_2_d2m(ds_plev), plevel=True)

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
psurf_interp = dw.sum(['longitude', 'latitude'], keep_attrs=True)    # compute horizontal inverse weighted horizontal interpolation

# ========= Converting z from [m**2 s**-2] to [m] asl =======
plev_interp.z.values = plev_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
plev_interp.z.attrs['units'] = 'm'  # modify z unit
plev_interp.z.attrs['standard_name'] = 'Elevation'  # modify z unit
plev_interp.z.attrs['Long_name'] = 'Elevation of plevel'  # modify z unit

psurf_interp.z.values = psurf_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
psurf_interp.z.attrs['units'] = 'm'  # modify z unit
psurf_interp.z.attrs['standard_name'] = 'Elevation'  # modify z unit
psurf_interp.z.attrs['Long_name'] = 'Elevation of ERA5 surface'  # modify z unit

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

# ======= logic  to compute ws, wd without loading data in memory, and maintaining the power of dask
down_pt['theta'] = np.arctan2(-down_pt.u, -down_pt.v)
down_pt['theta_neg'] = (down_pt.theta < 0)*(down_pt.theta + 2*np.pi)
down_pt['theta_pos'] = (down_pt.theta >= 0)*down_pt.theta
down_pt.drop('theta')
down_pt['wd'] = (down_pt.theta_pos + down_pt.theta_neg)  # direction in Rad
down_pt['ws'] = np.sqrt(down_pt.u ** 2 + down_pt.v**2)
down_pt.drop(['theta_pos', 'theta_neg'])





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




