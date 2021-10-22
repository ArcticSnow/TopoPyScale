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
'''

import xarray as xr
from pyproj import Transformer
import numpy as np

# Physical constants
g = 9.81    #  Acceleration of gravity [ms^-1]
R = 287.05  #  Gas constant for dry air [JK^-1kg^-1]
eps0=0.622  # Ratio of molecular weight of water and dry air [-]
S0=1370    # Solar constat (total TOA solar irradiance) [Wm^-2] used in ECMWF's IFS

def dewT_from_T_RH(temp, rh):
    return 243.04*(np.log(rh/100)+((17.625*temp)/(243.04+temp)))/(17.625-np.log(rh/100)-((17.625*temp)/(243.04+temp)))


def magnus(temp):
    '''
    A version of the Magnus formula with the AERK parameters. AERK from Alduchov and Eskridge (1996)
    Note, e=Magnus(tdc) and es=Magnus(tc)
    :param temp:
    :return:
    '''
    A1, B1, C1 =17.625, 243.04, 610.94
    return C1 * np.exp(A1 * temp / (B1 + temp))

def wp2e(w,p):
    # Vapor pressure from mixing ratio based on 2nd order Taylor series expansion
    eps0=0.622  # Ratio of molecular weight of water and dry air [-]
    return 0.5 * p *(-1 + np.sqrt(1 + 4 * w / eps0))

def q2w(q):
    return 0.5 * (1 - np.sqrt(1 - 4 * q))

path_forcing = '/home/arcticsnow/Desktop/topoTest/inputs/forcings/'
ds_plev = xr.open_mfdataset(path_forcing + 'PLEV*')
ds_surf = xr.open_mfdataset(path_forcing + 'SURF*')

# Convert lat lon to projected coordinates
trans = Transformer.from_crs("epsg:4326", "epsg:3844", always_xy=True)

nxv,  nyv = np.meshgrid(ds_surf.longitude.values, ds_surf.latitude.values)
nlons, nlats = trans.transform(nxv, nyv)
ds_surf = ds_surf.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})
ds_plev = ds_plev.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})
#ds_surf.t2m.isel(time=10).plot()


#selct the 9 era grid cell around a given point
xp = 289936
yp = 435589

row = df_centroids.iloc[0]

#ds_surf.sel(latitude=row.y, longitude=row.x, method='nearest')
ind_lat = np.abs(ds_surf.latitude-row.y).argmin()
ind_lon = np.abs(ds_surf.longitude-row.x).argmin()

ds_surf_pt = ds_surf.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])
ds_plev_pt = ds_plev.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])


## ====== Horizontal interpolation ====================

interp_method = 'idw'
Xs, Ys = np.meshgrid(ds_plev_pt.longitude.values, ds_plev_pt.latitude.values)
if interp_method == 'idw':
    dist = np.sqrt((xp - Xs)**2 + (yp - Ys)**2 )
    idw = 1/(dist**2)
    weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1


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
plev_interp.z.values = plev_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
plev_interp.z.attrs['units'] = 'm'  # modify z unit
plev_interp.z.attrs['standard_name'] = 'Elevation'  # modify z unit
plev_interp.z.attrs['Long_name'] = 'Elevation of plevel'  # modify z unit
# ========== Vertical interpolation at the DEM surface z  ===============

ind_z_bot = (plev_interp.where(plev_interp.z < row.elev).z - row.elev).argmin('level')
ind_z_top = (plev_interp.where(plev_interp.z > row.elev).z - row.elev).argmin('level')
top = plev_interp.isel(level=ind_z_top)
bot = plev_interp.isel(level=ind_z_bot)

dist = np.array([np.abs(bot.z - row.elev).values, np.abs(top.z - row.elev).values])
#idw = 1/dist**2
#weights = idw / np.sum(idw, axis=0)
weights = dist / np.sum(dist, axis=0)

down_temp = bot.t*weights[0] + top.t*weights[1]
down_u = bot.u*weights[0] + top.u*weights[1]
down_v = bot.v*weights[0] + top.v*weights[1]
# add function to copmute ws, wd here


# loop over time from here.
plevel_t = ds_plev_pt.isel(time=1)

'''
- compute distance to the 9 neighbors. compute weights based on distance (1/d^2)  see line 188, and normalize that sum(w)=1 (line 202)
- loop on time:
1. x Plevel for tstep, and reorganize array in (plevel, index)  (line 259) Zpl=geopentaitl hehght in m  
2. x line 262: miultiple by wiegths to get z of pressure levels above target point
3. x look for plevel right below and above the surface
4. x Ignore surface levels for vertical interpolation at the moment

do inverse distance weighting for each variable, then sample from that during next step
5. Z1 = plvel heigh below dem surf. z2= plevel above dem surf (line 292)
6. T(z_dem) = T(z1)+((T(z2)-T(z1))/(z2-z1))*(z_dem-z1)   for each function
rpeat for U, V, and compute Ws, Wd
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




