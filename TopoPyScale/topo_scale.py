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

path_forcing = '/home/arcticsnow/Desktop/topoTest/inputs/forcings/'
ds_plev = xr.open_mfdataset(path_forcing + 'PLEV*')
ds_surf = xr.open_mfdataset(path_forcing + 'SURF*')

# Convert lat lon to projected coordinates
trans = Transformer.from_proj(4326, 3844)

nxv,  nyv = np.meshgrid(ds_surf.longitude.values, ds_surf.latitude.values)
nlons, nlats = trans.transform(nxv, nyv)
ds_surf = ds_surf.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})
ds_plev = ds_plev.assign_coords({"latitude": nlats[:, 0], "longitude": nlons[0, :]})
ds_surf.t2m.isel(time=10).plot()


#selct the 9 era grid cell around a given point

xp = 289936
yp = 435589
ds_surf.sel(latitude=yp, longitude=xp, method='nearest')
ind_lat = np.abs(ds_surf.latitude-yp).argmin()
ind_lon = np.abs(ds_surf.longitude-xp).argmin()

ds_surf.isel(latitude=[ind_lat-1, ind_lat, ind_lat+1], longitude=[ind_lon-1, ind_lon, ind_lon+1])