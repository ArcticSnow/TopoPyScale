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

path_forcing = '/home/arcticsnow/Desktop/topoTest/inputs/forcings/'
ds_plev = xr.open_mfdataset(path_forcing + 'PLEV*')
ds_surf = xr.open_mfdataset(path_forcing + 'SURF*')

# Convert lat lon to projected coordinates


#selct the 9 era grid cell around a given point
