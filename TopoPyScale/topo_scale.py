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



