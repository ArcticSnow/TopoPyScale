'''
Script to convert TopoScale Xarray
S. Filhol, 2021

Mat file structure for one point. Loop over each point, and create one file per point

Time: days in proleptic ISO 1-> 1 Jan 2000
Temperature [C]
Humidity Specific [kg/kg]
Shortwave [W/m2]
Longwave [W/m2]
Pressure [Pa]
Rainfall [mm/day]
Snowfall [SWE mm/day]

# wind spedd threshold for stability
	df.VW[df.VW<0.5] = 0.5

TODO:
- split precip to rain/snow
-
'''

from scipy import io
import xarray as xr
import pandas as pd


fname = 'test.mat'

time = pd.date_range('')


def to_cryogrid_mat(ds, fname):
    '''
    Function to save toposcale dataset output to Cryogrid .mat file structure
    :param ds: xarray dataset, output of topo_scale
    :param fname: str, pathname of the files. name pattern in case of multipoints
    :return:
    '''

    # Code to split precip to Snow vs Rain precip. see https://github.com/CryoGrid/CryoGrid/blob/a0ac2cafbb3726c6da1c4a5633167100c16adf30/source/IO/FORCING/FORCING_slope_seb_readNc.m#L99
    # partition prate to rain snow (mm/hr)
    low_temp_thresh = 272.15    # deg Kelvin
    high_temp_thresh = 274.15  # deg Kelvin

    ds['snowfall'] = ds.tp.loc[ds.t2m < low_temp_thresh)
    ds['rainfall'] = ds.tp.loc[ds.t2m >= high_temp_thresh)

    # All wind speed below 0.5 must be equale to 0.5
    ds['ws_cryo'] = ds.ws * ds.ws[ds.ws>=0.5] + 0.5 * ds.ws[ds.ws<0.5]

    # Loop over point_id and save one file per point
    for point_id in ds.point_id.values:
        print(point_id)



point_id = 1
mdict = {
    't_span': ds.time,
    'Tair': ds.t2m.sel(point_id = point_id).values,
    'rainfall': ds.rainfall.sel(point_id=point_id),
    'snowfall': ds.snowfall.sel(point_id=point_id),
    'Lin': ds.LW.sel(point_id = point_id).values,
    'Sin': ds.SW.sel(point_id = point_id).values,
    'p': ds.sp.sel(point_id = point_id).values,
    'q':ds.q.sel(point_id = point_id).values,
    'Wind': ds.ws_cryo.sel(point_id = point_id).values
}

io.savemat(fname, mdict)