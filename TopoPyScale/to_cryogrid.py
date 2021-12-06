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
- implement function from era5/era5Land/CARRA to cryogrid
- implement function from toposcale to cryogrid

'''

from scipy import io
import xarray as xr
import pandas as pd


fname = 'test.mat'

time = pd.date_range('')
