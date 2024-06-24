# https://github.com/ecmwf/ecmwf-opendata
# https://www.ecmwf.int/en/forecasts/datasets/open-data

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Work in progress!! Use at own risk!
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Steps:

# For times 00z &12z: 0 to 144 by 3, 150 to 240 by 6.
# For times 06z & 18z: 0 to 90 by 3.
# Single and Pressure Levels (hPa): 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50

# 6 day forecast available at 3h steps

# do i need TOA?
#  [300, 500,600, 700,800, 850, 900, 1000]

# we compute z from gh on pressure levels
# renaming of 2t, 2d
# we comput z at surfeca from  msl and sp and t



from ecmwf.opendata import Client
import glob
import os
import pandas as pd

# data must be available by UTC+6 10am (Bishkek) eg 4am UTC, therefore need to use previous major fc step which is 12 UTC
fctime = 12

# removed implementation as the day is handle automatically based on last available fctime
# can be used to specify precise forecasts up to three day ago
mydate= -3 #  0 = today (default) -1 = yesterday -2 = day before yesterdaz -3 = day before that. Valid values 0-3.
outdir = '/home/joel/sim/TPS_2024/master/inputs/forecast'
os.chdir(outdir)

directory_path = "final"
# Check if the directory exists, and create it if it doesn't
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory created: {directory_path}")
else:
    print(f"Directory already exists: {directory_path}")

# clean up
# files2delete = glob.glob("PLEV*")
# for file in files2delete:
#     os.remove(file)

# files2delete = glob.glob("SURF*")
# for file in files2delete:
#     os.remove(file)


# print latest forecast data
client = Client(source="ecmwf")

print(client.latest(
    type="fc",
    step=24,
    param=["2t", "msl"],
    target="data.grib2",
))

# wait until this is 00 UTC (c. 9h UTC)

#NOTE: The data is available between 7 and 9 hours after the forecast starting date and time, depending on the forecasting system and the time step specified.
# - do forecast 9am  UTC ?

# download surface variables at fc steps 0-144 available at 3h setp
print("Downloading surface variables forecast steps 0-144")
client = Client()
 
client.retrieve(
	time=fctime,
	date = mydate,
     step=[i for i in range(0, 147, 3)], #147
     type="fc",
     param=["2t", "sp", "2d", "ssrd", "strd", "tp", "msl"],
     target="SURF_fc1.grib2",
 )

# download pressure variables at fc steps 0-144 available at 3h setp
client = Client()
print("Downloading pressure level variables forecast steps 0-144") 
client.retrieve(
	time=fctime,
	date = mydate,
     step=[i for i in range(0, 147, 3)], #147
     type="fc",
     param=["gh", "u", "v", "r", "q", "t"],
     levelist=[1000, 925, 850, 700, 500, 300],
     target="PLEV_fc1.grib2",
 )


# from ecmwf.opendata import Client

# client = Client(source="ecmwf")

# result = client.retrieve(
#     type="fc",
#     step=24,
#     param=["gh"],
#     target="data.grib2",
# )

# print(result.datetime)

# ssrd strd sp ,2d, sp
# gh, u, v, r,q,t

# download surface variables at steps 150-244 available at 6h setp
from ecmwf.opendata import Client
 
client = Client()
print("Downloading surface variables forecast steps 150-244") 
client.retrieve(
	time= fctime,
	date = mydate,
     step=[i for i in range(150, 241, 6)], # 144 start?
     type="fc",
     param=["2t", "sp", "2d", "ssrd", "strd", "tp", "msl"],
     target="SURF_fc2.grib2",
 )

# download pressure level variables at steps 150-244 available at 6h setp
from ecmwf.opendata import Client
 
client = Client()
print("Downloading pressure level variables forecast steps 150-244") 
client.retrieve(
	time= fctime,
	date = mydate,
     step=[i for i in range(150, 241, 6)],
     type="fc",
     param=["gh", "u", "v", "r", "q", "t"],
     levelist=[1000, 925, 850, 700, 500, 300],
     target="PLEV_fc2.grib2",
 )

print("converting grib to nc")
# convert all gribs to nc ( do this in pure python cdo )
cmd = "cdo -f nc copy SURF_fc1.grib2 SURF_fc1.nc"
os.system(cmd)
cmd = "cdo -f nc copy PLEV_fc1.grib2 PLEV_fc1.nc"
os.system(cmd)
cmd = "cdo -f nc copy SURF_fc2.grib2 SURF_fc2.nc"
os.system(cmd)
cmd = "cdo -f nc copy PLEV_fc2.grib2 PLEV_fc2.nc"
os.system(cmd)
#!ncview subset_SURF_fc1.nc
#!ncview PLEV_fc1.nc

# clean up big grib files
files2delete = glob.glob("*grib2")
for file in files2delete:
    os.remove(file)

files2delete = glob.glob("*grib2")
for file in files2delete:
    os.remove(file)


print("Spatial subsetting and deaccumulation of precip and rad params")

import xarray as xr

# Function to perform spatial subset on NetCDF file
def spatial_subset(nc_file, lat_range, lon_range):
    # Open the NetCDF file
    ds = xr.open_dataset(nc_file)

    # Extract latitudes and longitudes
    lat = ds['lat'].values
    lon = ds['lon'].values

    # Find the indices corresponding to the specified latitude and longitude ranges
    lat_indices = (lat >= lat_range[0]) & (lat <= lat_range[1])
    lon_indices = (lon >= lon_range[0]) & (lon <= lon_range[1])

    # Perform spatial subset
    subset = ds.sel(lat=lat[lat_indices], lon=lon[lon_indices])

    return subset




import numpy as np



# Function to calculate geopotential height
def calculate_geopotential(P, T, P0):
	# Constants
    R = 287  # Gas constant for dry air (J/kg/K)
    # Standard mean sea level pressure (hPa)
    P_hpa = P/100
    P0_hpa = P0/100

    Z = (R * T ) * np.log(P0_hpa / P_hpa)
    return Z


# import numpy as np

# # Constants
# R = 287  # Gas constant for dry air (J/kg/K)
# g = 9.81  # Acceleration due to gravity (m/s^2)
# P0 = 1013.25  # Standard mean sea level pressure (hPa)

# # Function to calculate geopotential height
# def calculate_geopotential_height(P, T):
#     Z = (R * T / g) * np.log(P0 / P)
#     return Z

# # Example usage
# P_surface = 1000  # Example surface pressure in hPa
# T_surface = 288  # Example surface temperature in Kelvin

# Z_surface = calculate_geopotential_height(P_surface, T_surface)
# print("Geopotential height at surface:", Z_surface, "m")






# Define the geographical subset (bounding box)
lat_range = (32, 45)  # Example latitude range (20N to 30N)
lon_range = (59, 81)  # Example longitude range (30E to 40E)


# Perform spatial subset on each NetCDF file
nc_files = glob.glob("SURF*.nc")  # List of NetCDF files
for nc_file in nc_files:
	subset = spatial_subset(nc_file, lat_range, lon_range)
   
	accumulated_var = subset["param193.1.0"]
    # Calculate the difference between consecutive forecast steps manually, step (n) - step (n-1) in shift first timestep is filled with 0.
	deaccumulated_var = accumulated_var - accumulated_var.shift(time=1, fill_value=0)
    # rename
	deaccumulated_rename = deaccumulated_var.rename('tp')
    # Update the Dataset with the calculated variable
	subset['tp'] = deaccumulated_rename
	# Drop the variable from the Dataset
	subset = subset.drop_vars('param193.1.0')

	accumulated_var = subset["ssrd"]
    # Calculate the difference between consecutive forecast steps manually, step (n) - step (n-1) in shift first timestep is filled with 0.
	deaccumulated_var = accumulated_var - accumulated_var.shift(time=1, fill_value=0)
    # Update the Dataset with the calculated variable
	subset['ssrd'] = deaccumulated_var

	accumulated_var = subset["strd"]
    # Calculate the difference between consecutive forecast steps manually, step (n) - step (n-1) in shift first timestep is filled with 0.
	deaccumulated_var = accumulated_var - accumulated_var.shift(time=1, fill_value=0)
    # Update the Dataset with the calculated variable
	subset['strd'] = deaccumulated_var

	# units tp = total precip over 3h timestep (to get mm/h divide by forecast step in h (3)
	# ssrd strd  total jm-2 over forecast step (to get W/m2 divide by forecast step in s (3600*3) 

	subset = subset.rename({'lon': 'longitude', 'lat': 'latitude'})
	subset = subset.rename({'2t': 't2m'})
	subset = subset.rename({'2d': 'd2m'})

	# compute geopotential z
	subset['z'] = calculate_geopotential(subset['sp'], subset['t2m'], subset['msl'])

	subset.to_netcdf(f'subset_{nc_file}')

    # check de accumulation
    # subset["param193.1.0"][:,40,40].plot()
    # accumulation[:,40,40].plot()  
    # plt.show() 

 

nc_files = glob.glob("PLEV*.nc")   # List of NetCDF files
for nc_file in nc_files:
    subset = spatial_subset(nc_file, lat_range, lon_range)
    subset= subset.rename({'lon': 'longitude', 'lat': 'latitude', 'plev': 'level'})
    subset['z'] = subset['gh']*9.81
    subset['level'] = subset['level']/100.  # pressure level pa to hpa
    subset = subset.isel(level=slice(None, None, -1) ) # reverse order of levels
    subset.to_netcdf(f'subset_{nc_file}')

os.remove("SURF_fc1.nc")
os.rename("subset_SURF_fc1.nc", "SURF_fc1.nc")

os.remove("SURF_fc2.nc")
os.rename("subset_SURF_fc2.nc", "SURF_fc2.nc")

os.remove("PLEV_fc1.nc")
os.rename("subset_PLEV_fc1.nc", "PLEV_fc1.nc")

os.remove("PLEV_fc2.nc")
os.rename("subset_PLEV_fc2.nc", "PLEV_fc2.nc")




import xarray as xr
# interpolat 6h - 3h
ds = xr.open_dataset("SURF_fc2.nc")
# Interpolate the data from 6h to 3h timestep
year = str(ds['time.year'][0].values)  # Extract hour and zero-pad if necessary
month = str(ds['time.month'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
day = str(ds['time.day'][0].values).zfill(2)  # Extract second and zero-pad if necessary
date_string = f"{year}-{month}-{day}"  # Concatenate into a single string

hour = str(ds['time.hour'][0].values).zfill(2)  # Extract hour and zero-pad if necessary
minute = str(ds['time.minute'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
second = str(ds['time.second'][0].values).zfill(2)  # Extract second and zero-pad if necessary
time_string = f"{hour}:{minute}:{second}"  # Concatenate into a single string

cmd = "cdo inttime,"+date_string+","+time_string+",3hour SURF_fc2.nc SURF_fc2_3h.nc"
os.system(cmd)

# interpolat 6h - 3h
ds = xr.open_dataset("PLEV_fc2.nc")
# Interpolate the data from 6h to 3h timestep
year = str(ds['time.year'][0].values)  # Extract hour and zero-pad if necessary
month = str(ds['time.month'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
day = str(ds['time.day'][0].values).zfill(2)  # Extract second and zero-pad if necessary
date_string = f"{year}-{month}-{day}"  # Concatenate into a single string

hour = str(ds['time.hour'][0].values).zfill(2)  # Extract hour and zero-pad if necessary
minute = str(ds['time.minute'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
second = str(ds['time.second'][0].values).zfill(2)  # Extract second and zero-pad if necessary
time_string = f"{hour}:{minute}:{second}"  # Concatenate into a single string

cmd = "cdo inttime,"+date_string+","+time_string+",3hour PLEV_fc2.nc PLEV_fc2_3h.nc"
os.system(cmd)

import xarray as xr
import numpy as np
from datetime import datetime, timedelta

# Open the first NetCDF file
ds1 = xr.open_dataset('SURF_fc1.nc')

# Open the second NetCDF file
ds2 = xr.open_dataset('SURF_fc2_3h.nc')

# Extract the last timestep of all variables in the first file
last_timestep_first_file = ds1.isel(time=-1)

# Extract the first timestep of all variables in the second file
first_timestep_second_file = ds2.isel(time=0)

# Calculate the average of the data
average_ds = (last_timestep_first_file + first_timestep_second_file) / 2

# Calculate the average between the last and first timesteps
t1 = pd.to_datetime((ds1['time']))[-1]
t2 = pd.to_datetime((ds2['time']))[0 ]

# Convert Timestamps to numerical representation (seconds since epoch)
t1_numeric = (t1 - datetime(1970, 1, 1)) / timedelta(seconds=1)
t2_numeric = (t2 - datetime(1970, 1, 1)) / timedelta(seconds=1)

# Calculate the average timestamp (seconds since epoch)
average_numeric = (t1_numeric + t2_numeric) / 2

# Convert average timestamp back to a Timestamp object
average_timestamp = datetime.utcfromtimestamp(average_numeric)

# Assign the time coordinate from the first file to the averaged dataset
average_ds['time'] = average_timestamp

ds_mean = average_ds.assign_coords(time=average_timestamp)

# Concatenate the averaged time coordinate with the existing data along the time dimension
ds_concatenated = xr.concat([ds1,ds_mean, ds2], dim='time')

# Write the averaged data to a new NetCDF file
ds_concatenated.to_netcdf(outdir + '/SURF_fc.nc')

# interpolat 3h - 1h
ds = xr.open_dataset(outdir +'/SURF_fc.nc')
# Interpolate the data from 6h to 3h timestep
year = str(ds['time.year'][0].values)  # Extract hour and zero-pad if necessary
month = str(ds['time.month'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
day = str(ds['time.day'][0].values).zfill(2)  # Extract second and zero-pad if necessary
date_string = f"{year}-{month}-{day}"  # Concatenate into a single string

hour = str(ds['time.hour'][0].values).zfill(2)  # Extract hour and zero-pad if necessary
minute = str(ds['time.minute'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
second = str(ds['time.second'][0].values).zfill(2)  # Extract second and zero-pad if necessary
time_string = f"{hour}:{minute}:{second}"  # Concatenate into a single string

cmd = "cdo inttime,"+date_string+","+time_string+",1hour "+ outdir +"/SURF_fc.nc " + outdir +"/SURF_fc_1h.nc"
os.system(cmd)



# Open the first NetCDF file
ds1 = xr.open_dataset('PLEV_fc1.nc')

# Open the second NetCDF file
ds2 = xr.open_dataset('PLEV_fc2_3h.nc')

# Extract the last timestep of all variables in the first file
last_timestep_first_file = ds1.isel(time=-1)

# Extract the first timestep of all variables in the second file
first_timestep_second_file = ds2.isel(time=0)

# Calculate the average of the data
average_ds = (last_timestep_first_file + first_timestep_second_file) / 2

# Calculate the average between the last and first timesteps
t1 = pd.to_datetime((ds1['time']))[-1]
t2 = pd.to_datetime((ds2['time']))[0 ]

# Convert Timestamps to numerical representation (seconds since epoch)
t1_numeric = (t1 - datetime(1970, 1, 1)) / timedelta(seconds=1)
t2_numeric = (t2 - datetime(1970, 1, 1)) / timedelta(seconds=1)

# Calculate the average timestamp (seconds since epoch)
average_numeric = (t1_numeric + t2_numeric) / 2

# Convert average timestamp back to a Timestamp object
average_timestamp = datetime.utcfromtimestamp(average_numeric)

# Assign the time coordinate from the first file to the averaged dataset
average_ds['time'] = average_timestamp

ds_mean = average_ds.assign_coords(time=average_timestamp)

# Concatenate the averaged time coordinate with the existing data along the time dimension
ds_concatenated = xr.concat([ds1,ds_mean, ds2], dim='time')

# Write the averaged data to a new NetCDF file
ds_concatenated.to_netcdf(outdir +'/PLEV_fc.nc')


# interpolat 3h - 1h
ds = xr.open_dataset(outdir +'/PLEV_fc.nc')
# Interpolate the data from 6h to 3h timestep
year = str(ds['time.year'][0].values)  # Extract hour and zero-pad if necessary
month = str(ds['time.month'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
day = str(ds['time.day'][0].values).zfill(2)  # Extract second and zero-pad if necessary
date_string = f"{year}-{month}-{day}"  # Concatenate into a single string

hour = str(ds['time.hour'][0].values).zfill(2)  # Extract hour and zero-pad if necessary
minute = str(ds['time.minute'][0].values).zfill(2)  # Extract minute and zero-pad if necessary
second = str(ds['time.second'][0].values).zfill(2)  # Extract second and zero-pad if necessary
time_string = f"{hour}:{minute}:{second}"  # Concatenate into a single string

cmd = "cdo inttime,"+date_string+","+time_string+",1hour "+ outdir +"/PLEV_fc.nc " + outdir +"/PLEV_fc_1h.nc"
os.system(cmd)

# cleanup
os.rename(outdir +"/SURF_fc_1h.nc", outdir +"/final/SURF_fc.nc")
os.rename(outdir +"/PLEV_fc_1h.nc", outdir +"/final/PLEV_fc.nc")




# creste hindcast product
ds1 = xr.open_dataset(outdir +"/final/PLEV_fc.nc")
# first day of forecast
thind = pd.to_datetime((ds1['time']))[0:24]
day = str(thind[0])[0:10]  
first_24_steps = ds1.isel(time=slice(0, 24))  
first_24_steps.to_netcdf(outdir + '/final/PLEV_'+day+'.nc')

ds1 = xr.open_dataset(outdir +"/final/SURF_fc.nc")
# first day of forecast
thind = pd.to_datetime((ds1['time']))[0:24]
day = str(thind[0])[0:10]  
first_24_steps = ds1.isel(time=slice(0, 24))  
first_24_steps.to_netcdf(outdir + '/final/SURF_'+day+'.nc')



# cleanup

# Define the pattern to match files (e.g., all .csv files in the current directory)
pattern = "*fc*"

# Find all files matching the pattern
files_to_remove = glob.glob(pattern)

# Loop through the list of files and remove each one
for file_path in files_to_remove:
    try:
        os.remove(file_path)
        print(f"Removed file: {file_path}")
    except OSError as e:
        print(f"Error removing file: {file_path}, {e}")


# Concatenate PLEV

cmd = "cdo mergetime  final/PLEV* final/PLEV_merged.nc"
os.system(cmd)

# remove duplicate timestamp

import xarray as xr

# Load the merged NetCDF file
ds = xr.open_dataset('final/PLEV_merged.nc')

# Convert the time coordinates to a pandas DateTimeIndex
time_index = pd.to_datetime(ds['time'].values)

# Identify duplicate timestamps
duplicates = time_index[time_index.duplicated()]

if len(duplicates) > 0:
    print("Duplicate timestamps found:")
    print(duplicates)
else:
    print("No duplicate timestamps found.")

# Identify duplicate timestamps
_, unique_indices = np.unique(time_index, return_index=True)

# Select only the unique timestamps
ds_unique = ds.isel(time=unique_indices)

# Save the cleaned dataset to a new NetCDF file
ds_unique.to_netcdf('../climate/PLEV_fc_cat.nc')

print("Duplicate timestamps removed. Cleaned Hindcast and forecast dataset '")



# Concatenate SURF

cmd = "cdo mergetime  final/SURF* final/SURF_merged.nc"
os.system(cmd)

# remove duplicate timestamp

import xarray as xr

# Load the merged NetCDF file
ds = xr.open_dataset('final/SURF_merged.nc')

# Convert the time coordinates to a pandas DateTimeIndex
time_index = pd.to_datetime(ds['time'].values)

# Identify duplicate timestamps
duplicates = time_index[time_index.duplicated()]

if len(duplicates) > 0:
    print("Duplicate timestamps found:")
    print(duplicates)
else:
    print("No duplicate timestamps found.")

# Identify duplicate timestamps
_, unique_indices = np.unique(time_index, return_index=True)

# Select only the unique timestamps
ds_unique = ds.isel(time=unique_indices)

# Save the cleaned dataset to a new NetCDF file
ds_unique.to_netcdf('../climate/SURF_fc_cat.nc')

print("Duplicate timestamps removed. Cleaned Hindcast and forecast dataset saved as 'cleaned.nc'")

os.remove('final/PLEV_merged.nc')
os.remove('final/SURF_merged.nc')

# could still be overlap with era5T?
# merge and clean with latest era5T?
# how do we adjust the moving window?
# always delete daily files that are more than 10days old or so?
# how do we determine which data to keep and which to delete on duplicates? 

