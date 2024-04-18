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

# download surface variables at fc steps 0-144 available at 3h setp
print("Downloading surface variables forecast steps 0-144")
client = Client()
 
client.retrieve(
     step=[i for i in range(0, 90, 3)], #147
     type="fc",
     param=["2t", "sp", "2d", "ssrd", "strd", "tp", "msl"],
     target="SURF_fc1.grib2",
 )

# download pressure variables at fc steps 0-144 available at 3h setp
client = Client()
print("Downloading pressure level variables forecast steps 0-144") 
client.retrieve(
     step=[i for i in range(0, 90, 3)], #147
     type="fc",
     param=["gh", "u", "v", "r", "q", "t"],
     levelist=[1000, 925, 850, 700, 500, 300],
     target="PLEV_fc1.grib2",
 )


from ecmwf.opendata import Client

client = Client(source="ecmwf")

result = client.retrieve(
    type="fc",
    step=24,
    param=["gh"],
    target="data.grib2",
)

print(result.datetime)

# ssrd strd sp ,2d, sp
# gh, u, v, r,q,t

# download surface variables at steps 150-244 available at 6h setp
from ecmwf.opendata import Client
 
client = Client()
print("Downloading surface variables forecast steps 150-244") 
client.retrieve(
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
     step=[i for i in range(150, 241, 6)],
     type="fc",
     param=["gh", "u", "v", "r", "q", "t"],
     levelist=[1000, 925, 850, 700, 500, 300],
     target="PLEV_fc2.grib2",
 )

print("converting grib to nc")
# convert all gribs to nc ( do this in pure python cdo )
!cdo -f nc copy SURF_fc1.grib2 SURF_fc1.nc
!cdo -f nc copy PLEV_fc1.grib2 PLEV_fc1.nc
!cdo -f nc copy SURF_fc2.grib2 SURF_fc2.nc
!cdo -f nc copy PLEV_fc2.grib2 PLEV_fc2.nc
#!ncview subset_SURF_fc1.nc
#!ncview PLEV_fc1.nc



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





# Define the geographical subset (bounding box)
lat_range = (32, 45)  # Example latitude range (20N to 30N)
lon_range = (59, 81)  # Example longitude range (30E to 40E)


# Perform spatial subset on each NetCDF file
nc_files = ['SURF_fc1.nc', 'SURF_fc2.nc']  # List of NetCDF files
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

 

nc_files = [ 'PLEV_fc1.nc',  'PLEV_fc2.nc']  # List of NetCDF files
for nc_file in nc_files:
    subset = spatial_subset(nc_file, lat_range, lon_range)
    subset= subset.rename({'lon': 'longitude', 'lat': 'latitude', 'plev': 'level'})
    subset['z'] = subset['gh']*9.81
    subset.to_netcdf(f'subset_{nc_file}')

os.remove("SURF_fc1.nc")
os.rename("subset_SURF_fc1.nc", "SURF_fc1.nc")

os.remove("SURF_fc2.nc")
os.rename("subset_SURF_fc2.nc", "SURF_fc2.nc")

os.remove("PLEV_fc1.nc")
os.rename("subset_PLEV_fc1.nc", "PLEV_fc1.nc")

os.remove("PLEV_fc2.nc")
os.rename("subset_PLEV_fc2.nc", "PLEV_fc2.nc")
























# import pygrib
# import xarray as xr
# import numpy as np

# # Open the GRIB2 file
# grbs = pygrib.open('fc2.grib2')

# for grip in grbs:                                                                                                                                                                                                                                                     
# 	print (grip) 

# # Extract data and metadata
# data = grbs[1].values  # Assuming you want to extract the first field, adjust index as needed
# lat, lon = grbs[1].latlons()

# # Define the latitude and longitude ranges for the subset
# lat_range = (32, 45)  # Example latitude range (20N to 30N)
# lon_range = (59, 81)  # Example longitude range (30E to 40E)

# # Find the indices corresponding to the specified latitude and longitude ranges
# lat_indices = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
# lon_indices = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[1]

# # Extract the subset of data and coordinates
# subset_data = data[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
# subset_lat = lat[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
# subset_lon = lon[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]

# # Create a new NetCDF file with the subset data
# ds_subset = xr.Dataset(
#     {
#         'data': (('lat', 'lon'), subset_data),
#     },
#     coords={'lat': subset_lat[:, 0], 'lon': subset_lon[0, :]},  # Use 1D arrays for coordinates
# )

# ds_subset.to_netcdf('subset_output.nc')


# !ncview subset_output.nc