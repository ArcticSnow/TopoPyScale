"""
Retrieve ecmwf data with cdsapi.

- J. Fiddes, Origin implementation
- S. Filhol adapted in 2021

"""
# !/usr/bin/env python
import pandas as pd
from datetime import datetime
import cdsapi, os, sys
from dateutil.relativedelta import *
import glob
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime, timedelta

def retrieve_era5(product, startDate, endDate, eraDir, latN, latS, lonE, lonW, step, num_threads=10, surf_plev='surf', plevels=None, realtime=False):
	""" Sets up era5 surface retrieval.
	* Creates list of year/month pairs to iterate through. 
	* MARS retrievals are most efficient when subset by time. 
	* Identifies preexisting downloads if restarted. 
	* Calls api using parallel function.

	Args:
		product: "reanalysis" (HRES) or "ensemble_members" (EDA)
		startDate:
		endDate:
		eraDir: directory to write output
		latN: north latitude of bbox
		latS: south latitude of bbox
		lonE: easterly lon of bbox
		lonW: westerly lon of bbox
		step: timestep to use: 1, 3, 6
		num_threads: number of threads to use for downloading data
		surf_plev: download surface single level or pressure level product: 'surf' or 'plev'

	Returns:
		Monthly era surface files stored in disk.		 

	"""
	print('\n')
	print('---> Loading ERA5 {} climate forcing'.format(surf_plev))
	bbox = [str(latN), str(lonW), str(latS), str(lonE)]
	time_step_dict = {'1H': ['00:00', '01:00', '02:00',
							 '03:00', '04:00', '05:00',
							 '06:00', '07:00', '08:00',
							 '09:00', '10:00', '11:00',
							 '12:00', '13:00', '14:00',
							 '15:00', '16:00', '17:00',
							 '18:00', '19:00', '20:00',
							 '21:00', '22:00', '23:00'],
					'3H': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
					'6H': ['00:00', '06:00', '12:00', '18:00']}

	df = pd.DataFrame()
	# date_range will make sure to include the month of the latest date (endDate) provided
	df['dates'] = pd.date_range(startDate, pd.Timestamp(endDate)-pd.offsets.Day()+pd.offsets.MonthEnd(), freq='M', inclusive='both')
	df['month'] = df.dates.dt.month
	df['year'] = df.dates.dt.year
	if surf_plev == 'surf':
		df['dataset'] = df.dates.apply(lambda x: 'reanalysis-era5-single-levels' if x.year >= 1979 else 'reanalysis-era5-single-levels-preliminary-back-extension')
		df['target_file'] = df.dates.apply(lambda x: eraDir + "SURF_%04d%02d.nc" % (x.year, x.month))
	elif surf_plev == 'plev':
		df['dataset'] = df.dates.apply(lambda x: 'reanalysis-era5-pressure-levels' if x.year >= 1979 else 'reanalysis-era5-pressure-levels-preliminary-back-extension')
		df['target_file'] = df.dates.apply(lambda x: eraDir + "PLEV_%04d%02d.nc" % (x.year, x.month))
		loc_list = []
		loc_list.extend([plevels]*df.shape[0])
		df['plevels'] = loc_list
	else:
		sys.exit('ERROR: surf_plev can only be surf or plev')
	df['file_exist'] = 0
	df.file_exist = df.target_file.apply(lambda x: os.path.isfile(x)*1)
	df['step'] = step
	df['time_steps'] = df.step.apply(lambda x: time_step_dict.get(x))
	df['bbox'] = df.step.apply(lambda x: bbox)
	df['product_type'] = product

	print("Start = ", df.dates[0].strftime('%Y-%b'))
	print("End = ", df.dates[len(df.dates) - 1].strftime('%Y-%b'))

	if df.file_exist.sum() > 0:
		print("ECWMF {} data found:".format(surf_plev.upper()))
		print(df.target_file.loc[df.file_exist == 1].apply(lambda x: x.split('/')[-1]))

	if (df.file_exist == 0).sum() > 0:
		print("Downloading {} from ECWMF:".format(surf_plev.upper()))
		print(df.target_file.loc[df.file_exist == 0].apply(lambda x: x.split('/')[-1]))

	download = df.loc[df.file_exist == 0]
	if download.shape[0] > 0:
		# ans = input('---> Download ERA5 {} data? (y/n)'.format(surf_plev.upper()))
		# if (ans.lower() == 'y') or (ans == '1'):
		if surf_plev == 'surf':
			pool = ThreadPool(num_threads)
			pool.starmap(era5_request_surf, zip(list(download.dataset),
												list(download.year),
												list(download.month),
												list(download.bbox),
												list(download.target_file),
												list(download.product_type),
												list(download.time_steps)))
			pool.close()
			pool.join()
		elif surf_plev == 'plev':
			pool = ThreadPool(num_threads)
			pool.starmap(era5_request_plev, zip(list(download.dataset),
												list(download.year),
												list(download.month),
												list(download.bbox),
												list(download.target_file),
												list(download.product_type),
												list(download.time_steps),
												list(download.plevels)))
			pool.close()
			pool.join()
		else:
			sys.exit('ERROR: surf_plev can only be surf or plev')
		#else:
		#	sys.exit('ERROR: Some forcing files are missing given the date range provided\n ---> or implement a method to modify start/end date of project to file available')

	if realtime:
		if surf_plev == 'surf':
			# redownload current month to catch missing days in realtime mode.
			era5_realtime_surf(eraDir, df.dataset[0], df.bbox[0], df.product_type[0])

		if surf_plev == 'plev':
			# redownload current month to catch missing days in realtime mode.
			era5_realtime_plev(eraDir, df.dataset[0], df.bbox[0], df.product_type[0], df.plevels[0])

def era5_request_surf(dataset, year, month, bbox, target, product, time):
	"""CDS surface api call

	Args:
		dataset (str): copernicus dataset (era5)
		year (str or list): year of interest
		month (str or list): month of interest
		bbox (list): bonding box in lat-lon
		target (str): filename
		product (str): type of model run. defaul: reanalysis
		time (str or list): hours for which to download data 

	Returns:
		Store to disk dataset as indicated

	"""
	c = cdsapi.Client()
	c.retrieve(
		dataset,
		{'variable': ['geopotential', '2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
					  'surface_solar_radiation_downwards','surface_pressure',
					  'Total precipitation', '2m_temperature', 'TOA incident solar radiation',
					  'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux'
					  ],
		 'product_type': product,
		 "area": bbox,
		 'year': year,
		 'month': '%02d'%(month),
		 'day': ['01', '02', '03',
				 '04', '05', '06',
				 '07', '08', '09',
				 '10', '11', '12',
				 '13', '14', '15',
				 '16', '17', '18',
				 '19', '20', '21',
				 '22', '23', '24',
				 '25', '26', '27',
				 '28', '29', '30',
				 '31'
				 ],
		 'time': time,
		 'grid': [0.25, 0.25],
		 'format': 'netcdf'
		 },
		target)
	print(target + " complete")

def era5_request_plev(dataset, year, month, bbox, target, product, time, plevels):
	"""CDS plevel api call
	
	Args:
		dataset (str): copernicus dataset (era5)
		year (str or list): year of interest
		month (str or list): month of interest
		bbox (list): bonding box in lat-lon
		target (str): filename
		product (str): type of model run. defaul: reanalysis
		time (str or list): hours to query
		plevels (str or list): pressure levels to query

	Returns:
		Store to disk dataset as indicated

	"""
	c = cdsapi.Client()
	c.retrieve(
		dataset,
		{
			'product_type': product,
			'format': 'netcdf',
			"area": bbox,
			'variable': [
				'geopotential', 'temperature', 'u_component_of_wind',
				'v_component_of_wind', 'relative_humidity', 'specific_humidity'
			],
			'pressure_level': plevels,
			'year': year,
			'month': '%02d'%(month),
			'day': [
				'01', '02', '03',
				'04', '05', '06',
				'07', '08', '09',
				'10', '11', '12',
				'13', '14', '15',
				'16', '17', '18',
				'19', '20', '21',
				'22', '23', '24',
				'25', '26', '27',
				'28', '29', '30',
				'31'
			],
			'time': time,
			'grid': [0.25, 0.25],
		},
		target)
	print(target + " complete")


def era5_realtime_surf(eraDir, dataset, bbox, product ):
	# a small routine thaty simply redownloads the current month
	# this is a simple (not most efficient) way of ensuring we have latest data for realtime applications
	# a better way woulfd be to compare date that data is available for with date in latest files, download the difference and append - tobe implemented!

	# get current month
	currentMonth = datetime.now().month 
	currentYear = datetime.now().year 
	print("Running realtime download " + str(currentYear) + "_" + str(currentMonth))

	time = ['00:00', '01:00', '02:00',
							 '03:00', '04:00', '05:00',
							 '06:00', '07:00', '08:00',
							 '09:00', '10:00', '11:00',
							 '12:00', '13:00', '14:00',
							 '15:00', '16:00', '17:00',
							 '18:00', '19:00', '20:00',
							 '21:00', '22:00', '23:00']

	target = eraDir +"/SURF_%04d%02d.nc" % (currentYear, currentMonth)
	era5_request_surf(dataset, currentYear, currentMonth, bbox, target, product, time)


def era5_realtime_plev(eraDir, dataset, bbox, product,plevels ):
	# a small routine thaty simply redownloads the current month
	# this is a simple (not most efficient) way of ensuring we have latest data for realtime applications
	# a better way woulfd be to compare date that data is available for with date in latest files, download the difference and append - tobe implemented!

	# get current month
	currentMonth = datetime.now().month
	currentYear = datetime.now().year
	print("Running realtime download " + str(currentYear) + "_" + str(currentMonth))

	time = ['00:00', '01:00', '02:00',
							 '03:00', '04:00', '05:00',
							 '06:00', '07:00', '08:00',
							 '09:00', '10:00', '11:00',
							 '12:00', '13:00', '14:00',
							 '15:00', '16:00', '17:00',
							 '18:00', '19:00', '20:00',
							 '21:00', '22:00', '23:00']

	target = eraDir + "/PLEV_%04d%02d.nc" % (currentYear, currentMonth)
	plevels = plevels
	era5_request_plev(dataset, currentYear, currentMonth, bbox, target, product, time, plevels)


def return_latest_date():
	# method to return latest full date available in CDS
	
	print("WARNING: Ignore the following warning - this is due to requesting todays date with the intention to harvest the date returned in error message. There must be a cleaner way to get latest date....")
	c = cdsapi.Client()

	# how to retrieve latest date that era5 data is available
	try:
	    # this will always fail but return the latest date in error message
	    # should be a cleaner way of doing this with api?

		c = cdsapi.Client()

		# Retrieve the list of available files for the ERA5 dataset
		res = c.retrieve("reanalysis-era5-single-levels", {
		    "variable": "2m_temperature",
		    "product_type": "reanalysis",
		    "format": "json"
		})

	except Exception as e:
	    # Extract the latest date from the exception object
	    exc_type = type(e).__name__
	    exc_msg = str(e)
	    print(f"Caught {exc_type} with message '{exc_msg}'")

	# Example string representing a date and time within a string
	date_string = exc_msg
	# Define the format string to match the input string
	format_string = 'the request you have submitted is not valid. None of the data you have requested is available yet, please revise the period requested. The latest date available for this dataset is: %Y-%m-%d %H:%M:%S.%f.' 
	# Parse the date string and create a datetime object
	latest_date = datetime.strptime(date_string, format_string)

	if latest_date.hour < 23:
		# Subtract one day from the datetime object
		latest_date = latest_date - timedelta(days=1)
		# Print the updated datetime object

	return(latest_date)
