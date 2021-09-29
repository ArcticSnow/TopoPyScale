''''
Retrieve ecmwf data with cdsapi.

J. Fiddes, Origin implementation
S. Filhol adapted in 2021


'''
#!/usr/bin/env python
from datetime import datetime, timedelta
from collections import OrderedDict
import calendar
import sys
#from ecmwfapi import ECMWFDataServer
import cdsapi
from dateutil.relativedelta import *
#from retrying import retry
import logging
import glob	
from joblib import Parallel, delayed 
import subprocess
#import multiprocessing 

def retrieve_era5_surf( product, startDate,endDate,eraDir, latN,latS,lonE,lonW, stepn num_cores):
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

	Returns:
		Monthly era surface files.		 

	"""
	# product  = config["forcing"]["product"]
	# print(product)
	# startDate = config["main"]["startDate"]
	# endDate = config["main"]["endDate"]
	#grd =   config["era-interim"]["grid"]
	#dataset = config["forcing"]["dataset"]
	#grid=str(grd) + "/" + str(grd)
	
	
	bbox=(str(latN) + "/" + str(lonW) + "/" + str(latS) + "/" + str(lonE) )
	
	if (step == "1"):
		time = [		'00:00','01:00','02:00',\
				'03:00','04:00','05:00',\
				'06:00','07:00','08:00',\
				'09:00','10:00','11:00',\
				'12:00','13:00','14:00',\
				'15:00','16:00','17:00',\
				'18:00','19:00','20:00',\
				'21:00','22:00','23:00']

	if (step == "3"):
		time = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00']

	if (step == "6"):
		time = ['00:00','06:00','12:00','18:00']

	# download buffer of +/- 1 month to ensure all necessary timestamps are there for interpolations and consistency between plevel and surf
	dates = [str(startDate), str(endDate)]
	start = datetime.strptime(dates[0], "%Y-%m-%d")
	end = datetime.strptime(dates[1], "%Y-%m-%d")
	
	# dont need to add buffer anymore
	#start = start+relativedelta(months=-1)
	#end = end+relativedelta(months=+1)

	dateList = OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m"), None) for _ in xrange((end - start).days)).keys()


	print("Start date = " , dateList[0])
	print("End date = " , dateList[len(dateList)-1])
	print("cores used = " + str(num_cores))
	
	requestDatesVec = []
	targetVec=[]
	yearVec=[]
	monthVec=[]
	for date in dateList:	
		strsplit = date.split('-' )
		year =  int(strsplit[0])
		month = int(strsplit[1])   
	  #  firstDate = "%04d%02d%02d" % (year, month, 1)
	  #  numberOfDays = calendar.monthrange(year, month)[1]
	  #  lastDate = "%04d%02d%02d" % (year, month, numberOfDays)
		target = eraDir + "SURF_%04d%02d.nc" % (year, month)
	#	requestDates = (firstDate + "/TO/" + lastDate)
	 #   requestDatesVec.append(requestDates)
		targetVec.append(target) 
		yearVec.append(year)
		monthVec.append(month) 

	# find files that already downloaded if any with exact matches (in case of restarts)
	dataExists = glob.glob(eraDir +"/SURF_??????.nc")

	# list only files that dont exist
	targetVecNew = [x for x in targetVec if x not in dataExists]
	logging.info("ECWMF SURF data found:" )
	logging.info(dataExists)
	logging.info("Downloading SURF from ECWMF:")
	logging.info(targetVecNew)

	# Amend requestDatesVec
	index = [targetVec.index(x) for x in targetVecNew]
	yearVecNew  = [yearVec[i] for i in index]
	monthVecNew  = [monthVec[i] for i in index]

	# https://zacharyst.com/2016/03/31/parallelize-a-multifunction-argument-in-python/	
	Parallel(n_jobs=int(num_cores))(delayed( era5_request_surf)(int(yearVecNew[i]), int(monthVecNew[i]), bbox, targetVecNew[i], product, time) for i in range(0,len(yearVecNew)))


#@retry(wait_random_min=10000, wait_random_max=20000)




def era5_request_surf(year, month, bbox, target, product, time):
	"""CDS surface api call"""
	c = cdsapi.Client()

	c.retrieve(
		'reanalysis-era5-single-levels',
		{
		'variable':['geopotential', '2m_dewpoint_temperature', 'surface_thermal_radiation_downwards', 'surface_solar_radiation_downwards',
		'Total precipitation','2m_temperature', 'TOA incident solar radiation',
			'friction_velocity','instantaneous_moisture_flux','instantaneous_surface_sensible_heat_flux'
		],
		'product_type': product,
			"area": bbox,
		'year':int(year),
		'month':[
			int(month)
		],
		'day':[
			'01','02','03',
			'04','05','06',
			'07','08','09',
			'10','11','12',
			'13','14','15',
			'16','17','18',
			'19','20','21',
			'22','23','24',
			'25','26','27',
			'28','29','30',
			'31'
		],
		'time':time,
		'grid': [0.25, 0.25],
		'format':'netcdf'

		},
		target)
	print(target+ " complete")

def retrieve_era5_plev(product, startDate,endDate,eraDir, latN,latS,lonE,lonW, step, plevels, num_cores):
	""" Sets up era5 pressure level retrieval.
	* Creates list of year/month pairs to iterate through. 
	* MARS retrievals are most efficient when subset by time. 
	* Identifies preexisting downloads if restarted. 
	* Calls api using parallel function.

	Args:
		config: config object defining INI
		eraDir: directory to write output
	latN: north latitude of bbox
		latS: south latitude of bbox
		lonE: easterly lon of bbox
		lonW: westerly lon of bbox

	Returns:
		Monthly era pressure level files.		 

	"""
	#product  = config["forcing"]["product"]
	#startDate = config["main"]["startDate"]
	#endDate = config["main"]["endDate"]
	#grd =   config["era-interim"]["grid"]
	#dataset = config["forcing"]["dataset"]
	#grid=str(grd) + "/" + str(grd)
	
	#num_cores = 4 #config['main']['num_cores']
	bbox=(str(latN) + "/" + str(lonW) + "/" + str(latS) + "/" + str(lonE) )

	if (step == "1"):
		time = [		'00:00','01:00','02:00',\
				'03:00','04:00','05:00',\
				'06:00','07:00','08:00',\
				'09:00','10:00','11:00',\
				'12:00','13:00','14:00',\
				'15:00','16:00','17:00',\
				'18:00','19:00','20:00',\
				'21:00','22:00','23:00']

	if (step == "3"):
		time = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00']

	if (step == "6"):
		time = ['00:00','06:00','12:00','18:00']

	# download buffer of +/- 1 month to ensure all necessary timestamps are there for interpolations and consistency between plevel and surf
	dates = [str(startDate), str(endDate)]
	start = datetime.strptime(dates[0], "%Y-%m-%d")
	end = datetime.strptime(dates[1], "%Y-%m-%d")
	# buffer not needed anymore
	#start = start+relativedelta(months=-1)
	#end = end+relativedelta(months=+1)
	dateList = OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m"), None) for _ in xrange((end - start).days)).keys()


	print("Start date = " , dateList[0])
	print("End date = " , dateList[len(dateList)-1])
	print("cores used = " + str(num_cores))

	requestDatesVec = []
	targetVec=[]
	yearVec=[]
	monthVec=[]
	for date in dateList:	
		strsplit = date.split('-' )
		year =  int(strsplit[0])
		month = int(strsplit[1])   
	  #  firstDate = "%04d%02d%02d" % (year, month, 1)
	  #  numberOfDays = calendar.monthrange(year, month)[1]
	  #  lastDate = "%04d%02d%02d" % (year, month, numberOfDays)
		target = eraDir + "PLEV_%04d%02d.nc" % (year, month)
	#	requestDates = (firstDate + "/TO/" + lastDate)
	 #   requestDatesVec.append(requestDates)
		targetVec.append(target) 
		yearVec.append(year)
		monthVec.append(month) 

	# find files that already downloaded if any with exact matches (in case of restarts)
	dataExists = glob.glob(eraDir +"/PLEV_??????.nc")

	# list only files that dont exist
	targetVecNew = [x for x in targetVec if x not in dataExists]
	logging.info("ECWMF PLEV data found:" )
	logging.info(dataExists)
	logging.info("Downloading PLEV from ECWMF:")
	logging.info(targetVecNew)

	# Amend requestDatesVec
	index = [targetVec.index(x) for x in targetVecNew]
	yearVecNew  = [yearVec[i] for i in index]
	monthVecNew  = [monthVec[i] for i in index]

	# https://zacharyst.com/2016/03/31/parallelize-a-multifunction-argument-in-python/	
	Parallel(n_jobs=int(num_cores))(delayed( era5_request_plev)(int(yearVecNew[i]), int(monthVecNew[i]), bbox, targetVecNew[i], product, time,plevels) for i in range(0,len(yearVecNew)))


#@retry(wait_random_min=10000, wait_random_max=20000)




def era5_request_plev(year, month, bbox, target, product, time,plevels):
	"""CDS plevel api call"""
	c = cdsapi.Client()

	c.retrieve(
		'reanalysis-era5-pressure-levels',
		{
		'product_type': product,
		'format':'netcdf',
		"area": bbox,
		'variable':[
			'geopotential','temperature','u_component_of_wind',
			'v_component_of_wind', 'relative_humidity' ,'specific_humidity'
		],
		'pressure_level':plevels,
		'year':int(year),
		'month':[
			int(month)
		],
		'day':[
			'01','02','03',
			'04','05','06',
			'07','08','09',
			'10','11','12',
			'13','14','15',
			'16','17','18',
			'19','20','21',
			'22','23','24',
			'25','26','27',
			'28','29','30',
			'31'
		],
		'time':time,
		'grid': [0.25, 0.25],
		'format':'netcdf'
		},
		target)
	print(target+ " complete")


#if __name__ == "__main__":

#	config	= sys.argv[1]
#	eraDir	 = sys.argv[2]
#	latNorth	= str(float(sys.argv[3]))
#	latSouth	=  str(float(sys.argv[4]))
#	lonEast	 = str(float(sys.argv[5]))
#	lonWest	 = str(float(sys.argv[6]))

#	retrieve_interim(config,eraDir, latNorth,latSouth,lonEast,lonWest)

def eraCat(wd, grepStr):

	"""
	*** CDO ***
	Concats monthly era (interim or 5) files by some keyword *grepStr*. Depends on CDO.
	- reads monthly ERA5 data files (MARS efficiency)
	- concats to single file timeseries

	Syntax: cdo -b F64 -f nc2 mergetime SURF* SURF.nc

	Args:
		wd: directory of era monthly datafiles
		grepStr: "PLEV" or "SURF"
	Example:
		wd=/home/eraDat/
		grepStr= "PLEV"
		eraCat("/home/joel/mnt/myserver/sim/wfj_era5/eraDat/", "PLEV")
	"""
	
	cmd	 = ("cdo -b F64 -f nc2 mergetime " 
				+ wd 
				+  grepStr
				+ "* " 
				+ wd 
				+"/"
				+grepStr
				+".nc")
	subprocess.check_output(cmd, shell = "TRUE")


def eraCat5d(wd, grepStr):

	"""
	*** NCO ***
	Updated method to deal with 5D of ensemble datasets NOT supported by CDO
	Concats monthly era (interim or 5) files by some keyword *grepStr*. Depends on NCO.
	sudo apt-get install nco
	- reads monthly ERA5 data files (MARS efficiency)
	- concats to single file timeseries
	- 2 steps 
		- assign time to "record" dimeanion in firstfile
		- concat files

	Args:
		wd: directory of era monthly datafiles
		grepStr: "PLEV" or "SURF"
	Example:
		wd=/home/eraDat/
		grepStr= "PLEV"
		eraCat("/home/joel/mnt/myserver/sim/wfj_era5/eraDat/", "PLEV")
	"""
	
	lst = glob.glob(grepStr+'*')
	lst.sort()

	firstfile = lst[0]
	cmd	 = cmd	 = ("ncks -O --mk_rec_dmn time " + firstfile + " " + firstfile)
	subprocess.check_output(cmd, shell = "TRUE")

	cmd	 = ("ncrcat " +grepStr+"*"+" " + grepStr+".nc")
	subprocess.check_output(cmd, shell = "TRUE")


	
def retrieve_era5_tpmm( startDate,endDate,eraDir, latN,latS,lonE,lonW):
	"""
	retrive monthly tp means to correct 6 or 3h TP retrieval
	documented era5 here: https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation#ERA5datadocumentation-Monthlymeans
	values are mean daily for that month, weight by number of days in month eg annualtotal = sum(wfj_m*c(31,28,31,30,31,30,31,31,30,31,30,31))
	"""
	# eraDir='/home/joel/sim/imis/forcing/'	
	# latN=47.9
	# latS=45.75
	# lonW=5.7
	# lonE=10.6

	# startDate = '1979-09-01'                        # adjust to your requirements - as of 2017-07 only 2010-01-01 onwards is available
	# endDate = '2018-09-01'   
	param='total_precipitation'                     # adjust to your requirements
	months = [1,2,3,4,5,6,7,8,9,10,11,12] 
	
	dates = [str(startDate), str(endDate)]
	start = datetime.strptime(dates[0], "%Y-%m-%d")
	end = datetime.strptime(dates[1], "%Y-%m-%d")
	start = start+relativedelta(months=-1)
	end = end+relativedelta(months=+1)
	dateList = OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m"), None) for _ in xrange((end - start).days)).keys()
	#target = eraDir + "tp_%04d%02d.nc" % (startDate, endDate)
	target = eraDir + "tpmm.nc"
	bbox=(str(latN) + "/" + str(lonW) + "/" + str(latS) + "/" + str(lonE) )
	
	yearVec=[]
	for date in dateList:	
		strsplit = date.split('-' )
		year =  int(strsplit[0])
		yearVec.append(year)
	

	myyears = list(sorted(set(yearVec)))

	c = cdsapi.Client()

	c.retrieve(
	    'reanalysis-era5-single-levels-monthly-means',
	    {
		'product_type':'reanalysis-monthly-means-of-daily-means',
		'area': bbox,
		'variable': param,
		'year':	myyears,
		'month':[
		    '01','02','03',
		    '04','05','06',
		    '07','08','09',
		    '10','11','12'
		],
		'time':'00:00',
		'format':'netcdf'
	    },
	    target)
