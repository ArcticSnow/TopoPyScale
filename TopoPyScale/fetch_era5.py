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
from TopoPyScale import topo_export as te
import xarray as xr
import cfgrib
import os
import zipfile
import shutil
import glob
from cdo import *

import era5_downloader as era5down


var_surf_name_google = {'geopotential_at_surface':'z',
                '2m_dewpoint_temperature':'d2m', 
                'surface_thermal_radiation_downwards':'strd',
                'surface_solar_radiation_downwards':'ssrd',
                'surface_pressure':'sp',
                'total_precipitation':'tp',
                '2m_temperature':'t2m', 
                'toa_incident_solar_radiation':'tisr',
                'friction_velocity':'zust', 
                'instantaneous_moisture_flux':'ie', 
                'instantaneous_surface_sensible_heat_flux':'ishf'}

var_plev_name_google = {'geopotential':'z', 
                'temperature':'t', 
                'u_component_of_wind':'u',
                'v_component_of_wind':'v', 
                'specific_humidity':'q'}


def fetch_era5_google_from_zarr(eraDir, startDate, endDate, lonW, latS, lonE, latN, plevels, 
    bucket='gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'):
    '''
    Function to download data from Zarr repository on Google Cloud Storage (https://github.com/google-research/arco-era5/tree/main)

    '''
    print('!!!!!!!!!WORK IN PROGRESS!!!!!!!!!')
    ds = xr.open_zarr(
        bucket,
        chunks=None,
        storage_options=dict(token='anon'),
    )
    ds_plev = ds[[
                'geopotential', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'specific_humidity'
            ]].sel(
        time=slice(startDate, endDate), 
        latitude=slice(latN,latS), 
        longitude=slice(lonE, lonW),
        levels=plevels)
    ds_plev.to_zarr('PLEV.zarr')
    ds_plev = None
    ds_surf = ds.sel(
        time=slice(startDate, endDate), 
        latitude=slice(latN,latS), 
        longitude=slice(lonE, lonW))[['geopotential_at_surface','2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                      'surface_solar_radiation_downwards','surface_pressure',
                      'total_precipitation', '2m_temperature', 'toa_incident_solar_radiation',
                      'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux'
                      ]]
    ds_surf.to_zarr('SURF.zarr')



def fetch_era5_google(eraDir, startDate, endDate, lonW, latS, lonE, latN, plevels, step='3H',num_threads=1):

    print('\n')
    print(f'---> Downloading ERA5 climate forcing from Google Cloud Storage')

    
    bbox = (lonW, latS, lonE, latN)
    time_step_dict = {'1H': (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23),
                    '3H': (0,3,6,9,12,15,18,21),
                    '6H': (0,6,12,18)}

    # prepare download of PLEVEL variables
    local_plev_path = eraDir + "PLEV_{time:%Y%m%d}.nc"
    era5_plevel = era5down.ERA5Downloader(
        bbox_WSEN=bbox,  # Bounding box: West, South, East, North
        dest_path=local_plev_path,  # can also be an S3 bucket with format "s3://bucket-name/era5-{region_name}/...
        levels=plevels,  # Pressure levels (can also be None)
        region_name='dummy',
        variables=[
                'geopotential', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'specific_humidity'
            ],
        time_steps = time_step_dict.get(step), 
    )

    # prepare download of SURFACE variables
    local_surf_path = eraDir + "SURF_{time:%Y%m%d}.nc"
    era5_surf = era5down.ERA5Downloader(
        bbox_WSEN=bbox,  # Bounding box: West, South, East, North
        dest_path=local_surf_path,  # can also be an S3 bucket with format "s3://bucket-name/era5-{region_name}/...
        levels=plevels,  # Pressure levels (can also be None)
        region_name='dummy',
        variables=['geopotential_at_surface','2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                      'surface_solar_radiation_downwards','surface_pressure',
                      'total_precipitation', '2m_temperature', 'toa_incident_solar_radiation',
                      'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux'
                      ],
        time_steps = time_step_dict.get(step), 
    )


    # date_range will make sure to include the month of the latest date (endDate) provided
    date_vec = pd.date_range(startDate, pd.Timestamp(endDate)-pd.offsets.Day()+pd.offsets.MonthEnd(), freq='D', inclusive='both')
    date_list = list(date_vec.strftime("%Y-%m-%d").values)
  
    if num_threads==1:
        print("Downloading data sequentially")
        for date in date_list:
            era5_surf.download(date)
            era5_plevel.download(date)

    elif num_threads>1:
        print(f"Downloading data in parallel with {num_threads} threads.")
        # pool multithread for fetching surface data
        pool = ThreadPool(num_threads)
        pool.starmap(era5_surf.download, zip(date_list))
        pool.close()
        pool.join()

        # pool multithread for fetching surface data
        pool = ThreadPool(num_threads)
        pool.starmap(era5_plev.download, zip(date_list))
        pool.close()
        pool.join()
    else:
        raise ValueError("download_type not available. Currently implemented: sequential, parallel")



def retrieve_era5(product, startDate, endDate, eraDir, latN, latS, lonE, lonW, step, 
                    num_threads=10, surf_plev='surf', plevels=None, realtime=False, 
                    output_format='netcdf', download_format="unarchived", new_CDS_API=True, rm_daily=False):
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
        output_format (str): default is "netcdf", can be "grib".
        download_format (str): default "unarchived". Can be "zip"
        new_CDS_API: flag to handle new formating of SURF files with the new CDS API (2024).
        rm_daily: remove folder containing all daily ERA5 file. Option to clear space of data converted to yearly files.

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
    #df['dates'] = pd.date_range(startDate, pd.Timestamp(endDate)-pd.offsets.Day()+pd.offsets.MonthEnd(), freq='M', inclusive='both')
    df['dates'] = pd.date_range(startDate, pd.Timestamp(endDate), freq='D', inclusive='both')
    df['month'] = df.dates.dt.month
    df['day'] = df.dates.dt.day
    df['year'] = df.dates.dt.year

    os.makedirs(eraDir / "daily", exist_ok=True)

    if surf_plev.lower() == 'surf':
        df['dataset'] = 'reanalysis-era5-single-levels'
        if output_format == "netcdf":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dSURF_%04d%02d%02d.nc" % (x.year, x.month, x.day)))
        if output_format == "grib":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dSURF_%04d%02d%02d.grib" % (x.year, x.month, x.day)))
    
    elif surf_plev.lower() == 'plev':
        df['dataset'] = 'reanalysis-era5-pressure-levels'
        if output_format == "netcdf":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dPLEV_%04d%02d%02d.nc" % (x.year, x.month, x.day)))
        if output_format == "grib":
            df['target_file'] = df.dates.apply(lambda x: eraDir / "daily" / ("dPLEV_%04d%02d%02d.grib" % (x.year, x.month, x.day)))
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
    df['output_format'] = output_format
    df['download_format'] = download_format

    print("Start = ", df.dates[0].strftime('%Y-%m-%d'))
    print("End = ", df.dates[len(df.dates) - 1].strftime('%Y-%m-%d'))

    if df.file_exist.sum() > 0:
        print("ECWMF daily {} data found:".format(surf_plev.upper()))
        print(df.target_file.loc[df.file_exist == 1].apply(lambda x: x.name))

    if (df.file_exist == 0).sum() > 0:
        print("Downloading daily {} from ECWMF:".format(surf_plev.upper()))
        print(df.target_file.loc[df.file_exist == 0].apply(lambda x: x.name))

    download = df.loc[df.file_exist == 0]
    if download.shape[0] > 0:
        if surf_plev == 'surf':
            pool = ThreadPool(num_threads)
            pool.starmap(era5_request_surf, zip(list(download.dataset),
                                                list(download.year),
                                                list(download.month),
                                                list(download.day),
                                                list(download.bbox),
                                                list(download.target_file),
                                                list(download.product_type),
                                                list(download.time_steps),
                                                list(download.output_format),
                                                list(download.download_format)))
            pool.close()
            pool.join()
        elif surf_plev == 'plev':
            pool = ThreadPool(num_threads)
            pool.starmap(era5_request_plev, zip(list(download.dataset),
                                                list(download.year),
                                                list(download.month),
                                                list(download.day),
                                                list(download.bbox),
                                                list(download.target_file),
                                                list(download.product_type),
                                                list(download.time_steps),
                                                list(download.plevels),
                                                list(download.output_format),
                                                list(download.download_format)))
            pool.close()
            pool.join()
        else:
            sys.exit('ERROR: surf_plev can only be surf or plev')
        #else:
        #	sys.exit('ERROR: Some forcing files are missing given the date range provided\n ---> or implement a method to modify start/end date of project to file available')

    if realtime:
        if surf_plev == 'surf':
            # redownload current month to catch missing days in realtime mode.
            era5_realtime_surf(eraDir / "daily", df.dataset[0], df.bbox[0], df.product_type[0])

        if surf_plev == 'plev':
            # redownload current month to catch missing days in realtime mode.
            era5_realtime_plev(eraDir / "daily", df.dataset[0], df.bbox[0], df.product_type[0], df.plevels[0])

    if new_CDS_API:
        # Since mid-2024, CDS API has changed variable name and the way it packages SURF variables. 
        # These two functions apply a small post process to get things in order.
        if surf_plev == 'surf':
            remap_CDSbeta(str(eraDir / "daily" / "dSURF*.nc"), file_type='surf')
        if surf_plev == 'plev':
            remap_CDSbeta(str(eraDir / "daily" / "dPLEV*.nc"), file_type='plev')

    # code to merge daily files to monthly
    # - [ ] write option earlier that checks also if monthly files already exist. improve logic that does not require to have both monthly and daily files
    cdo = Cdo()
    for year in df.year.unique():
        print(f"---> Merging daily {surf_plev.upper()} files from {year}")
        
        if surf_plev == 'surf':
            fpat = str(eraDir / "daily" / ("dSURF_%04d*.nc" % (year)))
            fout = str(eraDir / ("SURF_%04d.nc" % (year)))
            cdo.mergetime(input=fpat, output=fout)

        if surf_plev == 'plev':
            fpat = str(eraDir / "daily" / ("dPLEV_%04d*.nc" % (year)))
            fout = str(eraDir / ("PLEV_%04d.nc" % (year)))
            cdo.mergetime(input=fpat, output=fout)

    if rm_daily:
        shutil.rmtree(eraDir / "daily")


    print("===> ERA5 files ready")


def era5_request_surf(dataset, year, month, day, bbox, target, product, time, output_format= "netcdf", download_format="unarchived"):
    """CDS surface api call

    Args:
        dataset (str): copernicus dataset (era5)
        year (str or list): year of interest
        month (str or list): month of interest
        bbox (list): bonding box in lat-lon
        target (str): filename
        product (str): type of model run. defaul: reanalysis
        time (str or list): hours for which to download data
        output_format (str): default is "netcdf", can be "grib".
        download_format (str): default "unarchived". Can be "zip"

    Returns:
        Store to disk dataset as indicated

    """

    varnames = ['geopotential', '2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                      'surface_solar_radiation_downwards','surface_pressure',
                      'total_precipitation', '2m_temperature', 'toa_incident_solar_radiation',
                      'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux']

    c = cdsapi.Client()
    c.retrieve(
        dataset,
        {'variable': varnames,
         'product_type': [product],
         "area": bbox,
         'year': year,
         'month': '%02d'%(month),
         'day': '%02d'%(day),
         'time': time,
         'grid': "0.25/0.25",
         'data_format': output_format,
         'download_format': download_format
         },
        target)
    print(str(target) + " complete")
    unzip_file(str(target))


def era5_request_plev(dataset, year, month, day, bbox, target, product, time, plevels, output_format= "netcdf", download_format="unarchived"):
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
        output_format (str): default is "netcdf", can be "grib".
        download_format (str): default "unarchived". Can be "zip"

    Returns:
        Store to disk dataset as indicated

    """
    c = cdsapi.Client()
    c.retrieve(
        dataset,
        {
            'product_type': [product],
            "area": bbox,
            'variable': [
                'geopotential', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'specific_humidity'
            ],
            'pressure_level': plevels,
            'year': year,
            'month': '%02d'%(month),
            'day': '%02d'%(day),
            'time': time,
            'grid': "0.25/0.25",
            'data_format': output_format,
            'download_format': download_format
        },
        target)
    print(str(target) + " complete")
    unzip_file(str(target))



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

    target = str(eraDir / "SURF_%04d%02d.nc" % (currentYear, currentMonth))
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

    target = str(eraDir / "PLEV_%04d%02d.nc" % (currentYear, currentMonth))
    plevels = plevels
    era5_request_plev(dataset, currentYear, currentMonth, bbox, target, product, time, plevels)



def return_last_fullday():
    """
    TODO: NEED docstring and explanation 
    """


    # Get current UTC time
    current_time_utc = datetime.utcnow()

    # Round down to the nearest hour
    rounded_time_utc = current_time_utc.replace(minute=0, second=0, microsecond=0)

    # Get time 6 days ago in UTC
    six_days_ago_utc = rounded_time_utc - timedelta(days=6)

    # Format the time strings
    current_time_utc_str = rounded_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    six_days_ago_utc_str = six_days_ago_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Print the results
    six_days_ago_utc = datetime.strptime(six_days_ago_utc_str, "%Y-%m-%d %H:%M:%S UTC")

    # # if 5 days ago is an incomplete day h<23, Subtract one day from the datetime object to get last full day of data
    # if six_days_ago_utc.hour < 23:
    #     last_fullday_data = six_days_ago_utc - timedelta(days=1)
    # else:
    #     last_fullday_data = six_days_ago_utc
    last_fullday_data_str = six_days_ago_utc.strftime("%Y-%m-%d")
    print("Current time (rounded down) in UTC:", current_time_utc_str)
    print("Last full day ERA5T data in UTC:", last_fullday_data_str)

    return (last_fullday_data_str)


def grib2netcdf(gribname, outname=None):
    """
    Function to convert grib file to netcdf

    Args:
        gribname: filename fo grib file to convert
    """
    if outname is None:
        outname = gribname[:-5] + ".nc"

    cdo = Cdo()
    cdo.copy(option='-f nc4', input=gribname, output=outname)


def unzip_file(file_path):
    """
    Function to unpack and repack as NETCDF data sent by the new CDS API. CDS can send either weird zip file shipped as netcdf file for Surface data. 
    The option of zip format can be used to download the data. 
    
    Args:
        file_path: path of era5 data. Typically in TopoPyScale project it will be at: ./inputs/climate/daily
    """
        
    # Step 1: Try to open as a NetCDF file
    try:
        with xr.open_dataset(file_path) as ds:
            print(f"{file_path} is a valid NetCDF file. No processing needed.")

    except Exception:
        print(f"{file_path} is not a valid NetCDF file. Checking if it's a ZIP file.")
    
        # Step 2: Check if it's a ZIP file
        if zipfile.is_zipfile(file_path):
            if file_path[-3:]=='.nc':
                print(f"{file_path} is neither a valid NetCDF nor a ZIP file.")
            
                # Step 3: Rename the file if it's actually a ZIP
                zip_file_path = file_path.replace('.nc', '.zip')
                os.rename(file_path, zip_file_path)
                print(f"Renamed {file_path} to {zip_file_path} for processing.")
        
            # Step 4: Unzip the file
            unzip_dir = zip_file_path.replace('.zip', '')
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            print(f"Unzipped {zip_file_path} to {unzip_dir}.")
            
            # Step 5: Merge `.nc` files inside the unzipped directory
            nc_files = [os.path.join(unzip_dir, f) for f in os.listdir(unzip_dir) if f.endswith('.nc')]
            if not nc_files:
                print(f"No .nc files found in {unzip_dir}.")
                return
            
            merged_file_path = os.path.join(os.path.dirname(zip_file_path), os.path.basename(zip_file_path).replace('.zip', '.nc'))
            try:
                # Combine all `.nc` files
                datasets = [xr.open_dataset(nc_file) for nc_file in nc_files]
                merged_ds = xr.merge(datasets)  # Adjust dimension as needed
                merged_ds.to_netcdf(merged_file_path)
                print(f"Merged .nc files into {merged_file_path}.")

            finally:
                # Close datasets
                for ds in datasets:
                    ds.close()

            # Step 6: Clean up
            os.remove(zip_file_path)
            shutil.rmtree(unzip_dir)
            print(f"Deleted {zip_file_path} and {unzip_dir}.")


def remap_CDSbeta(file_pattern, file_type='SURF'):
    """
    Remapping of variable names from CDS beta to CDS legacy standard.
    
    Args:
        file_pattern (str): path and pattern of the file to be remapped. example: 'inputs/climate/daily/dSURF_*.nc'
        file_type (str): type of file: SURF or PLEV.  
    """

    flist = glob.glob(file_pattern)

    if file_type.lower() == 'plev':
        for nc_file in flist:
            try:
                ds = xr.open_dataset(nc_file)
                ds = ds.rename({ 'pressure_level': 'level', 'valid_time' : 'time'})
                ds = ds.isel(level=slice(None, None, -1))  # reverse order of levels

                try:
                    ds = ds.drop_vars('number')
                except:
                    print("variables not found")

                try:
                    ds = ds.drop_vars('expver')
                except:
                    print("variables not found")

                ds.to_netcdf(nc_file+ "_remap", mode='w')
                # move remap back to orig name
                os.rename(nc_file + "_remap", nc_file)
            except:
                print(nc_file+" already remapped or fc file.")

    if file_type.lower() == 'surf':
        for nc_file in flist:
            try:
                ds = xr.open_dataset(nc_file)
                ds = ds.rename({ 'valid_time' : 'time'})

                try:
                    #cdo delname,number,ishf,ie,zust,tisr SURF_20240925.nc SURF_clean.nc
                    #ds2 = ds.swap_dims({'valid_time': 'time'})
                    ds = ds.drop_vars('ishf')
                    ds = ds.drop_vars('ie')
                    ds = ds.drop_vars('zust')
                    ds = ds.drop_vars('tisr')
                except:
                    print("variables not found")

                try:
                    ds = ds.drop_vars('number')
                except:
                    print("variables not found")

                try:
                    ds = ds.drop_vars('expver')
                except:
                    print("variables not found")

                ds.to_netcdf(nc_file + "_remap", mode='w')
                # move remap back to orig name
                os.rename(nc_file + "_remap", nc_file)

            except:
                print(nc_file+" already remapped or fc file.")







def era5_request_surf_snowmapper( today, latN, latS, lonE, lonW, eraDir, output_format= "netcdf"):
    """CDS surface api call

    Args:
        dataset (str): copernicus dataset (era5)
        today (datetime object): day to download
        target (str): filename
        product (str): type of model run. defaul: reanalysis
        time (str or list): hours for which to download data
        format (str): "grib" or "netcdf"

    Returns:
        Store to disk dataset as indicated

    """
    time = ['00:00', '01:00', '02:00',
                             '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00',
                             '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00',
                             '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00',
                             '21:00', '22:00', '23:00']

    bbox = [str(latN), str(lonW), str(latS), str(lonE)]

    target = eraDir + "/forecast/SURF_%04d%02d%02d.nc" % (today.year, today.month, today.day)

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {'variable': ['geopotential', '2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                      'surface_solar_radiation_downwards','surface_pressure',
                      'Total precipitation', '2m_temperature', 'TOA incident solar radiation',
                      'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux'
                      ],
         'product_type': ['reanalysis'],
         "area": bbox,
         'year': [today.year],
         'month': '%02d'%(today.month),
         'day': [today.day],
         'time': time,
         'grid': [0.25, 0.25],
         'data_format': output_format,
         'download_format': 'unarchived'
         },
        target)
    print(target + " complete")


def era5_request_plev_snowmapper(today, latN, latS, lonE, lonW, eraDir, plevels, output_format= "netcdf"):
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
    time = ['00:00', '01:00', '02:00',
                             '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00',
                             '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00',
                             '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00',
                             '21:00', '22:00', '23:00']

    bbox = [str(latN), str(lonW), str(latS), str(lonE)]

    target = eraDir + "/forecast/PLEV_%04d%02d%02d.nc" % (today.year, today.month, today.day)


    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': ['reanalysis'],
            "area": bbox,
            'variable': [
                'geopotential', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'relative_humidity', 'specific_humidity'
            ],
            'pressure_level': plevels,
            'year': [today.year],
            'month': '%02d'%(today.month),
            'day': [today.day],
            'time': time,
            'grid': [0.25, 0.25],
            'data_format': output_format,
            'download_format': 'unarchived'
        },
        target)
    print(target + " complete")
