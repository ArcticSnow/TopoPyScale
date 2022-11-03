"""
Tools to download and compare Downsclaed timeseries to observation
All observation in folder inputs/obs/
S. Filhol December 2021
"""
import requests, os, glob
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")



def get_metno_obs(sources, voi, start_date, end_date, client_id=os.getenv('FROST_API_CLIENTID')):
    """
    Function to download observation data from MetNo FROST API (Norwegian Meteorological institute)

    Args
        sources (list): station code, e.g. 'SN25830'
        voi (list): variables to download
        start_date (str): starting date
        end_date (str): ending date
        client_id (str): FROST_API_CLIENTID

    Return: 
        dataframe: all data combined together
    
    List of variable: https://frost.met.no/element table
    Find out about stations: https://seklima.met.no/
	WARNING: Download max one year at the time to not reach max limit of data to download.

    TODO:
    - convert df to xarray dataset with stn_id, time as coordinates
    """
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    
    df_out = pd.DataFrame()
    for source in sources:
        print('.............................................')
        print('---> Retreiving data for {}'.format(source))
        for var in voi:
            parameters = {
                'sources': source,
                'elements': var,
                'referencetime': start_date + '/' + end_date,
            }

            # Issue an HTTP GET request
            r = requests.get(endpoint, parameters, auth=(client_id,''))
            json = r.json()
            # Check if the request worked, print out any errors
            if r.status_code == 200:
                data = json['data']
                print('-> got VOI: {}'.format(var))
                df = pd.DataFrame()
                for i in range(len(data)):
                    row = pd.DataFrame(data[i]['observations']) # [:1] # raw data = [:1]; corrected = [1:]
                    row['referenceTime'] = data[i]['referenceTime']
                    row['sourceId'] = data[i]['sourceId']
                    df = df.append(row)

                df = df.reset_index()
                df = df[['elementId', 'value', 'qualityCode', 'referenceTime', 'sourceId', 'unit']].copy()
                df['referenceTime'] = pd.to_datetime(df['referenceTime'])
                df_out = df_out.append(df)
            else:
                print('..... Error with variable {}! Returned status code {}'.format(var, r.status_code))
                print('..... Reason: %s' % json['error']['reason'])
                print('---> {} for {} skipped'.format(var, source))
    
    return df_out

def combine_metno_obs_to_xarray(fnames='metno*.pckl', path='inputs/obs/'):
    """
    Function to convert metno format to usable dataset

    Args:
        fnames (str pattern): pattern of the file to load
        path (str): path of the file to load

    Returns:
        dataset: dataset will all data organized in timeseries and station number

    """
    df_obs = pd.concat(map(pd.read_pickle, glob.glob(os.path.join('', path + fnames))))
    df = pd.pivot_table(df_obs, columns=['elementId'],values=['value'], index=['sourceId', 'referenceTime'])
    ds = df.xs('value', axis=1, level=0).to_xarray()
    return ds


def fetch_WMO_insitu_observations(year, month, bbox, target_path='./inputs/observations'):
    """
    Function to download WMO in-situ data from land surface in-situ observations from Copernicus.
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-surface-land?tab=overview

    Args:
        year (str or list): year(s) to download
        month (str or list): month(s) to download
        bbox (list): bonding box in lat-lon [lat_south, lon_west, lat_north, lon_east]
        target (str): filename

    Returns:
        Store to disk the dataset as zip file

    TODO:
        - [x] test function
        - [ ] check if can download large extent at once or need multiple requests?
        - [x] save data in individual files for each stations. store either as csv or netcdf (better)
    """
    import cdsapi
    import zipfile

    try:
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
    except:
        print(f'ERROR: path "{target_path}" does not exist')
        return False

    c = cdsapi.Client()

    c.retrieve(
        'insitu-observations-surface-land',
        {
            'time_aggregation': 'sub_daily',
            'variable': [
                'air_pressure', 'air_pressure_at_sea_level', 'air_temperature',
                'dew_point_temperature', 'wind_from_direction', 'wind_speed'
            ],
            'usage_restrictions': 'restricted',
            'data_quality': 'passed',
            'year': year,
            'month': month,
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
            'area': bbox,
            'format': 'zip',
        },
        target_path + os.sep + 'download.zip')
    print('---> download complete')

    try:
        with zipfile.ZipFile(target_path + os.sep + 'download.zip') as z:
            z.extractall(path=target_path)
            print('---> Observation extracted')
            os.remove(target_path + os.sep + 'download.zip')
            print(f'\t Stored in file {target_path + os.sep + "download.zip2"}')
            return
    except:
        print(f'ERROR: Invalid target path\n\t target path used: {target_path}')


def parse_WMO_insitu_observations(fname=None, file_pattern='inputs/observations/surf*subset_csv*.csv', path='./inputs/observations'):
    """
    Function to parse WMO files formated from CDS database. parse in single station file

    Args:
        fname:
        file_pattern:
        path:

    Returns:

    """

    if fname is None and file_pattern is not None:
        flist = glob.glob(file_pattern)
        if len(flist) == 1:
            fname = flist[0]
        else:
            pmp = 'List of files available: \n'
            for i, file in enumerate(flist):
                pmp += f'\t- [{i}] {file} \n'

            res = input(pmp + 'Pick file to load:\n')
            fname = flist[int(res)]
    elif fname is None and file_pattern is None:
        print('ERROR: choose filename or provide file naming pattern')
        return False

    df = pd.read_csv(fname)
    for stn in df.station_name.unique():
        dd = df.loc[df.station_name==stn]
        dd['time'] = pd.to_datetime(dd.date_time, utc=True)
        de = dd.pivot(index='time', columns=['observed_variable'],values=[ 'observation_value'])
        de = de.droplevel(level=0, axis=1)

        ds = xr.Dataset(coords=dict(
            time = de.index,
            reference_time = pd.to_datetime('1970-01-01T00:00:00Z'),
            latitude = dd.latitude.unique()[0],
            longitude = dd.longitude.unique()[0],
            station_name = stn,
            station_id = dd.primary_station_id.unique()[0]
        ))
        ds.time.attrs = {'units':'ns'}
        for col in de.columns:
            ds[col] = ('time', de[col])
            ds[col].attrs = {'units': dd.loc[dd.observed_variable==col].units.unique()[0], 'standard_name': col}

        stn_f = stn.replace(' ', '_')
        foutput = f'{stn_f}_{dd.time.min().strftime("%Y%m%d")}_{dd.time.max().strftime("%Y%m%d")}.nc'
        ds.to_netcdf(path + os.sep + foutput)
        print(f'---> Observation data for station {stn} parsed into file \n\t {foutput}')
        dd = None
        de = None
        ds = None




    # add here logic


    # store each observation by stations with





if __name__ == "__main__":
	#=================== code ========================
	df_stn = pd.read_csv('inputs/dem/station_list.csv')
	voi = ['air_temperature',
	       'sum(precipitation_amount PT12H)',
	       'wind_speed',
	       'wind_from_direction',
	       'relative_humidity',
	       'air_pressure_at_sea_level']
	sources = df_stn.stn_number.unique()

	dates = ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2021-08-31']
	for i, start in enumerate(dates[:-1]):
	    print('---> Downloading year {} {}'.format(start, dates[i+1]))
	    df = get_metno_obs(sources, voi, start, dates[i+1], os.getenv('FROST_API_CLIENTID'))
	    df.to_pickle('inputs/climate/metno_obs{}_{}.pckl'.format(start, dates[i+1]))