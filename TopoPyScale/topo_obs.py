'''
Tools to download and compare Downsclaed timeseries to observation
All observation in folder inputs/obs/
S. Filhol December 2021
'''

import requests, os
import pandas as pd
import numpy as np


def get_metno_obs(sources, voi, start_date, end_date, client_id=os.getenv('FROST_API_CLIENTID')):
    '''
    Function to download observation data from MetNo FROST API (Norwegian Meteorological institute)
    :param sources: list, station code, e.g. 'SN25830'
    :param voi: list of variables to download
    :param start_date: starting date
    :param end_date: ending date
    :param client_id: FROST_API_CLIENTID
    :return: dataframe with all data combined together
    
    
    List of variable: https://frost.met.no/element table
    Find out about stations: https://seklima.met.no/
	WARNING: Download max one year at the time to not reach max limit of data to download.

    TODO:
    - convert df to xarray dataset with stn_id, time as coordinates
    '''
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
    df_obs = pd.concat(map(pd.read_pickle, glob.glob(os.path.join('', path + fnames))))
    df = pd.pivot_table(df_obs, columns=['elementId'],values=['value'], index=['sourceId', 'referenceTime'])
    ds = df.xs('value', axis=1, level=0).to_xarray()
    return ds


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