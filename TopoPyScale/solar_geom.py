'''
Function to compute Solar angles
S. Filhol, Oct 2021

'''

import pvlib
import pandas as pd
import xarray as xr




df = pd.DataFrame()
df['time'] = pd.date_range('2014-09-01', '2015-08-31', freq='6H', tz='UTC')
df.index = df.time
df['longitude'] = 12
df['latitude'] = 65
df[['zenith','azimuth']] = pvlib.solarposition.get_solarposition(df.time, df.latitude, df.longitude)[['zenith','azimuth']]

