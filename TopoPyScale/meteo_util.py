import numpy as np

var_era_plevel = {
    'temp': 't',
    'pressure': 'p',
    'rh': 'r',
    'elev': 'z',
    'spec_h': 'q',
    'mix_ratio': 'w',
    'dew': 'dew',
    'vapor_pressure': 'wp'
}

var_era_surf = {
    'elev': 'z',
    'dew': 'd2m',
    'pressure': 'sp',
    'temp': 't2m',
    'spec_h': 'q',
    'mix_ratio': 'w',
    'vapor_pressure': 'wp'
}

# Physical constants
g = 9.81    #  Acceleration of gravity [ms^-1]
R = 287.05  #  Gas constant for dry air [JK^-1kg^-1]
eps0 = 0.622  # Ratio of molecular weight of water and dry air [-]
S0 = 1370    # Solar constat (total TOA solar irradiance) [Wm^-2] used in ECMWF's IFS


def partition_snow(precip, temp, tair_low_thresh=272.15, tair_high_thresh=274.15):
    '''
    Function to partition precipitation in between rain vs snow based on temperature threshold and mixing around freezing
    :param precip: 1D array, precipitation in mm/hr
    :param temp: 1S array, air temperature in K
    :param tair_low_thresh: float, lower temperature threshold under which all precip is snow. degree K
    :param tair_high_thresh: float, higher temperature threshold under which all precip is rain. degree K
    :return: 1D array rain, 1D array snow
    '''
    snow = precip * ((temp <= tair_low_thresh) +
                     ((temp > tair_low_thresh) & (temp <= tair_high_thresh)) *
                     (temp - tair_low_thresh) / np.max([1e-12, tair_high_thresh - tair_low_thresh]))

    rain = precip - snow

    return rain, snow

def q_2_rh(temp, pressure, qair):
    '''
    Function to convert specific humidity (q) to relative humidity (RH) following Bolton (1980)
    :param temp: temperature in degree K
    :param pressure: aire pressure in Pa
    :param qair: specific humidity in kg/kg
    :return: rh [0-1]
    '''
    mr = qair / (1-qair)
    e = mr * pressure / (0.62197 + mr)
    # convert temperature to saturated vapor pressure
    es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    rh = e / es
    rh[rh > 1] = 1
    rh[rh < 0] = 0
    return rh


def mixing_ratio(ds, var):
    '''
    Function to compute mixing ratio
    :param ds: dataset (following ERA5 variable convention) q-specific humidity in kg/kg
    :param var: dictionnary of variable name (for ERA surf / ERA plevel)
    :return: dataset with new variable 'w'
    '''
    ds[var['mix_ratio']] = 0.5 * (1 - np.sqrt(1 - 4 * ds[var['spec_h']]))
    return ds


def t_rh_2_dewT(ds, var):
    '''
    Function to compute dea temperature from air temperature and relative humidity
    :param ds: dataset with temperature ('t') and relative humidity ('r') variables
    :param var: dictionnary of variable name (for ERA surf / ERA plevel)
    :return: dataset with added variable
    '''
    ds['dew'] = 243.04 * (np.log(ds[var['rh']] / 100) + ((17.625 * ds[var['temp']]) / (243.04 + ds[var['temp']])))/\
                (17.625-np.log(ds[var['rh']] / 100) - ((17.625 * ds[var['temp']]) / (243.04 + ds[var['temp']])))
    return ds

def dewT_2_q_magnus(ds, var):
    '''
    A version of the Magnus formula with the AERK parameters. AERK from Alduchov and Eskridge (1996)
    Note, e=Magnus(tdc) and es=Magnus(tc)
    :param ds: dataset with
    :param var: dictionnary of variable name (for ERA surf / ERA plevel)
    :return: specific humidity in kg/kg
    '''
    A1, B1, C1 = 17.625, 243.04, 610.94
    vpsl = C1 * np.exp(A1 * (ds[var['dew']] - 273.15) / (B1 + (ds[var['dew']] - 273.15)))
    wsl = eps0 * vpsl / (ds[var['pressure']] - vpsl)
    ds[var['spec_h']] = wsl / (1 + wsl)
    return ds

def vapor_pressure(ds, var):
    '''
    Function to compute Vapor pressure from mixing ratio based on 2nd order Taylor series expansion
    :param ds:
    :param var:
    :return:
    '''
    eps0 = 0.622  # Ratio of molecular weight of water and dry air [-]
    ds['vp'] = 0.5 * ds[var['pressure']] * (-1 + np.sqrt(1 + 4 * ds[var['mix_ratio']] / eps0))
    return ds