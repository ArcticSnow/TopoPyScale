import numpy as np
from numpy.random import default_rng
import pdb

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


def partition_snow(precip, temp, rh=None, sp=None, method='continuous', tair_low_thresh=272.15, tair_high_thresh=274.15):
    """
    Function to partition precipitation in between rain vs snow based on temperature threshold and mixing around freezing.
    The methods to partition snow/rain precipitation are:
    - continuous: method implemented into Cryogrid.
    - jennings2018 methods are from the publication Jennings et al (2018). DOI: https://doi.org/10.1038/s41467-018-03629-7

    Args:
        precip (array): 1D array, precipitation in mm/hr
        temp (arrray): 1D array, air temperature in K
        rh (array): 1D array, relative humidity in %
        sp (array): 1D array, surface pressure in Pa
        method (str): 'continuous', 'Jennings2018_bivariate', 'Jennings2018_trivariate'.
        tair_low_thresh (float): lower temperature threshold under which all precip is snow. degree K
        tair_high_thresh (float): higher temperature threshold under which all precip is rain. degree K

    Returns: 
        array: 1D array rain
        array: 1D array snow
    """
    #
    def func(x):
            '''
            Function to choose if it snows or not randomly based on probability
            '''
            rng = default_rng()
            return rng.choice([0,1], 1, p=[1-x, x])

    if method.lower() == 'continuous':
        snow = precip * ((temp <= tair_low_thresh) +
                         ((temp > tair_low_thresh) & (temp <= tair_high_thresh)) *
                         (temp - tair_low_thresh) / np.max([1e-12, tair_high_thresh - tair_low_thresh]))

    elif method.lower() == 'jennings2018_bivariate':
        if rh is None:
            print('ERROR: Relative humidity is required')
        else:
            # Compute probability of snowfall
            psnow = 1/(1 + np.exp(-10.04 + 1.41 * (temp - 273.15) + 0.09 * rh))

            # sample random realization based on probability
            snow_IO = np.array([func(xi) for xi in psnow]).flatten()
            snow = precip * snow_IO

    elif method.lower() == 'jennings2018_trivariate':
        if rh is None:
            print('ERROR: Relative humidity is required')
        elif sp is None:
            print('ERROR: Surface pressure is required')
        else:

            # Compute probability of snowfall
            psnow = 1/(1 + np.exp(-12.80 + 1.41 * (temp - 273.15) + 0.09 * rh + 0.03 * (sp / 1000)))

            # sample random realization based on probability
            snow_IO = np.array([func(xi) for xi in psnow]).flatten()
            snow = precip * snow_IO
    else:
        print(f"ERROR, {method} is not available. Choose from: ['continuous', 'Jennings2018_bivariate', 'Jennings2018_trivariate'] ")

    rain = precip - snow
    return rain, snow

def q_2_rh(temp, pressure, qair):
    """
    Function to convert specific humidity (q) to relative humidity (RH) following Bolton (1980)

    Args:
        temp (array): temperature in degree K
        pressure (array): aire pressure in Pa
        qair (array): specific humidity in kg/kg
    Returns: 
        array: relative humidity in the range of [0-1]
    """
    mr = qair / (1-qair)
    e = mr * pressure / (0.62197 + mr)
    # convert temperature to saturated vapor pressure
    es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    rh = e / es
    rh[rh > 1] = 1
    rh[rh < 0] = 0
    return rh


def mixing_ratio(ds, var):
    """
    Function to compute mixing ratio

    Args:
        ds: dataset (following ERA5 variable convention) q-specific humidity in kg/kg
        var: dictionnary of variable name (for ERA surf / ERA plevel)
    Returns: dataset with new variable 'w'
    """
    ds[var['mix_ratio']] = 0.5 * (1 - np.sqrt(1 - 4 * ds[var['spec_h']]))
    return ds


def t_rh_2_dewT(ds, var):
    """
    Function to compute dea temperature from air temperature and relative humidity

    Args:
        ds: dataset with temperature ('t') and relative humidity ('r') variables
        var: dictionnary of variable name (for ERA surf / ERA plevel)
    Returns: 
        dataset: with added variable
    """
    ds['dew'] = 243.04 * (np.log(ds[var['rh']] / 100) + ((17.625 * ds[var['temp']]) / (243.04 + ds[var['temp']])))/\
                (17.625-np.log(ds[var['rh']] / 100) - ((17.625 * ds[var['temp']]) / (243.04 + ds[var['temp']])))
    return ds

def dewT_2_q_magnus(ds, var):
    """
    A version of the Magnus formula with the AERK parameters. AERK from Alduchov and Eskridge (1996)
    Note, e=Magnus(tdc) and es=Magnus(tc)

    Args:
        ds: dataset with
        var: dictionnary of variable name (for ERA surf / ERA plevel)
    Returns: 
        array: specific humidity in kg/kg
    """
    A1, B1, C1 = 17.625, 243.04, 610.94
    vpsl = C1 * np.exp(A1 * (ds[var['dew']] - 273.15) / (B1 + (ds[var['dew']] - 273.15)))
    wsl = eps0 * vpsl / (ds[var['pressure']] - vpsl)
    ds[var['spec_h']] = wsl / (1 + wsl)
    return ds

def vapor_pressure(ds, var):
    """
    Function to compute Vapor pressure from mixing ratio based on 2nd order Taylor series expansion

    Args:
        ds (dataset): dataset with pressure and mix_ratio variables
        var (dict): variable name correspondance 
    Returns:
        dataset: input dataset with new `vp` columns
    """
    eps0 = 0.622  # Ratio of molecular weight of water and dry air [-]
    ds['vp'] = 0.5 * ds[var['pressure']] * (-1 + np.sqrt(1 + 4 * ds[var['mix_ratio']] / eps0))
    return ds