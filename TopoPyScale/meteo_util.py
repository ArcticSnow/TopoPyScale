import numpy as np
import warnings

try:
    from numba import jit, float64, boolean
    _numba_available = True
except ImportError:
    _numba_available = False

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


def steep_snow_reduce_all(snow, slope):
    # ======================================================================================
    # # linearly reduce snow to zero in steep slopes
    # if steepSnowReduce=="TRUE": # make this an option if need that in future
    # ======================================================================================

    snowSMIN = 30.
    snowSMAX = 80.

    # partial snow removal
    k = (snowSMAX - slope) / (snowSMAX - snowSMIN)

    # no snow removal
    a = slope < snowSMIN
    k[~a] = 1

    # all snow removal
    a = slope > snowSMIN
    k[~a] = 0

    steepSnow = snow.transpose() * np.array(k).transpose()

    return (steepSnow)


def _sample_bernoulli_vect(probs):
    """
    Vectorized Bernoulli sampling — compares uniform random draws
    against probabilities instead of calling rng.choice per element.
    """
    return (np.random.random(len(probs)) < probs).astype(np.float64)


def _psnow_bivariate(temp, rh):
    """Compute snow probability (Jennings 2018, bivariate)."""
    return 1.0 / (1.0 + np.exp(-10.04 + 1.41 * (temp - 273.15) + 0.09 * rh))


def _psnow_trivariate(temp, rh, sp):
    """Compute snow probability (Jennings 2018, trivariate)."""
    return 1.0 / (1.0 + np.exp(-12.80 + 1.41 * (temp - 273.15) + 0.09 * rh + 0.03 * (sp / 1000.0)))


if _numba_available:

    @jit(float64[:](float64[:], float64[:], float64, float64), nopython=True, cache=True)
    def _partition_continuous_numba(precip, temp, tair_low, tair_high):
        """Numba-compiled continuous method for rain/snow partitioning."""
        n = len(precip)
        snow = np.empty(n, dtype=np.float64)
        inv_range = 1.0 / max(1e-12, tair_high - tair_low)
        for i in range(n):
            if temp[i] <= tair_low:
                snow[i] = precip[i]
            elif temp[i] >= tair_high:
                snow[i] = 0.0
            else:
                snow[i] = precip[i] * (temp[i] - tair_low) * inv_range
        return snow


def partition_snow(precip, temp, rh=None, sp=None, method='continuous',
                   tair_low_thresh=272.15, tair_high_thresh=274.15, seed=None):
    """
    Partition precipitation into rain vs snow.

    Methods:
      - continuous: Cryogrid / linear mixing around freezing
      - jennings2018_bivariate / jennings2018_trivariate: probabilistic
        partition from Jennings et al (2018). DOI: 10.1038/s41467-018-03629-7

    Parameters
    ----------
    precip : ndarray
        1-D precipitation (mm hr⁻¹)
    temp : ndarray
        1-D air temperature (K)
    rh : ndarray, optional
        Relative humidity (%) — required for Jennings methods
    sp : ndarray, optional
        Surface pressure (Pa) — required for trivariate method
    method : str
        'continuous' | 'jennings2018_bivariate' | 'jennings2018_trivariate'
    tair_low_thresh : float
        Below this temperature all precip is snow (K)
    tair_high_thresh : float
        Above this temperature all precip is rain (K)
    seed : int, optional
        Seed for the random number generator (Jennings methods only)

    Returns
    -------
    rain : ndarray
    snow : ndarray
    """
    if seed is not None:
        np.random.seed(seed)

    precip = np.asarray(precip, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    t_lo = np.float64(tair_low_thresh)
    t_hi = np.float64(tair_high_thresh)

    method_lc = method.lower()

    if method_lc == 'continuous':
        if _numba_available:
            snow = _partition_continuous_numba(precip, temp, t_lo, t_hi)
        else:
            ratio = np.clip((temp - t_lo) / max(1e-12, t_hi - t_lo), 0.0, 1.0)
            snow = precip * (1.0 - ratio)

    elif method_lc == 'jennings2018_bivariate':
        if rh is None:
            raise ValueError('Relative humidity is required for Jennings 2018 bivariate')
        rh = np.asarray(rh, dtype=np.float64)
        psnow = _psnow_bivariate(temp, rh)
        snow_mask = _sample_bernoulli_vect(psnow)
        snow = precip * snow_mask

    elif method_lc == 'jennings2018_trivariate':
        if rh is None:
            raise ValueError('Relative humidity is required for Jennings 2018 trivariate')
        if sp is None:
            raise ValueError('Surface pressure is required for Jennings 2018 trivariate')
        rh = np.asarray(rh, dtype=np.float64)
        sp = np.asarray(sp, dtype=np.float64)
        psnow = _psnow_trivariate(temp, rh, sp)
        snow_mask = _sample_bernoulli_vect(psnow)
        snow = precip * snow_mask

    else:
        raise ValueError(
            f"Method '{method}' not available. "
            "Choose from: 'continuous', 'jennings2018_bivariate', 'jennings2018_trivariate'"
        )

    rain = precip - snow
    return rain, snow


def q_2_rh(temp, pressure, qair):
    """
    Function to convert specific humidity (q) to relative humidity (RH) following Bolton (1980)

    Args:
        temp (array): temperature in degree K
        pressure (array): air pressure in Pa
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
    Function to compute dew temperature from air temperature and relative humidity

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
    ds.vp.attrs = {'units': '???????', 'long_name': 'Vapor pressure', 'standard_name': 'vapor_pressure'}
    return ds
