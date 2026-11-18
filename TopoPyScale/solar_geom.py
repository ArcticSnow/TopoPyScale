"""
Function to compute Solar angles
S. Filhol, Oct 2021

The solar position follows the NREL Solar Position Algorithm (Reda and Andreas, 2004) as implemented in pvlib
(method 'nrel_numpy'). Its time-dependent part (heliocentric position of the Earth, nutation, sidereal time, sun right
ascension and declination) does not depend on the location: it is computed once for all time steps, and only the
location-dependent part is evaluated for each point, as arrays (time x point).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import xarray as xr
from pyproj import Transformer

S0_DEFAULT = 1361.  # nominal total solar irradiance at 1 AU [W m-2] (IAU 2015 Resolution B3)


def _spa_module():
    """pvlib's NREL SPA implementation in plain numpy (not numba), whose functions broadcast over arrays."""
    try:
        from pvlib.solarposition import _spa_python_import
        return _spa_python_import('numpy')
    except ImportError:
        from pvlib import spa
        return spa


def _unixtime(times):
    return np.asarray((times - pd.Timestamp('1970-01-01', tz='UTC')).total_seconds(), dtype=float)


def spa_time_terms(times):
    """
    Location-independent part of the NREL SPA, for a DatetimeIndex in UTC.

    Returns:
        dict of 1D arrays (time): apparent sidereal time v, sun right ascension alpha and declination delta [deg],
        equatorial horizontal parallax xi [deg], Earth-Sun distance R [AU]
    """
    spa = _spa_module()
    times = pd.DatetimeIndex(times)
    unixtime = _unixtime(times)
    delta_t = spa.calculate_deltat(times.year.values, times.month.values)
    jd = spa.julian_day(unixtime)
    jde = spa.julian_ephemeris_day(jd, delta_t)
    jc = spa.julian_century(jd)
    jce = spa.julian_ephemeris_century(jde)
    jme = spa.julian_ephemeris_millennium(jce)
    R = spa.heliocentric_radius_vector(jme)
    L = spa.heliocentric_longitude(jme)
    B = spa.heliocentric_latitude(jme)
    theta = spa.geocentric_longitude(L)
    beta = spa.geocentric_latitude(B)
    x0 = spa.mean_elongation(jce)
    x1 = spa.mean_anomaly_sun(jce)
    x2 = spa.mean_anomaly_moon(jce)
    x3 = spa.moon_argument_latitude(jce)
    x4 = spa.moon_ascending_longitude(jce)
    nutation = np.empty((2, len(x0)))
    spa.longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4, nutation)
    epsilon = spa.true_ecliptic_obliquity(spa.mean_ecliptic_obliquity(jme), nutation[1])
    lamd = spa.apparent_sun_longitude(theta, nutation[0], spa.aberration_correction(R))
    v = spa.apparent_sidereal_time(spa.mean_sidereal_time(jd, jc), nutation[0], epsilon)
    return {'v': v,
            'alpha': spa.geocentric_sun_right_ascension(lamd, epsilon, beta),
            'delta': spa.geocentric_sun_declination(lamd, epsilon, beta),
            'xi': spa.equatorial_horizontal_parallax(R),
            'R': np.asarray(R, dtype=float)}


def spa_positions(terms, latitude, longitude, elevation):
    """
    Location-dependent part of the NREL SPA, broadcast over time and points.

    Args:
        terms (dict): from spa_time_terms()
        latitude, longitude (array): degrees, (point,)
        elevation (array): m, (point,)

    Returns:
        zenith, azimuth (clockwise from north), elevation angle: degrees, (time, point), without refraction
        (the 'zenith', 'azimuth' and 'elevation' columns of pvlib.solarposition.get_solarposition)
    """
    spa = _spa_module()
    v, alpha, delta, xi = (np.asarray(terms[k])[:, None] for k in ['v', 'alpha', 'delta', 'xi'])
    lat, lon, elev = (np.asarray(a, dtype=float)[None, :] for a in (latitude, longitude, elevation))
    H = spa.local_hour_angle(v, lon, alpha)
    u = spa.uterm(lat)
    x = spa.xterm(u, lat, elev)
    y = spa.yterm(u, lat, elev)
    delta_alpha = spa.parallax_sun_right_ascension(x, xi, H, delta)
    delta_prime = spa.topocentric_sun_declination(delta, x, y, xi, delta_alpha, H)
    H_prime = spa.topocentric_local_hour_angle(H, delta_alpha)
    e0 = spa.topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
    azimuth = spa.topocentric_azimuth_angle(spa.topocentric_astronomers_azimuth(H_prime, delta_prime, lat))
    return spa.topocentric_zenith_angle(e0), azimuth, e0


def cos_zenith_geocentric(terms, latitude, longitude):
    """
    Cosine of the geocentric solar zenith angle, (time, point): cheap, used to average the top-of-atmosphere
    irradiance over sub-steps. It neglects the parallax (< 0.003 deg) and the atmospheric refraction.
    """
    H = np.deg2rad(np.asarray(terms['v'])[:, None] + np.asarray(longitude, float)[None, :]
                   - np.asarray(terms['alpha'])[:, None])
    lat = np.deg2rad(np.asarray(latitude, float))[None, :]
    dec = np.deg2rad(np.asarray(terms['delta']))[:, None]
    return np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)


def sun_geocentric(terms, latitude, longitude):
    """
    Geocentric solar elevation and azimuth, (time, point): cheap, used for the shading sub-steps. It neglects the
    parallax (< 0.003 deg) and the atmospheric refraction.

    Returns:
        elevation [rad], azimuth [rad, 0 towards south, positive towards east, as ds_solar]
    """
    H = np.deg2rad(np.asarray(terms['v'])[:, None] + np.asarray(longitude, float)[None, :]
                   - np.asarray(terms['alpha'])[:, None])
    lat = np.deg2rad(np.asarray(latitude, float))[None, :]
    dec = np.deg2rad(np.asarray(terms['delta']))[:, None]
    elevation = np.arcsin(np.clip(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H), -1, 1))
    # astronomers' azimuth of the SPA (from south, positive towards west); ds_solar uses the opposite sign
    gamma = np.arctan2(np.sin(H), np.cos(H) * np.sin(lat) - np.tan(dec) * np.cos(lat))
    return elevation, -gamma


def substep_offsets(accumulation_window, n_substeps):
    """Offsets of the centres of n_substeps equal parts of the window (t - window, t], from the time stamp t."""
    window = pd.Timedelta(accumulation_window)
    return pd.to_timedelta(((np.arange(n_substeps) + 0.5) / n_substeps - 1) * window.total_seconds(), unit='s')


def _positions(times, terms, lat, lon, elev, method):
    if method == 'nrel_numpy':
        return spa_positions(terms, lat, lon, elev)
    # other pvlib methods: point by point
    cols = [pvlib.solarposition.get_solarposition(times, la, lo, el, method=method)[['zenith', 'azimuth', 'elevation']]
            for la, lo, el in zip(lat, lon, elev)]
    return tuple(np.stack([c[k].values for c in cols], axis=1) for k in ['zenith', 'azimuth', 'elevation'])


def get_solar_geom(df_position,
                   start_date,
                   end_date,
                   tstep,
                   sr_epsg="4326",
                   fname='ds_solar.nc',
                   project_ouput=Path('./'),
                   method_solar='nrel_numpy',
                   accumulation_window='1h',
                   solar_constant=S0_DEFAULT,
                   n_substeps=6):
    """
    Compute the solar geometry and top-of-atmosphere irradiance for each point.
    Azimuth is defined with 0 towards South, negative towards West and positive towards East.

    Args:
        df_position (dataframe): point_name, elevation, and either longitude/latitude (or lon/lat) or x/y in sr_epsg
        start_date (str): start date  "2014-05-10"
        end_date (str): end date   "2015-05-10"
        tstep (str): time step, ex: '6h'
        sr_epsg (str): EPSG code of x, y, used when longitude/latitude are not given
        fname (str): name of netcdf file to store solar geometry
        project_ouput (str): path to project root directory
        method_solar (str): pvlib solar position method. 'nrel_numpy' (default) is vectorized over points
        accumulation_window (str): length of the window over which the radiation fluxes are accumulated, ending at
            the time stamp (ERA5: '1h' whatever the sampling time step). Angles are evaluated at the centre of this
            window, and SWtoa is the mean over the window. Use '0h' for instantaneous fluxes.
        solar_constant (float): total solar irradiance at 1 AU [W m-2]. Default 1361
        n_substeps (int): number of sub-steps used to average SWtoa over the accumulation window

    Returns:
        dataset: zenith, azimuth, elevation [rad], mu0, SWtoa [W m-2], sunset (point_name, time)
    """
    print('\n---> Computing solar geometry')
    df = df_position.copy()
    if {'longitude', 'latitude'} <= set(df.columns):
        lon, lat = df.longitude.values, df.latitude.values
    elif {'lon', 'lat'} <= set(df.columns):
        lon, lat = df.lon.values, df.lat.values
    else:
        trans = Transformer.from_crs(f'epsg:{sr_epsg}', 'epsg:4326', always_xy=True)
        lon, lat = trans.transform(df.x.values, df.y.values)
    lon, lat, elev = np.asarray(lon, float), np.asarray(lat, float), df.elevation.values.astype(float)

    stamps = pd.date_range(start_date, pd.to_datetime(end_date) + pd.to_timedelta('1D'), freq=tstep, inclusive='left')
    times = stamps.tz_localize('UTC')
    window = pd.Timedelta(accumulation_window)
    eval_offset = window / 2
    t_mid = times - eval_offset
    n_sub = int(n_substeps) if (window > pd.Timedelta(0) and n_substeps and n_substeps > 1) else 1
    if n_sub > 1:
        # centres of n_sub equal parts of the window (t - window, t]
        offsets = substep_offsets(window, n_sub)
        t_sub = pd.DatetimeIndex((times.values[:, None] + offsets.values[None, :]).ravel()).tz_localize('UTC')
    else:
        t_sub = t_mid

    terms_mid = spa_time_terms(t_mid)
    terms_sub = spa_time_terms(t_sub) if n_sub > 1 else terms_mid
    nt, npts = times.size, len(df)
    zen, azi, ele, toa = (np.empty((npts, nt)) for _ in range(4))
    block = max(1, int(2e7 // (nt * n_sub)))
    for p0 in range(0, npts, block):
        sl = slice(p0, min(p0 + block, npts))
        z, a, e = _positions(t_mid, terms_mid, lat[sl], lon[sl], elev[sl], method_solar)
        zen[sl], azi[sl], ele[sl] = z.T, a.T, e.T
        if n_sub == 1:
            mu_sub = np.clip(np.cos(np.deg2rad(z)), 0, None)
        else:
            mu_sub = np.clip(cos_zenith_geocentric(terms_sub, lat[sl], lon[sl]), 0, None)
        mu_sub = mu_sub / terms_sub['R'][:, None] ** 2
        toa[sl] = (solar_constant * mu_sub.reshape(nt, n_sub, -1).mean(axis=1)).T

    ds = xr.Dataset(
        {
            "zenith": (["point_name", "time"], np.deg2rad(zen)),
            "azimuth": (["point_name", "time"], np.deg2rad(180 - azi)),
            "elevation": (["point_name", "time"], np.deg2rad(ele)),
        },
        coords={
            "point_name": (("point_name"), df.point_name.values),
            "time": stamps,
            "reference_time": pd.Timestamp(start_date),
        },
    )
    ds['mu0'] = np.cos(ds.zenith) * (np.cos(ds.zenith) > 0)
    ds['SWtoa'] = (["point_name", "time"], toa)
    ds['sunset'] = ds.mu0 < np.cos(89 * np.pi / 180)
    ds.attrs['solar_geometry_time_offset'] = f'-{eval_offset}'
    ds.attrs['solar_geometry_note'] = ('solar angles evaluated at time - offset, the centre of the '
                                       f'{accumulation_window} radiation accumulation window ending at time')
    ds.attrs['solar_constant'] = float(solar_constant)
    ds.attrs['accumulation_window'] = str(accumulation_window)
    ds.attrs['solar_substeps'] = n_sub
    ds.attrs['solar_position_method'] = method_solar
    ds.zenith.attrs = {'units': 'rad', 'long_name': 'Solar zenith angle at the centre of the accumulation window'}
    ds.azimuth.attrs = {'units': 'rad', 'long_name': 'Solar azimuth, 0 towards south, positive towards east'}
    ds.elevation.attrs = {'units': 'rad', 'long_name': 'Solar elevation angle at the centre of the accumulation window'}
    ds.mu0.attrs = {'units': '1', 'long_name': 'Cosine of the solar zenith angle (0 when the sun is down)'}
    ds.SWtoa.attrs = {'units': 'W m**-2',
                      'long_name': f'Top-of-atmosphere irradiance on a horizontal surface, mean over the '
                                   f'{accumulation_window} accumulation window ({n_sub} sub-steps), including the '
                                   f'Earth-Sun distance, solar constant {solar_constant} W m-2'}
    ds.sunset.attrs = {'units': 'bool', 'long_name': 'Sun below 1 degree elevation at the centre of the window'}

    # store as float32, not packed: packed int16 quantizes angles and SWtoa when the file is read back
    fout = Path(project_ouput) / fname
    encoding = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                for v in ['zenith', 'azimuth', 'elevation', 'mu0', 'SWtoa']}
    ds.to_netcdf(fout, engine='h5netcdf', encoding=encoding)
    print(f'---> File {fout} saved')
    return ds
