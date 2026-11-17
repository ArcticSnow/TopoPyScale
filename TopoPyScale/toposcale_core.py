"""
TopoSCALE physics on numpy arrays, for many points at once.

All functions are pure: they take numpy arrays and return numpy arrays, with no I/O. Arrays are time-major:
(time, point) for point fields, (time, level, point) after horizontal interpolation of pressure levels.

Formulas of TopoSCALE (Fiddes and Gruber, 2014), used by topo_scale_zarr.py.
"""
import numpy as np


# Monthly precipitation adjustment coefficients [km-1] (Liston and Elder, 2006, MicroMet)
PRECIP_LAPSE_COEF = np.array([0.35, 0.35, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35])
PRECIP_LAPSE_MAX_FACTOR = 5.


def _bracket(coords, value):
    """Indices of the two coordinates bracketing value and the normalized position of value between them."""
    coords = np.asarray(coords, dtype=float)
    order = np.argsort(coords)
    c = coords[order]
    k = int(np.clip(np.searchsorted(c, value) - 1, 0, c.size - 2))
    t = float(np.clip((value - c[k]) / (c[k + 1] - c[k]), 0, 1))
    return order[k], order[k + 1], t


def horizontal_weights(row, lats, lons, Xs, Ys, method='idw'):
    """
    Weights of the ERA5 nodes surrounding a point for the horizontal interpolation.

    Args:
        row: point with attributes x, y (DEM projection) and lat, lon (WGS84)
        lats (array): latitudes of the nodes (1D)
        lons (array): longitudes of the nodes (1D)
        Xs (array): x coordinates of the nodes in the DEM projection (2D, latitude x longitude)
        Ys (array): y coordinates of the nodes in the DEM projection (2D, latitude x longitude)
        method (str): 'idw' (inverse squared distance) or 'linear' (bilinear in latitude/longitude)

    Returns:
        array: 2D weights (latitude x longitude), non-negative and summing to 1
    """
    if method == 'idw':
        dist = np.sqrt((row.x - Xs) ** 2 + (row.y - Ys) ** 2)
        if dist.min() < 1e-6:
            # point on an ERA5 node: inverse distance is singular, the node value is the answer
            weights = (dist == dist.min()).astype(float)
        else:
            weights = 1 / dist ** 2
    elif method == 'linear':
        i0, i1, ti = _bracket(lats, row.lat)
        j0, j1, tj = _bracket(lons, row.lon)
        weights = np.zeros((len(lats), len(lons)))
        weights[i0, j0] += (1 - ti) * (1 - tj)
        weights[i1, j0] += ti * (1 - tj)
        weights[i0, j1] += (1 - ti) * tj
        weights[i1, j1] += ti * tj
    else:
        raise ValueError(f'ERROR: interpolation method {method} not available. Use idw or linear')
    return weights / weights.sum()


def precip_lapse_factor(elev_diff, months, max_factor=PRECIP_LAPSE_MAX_FACTOR):
    """
    Precipitation elevation adjustment factor (1 + c dz) / (1 - c dz) of Liston and Elder (2006).

    The factor diverges for c dz -> 1 (dz ~ 2.9 km with c = 0.35 km-1) and becomes negative beyond. c dz is therefore
    bounded so that the factor stays within [1/max_factor, max_factor].

    Args:
        elev_diff (array): elevation of the point minus elevation of the ERA5 surface [m]
        months (array): month of each time step (1-12)
        max_factor (float): upper bound of the factor (> 1)

    Returns:
        array: adjustment factor
    """
    x = PRECIP_LAPSE_COEF[np.asarray(months) - 1] * np.asarray(elev_diff) * 1e-3
    x_max = (max_factor - 1) / (max_factor + 1)
    x = np.clip(x, -x_max, x_max)
    return (1 + x) / (1 - x)


G = 9.81            # gravity [m s-2]
R = 287.05          # gas constant of dry air [J K-1 kg-1]
EPS = 0.622         # ratio of molecular weights water / dry air [-]
SIGMA = 5.67e-8     # Stefan-Boltzmann constant [W m-2 K-4]
LW_X1, LW_X2 = 0.43, 5.7   # clear-sky emissivity coefficients
T_TERRAIN = 273.15  # temperature of the surrounding terrain in the LW terrain term [K]


# ---------------------------------------------------------------------------------------------------------------------
#  horizontal interpolation
# ---------------------------------------------------------------------------------------------------------------------
def neighbours(points, lats, lons, transformer, method='idw'):
    """
    Indices of the 3x3 ERA5 nodes around each point and their interpolation weights.

    Args:
        points (dataframe): with columns x, y (DEM projection), lat, lon, point_name
        lats, lons (array): ERA5 latitude and longitude coordinates (1D)
        transformer (pyproj.Transformer): WGS84 -> DEM projection
        method (str): 'idw' or 'linear', see horizontal_weights()

    Returns:
        ilat (int array, npts x 3), ilon (int array, npts x 3), weights (float array, npts x 3 x 3)
    """
    lats, lons = np.asarray(lats), np.asarray(lons)
    lon2, lat2 = np.meshgrid(lons, lats)
    X, Y = transformer.transform(lon2.ravel(), lat2.ravel())
    X, Y = np.reshape(X, lon2.shape), np.reshape(Y, lon2.shape)
    n = len(points)
    ilat, ilon = np.empty((n, 3), dtype=int), np.empty((n, 3), dtype=int)
    weights = np.empty((n, 3, 3))
    for i, row in enumerate(points.itertuples(index=False)):
        il, jl = int(np.abs(lats - row.lat).argmin()), int(np.abs(lons - row.lon).argmin())
        if not (1 <= il <= lats.size - 2 and 1 <= jl <= lons.size - 2):
            raise ValueError(f'Point {row.point_name} (lat {row.lat:.3f}, lon {row.lon:.3f}) is at the edge of the '
                             'climate data domain: the 3x3 neighbouring nodes are not all available. '
                             'Extend the extent of the climate data.')
        ilat[i] = [il - 1, il, il + 1]
        ilon[i] = [jl - 1, jl, jl + 1]
        weights[i] = horizontal_weights(row, lats[ilat[i]], lons[ilon[i]],
                                        X[np.ix_(ilat[i], ilon[i])], Y[np.ix_(ilat[i], ilon[i])], method=method)
    return ilat, ilon, weights


def weight_matrix(ilat, ilon, weights, nlat, nlon):
    """
    Interpolation weights as a matrix (node x point) on a box of nlat x nlon ERA5 nodes.

    Args:
        ilat, ilon (int arrays): npts x 3, indices of the neighbours within the box
        weights (array): npts x 3 x 3
        nlat, nlon (int): size of the box

    Returns:
        array or scipy sparse matrix: (nlat * nlon, npts). Dense for small boxes, sparse otherwise.
    """
    npts = ilat.shape[0]
    rows = (ilat[:, :, None] * nlon + ilon[:, None, :]).reshape(-1)
    cols = np.repeat(np.arange(npts), 9)
    if nlat * nlon * npts <= 5e6:
        W = np.zeros((nlat * nlon, npts))
        np.add.at(W, (rows, cols), weights.reshape(-1))
        return W
    from scipy import sparse
    return sparse.csc_matrix((weights.reshape(-1), (rows, cols)), shape=(nlat * nlon, npts))


def apply_weights(field, W):
    """
    Horizontal interpolation of field (..., nlat, nlon) with a weight matrix from weight_matrix(): (..., npts).
    """
    lead = field.shape[:-2]
    flat = field.reshape(-1, field.shape[-2] * field.shape[-1])
    out = flat @ W if isinstance(W, np.ndarray) else (W.T @ flat.T).T
    return np.asarray(out).reshape(lead + (W.shape[1],))


def interpolate_horizontal(field, ilat, ilon, weights):
    """
    Weighted sum over the 3x3 neighbours of each point.

    Args:
        field (array): (..., nlat, nlon), the last two dimensions indexed by ilat and ilon
        ilat, ilon (int arrays): npts x 3, indices in the last two dimensions of field
        weights (array): npts x 3 x 3

    Returns:
        array: (..., npts)
    """
    return apply_weights(field, weight_matrix(ilat, ilon, weights, field.shape[-2], field.shape[-1]))


# ---------------------------------------------------------------------------------------------------------------------
#  vertical interpolation
# ---------------------------------------------------------------------------------------------------------------------
def interpolate_vertical(plev, elevation, level_hpa):
    """
    Linear interpolation in height between the two pressure levels bracketing each point, and pressure from the
    hypsometric equation. When a point is below the lowest pressure level, the lowest level is used (per time step).

    Args:
        plev (dict): 'z' (geopotential height, m), 't', 'q', 'u', 'v', each (time, level, point).
            Levels sorted by increasing pressure (the last level is the lowest in altitude)
        elevation (array): point elevation [m], (point,)
        level_hpa (array): pressure of the levels [hPa], (level,)

    Returns:
        dict: 't', 'q', 'u', 'v', 'p' (Pa), each (time, point)
    """
    z = plev['z']
    nlev = z.shape[1]
    k = (z > elevation[None, None, :]).sum(axis=1)  # number of levels above the point
    if (k == 0).any():
        raise ValueError(f'ERROR: Upper pressure level {np.min(level_hpa)} hPa geopotential is lower than the '
                         f'elevation of some points (max {elevation.max():.0f} m). Download higher pressure levels.')
    top = (k - 1)[:, None, :]
    bot = np.minimum(k, nlev - 1)[:, None, :]

    def take(a, idx):
        return np.take_along_axis(a, idx, axis=1)[:, 0, :]

    z_top, z_bot = take(z, top), take(z, bot)
    d_top, d_bot = np.abs(z_top - elevation), np.abs(z_bot - elevation)
    with np.errstate(invalid='ignore', divide='ignore'):
        w_bot = np.where(k < nlev, d_top / (d_top + d_bot), 0.)
    w_top = 1. - w_bot
    out = {v: take(plev[v], bot) * w_bot + take(plev[v], top) * w_top for v in ['t', 'q', 'u', 'v']}
    p_top = np.asarray(level_hpa, dtype=float)[top[:, 0, :]] * 100.
    out['p'] = p_top * np.exp(-(elevation - z_top) / (0.5 * (take(plev['t'], top) + out['t']) * R / G))
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  thermodynamics, wind, precipitation
# ---------------------------------------------------------------------------------------------------------------------
def q_from_dewpoint(d2m, sp):
    """Specific humidity [kg kg-1] from dew point [K] and pressure [Pa] (Magnus, as meteo_util.dewT_2_q_magnus)."""
    vpsl = 610.94 * np.exp(17.625 * (d2m - 273.15) / (243.04 + (d2m - 273.15)))
    wsl = EPS * vpsl / (sp - vpsl)
    return wsl / (1 + wsl)


def vapor_pressure_from_q(q, p):
    """Vapour pressure [Pa] from specific humidity and pressure (exact inverse of q_from_dewpoint)."""
    return q * p / (EPS + (1 - EPS) * q)


def dewpoint_from_vp(vp):
    """Dew point [K] from vapour pressure [Pa] (inverse of the Magnus formula of q_from_dewpoint)."""
    lnr = np.log(np.maximum(vp, 1e-3) / 610.94)
    return 273.15 + 243.04 * lnr / (17.625 - lnr)


def surface_signal(plev, surf, level_hpa):
    """
    ERA5 surface effect at the grid surface: 2 m temperature and dew point minus their pressure-level values at the
    elevation of the ERA5 surface (inversions, cold-air pools, diurnal cycle missing from free-atmosphere air).

    Args:
        plev (dict): pressure-level fields as in interpolate_vertical, geopotential height 'z' in m
        surf (dict): 'z_surf' (m2 s-2), 't2m', 'd2m', each (time, point)

    Returns:
        dT, dTd [K], both (time, point)
    """
    grid = interpolate_vertical(plev, surf['z_surf'][0] / G, level_hpa)
    td_grid = dewpoint_from_vp(vapor_pressure_from_q(grid['q'], grid['p']))
    return surf['t2m'] - grid['t'], surf['d2m'] - td_grid


def surface_correction(t, q, p, dT, dTd, weight, beta=0., dewpoint=True):
    """
    Add the terrain-weighted surface effect to pressure-level temperature and humidity (REDCAPP, DReaMIT):
    T = T_pl + w dT + beta and Td = Td_pl + w dTd, with Td capped at T. w = 0 and beta = 0 return t and q.

    Args:
        t, q, p: pressure-level temperature [K], specific humidity [kg kg-1] and pressure [Pa] at the points
        dT, dTd: surface_signal() at the points, (time, point)
        weight (array): surface-coupling weight w, (point,)
        beta (array or float): additive offset [K], (time, 1) or scalar
        dewpoint (bool): correct the humidity too

    Returns:
        corrected t, q
    """
    t_new = t + weight * dT + beta
    if not dewpoint:
        return t_new, q
    td = dewpoint_from_vp(vapor_pressure_from_q(q, p)) + weight * dTd
    return t_new, q_from_dewpoint(np.minimum(td, t_new), p)


def mixing_ratio(q):
    """As meteo_util.mixing_ratio."""
    return 0.5 * (1 - np.sqrt(1 - 4 * q))


def vapor_pressure(p, w):
    """As meteo_util.vapor_pressure [Pa]."""
    return 0.5 * p * (-1 + np.sqrt(1 + 4 * w / EPS))


def wind_speed_direction(u, v):
    """Wind speed [m s-1] and direction the wind blows from, clockwise from north [rad, 0-2pi]."""
    return np.sqrt(u ** 2 + v ** 2), np.mod(np.arctan2(-u, -v), 2 * np.pi)


def precipitation(tp_m, elev_diff, months, tstep_h, lapse_rate_flag, max_factor=PRECIP_LAPSE_MAX_FACTOR):
    """
    Args:
        tp_m (array): interpolated ERA5 precipitation [m] over the hour ending at the time stamp, (time, point)
        elev_diff (array): point elevation - ERA5 surface elevation [m], (time, point)
        months (array): month of each time step, (time,)
        tstep_h (int): time step in hours
    Returns:
        tp [mm per time step], precip_lapse_rate [-], both (time, point)
    """
    if lapse_rate_flag:
        factor = precip_lapse_factor(elev_diff, np.broadcast_to(months[:, None], elev_diff.shape), max_factor)
    else:
        factor = np.ones_like(tp_m)
    return factor * tp_m * tstep_h * 1e3, factor


# ---------------------------------------------------------------------------------------------------------------------
#  radiation
# ---------------------------------------------------------------------------------------------------------------------
def longwave(t, vp, t2m, vp_surf, strd, svf, slope, terrain_flag):
    """
    Downwelling longwave [W m-2]: clear-sky emissivity at the point plus the cloud emissivity of ERA5.

    Args:
        t, vp: air temperature [K] and vapour pressure [Pa] at the points, (time, point)
        t2m, vp_surf, strd: ERA5 surface temperature [K], vapour pressure [Pa], and strd [J m-2 over 1 h], (time, point)
        svf, slope: sky view factor [-] and slope [rad], (point,)
    """
    cse = 0.23 + LW_X1 * (vp / t) ** (1 / LW_X2)
    cse_grid = 0.23 + LW_X1 * (vp_surf / t2m) ** (1 / LW_X2)
    cle = (strd / 3600.) / (SIGMA * t2m ** 4) - cse_grid
    lw = svf * (cse + cle) * SIGMA * t ** 4
    if terrain_flag:
        lw = lw + 0.5 * (1 + np.cos(slope)) * (1 - svf) * 0.99 * SIGMA * T_TERRAIN ** 4
    return lw, cse


def horizon_at_sun(profiles, horizon_azimuth_deg, sun_azimuth):
    """
    Horizon angle in the direction of the sun (nearest azimuth bin, as xarray .sel(method='nearest')).

    Args:
        profiles (array): horizon angles [rad], (point, azimuth)
        horizon_azimuth_deg (array): azimuth of the bins [deg], sorted ascending
        sun_azimuth (array): solar azimuth [rad], (time, point)
    """
    az = np.asarray(horizon_azimuth_deg, dtype=float)
    s = np.rad2deg(sun_azimuth)
    idx = np.clip(np.searchsorted(az, s), 1, az.size - 1)
    # ties go to the larger azimuth, as pandas get_indexer(method="nearest")
    idx = np.where(np.abs(s - az[idx - 1]) < np.abs(az[idx] - s), idx - 1, idx)
    return profiles[np.arange(s.shape[1])[None, :], idx]


def shortwave(ssrd, sp, p, solar, slope, aspect, svf, shade):
    """
    Shortwave partition and terrain corrections.

    Args:
        ssrd (array): ERA5 ssrd [J m-2 over 1 h], (time, point)
        sp (array): ERA5 surface pressure [Pa], (time, point)
        p (array): pressure at the point [Pa], (time, point)
        solar (dict): 'mu0', 'SWtoa', 'zenith', 'azimuth' (rad), 'sunset' (bool), each (time, point)
        slope, aspect, svf (array): (point,), angles in rad, aspect 0 = south
        shade (bool array): sun behind the local horizon, (time, point)

    Returns:
        dict: 'SW', 'SW_direct', 'SW_diffuse', 'cos_illumination', each (time, point)
    """
    mu0, swtoa, day = solar['mu0'], solar['SWtoa'], ~solar['sunset']
    with np.errstate(invalid='ignore', divide='ignore'):
        kt = np.where(day, ssrd / 3600. / swtoa, 0.)
        kt = np.clip(kt, 0, 1)
        kd = 0.952 - 1.041 * np.exp(-np.exp(2.3 - 4.702 * kt))
        sw_grid = np.maximum(ssrd / 3600., 0.)
        diffuse_grid = kd * sw_grid
        direct_grid = np.minimum(sw_grid - diffuse_grid, swtoa)
        # Beer's law: extinction coefficient of the ERA5 column, applied with the pressure at the point
        ka = np.where(day, G * mu0 / sp * np.log(swtoa / direct_grid), 0.)
        cos_i = mu0 * np.cos(slope) + np.sin(solar['zenith']) * np.sin(slope) * np.cos(solar['azimuth'] - aspect)
        cos_i = np.maximum(cos_i, 0.)
        direct_flat = np.where(day, swtoa * np.exp(-ka * p / (G * mu0)), 0.)
        direct = np.where(day, direct_flat * cos_i / mu0 * (1 - shade), 0.)
    diffuse = svf * diffuse_grid
    return {'SW': diffuse + direct, 'SW_direct': direct, 'SW_diffuse': diffuse, 'cos_illumination': cos_i}


# ---------------------------------------------------------------------------------------------------------------------
#  full chain for one block of time steps and points
# ---------------------------------------------------------------------------------------------------------------------
def downscale_block(plev, surf, level_hpa, months, points, solar, shade, tstep_h, lw_terrain_flag,
                    precip_lapse_rate_flag, precip_lapse_rate_max_factor=PRECIP_LAPSE_MAX_FACTOR, surface=None):
    """
    Downscale one block of time steps for a set of points.

    Args:
        plev (dict): horizontally interpolated pressure-level fields 'z' (m2 s-2), 't', 'q', 'u', 'v',
            each (time, level, point)
        surf (dict): horizontally interpolated surface fields 'z_surf' (m2 s-2), 'sp', 't2m', 'd2m', 'tp', 'ssrd',
            'strd', each (time, point)
        level_hpa (array): pressure levels [hPa]
        months (array): month of each time step
        points (dict): 'elevation', 'slope', 'aspect', 'svf', each (point,)
        solar (dict): see shortwave(), each (time, point)
        shade (bool array): (time, point)
        tstep_h (int): time step of the data in hours
        surface (dict): optional surface-effect correction (surface_effect.py): 'weight' (point,) or None for the
            diagnostics only, 'beta' (time,) or scalar, 'dewpoint' (bool), 'lw_consistent' (bool, longwave from the
            corrected temperature and humidity). None: no correction, outputs unchanged

    Returns:
        dict of (time, point) arrays: t, q, u, v, p, ws, wd, tp, precip_lapse_rate, w, vp, cse, LW,
        SW, SW_direct, SW_diffuse, cos_illumination; with surface, also t_pl, dT_surf, dTd_surf
    """
    plev = dict(plev, z=plev['z'] / G)
    elev = points['elevation']
    out = interpolate_vertical(plev, elev, level_hpa)
    t_lw, q_lw = out['t'], out['q']
    if surface is not None:
        out['t_pl'] = out['t']
        out['dT_surf'], out['dTd_surf'] = surface_signal(plev, surf, level_hpa)
        if surface.get('weight') is not None:
            beta = np.reshape(surface.get('beta', 0.), (-1, 1))
            out['t'], out['q'] = surface_correction(out['t'], out['q'], out['p'], out['dT_surf'], out['dTd_surf'],
                                                    surface['weight'], beta, surface.get('dewpoint', True))
            if surface.get('lw_consistent', True):
                t_lw, q_lw = out['t'], out['q']
    out['ws'], out['wd'] = wind_speed_direction(out['u'], out['v'])
    out['tp'], out['precip_lapse_rate'] = precipitation(surf['tp'], elev[None, :] - surf['z_surf'] / G, months,
                                                        tstep_h, precip_lapse_rate_flag, precip_lapse_rate_max_factor)
    out['w'] = mixing_ratio(out['q'])
    out['vp'] = vapor_pressure(out['p'], out['w'])
    vp_surf = vapor_pressure(surf['sp'], mixing_ratio(q_from_dewpoint(surf['d2m'], surf['sp'])))
    vp_lw = out['vp'] if q_lw is out['q'] else vapor_pressure(out['p'], mixing_ratio(q_lw))
    out['LW'], out['cse'] = longwave(t_lw, vp_lw, surf['t2m'], vp_surf, surf['strd'], points['svf'],
                                     points['slope'], lw_terrain_flag)
    out.update(shortwave(surf['ssrd'], surf['sp'], out['p'], solar, points['slope'], points['aspect'],
                         points['svf'], shade))
    return out
