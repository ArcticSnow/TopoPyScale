"""
Surface-effect correction of the downscaled air temperature and dew point: REDCAPP and DReaMIT.

TopoSCALE takes air temperature from pressure levels, i.e. free-atmosphere air, without the inversions, cold-air
pools and diurnal cycle that ERA5 resolves at its own surface. Both methods add back a terrain-weighted share of the
ERA5 surface signal dT = t2m - T_pl(z_grid) (toposcale_core.surface_signal):

    T = T_pl(z_point) + w dT + beta(t),     Td = Td_pl(z_point) + w dTd  (capped at T)

- REDCAPP (Cao et al. 2017, GMD 10, 2905): w = alpha [H (1 - S) + S] + beta V (1 - S), S = exp(-R / gamma),
  with H the hypsometric position and R the elevation range within a 30 km wide circle, and V = MRVBF / 8
  (Gallant and Dowling 2003). Published coefficients: Swiss Alps.
- DReaMIT (Pozsgay et al. 2026, GMD 19, 7197): w = alpha_intercept + exp(alpha_slope h) - 1, with h the hypsometric
  position within 50 km, and beta(t) = beta_amp cos(2 pi (t - t_star)) + beta_bias (t: fraction of the year).
  Published coefficients: ERA5, Yukon and Northwest Territories (GlobSim). DReaMIT defines dT against a linear lapse
  rate fitted above the inversion; here T_pl is the TopoSCALE pressure-level interpolation, as in REDCAPP.

Hypsometric position = fraction of the DEM cells within the window that are higher than the point: 0 on the
highest summit, 1 at the bottom of the lowest valley (both papers).

Config (toposcale.surface_effect), off by default:
    method: none | redcapp | dreamit
    coefficients: file written by surface_calib.save_coefficients (default: published values)
    context_dem: DEM for the terrain indices, covering the points plus the window (default: project DEM)
    dewpoint: true               # correct the humidity too
    lw_consistent: true          # longwave from the corrected temperature and humidity
    allow_partial_window: false  # accept windows that leave the context DEM (flagged in df_centroids)
    diagnostics: false           # store t_pl, dT_surf and dTd_surf (also with method: none)
"""
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import ndimage

METHODS = ('none', 'redcapp', 'dreamit')
DEFAULTS = {
    'redcapp': {'radius_km': 15., 'alpha': 0.61, 'beta': 1.56, 'gamma': 465., 'mrvbf_slope_threshold': 50.,
                'source': 'Cao et al. 2017, Swiss Alps'},
    'dreamit': {'radius_km': 50., 'alpha_intercept': 0.449, 'alpha_slope': 0.732, 'beta_amp': 0.918,
                't_star': 0.958, 'beta_bias': 1.181,
                'source': 'Pozsgay et al. 2026, ERA5, Yukon and Northwest Territories (GlobSim DReaMIT_params.csv)'},
}
INDEX_COLUMNS = ['hypso_pos', 'relief', 'mrvbf', 'window_coverage']


# ---------------------------------------------------------------------------------------------------------------------
#  weights
# ---------------------------------------------------------------------------------------------------------------------
def weight_dreamit(hypso_pos, alpha_intercept, alpha_slope, **_):
    """DReaMIT surface-coupling weight alpha(h) (Pozsgay et al. 2026, eq. 5)."""
    return alpha_intercept + np.expm1(alpha_slope * np.asarray(hypso_pos, dtype=float))


def weight_redcapp(hypso_pos, relief, mrvbf, alpha, beta, gamma, **_):
    """REDCAPP land-surface correction factor LSCF (Cao et al. 2017)."""
    s = np.exp(-np.asarray(relief, dtype=float) / gamma)
    return alpha * (np.asarray(hypso_pos) * (1 - s) + s) + beta * np.asarray(mrvbf) / 8. * (1 - s)


def weight(method, df, coef):
    """Weight of each point (rows of df with the INDEX_COLUMNS) for a method and its coefficients."""
    if method == 'dreamit':
        return weight_dreamit(df['hypso_pos'].values, **coef)
    if method == 'redcapp':
        return weight_redcapp(df['hypso_pos'].values, df['relief'].values, df['mrvbf'].values, **coef)
    raise ValueError(f'Unknown surface-effect method {method}, use one of {METHODS}')


def beta_t(times, beta_amp=0., t_star=0., beta_bias=0., **_):
    """DReaMIT seasonal offset beta(t) [K] (Pozsgay et al. 2026, eq. 6), t the fraction of the year."""
    t = pd.DatetimeIndex(times)
    y0 = pd.to_datetime(t.year.astype(str), format='%Y')
    y1 = pd.to_datetime((t.year + 1).astype(str), format='%Y')
    frac = np.asarray((t - y0) / (y1 - y0), dtype=float)
    return beta_amp * np.cos(2 * np.pi * (frac - t_star)) + beta_bias


# ---------------------------------------------------------------------------------------------------------------------
#  coefficients
# ---------------------------------------------------------------------------------------------------------------------
def load_coefficients(method, path=None):
    """Published coefficients of a method, updated with a coefficients file (surface_calib.save_coefficients)."""
    if method not in DEFAULTS:
        raise ValueError(f'Unknown surface-effect method {method}, use one of {METHODS}')
    coef = dict(DEFAULTS[method])
    if path:
        with open(path) as f:
            d = yaml.safe_load(f)
        if d.get('method', method) != method:
            raise ValueError(f'{path} holds coefficients of {d.get("method")}, not {method}')
        coef.update(d.get('coefficients', {}))
        for k in ['radius_km', 'mrvbf_slope_threshold']:
            if k in d:
                coef[k] = d[k]
        coef['source'] = str(path)
    return coef


# ---------------------------------------------------------------------------------------------------------------------
#  terrain indices
# ---------------------------------------------------------------------------------------------------------------------
def _n(x, t, p):
    """Gallant and Dowling (2003) transform of x > 0 onto [0, 1]: 1 for x << t, 0 for x >> t."""
    return 1. / (1. + (x / t) ** p)


def _disk(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return [(dy, dx) for dy, dx in zip(yy.ravel(), xx.ravel()) if dy * dy + dx * dx <= radius * radius]


def _shift(z, dy, dx):
    out = np.full_like(z, np.nan)
    ny, nx = z.shape
    out[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)] = \
        z[max(0, dy):ny - max(0, -dy), max(0, dx):nx - max(0, -dx)]
    return out


def _lowness_pctl(z, radius):
    """Fraction of the cells within a circle of radius (cells) lower than the centre cell."""
    lower = np.zeros_like(z)
    valid = np.zeros_like(z)
    for dy, dx in _disk(radius):
        zs = _shift(z, dy, dx)
        ok = np.isfinite(zs)
        lower += ok & (zs < z)
        valid += ok
    with np.errstate(invalid='ignore'):
        return np.where(np.isfinite(z), lower / valid, np.nan)


def _slope_pct(z, res):
    gy, gx = np.gradient(z, res)
    return 100. * np.hypot(gx, gy)


def _resize(a, shape):
    return ndimage.zoom(np.nan_to_num(a), (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)[:shape[0], :shape[1]]


def mrvbf(z, res, slope_threshold=16., max_steps=8, pctl_radius=6):
    """
    Multiresolution index of valley bottom flatness (Gallant and Dowling 2003, Water Resour. Res. 39, 1347).

    Steps 1 and 2 at the DEM resolution, then the DEM is smoothed and coarsened 3 times per step and the slope
    threshold halved. Steps stop when the coarse DEM is smaller than the lowness window.

    Args:
        z (2d array): elevation [m], nan for no data
        res (float): cell size [m]
        slope_threshold (float): slope threshold of the first step [%]; 16 for a 25 m DEM (Gallant and Dowling),
            REDCAPP uses 50 for a smoother index
        max_steps (int): number of steps (8: index up to about 8)
        pctl_radius (int): radius of the elevation-percentile window [cells] from step 2 (3 at step 1)

    Returns:
        mrvbf (2d array), number of steps computed
    """
    z = np.asarray(z, dtype=float)
    f1 = _n(_slope_pct(z, res), slope_threshold, 4)
    vf1 = 1 - _n(f1 * _n(_lowness_pctl(z, pctl_radius / 2), 0.4, 3), 0.3, 4)
    f2 = _n(_slope_pct(z, res), slope_threshold / 2, 4)
    cf = f1 * f2
    vf2 = 1 - _n(cf * _n(_lowness_pctl(z, pctl_radius), 0.4, 3), 0.3, 4)
    w2 = 1 - _n(vf2, 0.4, np.log10(1.5 / 0.1) / np.log10(1.5))
    m = w2 * (1 + vf2) + (1 - w2) * vf1
    zc, resc, steps = z, res, 2
    for step in range(3, max_steps + 1):
        mask = np.isfinite(zc)
        smooth = ndimage.gaussian_filter(np.where(mask, zc, 0.), 1.5)
        norm = ndimage.gaussian_filter(mask.astype(float), 1.5)
        with np.errstate(invalid='ignore'):
            zs = np.where(norm > 0.5, smooth / norm, np.nan)
        zc, resc = zs[1::3, 1::3], resc * 3
        if min(zc.shape) < 2 * pctl_radius + 1:
            break
        f = _n(_slope_pct(zc, resc), slope_threshold / 2 ** (step - 1), 4)
        low = _n(_lowness_pctl(zc, pctl_radius), 0.4, 3)
        cf = cf * _resize(f, z.shape)
        vf = 1 - _n(cf * _resize(low, z.shape), 0.3, 4)
        w = 1 - _n(vf, 0.4, np.log10((step - 0.5) / 0.1) / np.log10(1.5))
        m = w * (step - 1 + vf) + (1 - w) * m
        steps = step
    return np.where(np.isfinite(z), m, np.nan), steps


def _open_dem(path, epsg=None):
    import rioxarray  # noqa: F401
    da = xr.open_dataarray(path, engine='rasterio').squeeze(drop=True)
    if epsg and da.rio.crs is not None and da.rio.crs.to_epsg() != int(epsg):
        da = da.rio.reproject(f'EPSG:{int(epsg)}')
    nodata = da.rio.nodata
    z = da.values.astype(float)
    if nodata is not None:
        z[z == nodata] = np.nan
    return z, da.x.values, da.y.values


def terrain_indices(df, dem_file, radius_km, epsg=None, with_mrvbf=False, mrvbf_slope_threshold=50.):
    """
    Hypsometric position, elevation range and MRVBF of points.

    Args:
        df (dataframe): points with x, y (projected, m) and elevation
        dem_file (str): context DEM, covering the points plus the radius; reprojected to epsg if needed
        radius_km (float): window radius [km]
        with_mrvbf (bool): compute MRVBF (REDCAPP), on the context DEM

    Returns:
        dataframe with hypso_pos, relief [m], mrvbf, window_coverage (share of the circle inside the DEM)
    """
    z, xs, ys = _open_dem(dem_file, epsg)
    rx, ry = abs(float(xs[1] - xs[0])), abs(float(ys[1] - ys[0]))
    R = radius_km * 1e3
    out = pd.DataFrame(index=df.index, columns=INDEX_COLUMNS, dtype=float)
    for i, (x, y, elev) in zip(df.index, df[['x', 'y', 'elevation']].values):
        ix, iy = np.argmin(np.abs(xs - x)), np.argmin(np.abs(ys - y))
        ny, nx = int(np.ceil(R / ry)), int(np.ceil(R / rx))
        y0, x0 = max(iy - ny, 0), max(ix - nx, 0)
        win = z[y0:iy + ny + 1, x0:ix + nx + 1]
        d2 = (((np.arange(y0, y0 + win.shape[0]) - iy) * ry) ** 2)[:, None] + \
            (((np.arange(x0, x0 + win.shape[1]) - ix) * rx) ** 2)[None, :]
        cells = win[(d2 <= R * R) & np.isfinite(win)]
        out.loc[i, 'hypso_pos'] = np.mean(cells > elev)
        out.loc[i, 'relief'] = cells.max() - cells.min()
        out.loc[i, 'window_coverage'] = min(1., cells.size * rx * ry / (np.pi * R * R))
    if with_mrvbf:
        m, _ = mrvbf(z, 0.5 * (rx + ry), mrvbf_slope_threshold)
        ix = np.abs(xs[None, :] - df.x.values[:, None]).argmin(axis=1)
        iy = np.abs(ys[None, :] - df.y.values[:, None]).argmin(axis=1)
        out['mrvbf'] = m[iy, ix]
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  setup for the downscaling
# ---------------------------------------------------------------------------------------------------------------------
def setup(df, cfg, dem_file, epsg):
    """
    Terrain indices and weights of the points, and the arguments of toposcale_core.downscale_block.

    Args:
        df (dataframe): df_centroids (x, y, elevation); index columns and w_surf are added in place
        cfg (dict): config toposcale.surface_effect
        dem_file (str): project DEM, used when cfg has no context_dem
        epsg (int): projection of x, y

    Returns:
        dict for downscale_block(surface=...), or None when the correction and its diagnostics are off
    """
    cfg = dict(cfg or {})
    method = str(cfg.get('method') or 'none').lower()
    if method not in METHODS:
        raise ValueError(f'Unknown surface-effect method {method}, use one of {METHODS}')
    if method == 'none':
        return {'weight': None, 'diagnostics': True} if cfg.get('diagnostics') else None
    coef = load_coefficients(method, cfg.get('coefficients'))
    idx = terrain_indices(df, cfg.get('context_dem') or dem_file, coef['radius_km'], epsg,
                          with_mrvbf=method == 'redcapp', mrvbf_slope_threshold=coef.get('mrvbf_slope_threshold', 50.))
    partial = idx.window_coverage < 0.95
    if partial.any():
        msg = (f'Surface effect: the {coef["radius_km"]} km window of {partial.sum()} points leaves the context DEM '
               f'(coverage down to {idx.window_coverage.min():.0%}). Provide toposcale.surface_effect.context_dem')
        if not cfg.get('allow_partial_window'):
            raise ValueError(msg + ' or set allow_partial_window: true')
        print('WARNING: ' + msg)
    for c in INDEX_COLUMNS:
        df[c] = idx[c].values
    df['w_surf'] = weight(method, df, coef)
    print(f'---> Surface effect {method} ({coef["source"]}): w from {df.w_surf.min():.2f} to {df.w_surf.max():.2f}')
    return {'weight': df['w_surf'].values.astype(float),
            'beta_coef': {k: coef[k] for k in ['beta_amp', 't_star', 'beta_bias']} if method == 'dreamit' else None,
            'dewpoint': cfg.get('dewpoint', True) is not False,
            'lw_consistent': cfg.get('lw_consistent', True) is not False,
            'diagnostics': bool(cfg.get('diagnostics'))}
