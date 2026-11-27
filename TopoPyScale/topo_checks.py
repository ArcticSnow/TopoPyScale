"""
Physics-consistency diagnostics for TopoPyScale downscaled outputs.

The downscaling chain (horizontal interpolation -> vertical interpolation -> radiation partitioning and
terrain corrections -> export) can silently break basic physical constraints: humidity above saturation,
direct beam in terrain shadow, precipitation created by an unbounded elevation factor, time misalignment
between accumulated ERA5 fluxes and the solar geometry, unit mix-ups between the netcdf attributes and the values.
This module checks the outputs against such constraints and returns a tidy report.

Every check returns rows with the columns:
    check, variable, status, n_fail, n_total, frac_fail, worst, note
where status is one of 'pass', 'info', 'warning', 'error'.

Typical usage after a downscaling run:

    from TopoPyScale import topo_checks as tc
    report = tc.run_all_checks(mp.downscaled_pts,
                               df_centroids=mp.toposub.df_centroids,
                               ds_param=mp.toposub.ds_param,
                               epsg=mp.config.dem.epsg)
    print(tc.summarize(report))
    tc.raise_on_failure(report, level='error')

or simply `mp.check_physics()` from a Topoclass object.

run_all_checks() reads the outputs by blocks of time steps (all points at once, as some checks compare points), so
its memory does not grow with the length of the period. The solar geometry of each block is computed from the
coordinates of the points (or read from ds_solar when given).
"""
import numpy as np
import pandas as pd

from TopoPyScale import solar_geom as sg

# Constants consistent with toposcale_core.py / meteo_util.py / solar_geom.py
G = 9.81            # gravity [m s-2]
R_D = 287.05        # gas constant dry air [J K-1 kg-1]
EPS = 0.622         # Mw / Md [-]
SIGMA = 5.67e-8     # Stefan-Boltzmann [W m-2 K-4]
S0 = 1361.          # solar constant used in solar_geom.py [W m-2]
S0_MAX = 1370 * 1.035  # upper bound: perihelion (1/0.983**2 = 1.035), with margin for older S0 = 1370
T_TERRAIN = 273.15  # temperature of surrounding terrain used in the LW terrain term

STATUS_ORDER = {'pass': 0, 'info': 1, 'warning': 2, 'error': 3}
REPORT_COLUMNS = ['check', 'variable', 'status', 'n_fail', 'n_total', 'frac_fail', 'worst', 'note']

# Plausible physical ranges, in the units produced by the downscaling (topo_scale_zarr.py)
DEFAULT_BOUNDS = {
    't': (180., 335.),                 # K
    'p': (2.5e4, 1.1e5),               # Pa
    'q': (0., 0.04),                   # kg kg-1
    'u': (-80., 80.),                  # m s-1
    'v': (-80., 80.),                  # m s-1
    'ws': (0., 80.),                   # m s-1
    'wd': (0., 2 * np.pi + 1e-6),      # rad, direction wind blows from
    'SW': (0., S0_MAX),                # W m-2
    'SW_direct': (0., S0_MAX),         # W m-2
    'SW_diffuse': (0., S0_MAX),        # W m-2
    'LW': (40., 650.),                 # W m-2
    'precip_lapse_rate': (0., np.inf), # -
}
MAX_PRECIP_RATE = 200.  # mm hr-1, used to bound tp (mm per time step)

# Value ranges (median, 99.5th percentile) expected for a given declared unit
UNIT_SIGNATURES = {
    'p': {'Pa': (2e4, 1.1e5), 'hPa': (200., 1100.), 'kPa': (20., 110.), 'bar': (0.2, 1.1)},
    't': {'K': (180., 340.), 'degC': (-90., 60.), '°C': (-90., 60.)},
    'wd': {'rad': (0., 2 * np.pi + 1e-6), 'radian': (0., 2 * np.pi + 1e-6),
           'deg': (0., 360.), 'degree': (0., 360.), 'degrees': (0., 360.)},
    'q': {'kg kg**-1': (0., 0.04), 'kg/kg': (0., 0.04), 'g kg**-1': (0., 40.), 'g/kg': (0., 40.)},
}
for _rad in ['SW', 'SW_direct', 'SW_diffuse', 'LW']:
    UNIT_SIGNATURES[_rad] = {u: (0., 1500.) for u in ['W m**-2', 'W m-2', 'W/m2', 'W/m**2']}


class PhysicsConsistencyError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------------------------------------------------
def _row(check, variable, n_fail, n_total, severity, worst=np.nan, note='', agg='max'):
    """
    One row of the report. agg says how the worst values of several blocks of time steps combine: 'max', 'min' or
    'absmax' (largest magnitude).
    """
    n_fail, n_total = int(n_fail), int(n_total)
    return {'check': check,
            'variable': variable,
            'status': severity if n_fail > 0 else 'pass',
            'n_fail': n_fail,
            'n_total': n_total,
            'frac_fail': n_fail / n_total if n_total else np.nan,
            'worst': np.nan if worst is None else float(worst),
            'note': note,
            '_severity': severity,
            '_agg': agg}


class _Rows:
    """Rows of the checks merged over blocks of time steps: counts summed, worst values combined."""

    def __init__(self):
        self.rows = {}

    def add(self, rows):
        for r in rows:
            key = (r['check'], r['variable'])
            old = self.rows.get(key)
            if old is None:
                self.rows[key] = dict(r)
                continue
            if r['note'].startswith('skipped') or '_severity' not in r:
                continue
            old['n_fail'] += r['n_fail']
            old['n_total'] += r['n_total']
            w = [x for x in (old['worst'], r['worst']) if np.isfinite(x)]
            if w:
                agg = r.get('_agg', 'max')
                old['worst'] = (min(w) if agg == 'min' else max(w, key=abs) if agg == 'absmax' else max(w))
            old['status'] = r['_severity'] if old['n_fail'] > 0 else 'pass'
            old['frac_fail'] = old['n_fail'] / old['n_total'] if old['n_total'] else np.nan

    def __iter__(self):
        return iter(self.rows.values())


def _skip(check, variable, note):
    return {'check': check, 'variable': variable, 'status': 'info', 'n_fail': 0, 'n_total': 0,
            'frac_fail': np.nan, 'worst': np.nan, 'note': f'skipped: {note}'}


def _as_str(values):
    """Fixed-width unicode array (zarr v3 returns numpy StringDType, which cannot be cast with astype(str))."""
    values = np.asarray(values)
    return np.array([str(v) for v in values.ravel()]).reshape(values.shape)


def _prepare(ds, df_centroids=None):
    """Return ds with a string `point_name` dimension (handles the zarr `point_ind` layout)."""
    if 'point_name' not in ds.dims:
        if 'point_ind' in ds.dims and df_centroids is not None:
            names = (df_centroids.set_index('point_ind').loc[ds.point_ind.values].point_name.astype(str).values)
            ds = ds.assign_coords(point_name=('point_ind', names)).swap_dims({'point_ind': 'point_name'})
        elif 'point_name' in ds.coords:  # single point dataset
            ds = ds.expand_dims('point_name')
        else:
            raise ValueError('Dataset must have a point_name (or point_ind + df_centroids) dimension')
    return ds.assign_coords(point_name=_as_str(ds.point_name.values))


def _terrain(df_centroids, point_names):
    if df_centroids is None:
        return None
    df = df_centroids.copy()
    df['point_name'] = df.point_name.astype(str)
    return df.set_index('point_name').loc[list(point_names)]


def _align_solar(ds, ds_solar):
    if ds_solar is None:
        return None
    sol = ds_solar.assign_coords(point_name=_as_str(ds_solar.point_name.values))
    sol = sol.sel(point_name=ds.point_name.values)
    return sol.reindex(time=ds.time.values)


def _covers(ds_solar, ds):
    """ds_solar holds the points and the time steps of ds."""
    names = set(_as_str(ds_solar.point_name.values))
    return set(_as_str(ds.point_name.values)) <= names and pd.DatetimeIndex(ds.time.values).isin(
        pd.DatetimeIndex(ds_solar.time.values)).all()


def _arr(ds, var):
    return np.asarray(ds[var].transpose('point_name', 'time').values, dtype=float)


def _sol(sol, var):
    """(point, time) array of a solar variable, from a dataset (ds_solar) or a dict of arrays."""
    if isinstance(sol, dict):
        return np.asarray(sol[var], dtype=float)
    return np.asarray(sol[var].transpose('point_name', 'time').values, dtype=float)


def _prev_cur_next(ext, pad, n):
    """
    Values at the previous, current and next time steps of n steps, from an array (point, time) holding pad = (before,
    after) extra steps around them. Without an extra step the first (last) value is repeated.
    """
    pb, pa = pad
    cur = ext[:, pb:pb + n]
    prev = ext[:, pb - 1:pb - 1 + n] if pb else np.concatenate([cur[:, :1], cur[:, :-1]], axis=1)
    nxt = ext[:, pb + 1:pb + 1 + n] if pa else np.concatenate([cur[:, 1:], cur[:, -1:]], axis=1)
    return prev, cur, nxt


def _window_from_attrs(attrs):
    """Radiation accumulation window recorded in the attributes of a downscaled store (default 1 h)."""
    if 'accumulation_window' in attrs:
        return pd.Timedelta(attrs['accumulation_window'])
    try:
        return 2 * pd.Timedelta(str(attrs.get('solar_geometry_time_offset')).lstrip('-'))
    except ValueError:
        return pd.Timedelta('1h')


class _Solar:
    """Solar geometry of the points of ds for blocks of time steps: from ds_solar, or computed from coordinates."""

    def __init__(self, ds, ds_solar=None, terrain=None, epsg=None):
        # computed from the coordinates of the points when possible: as the downscaling, with the sun at the
        # sub-steps of the window. A saved ds_solar is used otherwise, if it holds the points and time steps of ds
        self.sol, self.terms, self.note = None, None, ''
        try:
            if terrain is None:
                raise ValueError('no df_centroids')
            self.lon, self.lat, self.elev = sg.point_coordinates(terrain, epsg)
        except (ValueError, AttributeError):
            if ds_solar is not None and _covers(ds_solar, ds):
                self.sol = _align_solar(ds, ds_solar)
            else:
                self.note = 'needs lon/lat (or x/y and epsg) in df_centroids, or ds_solar of the points'
            return
        attrs = ds.attrs
        self.terms = sg.window_terms(pd.DatetimeIndex(ds.time.values), _window_from_attrs(attrs),
                                     int(attrs.get('solar_substeps', 6)))
        self.method = attrs.get('solar_position_method', 'nrel_numpy')
        self.solar_constant = float(attrs.get('solar_constant', sg.S0_DEFAULT))

    @property
    def available(self):
        return self.sol is not None or self.terms is not None

    def block(self, t0, t1):
        """dict of (point, time) arrays for time steps t0:t1, or None."""
        if self.sol is not None:
            sub = self.sol.isel(time=slice(t0, t1))
            return {v: _sol(sub, v) for v in ['zenith', 'azimuth', 'elevation', 'SWtoa'] if v in sub}
        if self.terms is None:
            return None
        out = sg.solar_block(self.terms, self.lon, self.lat, self.elev, t0, t1, method=self.method,
                             solar_constant=self.solar_constant)
        res = {v: out[v].T for v in ['zenith', 'azimuth', 'elevation', 'SWtoa']}
        if self.terms['n_sub'] > 1:
            # sun at the sub-steps of the window, as the downscaling shades the direct beam (sun_geocentric)
            el, az = [], []
            for k in range(self.terms['n_sub']):
                sub = {v: np.asarray(self.terms['sub'][v])[t0:t1, k] for v in ('v', 'alpha', 'delta')}
                e, a = sg.sun_geocentric(sub, self.lat, self.lon)
                el.append(e.T)
                az.append(a.T)
            res['sub_elevation'], res['sub_azimuth'] = np.stack(el, axis=-1), np.stack(az, axis=-1)
        return res


def _tstep_seconds(ds):
    t = pd.DatetimeIndex(ds.time.values)
    if t.size < 2:
        return np.nan
    return float(np.median(np.diff(t.values).astype('timedelta64[s]').astype(float)))


def esat_water(T):
    """Saturation vapour pressure over water [Pa], same formula as meteo_util.q_2_rh (Bolton 1980)."""
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def relative_humidity(T, p, q):
    """Unclipped relative humidity w.r.t. water [0-1] (meteo_util.q_2_rh clips to 1 and hides supersaturation)."""
    e = q * p / (EPS + (1 - EPS) * q)
    return e / esat_water(T)


# ---------------------------------------------------------------------------------------------------------------------
#  individual checks on downscaled outputs
# ---------------------------------------------------------------------------------------------------------------------
def check_time_axis(ds):
    rows = []
    t = pd.DatetimeIndex(ds.time.values)
    dt = np.diff(t.values).astype('timedelta64[s]').astype(float)
    rows.append(_row('time_monotonic', 'time', (dt <= 0).sum(), dt.size, 'error',
                     worst=dt.min() if dt.size else np.nan, note='time steps must be strictly increasing', agg='min'))
    if dt.size:
        med = np.median(dt)
        rows.append(_row('time_regular', 'time', (dt != med).sum(), dt.size, 'warning',
                         worst=np.abs(dt - med).max(), note=f'irregular steps vs median {med / 3600:g} h'))
        if 'tp' in ds and 'units' in ds.tp.attrs:
            u = ds.tp.attrs['units'].replace(' ', '')
            hourly_units = u in ['mmhr**-1', 'mm/h', 'mmh-1', 'mm/hr']
            rows.append(_row('tp_units_vs_timestep', 'tp', int(hourly_units and med != 3600), 1, 'error',
                             note=f"tp declared '{ds.tp.attrs['units']}' with a {med / 3600:g} h time step"))
    return rows


def check_finite(ds, variables=None):
    rows = []
    variables = variables or [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
    for var in variables:
        x = _arr(ds, var)
        rows.append(_row('finite', var, (~np.isfinite(x)).sum(), x.size, 'error', note='NaN or inf values'))
    return rows


def check_ranges(ds, bounds=None):
    rows = []
    bounds = dict(DEFAULT_BOUNDS if bounds is None else bounds)
    if 'tp' in ds:
        bounds.setdefault('tp', (0., MAX_PRECIP_RATE * _tstep_seconds(ds) / 3600))
    for var, (lo, hi) in bounds.items():
        if var not in ds:
            continue
        x = _arr(ds, var)
        bad = np.isfinite(x) & ((x < lo) | (x > hi))
        worst = x[bad][np.argmax(np.abs(x[bad]))] if bad.any() else np.nan
        rows.append(_row('range', var, bad.sum(), np.isfinite(x).sum(), 'error', worst=worst,
                         note=f'expected in [{lo:g}, {hi:g}]', agg='absmax'))
    return rows


def check_unit_attributes(ds):
    """Compare the declared `units` attribute with the magnitude of the values."""
    rows = []
    for var, signatures in UNIT_SIGNATURES.items():
        if var not in ds:
            continue
        unit = ds[var].attrs.get('units')
        x = _arr(ds, var)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        med, p995 = np.median(x), np.percentile(x, 99.5)

        def fits(rng):
            lo, hi = rng
            return lo <= med <= hi and p995 <= hi * 1.05 + 1e-9

        candidates = [u for u, rng in signatures.items() if fits(rng)]
        if var == 'wd' and p995 <= 2 * np.pi + 1e-6 and np.unique(np.round(x, 3)).size > 50:
            candidates = [u for u in candidates if u in ('rad', 'radian')]  # values never exceed 2*pi: radians
        if unit is None:
            rows.append(_skip('units_attr', var, f'no units attribute, values look like {candidates}'))
            continue
        if unit not in signatures:
            rows.append(_row('units_attr', var, 1, 1, 'warning', worst=med,
                             note=f"declared '{unit}' is not a unit of {var}; values look like {candidates}"))
            continue
        ok = unit in candidates
        rows.append(_row('units_attr', var, int(not ok), 1, 'error', worst=med,
                         note=f"declared '{unit}' but values (median {med:.4g}) look like {candidates}"))
    return rows


def check_humidity(ds, rh_tol=0.02):
    if not all(v in ds for v in ['t', 'p', 'q']):
        return [_skip('humidity', 'q', 'needs t, p, q')]
    T, p, q = _arr(ds, 't'), _arr(ds, 'p'), _arr(ds, 'q')
    rh = relative_humidity(T, p, q)
    rows = [_row('q_non_negative', 'q', (q < 0).sum(), q.size, 'error', worst=np.nanmin(q), agg='min')]
    bad = rh > 1 + rh_tol
    rows.append(_row('supersaturation', 'q', bad.sum(), np.isfinite(rh).sum(), 'warning', worst=np.nanmax(rh),
                     note=f'RH w.r.t. water > {1 + rh_tol:g} (q_2_rh clips this silently)'))
    return rows


def check_wind(ds):
    if not all(v in ds for v in ['u', 'v', 'ws', 'wd']):
        return [_skip('wind', 'ws/wd', 'needs u, v, ws, wd')]
    u, v, ws, wd = _arr(ds, 'u'), _arr(ds, 'v'), _arr(ds, 'ws'), _arr(ds, 'wd')
    rows = []
    err_ws = np.abs(ws - np.hypot(u, v))
    rows.append(_row('ws_equals_hypot_uv', 'ws', (err_ws > 1e-3 + 1e-4 * np.abs(ws)).sum(), ws.size, 'error',
                     worst=np.nanmax(err_ws)))
    expected = np.mod(np.arctan2(-u, -v), 2 * np.pi)
    diff = np.abs(np.angle(np.exp(1j * (wd - expected))))
    mask = np.hypot(u, v) > 0.1
    rows.append(_row('wd_matches_uv', 'wd', (diff[mask] > 1e-3).sum(), mask.sum(), 'error',
                     worst=np.nanmax(diff[mask]) if mask.any() else np.nan,
                     note='wd must be the direction the wind blows from, in radians'))
    return rows


def check_pressure_hypsometric(ds, terrain, tol=0.01):
    """Pressure differences between points must follow the hypsometric equation with the downscaled temperature."""
    if terrain is None or ds.point_name.size < 2:
        return [_skip('hypsometric', 'p', 'needs df_centroids and >= 2 points')]
    z = terrain.elevation.values.astype(float)
    T, p = _arr(ds, 't'), _arr(ds, 'p')
    iref = int(np.argmin(z))
    res = np.log(p / p[iref]) + G * (z - z[iref])[:, None] / (R_D * 0.5 * (T + T[iref]))
    res = np.delete(res, iref, axis=0)
    return [_row('hypsometric', 'p', (np.abs(res) > tol).sum(), res.size, 'warning',
                 worst=np.nanmax(np.abs(res)),
                 note=f'|ln(p/p_ref) + g dz / (R Tm)| > {tol:g} relative to lowest point '
                      '(includes horizontal pressure gradients between ERA5 cells)')]


def check_lapse_rate(ds, terrain, min_range=300.):
    """Effective temperature lapse rate across points should not be super-adiabatic."""
    if terrain is None or ds.point_name.size < 3:
        return [_skip('lapse_rate', 't', 'needs df_centroids and >= 3 points')]
    z = terrain.elevation.values.astype(float)
    if np.ptp(z) < min_range:
        return [_skip('lapse_rate', 't', f'elevation range < {min_range:g} m')]
    T = _arr(ds, 't')
    zc = z - z.mean()
    slope = (zc[:, None] * (T - T.mean(axis=0))).sum(axis=0) / (zc ** 2).sum()  # K m-1 per time step
    bad = slope < -(G / 1004.) - 5e-4
    return [_row('lapse_rate', 't', bad.sum(), slope.size, 'warning', worst=np.nanmin(slope) * 1e3,
                 note='super-adiabatic dT/dz across points (worst in K/km)', agg='min')]


def _horizon_at_sun(terrain, sun_azimuth):
    cols = [c for c in terrain.columns if str(c).startswith('hori_azi_')]
    if not cols:
        return None
    az = np.array([float(str(c)[len('hori_azi_'):]) for c in cols])
    order = np.argsort(az)
    az, hor = az[order], terrain[cols].values.astype(float)[:, order]
    inc = np.median(np.diff(az)) if az.size > 1 else 360.
    idx = np.mod(np.round((np.rad2deg(sun_azimuth) - az[0]) / inc), az.size).astype(int)
    return np.take_along_axis(hor, idx, axis=1)


def check_shortwave(ds, sol=None, terrain=None, tol=1.0, pad=(0, 0)):
    """
    Closure SW = direct + diffuse, no direct beam with the sun down or behind the local horizon, no SW at night and
    the direct beam below its geometric bound.

    Args:
        sol (dataset or dict): solar geometry of the points of ds (ds_solar aligned with _align_solar, or (point,
            time) arrays), holding pad = (before, after) extra time steps around those of ds
    """
    need = ['SW', 'SW_direct', 'SW_diffuse']
    if not all(v in ds for v in need):
        return [_skip('shortwave', 'SW', f'needs {need}')]
    SW, dr, df = _arr(ds, 'SW'), _arr(ds, 'SW_direct'), _arr(ds, 'SW_diffuse')
    rows = []
    err = np.abs(SW - (dr + df))
    rows.append(_row('sw_closure', 'SW', (err > tol).sum(), SW.size, 'error', worst=np.nanmax(err),
                     note='SW != SW_direct + SW_diffuse'))
    if sol is None:
        rows.append(_skip('sw_geometry', 'SW_direct', 'needs ds_solar or the coordinates of the points'))
        return rows
    n = SW.shape[1]
    elev_ext = _sol(sol, 'elevation')
    down_ext = elev_ext <= 0
    # sun down at the previous, current and next evaluation times: down over the whole accumulation window,
    # whether the geometry is evaluated at the end or at the centre of the window
    prev_down, sun_down, next_down = _prev_cur_next(down_ext, pad, n)
    # with the sun at the sub-steps of the window (solar geometry computed here), the tests of the direct beam
    # follow the downscaling: no direct beam if the sun is down, or behind the horizon, at every sub-step
    sub_el = sol.get('sub_elevation') if isinstance(sol, dict) else None
    if sub_el is not None:
        sub_el = sub_el[:, pad[0]:pad[0] + n]
        sun_down = (sub_el <= 0).all(axis=-1)
    rows.append(_row('direct_with_sun_down', 'SW_direct', ((dr > tol) & sun_down).sum(), sun_down.sum(), 'error',
                     worst=np.nanmax(np.where(sun_down, dr, 0)), note='direct beam while sun below astronomical horizon'))
    night = sun_down & prev_down & next_down
    rows.append(_row('sw_at_night', 'SW', ((SW > tol) & night).sum(), night.sum(), 'error',
                     worst=np.nanmax(np.where(night, SW, 0)),
                     note='SW > 0 although sun is down over the whole accumulation window'))
    azi_ext = _sol(sol, 'azimuth')
    azi = azi_ext[:, pad[0]:pad[0] + n]
    if 'cos_illumination' in ds:
        cos_i = _arr(ds, 'cos_illumination')
    elif terrain is not None:
        zen = _sol(sol, 'zenith')[:, pad[0]:pad[0] + n]
        s, a = terrain.slope.values[:, None], terrain.aspect.values[:, None]
        cos_i = np.clip(np.cos(zen) * np.cos(s) + np.sin(zen) * np.sin(s) * np.cos(azi - a), 0, None)
    else:
        cos_i = None
    if cos_i is not None:
        excess = dr - S0_MAX * cos_i
        rows.append(_row('direct_beam_bound', 'SW_direct', (excess > tol).sum(), dr.size, 'error',
                         worst=np.nanmax(excess), note='SW_direct > S0 * cos(illumination angle)'))
    if terrain is not None:
        hor = _horizon_at_sun(terrain, azi_ext)
        if hor is not None and sub_el is not None:
            # behind the horizon (or below the astronomical horizon) at every sub-step of the window, with the sun up
            # at one of them at least: the downscaling scales the direct beam by the sunlit fraction of the window
            sub_az = sol['sub_azimuth'][:, pad[0]:pad[0] + n]
            hor_sub = np.stack([_horizon_at_sun(terrain, sub_az[..., k]) for k in range(sub_az.shape[-1])], axis=-1)
            shade = ((sub_el <= 0) | (hor_sub > sub_el)).all(axis=-1) & ~sun_down
        elif hor is not None:
            # behind the horizon at the previous, current and next evaluation times: an approximation of shaded over
            # the whole accumulation window, which may miss the sun passing between two peaks
            prev_b, behind, next_b = _prev_cur_next(hor > elev_ext, pad, n)
            shade = behind & prev_b & next_b & ~sun_down
        if hor is not None:
            rows.append(_row('direct_in_terrain_shadow', 'SW_direct', ((dr > tol) & shade).sum(), shade.sum(), 'error',
                             worst=np.nanmax(np.where(shade, dr, 0)),
                             note='direct beam while sun is behind the local horizon over the whole window'))
    return rows


def _timing_setup(ds, sol, terrain, max_slope_deg):
    """Flat points and longitude column for the timing check, or the reason to skip it."""
    if sol is None or terrain is None or 'SW' not in ds:
        return None, 'needs SW, the solar geometry and df_centroids'
    flat = terrain.slope.values < np.deg2rad(max_slope_deg)
    if not flat.any():
        return None, f'no point with slope < {max_slope_deg:g} deg'
    lon_col = 'lon' if 'lon' in terrain.columns else ('longitude' if 'longitude' in terrain.columns else None)
    if lon_col is None:
        return None, 'df_centroids has no lon/longitude column'
    return (flat, lon_col), ''


def _timing_offsets(ds, toa, terrain, flat, lon_col):
    """Daily centroid offsets SW - SWtoa [min] of the flat points, for the time steps of ds."""
    times = pd.DatetimeIndex(ds.time.values)
    SW = _arr(ds, 'SW')
    offsets = []
    for i in np.where(flat)[0]:
        local = times + pd.to_timedelta(terrain[lon_col].values[i] / 15., unit='h')
        df = pd.DataFrame({'sw': SW[i], 'toa': toa[i], 'day': local.floor('D'),
                           'h': (local - local.floor('D')) / pd.Timedelta('1h')})
        df['sw_h'], df['toa_h'] = df.sw * df.h, df.toa * df.h
        d = df.groupby('day')[['sw', 'toa', 'sw_h', 'toa_h']].sum()
        d = d[(d.toa > 0) & (d.sw > 0.05 * d.toa)]
        offsets.append(((d.sw_h / d.sw) - (d.toa_h / d.toa)) * 60.)
    return pd.concat(offsets) if offsets else pd.Series(dtype=float)


def _timing_row(off, tol_minutes, min_days):
    if off.size < min_days:
        return [_skip('sw_timing', 'SW', f'fewer than {min_days} usable days')]
    med = float(np.median(off))
    return [_row('sw_timing', 'SW', int(abs(med) > tol_minutes), 1, 'warning', worst=med,
                 note='median daily centroid offset SW - SWtoa in minutes (> 0: SW lags solar geometry)')]


def check_shortwave_timing(ds, sol, terrain, max_slope_deg=10., tol_minutes=15., min_days=5):
    """
    Compare the daily centroid (in local solar time) of downscaled SW with the one of the TOA irradiance computed
    from the solar geometry, for nearly flat points. ERA5 ssrd at time t is the accumulation over [t - 1h, t]: if the
    solar geometry is evaluated at t instead of t - 30 min, SW lags the sun by about +30 min, which biases kt (low in
    the morning, clipped in the afternoon), the diffuse/direct split and the illumination on slopes.
    """
    setup, why = _timing_setup(ds, sol, terrain, max_slope_deg)
    if setup is None:
        return [_skip('sw_timing', 'SW', why)]
    return _timing_row(_timing_offsets(ds, _sol(sol, 'SWtoa'), terrain, *setup), tol_minutes, min_days)


def check_longwave(ds, terrain=None, tol=0.02, min_emissivity=0.4):
    if not all(v in ds for v in ['LW', 't']):
        return [_skip('longwave', 'LW', 'needs LW, t')]
    LW, T = _arr(ds, 'LW'), _arr(ds, 't')
    svf = terrain.svf.values[:, None] if (terrain is not None and 'svf' in terrain) else 1.
    upper = SIGMA * np.maximum(T, T_TERRAIN) ** 4  # emissivities of sky and terrain <= 1
    eps_eff = LW / (SIGMA * T ** 4)
    return [_row('lw_emissivity_upper', 'LW', (LW > (1 + tol) * upper).sum(), LW.size, 'warning',
                 worst=np.nanmax(LW / upper), note='implied all-sky emissivity > 1'),
            _row('lw_emissivity_lower', 'LW', (eps_eff < min_emissivity * svf).sum(), LW.size, 'warning',
                 worst=np.nanmin(eps_eff), note=f'implied emissivity < {min_emissivity:g} * svf', agg='min')]


def check_precipitation(ds, max_factor=5.):
    if 'tp' not in ds:
        return [_skip('precipitation', 'tp', 'needs tp')]
    tp = _arr(ds, 'tp')
    rows = [_row('tp_non_negative', 'tp', (tp < -1e-6).sum(), tp.size, 'error', worst=np.nanmin(tp), agg='min')]
    if 'precip_lapse_rate' in ds:
        f = _arr(ds, 'precip_lapse_rate')
        rows.append(_row('precip_factor_positive', 'precip_lapse_rate', (~np.isfinite(f) | (f <= 0)).sum(), f.size,
                         'error', worst=np.nanmin(f), note='elevation factor must be finite and > 0', agg='min'))
        rows.append(_row('precip_factor_bounded', 'precip_lapse_rate', (f > max_factor).sum(), f.size, 'warning',
                         worst=np.nanmax(f), note=f'elevation factor > {max_factor:g}'))
        with np.errstate(divide='ignore', invalid='ignore'):
            tp0 = tp / f
        rows.append(_row('era5_tp_non_negative', 'tp', (tp0 < -1e-6).sum(), tp0.size, 'error', worst=np.nanmin(tp0),
                         note='tp / precip_lapse_rate (interpolated ERA5 precipitation) < 0', agg='min'))
    return rows


def _area_weights(ds, ds_param=None, weights=None):
    """Area weight of each point: its number of pixels in ds_param (1 for all points without ds_param)."""
    names = ds.point_name.values
    if weights is None and ds_param is not None and 'point_name' in ds_param:
        counts = pd.Series(np.asarray(ds_param.point_name.values).ravel().astype(str)).value_counts()
        weights = counts.reindex(names).fillna(0).values
        if weights.sum() == 0:
            weights = None
    if weights is None:
        weights = np.ones(names.size)
    return np.asarray(weights, dtype=float)


def _precip_sums(ds, w):
    """Area-weighted sums of the downscaled and of the interpolated ERA5 precipitation."""
    tp, f = _arr(ds, 'tp'), _arr(ds, 'precip_lapse_rate')
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nansum(w[:, None] * tp), np.nansum(w[:, None] * tp / f)


def _precip_budget_row(num, den, bounds):
    ratio = num / den if den else np.nan
    bad = not (bounds[0] <= ratio <= bounds[1])
    return [_row('domain_precip_budget', 'tp', int(bad), 1, 'warning', worst=ratio,
                 note=f'area-weighted sum(tp) / sum(ERA5 tp), expected in [{bounds[0]:g}, {bounds[1]:g}]')]


def check_domain_precip_budget(ds, ds_param=None, weights=None, bounds=(0.67, 1.5)):
    """
    Area-weighted domain precipitation relative to the horizontally interpolated ERA5 precipitation. Without precip
    lapse rate the ratio is 1. With it, the ratio tells how much water the elevation factor adds to the domain.
    """
    if 'tp' not in ds or 'precip_lapse_rate' not in ds:
        return [_skip('domain_precip_budget', 'tp', 'needs tp and precip_lapse_rate')]
    return _precip_budget_row(*_precip_sums(ds, _area_weights(ds, ds_param, weights)), bounds)


# ---------------------------------------------------------------------------------------------------------------------
#  probes of the physics functions themselves (independent of any output)
# ---------------------------------------------------------------------------------------------------------------------
def check_centroid_aspect(df_centroids, min_slope_deg=5., tol_deg=10., tol_dispersed_deg=45., min_resultant=0.7):
    """
    The centroid `aspect` used for the illumination angle must agree with the circular mean carried by the
    clustering features (aspect_cos, aspect_sin). Aspect is in [0, 2 pi) with 0 = south, so an arithmetic mean of a
    south-facing cluster (pixels near 0 and near 2 pi) gives ~pi, i.e. a north-facing slope.

    The cluster centre (aspect_cos, aspect_sin) is the mean of the pixels only for k-means; for mini-batch k-means it
    is an estimate. Its length, the mean resultant length of the aspects, measures how concentrated they are. For
    clusters whose aspects are spread (length < min_resultant), the direction of the centre and the circular mean of
    the pixels can differ by tens of degrees: they are compared with tol_dispersed_deg instead of tol_deg.
    """
    need = ['aspect', 'aspect_cos', 'aspect_sin', 'slope']
    if df_centroids is None or not all(c in df_centroids for c in need):
        return [_skip('centroid_aspect', 'aspect', f'needs df_centroids with {need}')]
    circ = np.arctan2(df_centroids.aspect_sin.values, df_centroids.aspect_cos.values)
    diff = np.rad2deg(np.abs(np.angle(np.exp(1j * (df_centroids.aspect.values - circ)))))
    resultant = np.hypot(df_centroids.aspect_sin.values, df_centroids.aspect_cos.values)
    tol = np.where(resultant < min_resultant, tol_dispersed_deg, tol_deg)
    mask = df_centroids.slope.values > np.deg2rad(min_slope_deg)
    return [_row('centroid_aspect', 'aspect', (diff[mask] > tol[mask]).sum(), mask.sum(), 'error',
                 worst=diff[mask].max() if mask.any() else np.nan,
                 note='aspect differs from atan2(aspect_sin, aspect_cos) (worst in degrees)')]


def check_dem_orientation(ds_param):
    """slope/aspect (gradient_d8 with dy < 0) assume a north-up raster."""
    if ds_param is None or 'y' not in ds_param.coords:
        return [_skip('dem_orientation', 'y', 'needs ds_param')]
    dy = float(ds_param.y.diff('y').median())
    return [_row('dem_orientation', 'y', int(dy > 0), 1, 'error', worst=dy,
                 note='y must decrease with row index (north-up raster), else aspect flips N/S')]


def check_snow_partition(method='continuous', n_draws=2000):
    """Probe meteo_util.partition_snow: closure, bounds, monotonic in T, and RH unit convention used by exporters."""
    from TopoPyScale import meteo_util as mu
    rows = []
    T = np.linspace(263.15, 283.15, 201)
    P = np.ones_like(T)
    if method.lower() == 'continuous':
        rain, snow = mu.partition_snow(P, T, method=method)
        rows.append(_row('partition_closure', method, (np.abs(rain + snow - P) > 1e-9).sum(), T.size, 'error'))
        rows.append(_row('partition_bounds', method, ((snow < -1e-9) | (snow > 1 + 1e-9)).sum(), T.size, 'error'))
        inc = np.diff(snow) > 1e-9
        rows.append(_row('partition_monotonic', method, inc.sum(), inc.size, 'error',
                         worst=np.max(np.diff(snow)), note='snow fraction must not increase with temperature'))
        return rows
    # Jennings et al. (2018): stochastic, probe with many draws, RH in % as passed by the exporters (100 * q_2_rh)
    p = np.full(n_draws, 80000.)
    for name, t_air, expect_snow in [('partition_warm_humid', 277.15, False), ('partition_cold_humid', 269.15, True)]:
        t = np.full(n_draws, t_air)
        e = 0.95 * esat_water(t)
        q = EPS * e / (p - (1 - EPS) * e)
        rh = 100 * mu.q_2_rh(t.copy(), p, q)
        rain, snow = mu.partition_snow(np.ones_like(t), t, rh, p, method=method)
        rows.append(_row('partition_closure', method, (np.abs(rain + snow - 1) > 1e-9).sum(), t.size, 'error'))
        frac = snow.mean()
        rows.append(_row(name, method, int((frac > 0.5) != expect_snow), 1, 'error', worst=frac,
                         note=f'snow fraction at {t_air - 273.15:+.0f} degC, RH 95 % (RH passed in %)'))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
#  orchestration
# ---------------------------------------------------------------------------------------------------------------------
def _time_blocks(ds, n_vars, max_block_mb):
    """Blocks of whole days of time steps holding all points within max_block_mb, aligned on the store chunks."""
    nt, npts = ds.sizes['time'], ds.sizes['point_name']
    per_day = max(1, int(round(86400 / _tstep_seconds(ds)))) if nt > 1 else 1
    # float64 copies of the variables and a few intermediate arrays of the checks
    n = max(per_day, int(max_block_mb * 1e6 // (npts * (n_vars + 12) * 8)))
    n = max(per_day, n // per_day * per_day)
    chunks = [ds[v].chunks for v in ds.data_vars if ds[v].chunks is not None]
    if chunks:
        c = chunks[0][ds[next(v for v in ds.data_vars if ds[v].chunks is not None)].dims.index('time')][0]
        if c <= n:
            n = n // c * c
    return [(t0, min(t0 + n, nt)) for t0 in range(0, nt, n)]


def run_all_checks(ds_down, ds_solar=None, df_centroids=None, ds_param=None, snow_partition_method=None,
                   points=None, bounds=None, epsg=None, max_block_mb=1000):
    """
    Run all physics-consistency checks on downscaled outputs, by blocks of time steps.

    Args:
        ds_down (dataset): downscaled points (e.g. Topoclass.downscaled_pts), dims (point_name, time)
        ds_solar (dataset): optional precomputed solar geometry (solar_geom.get_solar_geom). Default: computed for
            each block from the coordinates of the points in df_centroids and the attributes of ds_down
        df_centroids (dataframe): Topoclass.toposub.df_centroids (elevation, slope, aspect, svf, lon/lat or x/y,
            hori_azi_*)
        ds_param (dataset): Topoclass.toposub.ds_param, used for area weights of the precipitation budget
        snow_partition_method (str or list): partition method(s) to probe, e.g. 'continuous'
        points (list): optional subset of point_name to check
        bounds (dict): optional override of DEFAULT_BOUNDS
        epsg (int): EPSG code of x, y in df_centroids, used when it has no lon/lat columns
        max_block_mb (float): memory of the block of time steps read at once (all points)

    Returns:
        dataframe: report, worst status first
    """
    ds = _prepare(ds_down, df_centroids)
    if points is not None:
        ds = ds.sel(point_name=[str(p) for p in points])
    terrain = _terrain(df_centroids, ds.point_name.values)
    solar = _Solar(ds, ds_solar, terrain, epsg)
    bounds = dict(DEFAULT_BOUNDS if bounds is None else bounds)
    if 'tp' in ds:
        bounds.setdefault('tp', (0., MAX_PRECIP_RATE * _tstep_seconds(ds) / 3600))
    numeric = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number) and
               {'point_name', 'time'} <= set(ds[v].dims)]
    timing, why = _timing_setup(ds, {} if solar.available else None, terrain, 10.)
    budget = 'tp' in ds and 'precip_lapse_rate' in ds
    w = _area_weights(ds, ds_param) if budget else None

    rows = _Rows()
    rows.add(check_dem_orientation(ds_param))
    rows.add(check_centroid_aspect(df_centroids))
    rows.add(check_time_axis(ds))
    offsets, sums = [], [0., 0.]
    nt = ds.sizes['time']
    for i, (t0, t1) in enumerate(_time_blocks(ds, len(numeric), max_block_mb)):
        blk = ds[numeric].isel(time=slice(t0, t1)).load()
        a, b = max(0, t0 - 1), min(nt, t1 + 1)
        sol = solar.block(a, b)
        if i == 0:
            rows.add(check_unit_attributes(blk))
        rows.add(check_finite(blk))
        rows.add(check_ranges(blk, bounds))
        rows.add(check_humidity(blk))
        rows.add(check_wind(blk))
        rows.add(check_pressure_hypsometric(blk, terrain))
        rows.add(check_lapse_rate(blk, terrain))
        rows.add(check_shortwave(blk, sol, terrain, pad=(t0 - a, b - t1)))
        if timing is not None:
            offsets.append(_timing_offsets(blk, sol['SWtoa'][:, t0 - a:t0 - a + t1 - t0], terrain, *timing))
        rows.add(check_longwave(blk, terrain))
        rows.add(check_precipitation(blk))
        if budget:
            num, den = _precip_sums(blk, w)
            sums = [sums[0] + num, sums[1] + den]
    if not solar.available:
        rows.add([_skip('sw_geometry', 'SW_direct', solar.note)])
    if timing is None:
        rows.add([_skip('sw_timing', 'SW', why)])
    else:
        rows.add(_timing_row(pd.concat(offsets) if offsets else pd.Series(dtype=float), 15., 5))
    rows.add(_precip_budget_row(*sums, (0.67, 1.5)) if budget else
             [_skip('domain_precip_budget', 'tp', 'needs tp and precip_lapse_rate')])
    if snow_partition_method is not None:
        methods = [snow_partition_method] if isinstance(snow_partition_method, str) else snow_partition_method
        for m in methods:
            rows.add(check_snow_partition(m))

    report = pd.DataFrame(list(rows), columns=REPORT_COLUMNS)
    report['_order'] = report.status.map(STATUS_ORDER)
    return report.sort_values(['_order', 'check'], ascending=[False, True]).drop(columns='_order').reset_index(drop=True)


def summarize(report, min_status='info'):
    """Human-readable summary of the checks at or above `min_status`."""
    sub = report[report.status.map(STATUS_ORDER) >= STATUS_ORDER[min_status]]
    counts = report.status.value_counts().reindex(['error', 'warning', 'info', 'pass']).fillna(0).astype(int)
    head = ' | '.join(f'{k}: {v}' for k, v in counts.items())
    if sub.empty:
        return f'Physics checks: {head}'
    with pd.option_context('display.max_colwidth', 90, 'display.width', 250):
        return f'Physics checks: {head}\n' + sub.to_string(index=False)


def raise_on_failure(report, level='error'):
    bad = report[report.status.map(STATUS_ORDER) >= STATUS_ORDER[level]]
    if not bad.empty:
        raise PhysicsConsistencyError(summarize(report, min_status=level))
