"""
Calibration of the surface-effect correction (surface_effect.py) from station records.

1. Run TopoPyScale on the stations as a points project with toposcale.surface_effect: {method: none,
   diagnostics: true}: the store then holds t_pl, dT_surf and dTd_surf at each station.
2. Read the observations (read_meteoswiss, read_csv) and pair them with the store (pair).
3. calibrate(): per-station weights (diagnostic), coefficients of the method by pooled least squares,
   cross-validation (leave one station out, or contiguous time blocks with fewer than 3 stations) and
   moving-block bootstrap intervals.
4. save_coefficients(): YAML file for toposcale.surface_effect.coefficients.

Model fitted, per station s and hour t:  T_obs - T_pl = w(theta; indices_s) dT + beta(t) [+ c_s]
"""
import numpy as np
import pandas as pd
import yaml

from TopoPyScale import surface_effect as se
from TopoPyScale import toposcale_core as core

DREAMIT_SLOPES = np.round(np.arange(-4., 4.0001, 0.01), 3)
REDCAPP_GAMMAS = np.logspace(1, 4, 301)


# ---------------------------------------------------------------------------------------------------------------------
#  observations
# ---------------------------------------------------------------------------------------------------------------------
def dewpoint_from_rh(t, rh):
    """Dew point [K] from temperature [K] and relative humidity [%] (Magnus, as toposcale_core)."""
    tc = np.asarray(t) - 273.15
    es = 610.94 * np.exp(17.625 * tc / (243.04 + tc))
    return core.dewpoint_from_vp(np.clip(rh, 1, 100) / 100. * es)


def read_meteoswiss(path):
    """
    MeteoSwiss open-data hourly file (ogd-smn_<sta>_h_*.csv): t [K] and td [K], indexed by the END of the hour (UTC).
    """
    df = pd.read_csv(path, sep=';', usecols=['reference_timestamp', 'tre200h0', 'ure200h0'])
    df.index = pd.to_datetime(df.pop('reference_timestamp'), format='%d.%m.%Y %H:%M')
    t = pd.to_numeric(df.tre200h0, errors='coerce') + 273.15
    return pd.DataFrame({'t': t, 'td': dewpoint_from_rh(t, pd.to_numeric(df.ure200h0, errors='coerce'))})


def read_csv(path, time, t, rh=None, td=None, celsius=True, **kwargs):
    """Generic station file: column names of time, temperature and relative humidity [%] or dew point."""
    df = pd.read_csv(path, **kwargs)
    out = pd.DataFrame(index=pd.to_datetime(df[time]))
    out['t'] = df[t].values + (273.15 if celsius else 0.)
    if td is not None:
        out['td'] = df[td].values + (273.15 if celsius else 0.)
    elif rh is not None:
        out['td'] = dewpoint_from_rh(out['t'].values, df[rh].values)
    return out


def pair(ds_down, obs, hour_ending=True):
    """
    Pair station observations with a downscaled store run with surface-effect diagnostics.

    Args:
        ds_down (dataset): downscaled points (point_name, time) with t_pl, dT_surf, dTd_surf, q, p
        obs (dict): point_name -> dataframe of t and td [K] (read_meteoswiss, read_csv)
        hour_ending (bool): observations are means over the hour ending at the time stamp; the model is then
            averaged over the two time steps bracketing that hour

    Returns:
        dataframe with point_name, time, y (t obs - t_pl), dT, yd (td obs - td_pl), dTd
    """
    rows = []
    for name, o in obs.items():
        m = ds_down.sel(point_name=name)[['t_pl', 'dT_surf', 'dTd_surf', 'q', 'p']].to_dataframe()
        m['td_pl'] = core.dewpoint_from_vp(core.vapor_pressure_from_q(m.q.values, m.p.values))
        m = m[['t_pl', 'dT_surf', 'td_pl', 'dTd_surf']]
        if hour_ending:
            m = m.rolling(2).mean()
        d = m.join(o, how='inner').dropna(subset=['t', 't_pl', 'dT_surf'])
        rows.append(pd.DataFrame({'point_name': name, 'time': d.index, 't_pl': d.t_pl.values,
                                  'y': (d.t - d.t_pl).values, 'dT': d.dT_surf.values,
                                  'yd': (d.td - d.td_pl).values if 'td' in d else np.nan,
                                  'dTd': d.dTd_surf.values}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------
#  fits
# ---------------------------------------------------------------------------------------------------------------------
def fit_stations(data, min_abs_dT=1.):
    """Per-station weight: slope of T_obs - T_pl on dT (with intercept), over hours with |dT| > min_abs_dT."""
    out = {}
    for name, d in data.groupby('point_name'):
        d = d[np.abs(d.dT) > min_abs_dT]
        A = np.c_[d.dT, np.ones(len(d))]
        (w, c), *_ = np.linalg.lstsq(A, d.y, rcond=None)
        out[name] = {'w': w, 'intercept': c, 'n': len(d), 'r': np.corrcoef(d.y, d.dT)[0, 1]}
    return pd.DataFrame(out).T


def _beta_columns(time, fit_beta):
    if fit_beta in (None, 'none'):
        return np.empty((len(time), 0))
    cols = [np.ones(len(time))]
    if fit_beta == 'seasonal':
        frac = (se.beta_t(time, beta_amp=1., t_star=0.), se.beta_t(time, beta_amp=1., t_star=0.25))  # cos, sin
        cols += list(frac)
    return np.column_stack(cols)


def _design(method, data, idx, fit_beta, station_offsets):
    """Columns of the linear part and the function giving, for a nonlinear parameter, the fixed term and w columns."""
    ix = idx.loc[data.point_name.values]
    extra = [_beta_columns(pd.DatetimeIndex(data.time), fit_beta)]
    if station_offsets:
        extra.append(pd.get_dummies(data.point_name).values.astype(float))
    extra = np.column_stack(extra) if extra else np.empty((len(data), 0))
    dT = data.dT.values
    h = ix.hypso_pos.values
    if method == 'dreamit':
        def terms(k):  # w = a + expm1(k h): fixed part expm1(k h) dT, linear column dT for a
            return np.expm1(k * h) * dT, np.c_[dT]
        return DREAMIT_SLOPES, terms, extra
    s_of = lambda g: np.exp(-ix.relief.values / g)  # noqa: E731

    def terms(g):  # w = alpha (h (1-s) + s) + beta V (1-s)
        s = s_of(g)
        return np.zeros_like(dT), np.c_[(h * (1 - s) + s) * dT, ix.mrvbf.values / 8. * (1 - s) * dT]
    return REDCAPP_GAMMAS, terms, extra


def fit(method, data, idx, fit_beta='none', station_offsets=False):
    """
    Coefficients of a method by pooled least squares over all stations and hours: profile over the nonlinear
    parameter (DReaMIT alpha_slope, REDCAPP gamma), linear least squares for the others.

    Args:
        method (str): 'dreamit' or 'redcapp'
        data (dataframe): pair() output
        idx (dataframe): terrain indices of the stations (surface_effect.terrain_indices), indexed by point_name
        fit_beta (str): 'none', 'bias' (constant offset) or 'seasonal' (DReaMIT beta(t)); else beta = 0
        station_offsets (bool): one free offset per station (fits w on the variation of dT only)

    Returns:
        dict of coefficients (and 'offsets' per station if station_offsets)
    """
    grid, terms, extra = _design(method, data, idx, fit_beta, station_offsets)
    y = data.y.values
    best = (np.inf, None, None)
    for g in grid:
        fixed, cols = terms(g)
        A = np.c_[cols, extra]
        coef, *_ = np.linalg.lstsq(A, y - fixed, rcond=None)
        sse = np.sum((y - fixed - A @ coef) ** 2)
        if sse < best[0]:
            best = (sse, g, coef)
    _, g, coef = best
    if method == 'dreamit':
        out = {'alpha_slope': float(g), 'alpha_intercept': float(coef[0])}
        rest = coef[1:]
    else:
        out = {'gamma': float(g), 'alpha': float(coef[0]), 'beta': float(coef[1])}
        rest = coef[2:]
    out.update({'beta_amp': 0., 't_star': 0., 'beta_bias': 0.})
    nb = _beta_columns(pd.DatetimeIndex(data.time[:1]), fit_beta).shape[1]
    if nb:
        out['beta_bias'] = float(rest[0])
    if nb == 3:
        a, b = rest[1], rest[2]
        out['beta_amp'] = float(np.hypot(a, b))
        out['t_star'] = float(np.mod(np.arctan2(b, a) / (2 * np.pi), 1.))
    if station_offsets:
        names = sorted(data.point_name.unique())
        out['offsets'] = dict(zip(names, map(float, rest[nb:])))
    return out


def predict(method, coef, data, idx):
    """Corrected temperature minus T_pl for the rows of data (offsets per station if present in coef)."""
    ix = idx.loc[data.point_name.values].reset_index(drop=True)
    w = se.weight(method, ix, coef)
    out = w * data.dT.values + se.beta_t(pd.DatetimeIndex(data.time), **coef)
    if 'offsets' in coef:
        out = out + data.point_name.map(coef['offsets']).fillna(0.).values
    return out


def scores(sim, obs):
    """Bias, RMSE and correlation."""
    sim, obs = np.asarray(sim), np.asarray(obs)
    return {'bias': np.mean(sim - obs), 'rmse': np.sqrt(np.mean((sim - obs) ** 2)), 'r': np.corrcoef(sim, obs)[0, 1]}


def cross_validate(method, data, idx, by='station', n_blocks=2, **kw):
    """
    Out-of-sample predictions of T_obs - T_pl: leave one station out (by='station'), or leave one of n_blocks
    contiguous time blocks out (by='time').

    Returns:
        series of predictions aligned with data
    """
    pred = pd.Series(np.nan, index=data.index)
    if by == 'station':
        folds = [data.point_name == s for s in data.point_name.unique()]
    else:
        t = pd.DatetimeIndex(data.time)
        edges = pd.date_range(t.min(), t.max() + pd.Timedelta('1h'), periods=n_blocks + 1)
        folds = [(t >= a) & (t < b) for a, b in zip(edges[:-1], edges[1:])]
    for test in folds:
        test = np.asarray(test)
        coef = fit(method, data[~test], idx, **kw)
        coef.pop('offsets', None)
        pred[test] = predict(method, coef, data[test], idx)
    return pred


def bootstrap(method, data, idx, n=200, block='5D', seed=0, **kw):
    """Moving-block bootstrap over time (blocks shared by all stations): 2.5 and 97.5 % percentiles."""
    rng = np.random.default_rng(seed)
    t = pd.DatetimeIndex(data.time)
    blocks = t.floor(block)
    keys = blocks.unique()
    by_block = {k: np.flatnonzero(blocks == k) for k in keys}
    samples = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([by_block[k] for k in pick])
        c = fit(method, data.iloc[rows].reset_index(drop=True), idx, **kw)
        c.pop('offsets', None)
        samples.append(c)
    df = pd.DataFrame(samples)
    return {k: [float(df[k].quantile(0.025)), float(df[k].quantile(0.975))] for k in df.columns}


def calibrate(method, data, indices, fit_beta='none', station_offsets=False, n_boot=200, cv_by=None):
    """
    Full calibration: per-station weights, pooled fit, cross-validation and bootstrap, for each window radius.

    Args:
        method (str): 'dreamit' or 'redcapp'
        data (dataframe): pair() output
        indices (dict): radius_km -> terrain indices of the stations (indexed by point_name); the radius with the
            lowest cross-validated RMSE is kept
        cv_by (str): 'station' or 'time'; default 'station' with 3 stations or more, else 'time'

    Returns:
        dict: method, radius_km, coefficients, intervals_95, cv (scores per station, out of sample),
        fit (scores in sample), stations (per-station weights), cv_by
    """
    cv_by = cv_by or ('station' if data.point_name.nunique() >= 3 else 'time')
    kw = {'fit_beta': fit_beta, 'station_offsets': station_offsets}
    best = None
    for radius, idx in indices.items():
        pred = cross_validate(method, data, idx, by=cv_by, **kw)
        rmse = np.sqrt(np.mean((pred - data.y) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, radius, idx, pred)
    _, radius, idx, pred = best
    coef = fit(method, data, idx, **kw)
    insample = predict(method, coef, data, idx)
    per = lambda p: pd.DataFrame({s: scores(p[g.index] + data.t_pl[g.index], g.y + g.t_pl)  # noqa: E731
                                  for s, g in data.groupby('point_name')}).T
    return {'method': method, 'radius_km': float(radius), 'coefficients': coef,
            'intervals_95': bootstrap(method, data, idx, n=n_boot, **kw) if n_boot else {},
            'cv': per(pd.Series(pred, index=data.index)), 'fit': per(pd.Series(insample, index=data.index)),
            'stations': fit_stations(data), 'cv_by': cv_by,
            'period': [str(pd.Timestamp(data.time.min())), str(pd.Timestamp(data.time.max()))]}


def save_coefficients(result, path):
    """Write a calibrate() result as a coefficients file for toposcale.surface_effect.coefficients."""
    coef = {k: v for k, v in result['coefficients'].items() if k != 'offsets'}
    d = {'method': result['method'], 'radius_km': result['radius_km'],
         'coefficients': {k: round(float(v), 4) for k, v in coef.items()},
         'intervals_95': {k: [round(float(a), 4), round(float(b), 4)] for k, (a, b) in result['intervals_95'].items()
                          if k in coef},
         'fit': {'stations': list(result['cv'].index), 'period': result['period'], 'cv_by': result['cv_by'],
                 'cv_rmse_K': {k: round(float(v), 3) for k, v in result['cv'].rmse.items()},
                 'cv_bias_K': {k: round(float(v), 3) for k, v in result['cv'].bias.items()}}}
    if result['method'] == 'redcapp':
        d['mrvbf_slope_threshold'] = se.DEFAULTS['redcapp']['mrvbf_slope_threshold']
    with open(path, 'w') as f:
        yaml.safe_dump(d, f, sort_keys=False)
    return path
