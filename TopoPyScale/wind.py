"""
Wind downscaling options: blending with the ERA5 10 m wind, and the terrain modification of MicroMet.

TopoSCALE takes the wind of the pressure levels at the elevation of the point: free-atmosphere wind, without the
drag of the surface. Two options change it, alone or together (config toposcale.wind_downscaling):

- surface_blend: u = u_pl + w (u10 - u_pl), and the same for v, with u10, v10 the ERA5 10 m wind interpolated to
  the point. w is a constant in [0, 1], or 'surface_effect': the terrain weight of the surface-effect correction of
  temperature (toposcale.surface_effect, REDCAPP or DReaMIT), clipped to [0, 1]: points coupled to the ERA5 surface
  (valleys) get its wind, exposed points (ridges) keep the free-atmosphere wind. This reuse of the temperature
  weight is a TopoPyScale assumption, not a published method.
- terrain: liston_elder, the wind weighting and diverting factors of MicroMet (Liston and Elder, 2006, J.
  Hydrometeorol. 7, 217-234), applied after the blending:

      Ws = W (1 + gs Os + gc Oc),       direction + Od,     Od = -0.5 Os sin(2 (xi - theta))

  with theta the direction the wind blows from, xi the azimuth the slope faces (both clockwise from north),
  Os = beta cos(theta - xi) the slope in the direction of the wind (beta: slope), and Oc the curvature of the terrain
  over the length scale eta, both scaled to [-0.5, 0.5] over the DEM: Os by twice the steepest slope of the DEM, Oc
  by twice its largest absolute curvature. gs = 0.58, gc = 0.42 and eta = 500 m are the values of MicroMet and
  SnowModel.

Config:
    toposcale:
      wind_downscaling:
        surface_blend: 0            # weight of the ERA5 10 m wind: 0 (off), a value in [0, 1], or surface_effect
        terrain: none               # none or liston_elder
        slope_weight: 0.58          # gs
        curvature_weight: 0.42      # gc
        curvature_length: 500       # eta [m]
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

TERRAIN_METHODS = ('none', 'liston_elder')
DEFAULTS = {'surface_blend': 0., 'terrain': 'none', 'slope_weight': 0.58, 'curvature_weight': 0.42,
            'curvature_length': 500.}
# ERA5 10 m wind, fetched with the surface fields when surface_blend is on
U10, V10 = '10m_u_component_of_wind', '10m_v_component_of_wind'


def config(cfg):
    """Options of toposcale.wind_downscaling with their defaults; checks the values."""
    cfg = {**DEFAULTS, **{k: v for k, v in dict(cfg or {}).items() if v is not None}}
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise ValueError(f'Unknown wind_downscaling options {unknown}: use {sorted(DEFAULTS)}')
    blend = cfg['surface_blend']
    if isinstance(blend, str):
        if blend.lower() not in ('surface_effect', 'none', 'off'):
            raise ValueError(f"wind_downscaling.surface_blend {blend} not supported: a weight in [0, 1] or "
                             f"surface_effect")
        cfg['surface_blend'] = 0. if blend.lower() in ('none', 'off') else 'surface_effect'
    elif not 0 <= float(blend) <= 1:
        raise ValueError(f'wind_downscaling.surface_blend {blend} must be in [0, 1]')
    else:
        cfg['surface_blend'] = float(blend)
    cfg['terrain'] = str(cfg['terrain'] or 'none').lower()
    if cfg['terrain'] not in TERRAIN_METHODS:
        raise ValueError(f'wind_downscaling.terrain {cfg["terrain"]} not supported, use {" or ".join(TERRAIN_METHODS)}')
    return cfg


def needs_surface_wind(cfg):
    blend = config(cfg)['surface_blend']
    return blend == 'surface_effect' or blend > 0


def curvature(z, dx, eta):
    """
    Terrain curvature of MicroMet on a grid: the mean over the four directions (S-N, W-E, SW-NE, NW-SE) of the
    elevation of the cell minus the mean of its two neighbours at distance eta, divided by 2 eta (2 sqrt(2) eta on
    the diagonals). Neighbours beyond the edge take the value of the edge.

    Args:
        z (array): elevation [m], (y, x), NaN for no data
        dx (float): cell size [m]
        eta (float): length scale [m], about half the wavelength of the topographic features

    Returns:
        array: curvature [m-1 times m / m], (y, x)
    """
    n = max(1, int(round(eta / abs(dx))))
    zf = np.where(np.isfinite(z), z, np.nanmean(z))
    p = np.pad(zf, n, mode='edge')
    ny, nx = zf.shape

    def at(dy, dxc):
        return p[n + dy:n + dy + ny, n + dxc:n + dxc + nx]

    d = n * abs(dx)
    c = ((zf - 0.5 * (at(-n, 0) + at(n, 0))) / (2 * d)
         + (zf - 0.5 * (at(0, -n) + at(0, n))) / (2 * d)
         + (zf - 0.5 * (at(n, -n) + at(-n, n))) / (2 * np.sqrt(2) * d)
         + (zf - 0.5 * (at(-n, -n) + at(n, n))) / (2 * np.sqrt(2) * d)) / 4
    return np.where(np.isfinite(z), c, np.nan)


def setup(df, cfg, ds_param, per_cluster, surface=None):
    """
    Terrain parameters of the points for the wind options, and the argument wind of toposcale_core.downscale_block.

    Args:
        df (dataframe): df_centroids (point_name, x, y, slope, aspect); the column wind_curvature is added in place
            for liston_elder
        cfg (dict): config toposcale.wind_downscaling
        ds_param (dataset): DEM parameters (elevation, slope on the DEM grid, and point_name for clusters)
        per_cluster (bool): curvature of a point = mean over the cells of its cluster (toposub), else the cell of
            the point
        surface (dict): the surface-effect setup (surface_effect.setup), for surface_blend: surface_effect

    Returns:
        dict for downscale_block(wind=...), or None when both options are off
    """
    cfg = config(cfg)
    blend, terrain = cfg['surface_blend'], cfg['terrain']
    if not (blend == 'surface_effect' or blend > 0) and terrain == 'none':
        return None
    out = {'terrain': terrain}
    if blend == 'surface_effect':
        if surface is None or surface.get('weight') is None:
            raise ValueError('wind_downscaling.surface_blend: surface_effect needs toposcale.surface_effect with '
                             'method redcapp or dreamit')
        out['blend'] = np.clip(np.asarray(surface['weight'], dtype=float), 0., 1.)
    elif blend > 0:
        out['blend'] = np.full(len(df), float(blend))
    if terrain == 'liston_elder':
        if ds_param is None or 'elevation' not in ds_param or 'slope' not in ds_param:
            raise ValueError('wind_downscaling.terrain: liston_elder needs the DEM parameters (compute_dem_param)')
        dx = float(np.abs(np.median(np.diff(ds_param.x.values))))
        z = np.asarray(ds_param.elevation.transpose('y', 'x').values, dtype=float)
        c = curvature(z, dx, float(cfg['curvature_length']))
        c_scale = 2 * np.nanmax(np.abs(c))
        slope_scale = 2 * float(np.nanmax(ds_param.slope.values))
        c = c / c_scale if c_scale > 0 else np.zeros_like(c)
        if per_cluster and 'point_name' in ds_param:
            labels = np.asarray(ds_param.point_name.transpose('y', 'x').values).ravel().astype(str)
            means = pd.Series(c.ravel()).groupby(labels).mean()
            oc = means.reindex(df.point_name.astype(str)).values
        else:
            iy = np.abs(ds_param.y.values[:, None] - df.y.values[None, :]).argmin(axis=0)
            ix = np.abs(ds_param.x.values[:, None] - df.x.values[None, :]).argmin(axis=0)
            oc = c[iy, ix]
        if not np.isfinite(oc).all():
            raise ValueError('wind_downscaling: no DEM curvature at some points')
        df['wind_curvature'] = oc
        out.update(curvature=oc.astype(float), slope_scale=slope_scale, slope_weight=float(cfg['slope_weight']),
                   curvature_weight=float(cfg['curvature_weight']))
        logger.info(f'---> Wind: MicroMet terrain factors, scaled curvature from {oc.min():.2f} to {oc.max():.2f} '
                    f'(eta {cfg["curvature_length"]:g} m), steepest slope of the DEM '
                    f'{np.rad2deg(slope_scale / 2):.0f} deg')
    if 'blend' in out:
        logger.info(f'---> Wind: blended with the ERA5 10 m wind, weight from {out["blend"].min():.2f} to '
                    f'{out["blend"].max():.2f}')
    return out
