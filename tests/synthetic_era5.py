"""
Synthetic, physically consistent ERA5-like forcing on a 3x3 grid, used to run the real TopoPyScale downscaling
functions (topo_scale.py and topo_scale_zarr.py) on cases with a known answer.

Atmosphere: hydrostatic, constant lapse rate gamma, sea-level temperature with a diurnal cycle, constant RH.
Fields are horizontally homogeneous unless `tp_weights` is given.
ERA5 conventions reproduced: geopotential in m2 s-2, accumulated fields (tp, ssrd, strd) at time t are the
accumulation over (t - 1h, t], pressure levels in hPa sorted ascending (level=-1 is the lowest altitude).
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pvlib
import xarray as xr
from pyproj import Transformer

G, R, EPS, SIGMA = 9.81, 287.05, 0.622, 5.67e-8
S0 = 1370.
LEVELS = np.array([500., 600., 700., 775., 850., 925., 1000.])  # hPa
EPSG = 32632


def esat_bolton(T):
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def esat_magnus(T):
    Tc = T - 273.15
    return 610.94 * np.exp(17.625 * Tc / (243.04 + Tc))


def z_of_p(T0, p, gamma, p0):
    """Height [m] of pressure p [Pa] in a constant lapse-rate atmosphere, broadcast over time (T0)."""
    return (np.asarray(T0)[:, None] / gamma) * (1 - (np.asarray(p)[None, :] / p0) ** (R * gamma / G))


def p_of_z(T0, z, gamma, p0):
    return p0 * (1 - gamma * z / np.asarray(T0)) ** (G / (R * gamma))


def make_era5(start='2020-06-18', n_days=6, lat0=60.5, lon0=7.5, d=0.25, z_surf=1200., T0_mean=290.,
              T0_amp=4., gamma=0.0065, p0=101325., rh=0.7, transmissivity=0.75, cloud=None, eps_sky=0.8,
              tp_mm=None, tp_weights=None, u0=4., v0=-3.):
    tvec = pd.date_range(start, periods=24 * n_days, freq='1h')
    nt = tvec.size
    T0 = T0_mean + T0_amp * np.sin(2 * np.pi * (tvec.hour.values - 9) / 24)
    lats = np.array([lat0 + d, lat0, lat0 - d])
    lons = np.array([lon0 - d, lon0, lon0 + d])

    # ---- pressure levels
    p_lev = LEVELS * 100
    z_lev = z_of_p(T0, p_lev, gamma, p0)
    T_lev = T0[:, None] - gamma * z_lev
    e_lev = rh * esat_bolton(T_lev)
    q_lev = EPS * e_lev / (p_lev[None, :] - (1 - EPS) * e_lev)

    def g4(a):
        return np.broadcast_to(np.asarray(a)[..., None, None], np.shape(a) + (3, 3)).copy()

    # wind speed increasing with height, direction turning 1.5 times over the period
    speed = np.hypot(u0, v0) + 0.002 * z_lev
    theta = np.linspace(0, 3 * np.pi, nt)[:, None]

    dims4 = ('time', 'level', 'latitude', 'longitude')
    plev = xr.Dataset({'z': (dims4, g4(z_lev)),  # already in m (as written by _subset_climate_dataset)
                       't': (dims4, g4(T_lev)),
                       'q': (dims4, g4(q_lev)),
                       'r': (dims4, g4(np.full_like(T_lev, rh * 100))),
                       'u': (dims4, g4(speed * np.cos(theta))),
                       'v': (dims4, g4(speed * np.sin(theta)))},
                      coords={'time': tvec, 'level': LEVELS, 'latitude': lats, 'longitude': lons})

    # ---- surface
    sp = p_of_z(T0, z_surf, gamma, p0)
    t2m = T0 - gamma * z_surf
    e2m = rh * esat_magnus(t2m)
    lne = np.log(e2m / 610.94)
    d2m = 243.04 * lne / (17.625 - lne) + 273.15

    minutes = pd.date_range(tvec[0] - pd.Timedelta('1h'), tvec[-1], freq='1min', inclusive='right')
    sol = pvlib.solarposition.get_solarposition(minutes, lat0, lon0, z_surf, method='nrel_numpy')
    mu = np.clip(np.cos(np.deg2rad(sol.zenith.values)), 0, None)
    cloud = np.ones(n_days) if cloud is None else np.asarray(cloud, float)
    ghi = S0 * mu * transmissivity * np.repeat(cloud, 24 * 60)
    ssrd = ghi.reshape(nt, 60).sum(axis=1) * 60.  # J m-2 over (t-1h, t]
    strd = SIGMA * eps_sky * t2m ** 4 * 3600.
    tp_mm = np.zeros(nt) if tp_mm is None else np.asarray(tp_mm, float)
    tp_weights = np.ones((3, 3)) if tp_weights is None else np.asarray(tp_weights, float)

    dims3 = ('time', 'latitude', 'longitude')

    def g3(a):
        return np.broadcast_to(np.asarray(a, float)[:, None, None], (nt, 3, 3)).copy()

    surf = xr.Dataset({'z': (dims3, g3(np.full(nt, z_surf))),
                       'sp': (dims3, g3(sp)), 't2m': (dims3, g3(t2m)), 'd2m': (dims3, g3(d2m)),
                       'tp': (dims3, g3(tp_mm / 1000.) * tp_weights[None]),
                       'ssrd': (dims3, g3(ssrd)), 'strd': (dims3, g3(strd))},
                      coords={'time': tvec, 'latitude': lats, 'longitude': lons})

    to_xy = Transformer.from_crs('epsg:4326', f'epsg:{EPSG}', always_xy=True)
    x0, y0 = to_xy.transform(lon0, lat0)
    return SimpleNamespace(plev=plev, surf=surf, tvec=tvec, T0=T0, gamma=gamma, p0=p0, z_surf=z_surf, rh=rh,
                           lat0=lat0, lon0=lon0, x0=x0, y0=y0, start=str(tvec[0].date()),
                           end=str(tvec[-1].date()), transmissivity=transmissivity)


def make_point(case, name='001', dx=1., dy=0., elevation=None, slope_deg=0., aspect_deg=0., svf=1.):
    """aspect_deg follows the TopoPyScale convention (same as solar azimuth): 0 = south, +90 = east."""
    x, y = case.x0 + dx, case.y0 + dy
    lon, lat = Transformer.from_crs(f'epsg:{EPSG}', 'epsg:4326', always_xy=True).transform(x, y)
    return SimpleNamespace(point_name=name, point_ind=int(name), x=x, y=y, lon=lon, lat=lat,
                           elevation=case.z_surf if elevation is None else float(elevation),
                           slope=np.deg2rad(slope_deg), aspect=np.deg2rad(aspect_deg), svf=svf)


def centroids_frame(rows, horizon_deg=0.):
    df = pd.DataFrame([vars(r) for r in rows])
    df['aspect_cos'], df['aspect_sin'] = np.cos(df.aspect), np.sin(df.aspect)
    for az in np.arange(-165, 180, 30.):
        df[f'hori_azi_{az}'] = np.deg2rad(horizon_deg)
    return df


def solar_geometry(case, rows, workdir):
    from TopoPyScale import solar_geom as sg
    df = pd.DataFrame({'point_name': [r.point_name for r in rows], 'x': [r.x for r in rows],
                       'y': [r.y for r in rows], 'elevation': [r.elevation for r in rows]})
    return sg.get_solar_geom(df, case.start, case.end, '1h', sr_epsg=str(EPSG), num_threads=2,
                             fname='ds_solar.nc', project_ouput=Path(workdir))


def horizon_da(rows, angle_deg=0.):
    az = np.arange(-165, 180, 30.)
    xs = np.unique(np.concatenate([[r.x - 100, r.x, r.x + 100] for r in rows]))
    ys = np.unique(np.concatenate([[r.y - 100, r.y, r.y + 100] for r in rows]))[::-1]
    return xr.DataArray(np.full((az.size, ys.size, xs.size), np.deg2rad(angle_deg)),
                        coords={'azimuth': az, 'y': ys, 'x': xs}, dims=['azimuth', 'y', 'x'], name='horizon')


def run(case, row, ds_solar, hor, workdir, backend='netcdf', interp_method='idw', lw_terrain=False,
        precip_lapse=False):
    """Downscale one point with the real pipeline functions. Returns a loaded dataset with a point_name coord."""
    meta = {'interp_method': interp_method, 'lw_terrain_flag': lw_terrain, 'tstep': 1, 'n_digits': 3,
            'file_pattern': 'down_pt_*.nc', 'target_epsg': EPSG, 'precip_lapse_rate_flag': precip_lapse}
    sol = ds_solar.sel(point_name=row.point_name)
    if backend == 'netcdf':
        from TopoPyScale import topo_scale as ta
        out = Path(workdir) / f'out_{row.point_name}'
        (out / 'tmp').mkdir(parents=True, exist_ok=True)
        (out / 'downscaled').mkdir(parents=True, exist_ok=True)
        meta.update(output_directory=out,
                    transformer=Transformer.from_crs('epsg:4326', f'epsg:{EPSG}', always_xy=True))
        ta.pt_downscale_interp(row, case.plev.copy(deep=True), case.surf.copy(deep=True), meta)
        ta.pt_downscale_radiations(row, sol, hor, meta, out)
        with xr.open_dataset(out / 'downscaled' / f'down_pt_{row.point_name}.nc', engine='h5netcdf') as ds:
            ds = ds.load()
    elif backend == 'zarr':
        from TopoPyScale import topo_scale_zarr as tz
        plev = case.plev.copy(deep=True)
        plev['z'] = plev.z * G
        surf = case.surf.rename({'z': 'z_surf'}).copy(deep=True)
        surf['z_surf'] = surf.z_surf * G
        subset = xr.merge([plev, surf])
        meta.update(output_directory=Path(workdir))
        ds = tz.ClimateDownscaler.downscale_atmo(None, row, subset, meta, sol, hor).load()
        ds = ds.drop_vars(['point_ind', 'reference_time'], errors='ignore')
    else:
        raise ValueError(backend)
    return ds.assign_coords(point_name=row.point_name)
