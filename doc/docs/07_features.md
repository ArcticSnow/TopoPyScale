# Features

`TopoPyScale` (TPS) has a number of features to customize your downscaling. This pages is an attempt to describe all features currently implemented.

## Spatial vs. Point downscaling

TPS can be used to downscaled in two modes. The simplest case is to use the library to downscale weather timeseries at a list of points. This list of points id specified in a `.csv` file using coordinate system of the same projection (*i.e.* EPSG code). At minimum each line of the `.csv` file should contain:

```text
name,x,y
```

Using the setting spatial in `toposub` will lead you using the clustering method decribed in more details here below. 

## Fetching ERA5 data from CDS

ERA5 can also be fetched from NCAR GDEX instead of the CDS (`climate.era5.data_repository: gdex`), see [Dataset Sources](04_datasetSources.md#from-ncar-gdex).

As of now, TPS includes a class to implementing all the functionalities in dealing with the ERA5 data independent from `Topoclass`. This makes the process, now very long, of fetching data more flexible and robuts. The class eith inherits the config from `Topoclass`, or can be directly pointed towards the `config.yml` file.

```python
from TopoPyScale import fetch_era5 as fe 
from TopoPyScale import topoclass as tc 

config_file = 'config.yml'
mp = tc.Topoclass(config_file)
era = fe.FetchERA5(config=mp.config)
era.go_fetch()
```

The class integrate a number of options to make the downloading more flexible. For instance only download Surface Level data, with a given set of variables.

```python
from TopoPyScale import fetch_era5 as fe 
era = fe.FetchERA5(config='config.yml')

varoi = ['geopotential_at_surface',
			'surface_thermal_radiation_downwards',
            'surface_solar_radiation_downwards',
            'surface_pressure',
            'total_precipitation', 
            '2m_temperature'
        ]

era.varoi_surf = varoi
era.go_fetch()
```

This way it is possible to use routines for TPS project but not only.

**Time steps of 3 h and 6 h.** ERA5 gives the accumulated fields (precipitation, radiation) as the accumulation over the hour ending at each time stamp. With `climate.era5.timestep: 3h` or `6h`, the pressure levels and the instantaneous surface fields are fetched at the time step, and the accumulated fields hourly: the store holds their sums over the time step ending at the stamp (store attribute `accumulation_hours`). The first window of each request starts on the previous day: from the CDS, its hours come with a small second request (the accumulated fields, for 2 or 5 hours); from GDEX, they are read with the day. The downscaling then gives precipitation totals and mean radiation over each time step, and computes the solar geometry over the same window. An hourly store downscaled at 3 h or 6 h is summed the same way. A 3 h store fetched by an earlier version holds only the hour ending at each stamp: it is used with a warning, and refused by `get_era5()` (delete it to fetch the sums).

**Variables added to an existing store.** A store keeps fetching every variable it holds. When a variable is added to a stream (e.g. the direct solar radiation `fdir` for `toposcale.direct_diffuse: era5`), that stream is fetched again for the whole period and the other one is kept.


## Convert ERA5 to Zarr

As explained in the (Datasets)[04_datasetSources.md], TPS relies on ERA5 data. In the future more datasets could be included. The minimum data required are at pressure levels `t,p,q,u,v`, and the surface level:
```
- 'geopotential_at_surface'
- '2m_dewpoint_temperature'
- 'surface_thermal_radiation_downwards',
- 'surface_solar_radiation_downwards','surface_pressure',
- 'total_precipitation'
- '2m_temperature'
- 'toa_incident_solar_radiation',
- 'friction_velocity'
- 'instantaneous_moisture_flux'
- 'instantaneous_surface_sensible_heat_flux'
```

### ERA5 download

`mp.get_era5()` writes ERA5 straight into the zarr store read by the downscaling, `inputs/climate/ERA5.zarr`, with no intermediate daily or yearly files:

- **One request per stream and month**, the request size ECMWF advises for hourly ERA5: about 24 requests per year instead of 730 daily ones, each of which waits in the CDS queue. A request over the CDS cost limit is split in two (by days, then by variables).
- **At most 10 requests at a time** (`download_threads`, default 8): the CDS runs 10 requests of a user in parallel, and lowers the priority of users who send more.
- **Workers only download.** Each file is read by the main thread as it arrives, one at a time (netCDF-C and HDF5 are not thread safe), checked, written to its time slice of the store, and deleted. Each value is written once, and memory holds one month.
- **Checks** before writing: all time steps of the month, the variables, the pressure levels, the grid of the store, no missing value. Downloads go to a `.part` file renamed once complete, so a cut download is never taken for a finished one.
- **Retries**: a failed request is sent again after 1, 2 and 4 minutes (not for errors a new attempt cannot cure, such as a licence not accepted). Months that still fail are listed at the end, after all the others are written.
- **Restart**: the store records the time steps it holds (`filled_plev`, `filled_surf`). Running `mp.get_era5()` again fetches only what is missing; a later `project.end` extends the store and fetches the new months only. A store holding another box, other levels or another time step is refused, rather than mixed.
- **Less data**: the five pressure-level variables and six surface fields the downscaling reads (relative humidity is no longer downloaded), and the surface geopotential once instead of every hour.
- **Earlier downloads** in `yearly/PLEV_*.nc`, `yearly/SURF_*.nc` or `daily/` are read into the store instead of being downloaded again, when they hold the variables and the grid of the request.

The box is the project extent plus 0.4°, reduced to the nodes of the 0.25° ERA5 grid inside it, as the CDS returns them. The store holds `z, t, q, u, v` on pressure levels, `sp, t2m, d2m, tp, ssrd, strd` at the surface and the static `z_surf`, as float32 compressed with Blosc. The same function fetches from NCAR GDEX (see [Dataset Sources](04_datasetSources.md#from-ncar-gdex)), or can be called directly:

```python
from TopoPyScale import fetch_era5 as fe

fe.fetch_era5_to_zarr('2020-01-01', '2020-12-31', 'inputs/climate', latN=47.3, latS=45.6, lonE=10.9, lonW=5.6,
                      step='1h', plevels=[700, 750, 800, 850, 900, 950, 1000])
```

### ERA5 from netcdf files

netcdf files placed in the climate folder by hand are converted to the store the first time `mp.downscale_climate()` runs (and again if the store does not cover the project period). The conversion opens one pair of PLEV/SURF files at a time and appends it to the store. It can also be run explicitly:

```python
from TopoPyScale import fetch_era5 as fe

fe.netcdf_to_zarr(path_to_netcdfs='inputs/climate/yearly',
                  zarrout='inputs/climate/ERA5.zarr',
                  plev_name='PLEV_*.nc',
                  surf_name='SURF_*.nc',
                  time_chunk=720)
```

The store holds the variables of the files (`z, t, q, u, v` on pressure levels and `z, sp, t2m, d2m, tp, ssrd, strd` at the surface at least), as float32 compressed with Blosc. The function also harmonizes files downloaded from the new CDS (`valid_time`, `pressure_level`, `expver`).

## Parallelization

TPS uses parallelization for a number of steps: threads sending download requests to the CDS (at most 10), threads for the sky view factor, threads over blocks of points for the horizons, multiple cores for clustering, and threads over blocks of points for the downscaling (`project.parallelization.setting.multicore.CPU_cores`).

## Solar geometry

The downscaling computes the solar geometry of each block of points and time steps when it downscales the block (`solar_geom.window_terms()` once for the period, then `solar_geom.solar_block()`), so it never holds the geometry of all points for the whole period: for 1000 points and 30 years of hourly data that would be about 8 GB. `mp.compute_solar_geometry()` only saves the same values to `outputs/ds_solar.nc` when `toposcale.save_solar_geometry` is true (`solar_geom.get_solar_geom()`, by blocks of points), for plots and analysis. The output store records the settings in its attributes (`accumulation_window`, `solar_substeps`, `solar_constant`, `solar_position_method`), which the physics checks read to recompute the geometry.

For every point and time step, the solar geometry holds:

- the solar zenith, azimuth and elevation angles at the centre of the radiation accumulation window (`t − 30 min` for ERA5), with the NREL Solar Position Algorithm of pvlib (`nrel_numpy`, about 0.0003° accuracy). The time-dependent part of the algorithm is computed once for all points; only the location-dependent part is evaluated per point, as arrays. For 2400 points and one year of hourly data this takes about 12 s, instead of about 65 s point by point;
- the top-of-atmosphere irradiance on a horizontal surface `SWtoa`, averaged over the accumulation window with `toposcale.solar_substeps` sub-steps (default 6). It uses the solar constant `toposcale.solar_constant` (default 1361 W m⁻²) scaled by the Earth–Sun distance, consistently with the accumulated ERA5 fluxes it is compared to (clearness index). Averaging over the window matters at sunrise and sunset, where the value at the window centre differs from the hourly mean by up to about 35 W m⁻².

`ds_solar.nc` is stored as float32. A file computed by an older version prints a warning when loaded; delete it to recompute it. The downscaling no longer reads it.

**Terrain shading within the window.** ERA5 shortwave is a mean over the accumulation window, so a point can be lit for part of it: the sun clears or drops behind a ridge during the hour. The downscaling tests the horizon at the same `toposcale.solar_substeps` sub-steps of the window and scales the direct beam by the sunlit fraction, each sub-step weighted by the beam it brings to the slope (cosine of the illumination angle). The sun positions of the sub-steps are geocentric (within 0.01° of the SPA); only terrain shading counts, not the sun below the astronomical horizon. With `solar_substeps: 1` the horizon is tested once, at the centre of the window, and the beam is kept or removed for the whole window. On the Davos points (January 2020), the 57 hours the sun crosses the horizon of DAV get an RMSE of 20.6 W m⁻² against the station instead of 30.3; the cost does not show in the downscaling time.


## Direct and diffuse shortwave

TopoSCALE splits the ERA5 shortwave of the grid into direct and diffuse before the terrain corrections: the diffuse part is scaled by the sky view factor, the direct part is extrapolated to the elevation of the point (Beer's law with the extinction of the ERA5 column), projected on the slope and removed in terrain shadow. `toposcale.direct_diffuse` chooses the split:

- `empirical` (default): the diffuse fraction from the clearness index kt = SW / SWtoa, kd = 0.952 − 1.041 exp(−exp(2.3 − 4.702 kt)) (Ruiz-Arias et al., 2010). It depends on the hourly kt only: a thin cloud layer and a broken cloud field with the same kt get the same split.
- `era5`: the direct radiation on a horizontal surface of the ERA5 radiation scheme (`fdir`, *total sky direct solar radiation at surface*), which accounts for the clouds, aerosols and water vapour of the reanalysis. The diffuse part is SW − fdir. `fdir` is fetched with the surface fields when the option is set (`get_era5()`; an existing store gets its surface stream fetched again). In the ECMWF radiation scheme, the direct flux includes part of the radiation scattered in the forward direction, so `fdir` is somewhat larger than what a pyrheliometer would measure (ECMWF, *Radiation quantities in the ECMWF model and MARS*).

**Sky diffuse on slopes.** `toposcale.diffuse_model: isotropic` (default) scales the grid diffuse by the sky view factor of the point. `hay_davies` (Hay and Davies, 1980) treats part of it as circumsolar: a share equal to the anisotropy index A = direct / SWtoa (both on a horizontal surface) comes from the direction of the sun, so it is projected on the slope with the beam ratio cos i / cos z (cos z floored at cos 85°) and removed in terrain shadow like the direct beam; the rest, 1 − A, is isotropic:

D = D_grid [(1 − A) svf + A (cos i / cos z) (1 − shade)].

On level open ground both give the grid diffuse. Under clear skies the circumsolar part brightens slopes facing the sun and darkens slopes facing away from it; under overcast skies (A → 0) the two models agree.

The output store records the options in its attributes `direct_diffuse` and `diffuse_model`.

## Terrain radiation

The terrain around a point fills the part of its view that the sky does not, 1 − svf (the sky view factor is cosine weighted and counts the tilt of the slope, so 1 − svf is the configuration factor of the terrain).

**Longwave.** With `toposcale.LW_terrain_contribution: True`, the terrain emits 0.99 σ T⁴ towards the point, weighted by ½ (1 + cos S)(1 − svf) as in TopoSCALE. `toposcale.LW_terrain_temperature` sets T: `air` (default), the air temperature at the point; `fixed`, 273.15 K, the value of earlier versions, which overestimates the longwave of cold snow-covered terrain by up to about 60 W m⁻² × the weight at −15 °C; or a temperature in K. `topo_scale_zarr.ClimateDownscaler` also takes a DataArray (point_name, time) of temperatures, the hook for surface temperatures of a snow model run on the points (e.g. FSM iterations).

**Shortwave.** With `toposcale.SW_terrain_albedo` above 0, the terrain reflects the global radiation of a level surface at the point (the direct beam at the elevation of the point, unshaded, plus the grid diffuse) with that albedo: SW_terrain = α (1 − svf) (B_flat + D_grid). It is added to `SW_diffuse` and stored as `SW_terrain`. It is largest in deep valleys and on steep slopes facing bright slopes: with α = 0.8 (fresh snow), svf = 0.7 and 500 W m⁻², it adds 120 W m⁻². `ClimateDownscaler` also takes a DataArray (point_name, time) of albedos, the hook for the albedo of a snow model.


## Wind

TopoSCALE takes the wind of the pressure levels at the elevation of the point: free-atmosphere wind, without the drag of the surface. In valleys it is much too strong: at Davos (1594 m, January 2020) the downscaled mean wind speed is 3.1 m s⁻¹ against 1.1 m s⁻¹ measured, while at Weissfluhjoch (2691 m, a summit) it is 5.6 against 5.2. `toposcale.wind_downscaling` offers two options, alone or together:

- `surface_blend`: the wind is blended with the ERA5 10 m wind interpolated to the point, u = u_pl + w (u10 − u_pl) (and the same for v), with w a constant in [0, 1] or `surface_effect`, the terrain weight of the [surface-effect correction](#surface-effect-correction-of-air-temperature) clipped to [0, 1]: points coupled to the ERA5 surface (valleys) get its wind, exposed points (ridges) keep the free-atmosphere wind. Using the temperature weight for the wind is an assumption of TopoPyScale, not a published method. `get_era5()` fetches the 10 m wind with the surface fields when the option is set.
- `terrain: liston_elder`: the wind weighting and diverting factors of MicroMet (Liston and Elder, 2006), applied after the blending. The speed is multiplied by 1 + γs Ωs + γc Ωc, with Ωs = β cos(θ − ξ) the slope in the direction the wind comes from (β slope, ξ azimuth the slope faces, θ wind direction) and Ωc the curvature of the terrain over the length η (`curvature_length`, default 500 m), each scaled to [−0.5, 0.5] over the DEM (by twice the steepest slope and twice the largest absolute curvature); γs = 0.58 and γc = 0.42 (`slope_weight`, `curvature_weight`). The direction turns by −0.5 Ωs sin(2(ξ − θ)) (Ryan, 1977). The curvature of a cluster is the mean over its cells, the one of a point the value of its cell; it is stored in `df_centroids` as `wind_curvature`. The factors stay within about ±30 %: at Davos the valley wind decreases by 2 % only, and the summit wind of Weissfluhjoch increases by 6 %.

The terrain parameterization of Helbig et al. (2017, Laplacian of the terrain and mean square slope) is not implemented yet.


## Slope, aspect and sky view factor

`mp.compute_dem_param()` computes the elevation, slope, aspect and sky view factor of every DEM cell and saves them to `outputs/ds_param.nc`. An existing `ds_param.nc` is loaded as is: delete it to recompute it (files computed with topocalc by earlier versions differ by about 0.003 in svf, see below).

**Slope and aspect** come from the 3 × 3 neighbourhood of each cell (Horn's finite differences, `topo_param.gradient_d8()`, the same code as topocalc's). The aspect is in radians, 0 = south, +π/2 = east, the convention of the solar azimuth.

**Sky view factor.** The svf is the part of the sky hemisphere seen by the ground surface, each direction weighted by the cosine of its angle to the normal of the surface. It is 1 on flat open ground and (1 + cos S)/2 on a plane of slope S with nothing above it. It scales the diffuse shortwave and the longwave from the sky; 1 − svf weights the longwave emitted by the surrounding terrain (`toposcale.LW_terrain_contribution`). `topo_param.sky_view_factor()` integrates it from the horizon angle H in 72 directions, every 5° (Dozier and Frew, 1990, eq. 7b), keeping the positive part of the integrand:

svf = 1/(2π) ∫ [cos S sin²H' + sin S cos(φ − A) (H' − sin H' cos H')] dφ, with H' = 90° − H the horizon angle from the zenith, S the slope and A the aspect.

**Horizons of every cell: line sweep.** One azimuth at a time, `topo_param.horizon_sweep()` computes the horizon of all cells of the DEM:

1. The grid axis closest to the direction is the major axis. Rays advance one column per step (columns stand for the major axis here) and drift by t rows per column, |t| ≤ 1.
2. Parallel straight lines in the direction of the rays cross the grid, one per row. At each column, the elevation of a line is interpolated linearly between the two rows it passes between.
3. Along each line, from its end back to its start, each node gets a pointer to the node ahead seen at the steepest angle. Following the pointers from any node walks along the upper convex hull of the line ahead of it (Dozier and Frew, 1990). From an observer, the elevation angle to the hull nodes rises to the horizon, then decreases, so the walk stops at the first decrease. The pointers of a line cost a number of steps proportional to its length. All lines are processed together as arrays.
4. The ray of each cell passes between two of these lines, at a distance f from one and 1 − f from the other (in rows). Over its first 32 cells (`near`), the ray is traced exactly, interpolating the DEM between rows at each column. Beyond, the horizon is the steepest angle from the elevation of the cell to the hull of each of the two lines, weighted by 1 − f and f. The horizon of the cell is the higher of the two parts.

![Line sweep](horizon_sweep.png){ width="900" }

Along the grid axes and diagonals the lines pass through the cell centres, and the result is exact. In other directions, distant terrain changes little from one line to the next, so the two lines around a ray give its horizon well. Near terrain changes much faster from one line to the next, which is why the first 32 cells are traced along the ray itself. Without them (`near=0`), horizons are wrong by more than 3° for 5% of the cells of the Davos example.

On the Davos example (SRTM 30 m, 747 × 500 cells), compared with exact straight rays from every cell in the 72 directions:

| | svf difference: median | 99th percentile | maximum |
|---|---|---|---|
| line sweep (`near` = 32) | 0.00001 | 0.0001 | 0.001 |
| topocalc (earlier versions) | 0.003 | 0.016 | 0.046 |

topocalc follows lines of sight shaped as staircases, which step at the same columns for all cells. The cells just before a step read their nearest terrain from the adjacent row, which leaves streaks in its horizon and svf maps for directions a few degrees off the grid axes. The straight lines of the sweep have no steps.

The sweep runs as a [numba](https://numba.pydata.org) kernel, compiled on first use (about 10 s once, then cached), with the lines and rows of each direction split between `project.parallelization.setting.multicore.CPU_cores` threads. Each direction takes about 0.15 s on the Davos DEM (747 × 500 cells) on one core: the svf takes 11 s on one core and 6.6 s on two (topocalc: 18 s on one core). Most of this time goes to the near part, so the cost grows with the number of cells times `near`. Without numba, or with `engine='numpy'` in `topo_param.sky_view_factor()`, the same algorithm runs with numpy, about 4 times slower, the directions shared between processes. Both engines give the same horizons, and the directions are summed in order: the svf depends neither on the engine nor on the number of threads or processes. No-data cells (NaN) never form a horizon. Their svf is NaN, as is that of their neighbours, which have no slope. Earth curvature is ignored (about 0.1° for terrain 20 km away).

For maps of horizons, `topo_param.compute_horizon(dem_file, azimuth_inc=30)` runs the same sweep for each direction and writes `outputs/da_horizon_grid.nc`, with dimensions (`azimuth`, `y`, `x`).


## Horizons

`mp.compute_horizon()` computes the horizon angle of each point in the directions `dem.horizon_increments` degrees apart (the centres of the bins: -175, -165, ..., 175 for 10°; 0 = south, 90 = east, ±180 = north). The downscaling removes the direct beam while the sun is below the horizon of the nearest direction, in proportion to the part of the accumulation window it spends there (Solar geometry). The horizons are saved to `outputs/da_horizon.nc`, with dimensions (`azimuth`, `point_name`), and as the columns `hori_azi_<azimuth>` of `df_centroids`.

From each point, `topo_param.horizon_rays()` follows a straight ray in each direction, one cell at a time along the major grid axis, interpolating the DEM bilinearly. The observer is on the DEM surface at the exact position of the point, not at the centre of the nearest cell. The horizon is the steepest elevation angle to the terrain along the ray, or 0 if no terrain is higher than the point. A ray stops at the edge of the DEM, or as soon as no terrain further away could be steeper than the horizon already found: the highest elevation of the DEM seen from that distance is below it. This shortcut does not change the result. No-data cells never form a horizon. Earth curvature is ignored.

For the 50 centroids of the Davos example (747 × 500 cells), this takes 0.15 s and writes 24 kB. Each point and direction costs about 10 µs on that DEM. With `sampling.method: grid`, every cell is a point: about 2 minutes for the 373,500 cells of Davos on one core. Blocks of points run in `project.parallelization.setting.multicore.CPU_cores` threads.

An existing `da_horizon.nc` is loaded if it holds the same points (names and positions), and recomputed otherwise, including files computed on the full DEM grid by earlier versions.


## Surface-effect correction of air temperature

TopoSCALE takes the air temperature and humidity of a point from the pressure levels: free-atmosphere air, without the inversions, cold-air pools and daily cycle that ERA5 resolves at its own surface. With `toposcale.surface_effect.method: redcapp` or `dreamit`, the downscaling adds back a terrain-weighted share of the ERA5 surface signal ΔT = t2m − T_pl(z_ERA5 surface):

T = T_pl(z_point) + w ΔT + β(t), and Td = Td_pl(z_point) + w ΔTd, capped at T.

w = 0 gives the plain TopoSCALE values; w = 1 transfers the whole grid-surface difference. The longwave is then computed from the corrected temperature and humidity (`lw_consistent`). The weight of each point comes from its terrain, computed once on `context_dem`, which must cover the points plus the window:

| Method | Weight | Terrain | Published coefficients |
| --- | --- | --- | --- |
| `redcapp` ([Cao et al. 2017](https://doi.org/10.5194/gmd-10-2905-2017)) | w = α [H (1 − S) + S] + β V (1 − S), S = exp(−R / γ) | hypsometric position H and relief R in a 30 km wide circle; V = MRVBF / 8 | Swiss Alps: α 0.61, β 1.56, γ 465 m |
| `dreamit` ([Pozsgay et al. 2026](https://doi.org/10.5194/gmd-19-7197-2026)) | w = α_intercept + exp(α_slope h) − 1, plus β(t) = β_amp cos(2π(t − t*)) + β_bias | hypsometric position h within 50 km | ERA5, Yukon: α_intercept 0.449, α_slope 0.732, β_amp 0.918, t* 0.958, β_bias 1.181 K |

The hypsometric position is the fraction of the DEM cells in the window that are higher than the point: 0 on the highest summit, 1 at the bottom of the lowest valley. The indices and `w_surf` are added to `df_centroids`. DReaMIT defines ΔT against a lapse rate fitted above the inversion; TopoPyScale uses its own pressure-level interpolation, as REDCAPP does.

Published coefficients may not transfer: at Davos (January 2020), the Yukon DReaMIT coefficients over-correct the Weissfluhjoch crest. `TopoPyScale.surface_calib` fits them for a new region from station records:

```python
from TopoPyScale import surface_calib as sc, surface_effect as se
# 1. downscale the stations as a points project with surface_effect: {method: none, diagnostics: true}
data = sc.pair(mp.downscaled_pts.load(), {name: sc.read_meteoswiss(file) for name, file in station_files.items()})
indices = {r: se.terrain_indices(df_stations, 'glo90.tif', r, epsg=32632).set_index(df_stations.point_name)
           for r in [5, 15, 30, 50]}
res = sc.calibrate('dreamit', data, indices, fit_beta='bias')   # pooled fit, cross-validation, bootstrap
sc.save_coefficients(res, 'surface_coeffs.yml')                  # then surface_effect.coefficients: surface_coeffs.yml
```

`calibrate` fits w per station (diagnostic), then the coefficients by pooled least squares over all stations and hours, keeps the window radius with the lowest cross-validated error (leave one station out, or contiguous time blocks with fewer than 3 stations), and gives moving-block bootstrap intervals. `fit_beta` is `none`, `bias` (regional offset) or `seasonal` (DReaMIT β(t), needs a year of data).


## Clustering

Clustering offer a number of possibilities. First, any DEM can by segmented by clustering (k-means) using a list of features. These features can be topographic or anything which can be labeled to the pixel level (e.g. watershed, glaciated land, etc.). The clustering will return a list of groups of pixels that fall within the same "cluster". The method implemented is K-means from [Scikit-learn](https://sklearn.org/stable/getting_started.html). To speed up the processing, TPS will use the routine `minibatch_kmean()` to spread the computational load on multiple core. 

### Clustering features

In the config file, the line `clustering_features: {'x':1, 'y':1, 'elevation':1, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1}` will define which variables are used to perform the clustering. A variable must be defined in `ds_param` to be used. Each feature are associated to a weight (default 1), but can be modulated higher or lower than one to diminish or enhance the effect of the selected variable. 
The value in `clustering_features` is a multiplier applied during the feature scaling prior to the K-means clustering.
This option allows to apply feature importance relative to the others. **This multiplier must be $>0$.** 
It may be useful for instance to emphasize the role of elevation in case of sharp terrain or any other variable.  


With the example below, the elevation as a factor of 4 relative to other features. 
```python
clustering_features: {'x':1, 'y':1, 'elevation':4, 'slope':1, 'aspect_cos':1, 'aspect_sin':1, 'svf':1}
```

If you add new features specific to your project, be aware that all features are standardize using the `scikit-learn` function `StandardScaler()`. Other types of scaler not yet available.

It is recommanded to keep the bare minimum variable: `x,y,elevation`.

### Searching optimum number of clusters

There are no clear cut method to decide the ideal number of clusters. However there exists a number of technics to find a compromise between a sufficiently fine segmentation and the lowest number of clusters (less computation). These methods are implemented `Topo_sub`. It includes a method to search for the optimal number of clusters: `search_optimum_number_of_clusters()`. This may be used in the following manner:

```python
from TopoPyScale import topoclass as tc
import numpy as np

config_file = './config.yml'
mp = tc.Topoclass(config_file)
mp.compute_dem_param()
df = mp.search_optimum_number_of_clusters(cluster_range=np.arange(100,1000,50),plot=False)
```

This will compute the scores Within-Cluster-Sum-of-Squares (WCSS) Davies-Bouldin, Calinski-Harabasz, and the RMSE between the original DEM elevation and the clustered DEM elevation. Any of these may be used to decide of an appropriate number of clusters  

### Masking

While the input DEM is a raster file and have rectangular boundings, TPS allows to mask part of the DEM. The mask should be a GeoTiff of the same grid (same boundings) as the mains DEM. Pixels categorized as `1` will be considered for downscaling, the one as `0` will be ignored. This should be specified into the config file with the following: `clustering_mask: clustering/catchment_mask.tif` within the section `toposub`.

### Groups

The clustering can be defined in subgroups. This may be usefull for instance when one wants to split a watershed in different section (i.g. glacier, alpine, fan, lake, *etc.*). This can be done by creating a raster file of the same size and bounds as the DEM with an arbitrary set of integers. Each integers will define a group. The path to the raster file must be indicated in the config file as part of the `toposub` section as follow: `clustering_groups: clustering/groups.tif`.

Now one can set relative number of clusters per group. Default, it maintain cluster density per area homogenous across groups.

To trigger the option.
1 in config.yml:
add clustering_group_weights: 'inputs/dem/gw.csv' in toposub
2. create a csv file, here called gw.csv
```
group,weight
8,0.8
4,0.2
```
This will use 20% of clusters to group 4 and 80% for group 8.

## FSM integration

As part of TPS, we integrated a number of functions to interact with the snow model **FSM**. 

## Realtime

**@Joel can you write smthg?**
