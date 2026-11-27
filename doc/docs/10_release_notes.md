# Release notes

## 1.1.0 (in development)

TopoPyScale 1.1 keeps the memory of long runs bounded, sums the accumulated ERA5 fields at 3 h and 6 h time steps, and adds options for the radiation and the wind. It also reports what it does through a log, checks `config.yml`, and runs its tests on GitHub.

### Highlights

- **Messages and log file**: every module reports through Python `logging`. `project.verbose` sets what the terminal shows (default: progress), and each run writes a time-stamped log to `outputs/logs/` (`project.log_file`), with the warnings of the libraries. Warnings that earlier versions silenced globally are visible again (see [Configuration](03_configurationFile.md#messages-and-log-file)).
- **config.yml checked**: unknown keys are reported with the closest documented key, renamed or removed keys with what replaces them. A missing config file raises a clear error.
- **Tests on GitHub**: the test suite and a lint for undefined names and syntax errors run on every push and pull request (GitHub Actions).
- **Solar geometry block by block**: the downscaling computes the solar geometry of each block of points and time steps and no longer needs `ds_solar.nc`, which held points × time steps values (about 8 GB for 1000 points and 30 years). `toposcale.save_solar_geometry: true` still writes it for plots and analysis. The physics checks read the outputs by blocks of time steps and compute the geometry the same way; the exports cap the memory of a block of points.
- **3 h and 6 h time steps**: the accumulated fields (precipitation, radiation) are fetched hourly and summed over each time step, instead of taking the hour ending at the stamp times the time step. An hourly store downscaled at 3 h or 6 h is summed the same way (see [Features](07_features.md#fetching-era5-data-from-cds)).
- **Precipitation factors in the Southern Hemisphere**: the monthly coefficients of MicroMet follow the season (shifted by six months south of the equator).
- **Radiation options** (see [Features](07_features.md#direct-and-diffuse-shortwave)): direct/diffuse split from the ERA5 direct radiation `fdir` (`toposcale.direct_diffuse: era5`); Hay–Davies circumsolar diffuse on slopes (`diffuse_model: hay_davies`); shortwave reflected by the surrounding terrain (`SW_terrain_albedo`, output `SW_terrain`); terrain longwave from the air temperature of the point (`LW_terrain_temperature: air`, the new default). The terrain temperature and albedo can also be given per point and time step, the hook for a snow model run on the points.
- **Wind options** (see [Features](07_features.md#wind)): blending with the ERA5 10 m wind (`wind_downscaling.surface_blend`) and the slope and curvature factors of MicroMet (`wind_downscaling.terrain: liston_elder`).
- **Fixes**: `to_netcdf()` wrote missing values of packed variables as plausible numbers (e.g. 2000 m for a missing elevation): they are now written as missing. `fetch_dem` and `sim_fsm` referred to undefined names on some paths; `to_fsm2()`, unfinished, now says so (use `to_fsm2oshd()`). `get_era5()` reads earlier ERA5 files that cover a larger box or more pressure levels than the request, such as those of the examples, instead of downloading them again. The maps of downscaled points (`topo_plot.map_variable`, `map_clusters`, `mp.toposub.plot_clusters_map()`) and `write_landform()` work with the zarr store and the cluster labels stored as point names. `to_cryogrid()`, `to_crocus()` and `to_surfex()` take the `lat`/`lon` columns of `df_centroids`. The physics check of the direct beam in terrain shadow tests the sun at the sub-steps of the window, as the downscaling shades it: the sun passing between two ridges was reported as an error.

### Upgrading from 1.0

1. **Solar geometry**: `mp.compute_solar_geometry()` only writes `outputs/ds_solar.nc` with `toposcale.save_solar_geometry: true` (it logs a message otherwise), and `mp.ds_solar` stays `None`. Pass `epsg=` to `topo_checks.run_all_checks()` when `df_centroids` has no `lon`/`lat` columns.
2. **Terrain longwave**: with `LW_terrain_contribution: True`, the terrain now emits at the air temperature of the point. Set `LW_terrain_temperature: fixed` for the results of 1.0.
3. **3 h and 6 h ERA5 stores** written by 1.0 hold the hour ending at each stamp: the downscaling uses them with a warning, and `get_era5()` refuses them. Delete the store to fetch the sums.
4. **Southern Hemisphere** projects with `precip_lapse_rate: True` get different precipitation factors.
5. **Example configs**: `project.CPU_cores` is not read, so the examples ran on one core. Use `project.parallelization.setting.multicore.CPU_cores`; the config check reports the key. The [examples](https://github.com/ArcticSnow/TopoPyScale_examples) are updated for 1.1.
6. **Messages** go to the log instead of `print()`. Set `project.verbose: False` for warnings only.

## 1.0.0

TopoPyScale 1.0 checks the physics of its outputs, fixes the errors those checks found, and replaces the slow or fragile parts of the workflow.

### Highlights

- **Physics-consistency checks** of the downscaled outputs (`mp.check_physics()`, `toposcale.physics_checks`, see [Physics Checks](09_physics_checks.md)), and a test suite running the code on synthetic ERA5 with known answers.
- **Corrected physics**: elevation scaling and timing of the shortwave, terrain shading scaled by the sunlit part of each hour, aspect of clusters, rain/snow partition, bilinear interpolation, bounded precipitation factor, units, humidity precision and the conversions of the exporters.
- **Zarr downscaling engine**, vectorized over blocks of points: one year of 2,400 points in 13 s on 2 cores, written to one store, `outputs/downscaled/down.zarr`.
- **Terrain without topocalc**: sky view factor and horizon maps by a line sweep (numba kernel, 11 s for the Davos DEM on one core), horizons of the points along straight rays from their exact positions.
- **Surface-effect correction** of temperature and humidity (REDCAPP, DReaMIT), off by default, and `surface_calib` to fit it for a new region (see [Features](07_features.md#surface-effect-correction-of-air-temperature)).
- **ERA5 straight to zarr**: one CDS request per month instead of per day, at most 10 at a time, each file checked and written once to `ERA5.zarr` by the main thread, failed downloads retried, and reruns fetching only what is missing. No daily or yearly files, no CDO (see [Features](07_features.md#fetching-era5-data-from-cds)).
- **ERA5 from NCAR GDEX** as an alternative to the CDS (`climate.era5.data_repository: gdex`).
- **Parallel exports**: the per-point exports (FSM, FSM2oshd, CryoGrid, CROCUS, SnowModel, SNOWPACK, GEOtop, T&C) read the store by blocks of points and write their files from `CPU_cores` processes: `to_fsm()` 2× faster in one process, and faster with each core (see [Outputs](05_outputs.md#writing-one-file-per-point-in-parallel)). `to_crocus()` and `to_micromet_single_station()` / `mp.to_snowmodel()` work again (they failed on undefined variables), and CROCUS files hold their point on a `Number_of_points` dimension.

### Upgrading from 0.3

1. **Environment**: topocalc and CDO are no longer needed. numba, scipy, pyyaml and joblib are new dependencies, installed with TopoPyScale.
2. **config.yml**: see [Updating an older config.yml](03_configurationFile.md#updating-an-older-configyml): keys no longer read, `1H` becomes `1h`, `project.split` and `outputs.file.downscaled_pt` are gone.
3. **Delete `outputs/ds_solar.nc`**: the solar geometry is now averaged over the accumulation window and stored as float32. An old file prints a warning.
4. **Delete `outputs/ds_param.nc`** to get the new sky view factor (median change about 0.003 from topocalc). An existing file is kept as is.
5. **Horizons**: `outputs/da_horizon.nc` now holds the horizons of the points, `(azimuth, point_name)`. A full-grid file of 0.3 is recomputed automatically.
6. **Outputs**: the downscaled points are in `outputs/downscaled/down.zarr` instead of one netcdf file per point. Read them with `mp.downscaled_pts` or `topo_utils.read_downscaled()`; the exporters read the store.
7. **Results change**: shortwave (elevation, timing, cluster aspect, top-of-atmosphere irradiance, hours in which the sun crosses the horizon), `interp_method: linear` (now bilinear) and precipitation at large elevation differences (factor bounded within [1/5, 5]). Re-run the downscaling before comparing with results of 0.3.
8. **ERA5**: `mp.get_era5()` writes `inputs/climate/ERA5.zarr` directly. Data already in `yearly/` or `daily/` are read into it instead of being downloaded again; the folders can be deleted afterwards. `retrieve_era5()` and `retrieve_era5_gdex()` are replaced by `fetch_era5.fetch_era5_to_zarr()`. Relative humidity on pressure levels is no longer downloaded (pass `varoi_plev` to `FetchERA5` to get it), and `rm_daily` and `cds_output_format` are no longer read.
