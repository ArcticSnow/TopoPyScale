# Release notes

## 1.0.0

TopoPyScale 1.0 checks the physics of its outputs, fixes the errors those checks found, and replaces the slow or fragile parts of the workflow.

### Highlights

- **Physics-consistency checks** of the downscaled outputs (`mp.check_physics()`, `toposcale.physics_checks`, see [Physics Checks](09_physics_checks.md)), and a test suite running the code on synthetic ERA5 with known answers.
- **Corrected physics**: elevation scaling and timing of the shortwave, terrain shading scaled by the sunlit part of each hour, aspect of clusters, rain/snow partition, bilinear interpolation, bounded precipitation factor, units, humidity precision and the conversions of the exporters.
- **Zarr downscaling engine**, vectorized over blocks of points: one year of 2,400 points in 13 s on 2 cores, written to one store, `outputs/downscaled/down.zarr`.
- **Terrain without topocalc**: sky view factor and horizon maps by a line sweep (numba kernel, 11 s for the Davos DEM on one core), horizons of the points along straight rays from their exact positions.
- **Surface-effect correction** of temperature and humidity (REDCAPP, DReaMIT), off by default, and `surface_calib` to fit it for a new region (see [Features](07_features.md#surface-effect-correction-of-air-temperature)).
- **ERA5 straight to zarr**: one CDS request per month instead of per day, at most 10 at a time, each file checked and written once to `ERA5.zarr` by the main thread, failed downloads retried, and reruns fetching only what is missing. No daily or yearly files, no CDO (see [Features](07_features.md#era5-download)).
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
