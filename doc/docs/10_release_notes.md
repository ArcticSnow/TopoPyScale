# Release notes

## 1.0.0

TopoPyScale 1.0 checks the physics of its outputs, fixes the errors those checks found, and replaces the slow or fragile parts of the workflow.

### Highlights

- **Physics-consistency checks** of the downscaled outputs (`mp.check_physics()`, `toposcale.physics_checks`, see [Physics Checks](09_physics_checks.md)), and a test suite running the code on synthetic ERA5 with known answers.
- **Corrected physics**: elevation scaling and timing of the shortwave, terrain shading scaled by the sunlit part of each hour, aspect of clusters, rain/snow partition, bilinear interpolation, bounded precipitation factor, units, humidity precision and the conversions of the exporters.
- **Zarr downscaling engine**, vectorized over blocks of points: one year of 2,400 points in 13 s on 2 cores, written to one store, `outputs/downscaled/down.zarr`.
- **Terrain without topocalc**: sky view factor and horizon maps by a line sweep (numba kernel, 11 s for the Davos DEM on one core), horizons of the points along straight rays from their exact positions.
- **Surface-effect correction** of temperature and humidity (REDCAPP, DReaMIT), off by default, and `surface_calib` to fit it for a new region (see [Features](07_features.md#surface-effect-correction-of-air-temperature)).
- **ERA5 from NCAR GDEX** as an alternative to the CDS (`climate.era5.data_repository: gdex`).

### Upgrading from 0.3

1. **Environment**: topocalc is no longer needed. numba, scipy, pyyaml and joblib are new dependencies, installed with TopoPyScale.
2. **config.yml**: see [Updating an older config.yml](03_configurationFile.md#updating-an-older-configyml): keys no longer read, `1H` becomes `1h`, `project.split` and `outputs.file.downscaled_pt` are gone.
3. **Delete `outputs/ds_solar.nc`**: the solar geometry is now averaged over the accumulation window and stored as float32. An old file prints a warning.
4. **Delete `outputs/ds_param.nc`** to get the new sky view factor (median change about 0.003 from topocalc). An existing file is kept as is.
5. **Horizons**: `outputs/da_horizon.nc` now holds the horizons of the points, `(azimuth, point_name)`. A full-grid file of 0.3 is recomputed automatically.
6. **Outputs**: the downscaled points are in `outputs/downscaled/down.zarr` instead of one netcdf file per point. Read them with `mp.downscaled_pts` or `topo_utils.read_downscaled()`; the exporters read the store.
7. **Results change**: shortwave (elevation, timing, cluster aspect, top-of-atmosphere irradiance, hours in which the sun crosses the horizon), `interp_method: linear` (now bilinear) and precipitation at large elevation differences (factor bounded within [1/5, 5]). Re-run the downscaling before comparing with results of 0.3.
