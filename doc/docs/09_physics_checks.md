# Physics Consistency Checks

The downscaling chain applies one step after another: horizontal interpolation of ERA5, vertical interpolation to the point elevation, partitioning of the radiation, terrain corrections and export to model formats. Each step can break a basic physical constraint without raising an error. Examples are humidity above saturation, direct beam in terrain shadow, precipitation created by an unbounded elevation factor, accumulated fluxes out of phase with the sun, and unit attributes that disagree with the values.

`TopoPyScale` has two tools to detect such problems:

- **`topo_checks.py`** checks the outputs of a project run and writes a report. It never modifies the outputs.
- **The test suite in `tests/`** runs the downscaling functions on synthetic ERA5 fields whose correct output is known. It protects the code itself against regressions.

## Running the checks on a project

### From the configuration file

Add `physics_checks` to the `toposcale` section of the configuration file. The checks then run at the end of `mp.downscale_climate()`:

```yaml
toposcale:
  interpolation_method: idw
  LW_terrain_contribution: True
  physics_checks: True        # True: report only, 'error' or 'warning': stop when a check fails at that level
```

### From Python

```python
from TopoPyScale import topoclass as tc

mp = tc.Topoclass('config.yml')
# ... usual workflow up to mp.downscale_climate()

report = mp.check_physics()                                   # report only
report = mp.check_physics(strict='error')                     # raise if any check has status 'error'
report = mp.check_physics(snow_partition_method='continuous') # also probe the rain/snow partition method
report = mp.check_physics(points=['001', '014'])              # subset of points
```

`check_physics()` reads the downscaled points, the solar geometry, the cluster centroids and `ds_param`. It saves the report to `outputs/physics_checks.csv` and prints a summary of the checks that did not pass.

The module can also be used without a `Topoclass` object, for instance on outputs of a previous run:

```python
import pandas as pd
import xarray as xr
from TopoPyScale import topo_checks as tc
from TopoPyScale import topo_utils as tu

ds_down = tu.read_downscaled('outputs/downscaled/down.zarr')
report = tc.run_all_checks(ds_down,
                           ds_solar=xr.open_dataset('outputs/ds_solar.nc'),
                           df_centroids=pd.read_pickle('outputs/df_centroids.pck'),
                           ds_param=xr.open_dataset('outputs/ds_param.nc'))
print(tc.summarize(report))
tc.raise_on_failure(report, level='error')
```

The optional inputs widen the set of checks: `ds_solar` enables the shortwave geometry and timing checks, `df_centroids` the checks across points and on the clusters, and `ds_param` the DEM orientation check and the area weights of the precipitation budget. A check whose inputs are missing is reported with status `info` and a note starting with `skipped:`. Older zarr outputs with a `point_ind` dimension are supported when `df_centroids` is given.

### Reading the report

The report is a table with one row per check and variable:

| Column | Meaning |
|---|---|
| `check` | name of the check (see the list below) |
| `variable` | variable checked |
| `status` | `pass`, `info`, `warning` or `error` |
| `n_fail`, `n_total`, `frac_fail` | number and fraction of values (point × time step) failing the check |
| `worst` | worst value found, in the unit given in `note` (for diagnostics such as `sw_timing`, the diagnostic value itself) |
| `note` | what was tested |

`error` marks a violation of a physical constraint or of a convention of the code. `warning` marks a value that is physically possible but suspicious. Example of a summary:

```text
Physics checks: error: 1 | warning: 1 | info: 1 | pass: 62
          check variable  status  n_fail  n_total  frac_fail        worst                                                         note
     units_attr        p   error       1        1        1.0 80601.624440 declared 'hPa' but values (median 8.06e+04) look like ['Pa']
supersaturation        q warning     432      432        1.0     1.128541          RH w.r.t. water > 1.02 (q_2_rh clips this silently)
```

## List of checks

### Inputs, metadata and time axis

| Check | Status if failing | Tests that |
|---|---|---|
| `dem_orientation` | error | the DEM is north-up (`y` decreasing with row index). Aspect (from `gradient_d8`) assumes it; otherwise north and south are swapped |
| `centroid_aspect` | error | for clusters steeper than 5°, `aspect` agrees within 10° with `atan2(aspect_sin, aspect_cos)`, the circular mean carried by the clustering features (within 45° for clusters with spread aspects, a centre shorter than 0.7: mini-batch k-means centres are not exactly the mean of their pixels) |
| `time_monotonic`, `time_regular` | error, warning | time steps are strictly increasing and regular |
| `tp_units_vs_timestep` | error | the `units` of `tp` agree with the time step (e.g. not `mm hr**-1` for 3 h data) |
| `finite` | error | no NaN or inf in any variable |
| `range` | error | each variable lies within physical bounds (table below) |
| `units_attr` | error or warning | the `units` attribute of `t`, `p`, `q`, `wd`, `SW`, `SW_direct`, `SW_diffuse` and `LW` is consistent with the magnitude of the values |

Default bounds of the `range` check, in the units written by `TopoPyScale`. They can be changed with `run_all_checks(..., bounds={...})`:

| Variable | Unit | Bounds |
|---|---|---|
| `t` | K | 180 – 335 |
| `p` | Pa | 25 000 – 110 000 |
| `q` | kg kg⁻¹ | 0 – 0.04 |
| `u`, `v` | m s⁻¹ | −80 – 80 |
| `ws` | m s⁻¹ | 0 – 80 |
| `wd` | rad | 0 – 2π |
| `SW`, `SW_direct`, `SW_diffuse` | W m⁻² | 0 – 1418 (1370 × 1.035) |
| `LW` | W m⁻² | 40 – 650 |
| `tp` | mm per time step | 0 – 200 mm h⁻¹ × time step |
| `precip_lapse_rate` | – | ≥ 0 |

### Thermodynamics and wind

| Check | Status if failing | Tests that |
|---|---|---|
| `q_non_negative` | error | `q ≥ 0` |
| `supersaturation` | warning | relative humidity with respect to water, computed without clipping, stays below 1.02. `meteo_util.q_2_rh` clips RH to 1, which hides supersaturation in the exported files |
| `hypsometric` | warning | between each point and the lowest point, `|ln(p/p_ref) + g Δz / (R T̄)| ≤ 0.01`, with `T̄` the mean temperature of the two points. Horizontal pressure gradients between ERA5 cells are part of the residual |
| `lapse_rate` | warning | the temperature lapse rate across points (least-squares fit per time step) is not super-adiabatic: not steeper than the dry adiabatic −9.8 K km⁻¹, with 0.5 K km⁻¹ tolerance. Needs at least 3 points spanning 300 m |
| `ws_equals_hypot_uv` | error | `ws = sqrt(u² + v²)` |
| `wd_matches_uv` | error | `wd = atan2(−u, −v)` modulo 2π, i.e. the direction the wind blows from, clockwise from north, in radians |

### Shortwave radiation

| Check | Status if failing | Tests that |
|---|---|---|
| `sw_closure` | error | `SW = SW_direct + SW_diffuse` (1 W m⁻² tolerance) |
| `direct_with_sun_down` | error | no direct beam when the sun is below the astronomical horizon |
| `sw_at_night` | error | no SW when the sun is down over the whole accumulation window |
| `direct_beam_bound` | error | `SW_direct ≤ S0 · cos(i)` with `i` the illumination angle on the slope: the beam cannot exceed the top-of-atmosphere beam |
| `direct_in_terrain_shadow` | error | no direct beam when the sun is behind the local horizon over the whole accumulation window (at the previous, current and next time steps). Within the window, the direct beam is scaled by its sunlit fraction |
| `sw_timing` | warning | the daily centroid of SW (in local solar time, for points with slope < 10°) matches the one of the top-of-atmosphere irradiance within 15 min. `worst` is the median offset in minutes; positive means SW lags the sun |

ERA5 `ssrd` at time `t` is the accumulation over `(t − 1 h, t]`, whatever the time step of the downloaded data. The solar geometry is therefore computed at the centre of this window (`t − 30 min`). The length of the window is set by `climate.<source>.accumulation_window` (default `1h`). The top-of-atmosphere irradiance `SWtoa`, used for the clearness index, is the mean over the same window, including the Earth–Sun distance (see [Features](07_features.md)). A solar file `ds_solar.nc` computed by an older version of `TopoPyScale` uses the time stamps; `sw_timing` then reports a lag of about +30 min, and `compute_solar_geometry()` prints a warning. Delete the file to recompute it.

### Longwave radiation

| Check | Status if failing | Tests that |
|---|---|---|
| `lw_emissivity_upper` | warning | `LW ≤ 1.02 · σ max(T, 273.15 K)⁴`: sky and terrain emissivities do not exceed 1 (273.15 K is the temperature of the surrounding terrain in the terrain term) |
| `lw_emissivity_lower` | warning | the implied emissivity `LW / (σ T⁴)` is above `0.4 · svf` |

### Precipitation

| Check | Status if failing | Tests that |
|---|---|---|
| `tp_non_negative` | error | `tp ≥ 0` |
| `precip_factor_positive` | error | the elevation factor `precip_lapse_rate` is finite and positive |
| `precip_factor_bounded` | warning | the elevation factor is ≤ 5 |
| `era5_tp_non_negative` | error | `tp / precip_lapse_rate`, the horizontally interpolated ERA5 precipitation, is ≥ 0 |
| `domain_precip_budget` | warning | the area-weighted precipitation over the domain divided by the interpolated ERA5 precipitation is between 0.67 and 1.5. Weights are the number of DEM pixels of each cluster (from `ds_param`), or uniform |

The elevation factor `(1 + c Δz) / (1 − c Δz)` of MicroMet [@listonMicroMet2006], with `c` between 0.20 and 0.35 km⁻¹ depending on the month, diverges when `c Δz` approaches 1 (Δz ≈ 2.9 km in winter) and becomes negative beyond. `TopoPyScale` keeps the factor within `[1/max, max]`. Set `max` with `climate.precip_lapse_rate_max_factor` (default 5). When the precipitation lapse rate is off, the budget ratio is exactly 1.

### Rain/snow partition

These checks probe `meteo_util.partition_snow` itself with synthetic inputs, for the method(s) given in `snow_partition_method`:

| Check | Method | Tests that |
|---|---|---|
| `partition_closure` | all | `rain + snow = precipitation` |
| `partition_bounds`, `partition_monotonic` | `continuous` | the snow fraction is between 0 and 1 and does not increase with temperature |
| `partition_warm_humid`, `partition_cold_humid` | `jennings2018_*` | precipitation is mostly rain at +4 °C and mostly snow at −4 °C, with RH 95 % passed in % as the exporters do [@jennings2018] |

## Conventions the checks rely on

| Quantity | Convention |
|---|---|
| `t` | K |
| `p` | Pa |
| `q`, `w` | kg kg⁻¹, not rounded |
| `wd` | rad, direction the wind blows from, clockwise from north |
| `tp` | mm per time step, accumulated over the time step ending at the time stamp |
| `SW`, `SW_direct`, `SW_diffuse`, `LW` | W m⁻², mean over the accumulation window ending at the time stamp |
| `slope`, `aspect` | rad. Aspect 0 = south, +π/2 = east, −π/2 (or 3π/2) = west, the convention of the solar azimuth in `ds_solar` (`180° − pvlib azimuth`) |
| horizon angles | rad, azimuth bins in degrees with the same convention as the aspect |
| relative humidity passed to `partition_snow` | % (0–100), i.e. `100 * q_2_rh(t, p, q)` |

The exporters convert precipitation to rates with the actual time step of the data.

## What to do when a check fails

- **`units_attr`, `range` or `finite` errors** usually come from a modified or outdated version of the code, or from data written by another tool. Compare with the conventions above.
- **`sw_timing`**: delete `outputs/ds_solar.nc` and rerun `compute_solar_geometry()` and `downscale_climate()`. For a forcing other than ERA5, set `accumulation_window` to its accumulation period.
- **`supersaturation` warnings** of a few percent can come from interpolating `q` linearly between pressure levels, since saturation humidity is not linear in temperature. Larger values point to inconsistent inputs.
- **`hypsometric` warnings** are expected when points lie in different ERA5 cells with a strong synoptic pressure gradient. They indicate a problem when they occur for points within one ERA5 cell.
- **`domain_precip_budget` or `precip_factor_bounded` warnings** show how much water the precipitation lapse rate adds or removes compared with ERA5. Large ratios suggest lowering `precip_lapse_rate_max_factor` or turning the lapse rate off. ERA5 already resolves part of the orographic enhancement at its own resolution.
- **`centroid_aspect` or `dem_orientation` errors** affect the illumination angle and the shading of every time step. Recompute `ds_param.nc` and the clusters with a north-up DEM.

## Test suite

The folder `tests/` runs the actual downscaling code on synthetic ERA5 fields built in `tests/synthetic_era5.py`: the zarr engine (`topo_scale_zarr.py` with the array functions of `toposcale_core.py`). The synthetic atmosphere is hydrostatic with a constant lapse rate, a diurnal cycle and a known relative humidity. Accumulated fluxes are integrated minute by minute over the hour preceding each time stamp, as in ERA5. The correct downscaled values are therefore known analytically.

```bash
pip install pytest
pytest tests -q
```

The suite covers:

- **Null downscaling**: a flat, unobstructed point at an ERA5 node and at the ERA5 elevation must reproduce the ERA5 surface pressure, temperature, precipitation and longwave, and the daily shortwave energy within 0.1 %.
- **Interpolation**: no NaN when a point sits on an ERA5 node; the nearest node dominates the weights; bilinear weights reproduce a linear field exactly; temperature and pressure follow the analytical profile at altitude; humidity keeps its precision in cold, dry air.
- **Radiation**: the direct beam increases with altitude (Beer's law with the ERA5 surface pressure, as in TopoSCALE [@fiddes2014]); SW closure, beam bound, no SW at night and no direct beam behind the horizon on steep slopes; SW in phase with the sun; LW emissivity bounds.
- **Precipitation**: the elevation factor is bounded, monotonic and symmetric (`f(−Δz) = 1/f(Δz)`); the rain/snow partition is closed and monotonic.
- **DEM and clusters**: the aspect and horizon conventions match the solar azimuth; cluster aspects are circular means (on a synthetic cone DEM).
- **Horizons and sky view factor** (`tests/test_horizon_points.py`, `tests/test_svf.py`): rays are exact on a tilted plane and a wall, and equal an exhaustive search along the grid axes; no streaks at azimuths a few degrees off the axes; the line sweep equals the rays along the axes and diagonals and stays within 0.05° of them for 95% of the cells elsewhere; slope and aspect of a plane; svf = 1 on flat ground and (1 + cos S)/2 on a plane; svf within 0.005 of exact rays on a rough synthetic DEM, independent of the number of processes and of the raster orientation; no-data handling.
- **Exports**: FSM conserves precipitation with 3 h data and receives RH in % for the Jennings methods; SNOWPACK gets wind direction in degrees and ISO time stamps.
- **Zarr pipeline** (`tests/test_zarr_pipeline.py`): the netcdf-to-zarr conversion (including the new CDS naming and the ERA5/ERA5T `expver` dimension), identical results whatever the block sizes and number of threads, the layout of the output store, an error for points at the edge of the ERA5 domain, and the `Topoclass` workflow from yearly netcdf files to `down.zarr`.
- **Report**: `run_all_checks` reports no failure on a consistent multi-point run.

To document a constraint that the code is known to violate, decorate its test with `known_violation('reason')` (a strict `xfail`). The suite then passes while the issue is open and fails as soon as the issue is fixed, as a reminder to remove the marker.
