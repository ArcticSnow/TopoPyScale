# Output Formats

`TopoPyScale` includes a number of functions to export the downscaled timeseries to be compatible with some established energy and mass balance land-surface models for snow and permafrost physics, or hydrology. All these functions are available into the file `topo_export.py` and are integrated to the class `topoclass`. `TopoPyScale` also exports the topographical parameters it relies on to compute corrections in the file `ds_param.nc`, with the corresponding list of points (or clusters) in the file `df_centroids.pckl`. 

## Downscaled Timeseries

### Netcdf files
It is possible to export netcdf files for a list of given variables. Otherwise, by default all downscaled timeseries are available in `outputs/`

### FSM Format
`TopoPyScale` includes tools to interact and work with [FSM](https://github.com/RichardEssery/FSM). Those are available in the file `topo_sim.py`.

### Cryogrid Format
[Cryogrid](https://github.com/CryoGrid/CryoGrid) is a model used for simulating the thermal regime of the ground, particularly applied to permafrost and hydrological application.

### MuSa Format
[MuSa](https://github.com/ealonsogzl/MuSA) is an assimilation toolbox for snow modeling. `mp.to_musa()` writes a folder `outputs/musa_<layout>/` that MuSA reads as it is. Two layouts are available:

| `layout` | MuSA grid | coordinates (`forcing_dim_names`) | MuSA result file `cell_<i>_<j>.pkl.blp` |
|---|---|---|---|
| `grid` (default) | the DEM grid of `ds_param.nc`: every DEM cell gets the forcing of its TopoSUB cluster | `northing`, `easting` (DEM coordinates) | DEM cell at `northing[i]`, `easting[j]` of `mask.nc` |
| `points` | a list of the points or TopoSUB clusters | `y` = point id, `x` = [0] | `i`-th point, sorted by point id (`j` = 0) |

- `grid` is fully distributed and needs the `toposub` sampling method. DEM cells without a cluster get NaN forcing and are not simulated. `bbox=[xmin, xmax, ymin, ymax]`, `stride` and `cell_mask` (a boolean array on the DEM grid, e.g. a catchment or the footprint of observations) restrict the cells to simulate.
- `points` is the layout for point sampling, or to run each TopoSUB cluster once. The point id is the number in `point_name` (and in the `down_pt_<point_name>.nc` files): 0..N-1 for `toposub`, 1..N for `points` sampling. So with `toposub`, `cell_<i>_0` is cluster `i`.

A grid cell holds exactly the timeseries of its cluster in `points`. With the same forcing and perturbations and without observations, MuSA gives every cell of a cluster the same result, so `points` is much cheaper for open-loop runs: map its results back to the DEM with `point_name` in `ds_param.nc`. `grid` is for assimilating observations that differ between DEM cells.

Files in the folder:

- `forcing/YYYY-MM-DD.nc`: daily files with `SW`, `LW` (W m-2), `precip` (kg m-2 s-1), `t` (K), `RH` (%), `ws` (m s-1) and `p` (Pa), the units MuSA expects with its default `forcing_multiplier`. MuSA reads every `*.nc` file in this folder (`nc_forcing_path`), so keep other files out of it.
- `mask.nc`: 1 for the cells to simulate, NaN for the cells to skip (`nc_maks_path`). Set cells to NaN to run a subset.
- `DEM.nc` (`grid`): elevation on the same grid (`dem_path`).
- `points_meta.csv` (`points`): for each point the MuSA result file (`musa_cell`), the point id (`musa_y`), `musa_x` = 0, `point_name`, and its x, y, lon, lat, elevation, slope, aspect and svf.

The variables are in the units MuSA expects, so its `forcing_multiplier` and `forcing_offset` keep their defaults: `RH` is `100 * q_2_rh(t, p, q)` and `precip` is `tp` divided by the timestep in seconds. The matching settings in MuSA's `config.py` are:

```python
nc_forcing_path = "<path>/forcing/"
nc_maks_path = "<path>/mask.nc"
dem_path = "<path>/DEM.nc"  # grid, if MuSA needs the DEM
nc_dem_varname = "DEM"
forcing_var_names = {"SW_var_name": "SW", "LW_var_name": "LW", "Precip_var_name": "precip",
                     "Press_var_name": "p", "RH_var_name": "RH", "Temp_var_name": "t", "Wind_var_name": "ws"}
forcing_dim_names = {"lat_forz_var_name": "northing",  # "y" for points
                     "lon_forz_var_name": "easting",   # "x" for points
                     "time_forz_var_name": "time"}
date_ini, date_end, dt  # as returned by to_musa(): first and last timestep, timestep in seconds
```

MuSA also needs observation files on the same grid as `mask.nc`, even for an open-loop run. They are not written by `to_musa()`: set them with `nc_obs_path`, `obs_var_names` and `dates_obs`.

```python
meta = mp.to_musa()                           # grid, full DEM, all timesteps
mp.to_musa(bbox=[xmin, xmax, ymin, ymax], start='2020-01-01', end='2020-01-05 23:00')
mp.to_musa(layout='points')
```

### Snowmodel Format
Snowmodel is a snow model focused on wind redistribution. 

### CROCUS Format
[CROCUS](https://gmd.copernicus.org/articles/5/773/2012/gmd-5-773-2012.pdf) is a 1D physical model developed by Meteo France to simulate the evolution of snowpack state variables and snow metamorphic path.

### SURFEX Format
[Surfex](http://www.cnrm.meteo.fr/surfex/) is a land -surface model devellopped at Météo-France by the Centre Nationale de Recherche Mtéorologique (CNRM). CROCUS is a snow module within Surfex. 

### SNOWPACK Format
[SNOWPACK](https://www.slf.ch/en/services-and-products/snowpack.html) is a physical model developped at SLF, Switzerland to simulate the evolution a snowpack state variables and snow metamorphic path.

### GEOTOP Format
Geotop is a hydrological model

## Supplementary files

- `ds_param.nc`: this file includes elevation, slope, aspect (zero pointed to South), sky view fraction, and cluster map when using `toposub`.
- `df_centroids.pckl`: a pandas dataframe (saved as pickle) containing information relevant at the point scale.In case of `toposub`, this table is the list of the clusters' centroids.
- `ds_solar.nc`: timeseries of incoming SW radiations (top of atmosphere), and sun geometry
- `da_horizon.nc`: maps of horizon angles.


