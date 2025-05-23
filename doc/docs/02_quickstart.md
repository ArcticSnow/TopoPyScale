# Quick Start

## Basic usage

1. Setup your Python environment
2. Create your project directory
3. Configure the file `config.yml` to fit your problem (see [`config.yml`](./03_configurationFile.md) for an example)
4. Run TopoPyScale

```python
import pandas as pd
from TopoPyScale import topoclass as tc
from matplotlib import pyplot as plt

# ========= STEP 1 ==========
# Load Configuration
config_file = './config.yml'
mp = tc.Topoclass(config_file)
mp.get_era5()

# ======== STEP 2 ===========
# Compute parameters of the DEM (slope, aspect, sky view factor)
mp.compute_dem_param()
mp.extract_topo_param()

# ========= STEP 3 ==========
# compute solar geometry and horizon angles
mp.compute_solar_geometry()
mp.compute_horizon()

# ========= STEP 4 ==========
# Perform the downscaling
mp.downscale_climate()

# ========= STEP 5 ==========
# explore the downscaled dataset. For instance the temperature difference between each point and the first one
(mp.downscaled_pts.t-mp.downscaled_pts.t.isel(point_id=0)).plot()
plt.show()

# ========= STEP 6 ==========
# Export output to desired format
mp.to_netcdf()
```

TopoClass will create a file structure in the project folder (see below). `TopoPyScale` assumes you have a DEM in GeoTiFF, and a set of climate data in netcdf (following ERA5 variable conventions). 
TopoPyScale can either segment the DEM using clustering (e.g. K-means), or use a list of predefined point coordinates in `pts_list.csv`. Make sure all parameters in `config.yml` are correct.
```
my_project/
    ├── inputs/
        ├── dem/ 
            ├── my_dem.tif
            └── pts_list.csv  (OPTIONAL: to downscale to specific points)
        └── climate/
            ├── PLEV*.nc
            └── SURF*.nc
    ├── outputs/
            ├── tmp/
    ├── pipeline.py (OPTIONAL: script for the downscaling instructions)
    └── config.yml
```

## Plotting

TopoPyScale incudes a number of plotting tools. Some are wrapped in the `topoclass` object:
```python

# To plot cluster map in the case clustering was used
mp.toposub.plot_clusters_map()

# To plot sky view factor or any other variable
mp.toposub.plot_clusters_map(var='svf', cmap=plt.cm.viridis)

# to plot a map version of a toposub downscaled output
mp.plot.map_variable()
```

However, the basic plotting functions are defined within the file [`topo_plot.py`](./TopoPyScale.topo_plot.md). So after having ran the downscaling in spatial (`toposub`) mode, run the following:
```python
from TopoPyScale import topo_plot as plot
import matplotlib.pyplot as plt

# to plot one single timetep
plot.map_variable(mp.downscaled_pts, mp.toposub.ds_param, time_step=100, var='t_surface')
plt.show()

# to plot the mean air temperature across the entire timeseries
plot.map_variable(mp.downscaled_pts.t.mean(dim='time').to_dataset(), mp.toposub.ds_param)
plt.title('Mean Air Temperature')
plt.show()
```

## Comparison to Observations

`TopoPyScale` includes tools to fetch weather station observations from public databases:

- [WMO](https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-surface-land?tab=overview)
- Norwegian meteoroligical Institue [MetNO FROST API](https://frost.met.no/index.html)

These functions are available into `topo_obs.py`. `topo_compare.py` includes functions to estimate and also correct bias between downscaled and a reference timeseries. 
