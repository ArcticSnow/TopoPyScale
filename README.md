<a style="border-width:0" href="https://doi.org/10.21105/joss.05059">
  <img src="https://joss.theoj.org/papers/10.21105/joss.05059/status.svg" alt="DOI badge" ></a>
  
[![DOI](https://zenodo.org/badge/411249045.svg)](https://zenodo.org/badge/latestdoi/411249045)
[![GitHub license](https://img.shields.io/github/license/ArcticSnow/TopoPyScale)](https://github.com/ArcticSnow/TopoPyScale/blob/main/LICENSE)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/ArcticSnow/TopoPyScale)
[![][docs-dev-img]][docs-dev-url]
[![PyPI Downloads](https://static.pepy.tech/badge/topopyscale)](https://pepy.tech/projects/topopyscale)
![Test](https://github.com/ArcticSnow/TopoPyScale/actions/workflows/test_topopyscale.yml/badge.svg)

Binder Notebooks Examples: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ArcticSnow/TopoPyScale_examples/HEAD)

[docs-dev-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-dev-url]: https://topopyscale.readthedocs.io

# TopoPyScale
Python version of Toposcale packaged as a Pypi library. Toposcale is an original idea of Joel Fiddes to perform topography-based downscaling of climate data to the hillslope scale.

Documentation avalaible: https://topopyscale.readthedocs.io

![](https://github.com/ArcticSnow/TopoPyScale/blob/main/JOSS/temperature_comparison_crop_scaled.jpg)

**References:**

- Filhol et al., (2023). TopoPyScale: A Python Package for Hillslope Climate Downscaling. Journal of Open Source Software, 8(86), 5059, https://doi.org/10.21105/joss.05059

And the original method it relies on:
- Fiddes, J. and Gruber, S.: TopoSCALE v.1.0: downscaling gridded climate data in complex terrain, Geosci. Model Dev., 7, 387–405, https://doi.org/10.5194/gmd-7-387-2014, 2014.
- Fiddes, J. and Gruber, S.: TopoSUB: a tool for efficient large area numerical modelling in complex topography at sub-grid scales, Geosci. Model Dev., 5, 1245–1257, https://doi.org/10.5194/gmd-5-1245-2012, 2012. 

Kristoffer Aalstad has a Matlab implementation: https://github.com/krisaalstad/TopoLAB

## Contribution Workflow
**Please follow these simple rules:**
1. a bug -> fix it! 
2. an idea or a bug you cannot fix? -> create a new [issue](https://github.com/ArcticSnow/TopoPyScale/issues) if none doesn't already exist. If one exist, then add material to it.
3. wanna develop a new feature/idea? -> create a new branch. Go wild. Merge with main branch when accomplished.
4. Create release version when significant improvements and bug fixes have been done. Coordinate with others on [Discussions](https://github.com/ArcticSnow/TopoPyScale/discussions)

**Create a new release:**
Follow procedure and conventions described in: https://www.youtube.com/watch?v=Ob9llA_QhQY

Our forum is now on [Github Discussions](https://github.com/ArcticSnow/TopoPyScale/discussions). Come visit!


## Design

1. Inputs
    - Climate data from reanalysis (ERA5, etc)
    - Climate data from future projections (CORDEX) (TBD)
    - DEM from local source, or fetch from public repository: SRTM, ArcticDEM, ASTER
2. Run TopoScale
    - compute derived values (from DEM)
    - toposcale (k-mean clustering)
    - interpolation (bilinear, inverse square dist.)
3. Output
    - Cryogrid format
    - FSM format
    - CROCUS format
    - Snowmodel format
    - basic netcfd
    - For each method, have the choice to output either the abstract cluster points, or the gridded product after interpolation
4. Validation toolset
    - validation to local observation timeseries
    - plotting
5. Gap filling algorithm
    - random forest temporal gap filling (TBD)

Validation (4) and Gap filling (4) are future implementation.

## Installation

We have now added an environments.yml file to handle versions of depencencies that are tested with the current codebase, to use this run:

`conda env create -f environment.yml`

Alternatively you can follow this method for dependencies (to be deprecated):

```bash
conda create -n downscaling python=3.9 ipython
conda activate downscaling

# Recomended way to install dependencies:
conda install -c conda-forge xarray matplotlib scikit-learn pandas numpy netcdf4 h5netcdf rasterio pyproj dask rioxarray
```

Then install the code:

```
# OPTION 1 (Pypi release):
pip install TopoPyScale

# OPTION 2 (development):
cd github  # navigate to where you want to clone TopoPyScale
git clone git@github.com:ArcticSnow/TopoPyScale.git
pip install -e TopoPyScale    #install a development version

#----------------------------------------------------------
#            OPTIONAL: if using jupyter lab
# add this new Python kernel to your jupyter lab PATH
python -m ipykernel install --user --name downscaling

# Tool for generating documentation from code docstring
pip install lazydocs
```

Then you need to setup your `cdsapi` with the Copernicus API key system. Follow [this tutorial](https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key) after creating an account with [Copernicus](https://cds.climate.copernicus.eu/). On Linux, create a file `nano ~/.cdsapirc` with inside:

```
url: https://cds.climate.copernicus.eu/api/v2
key: {uid}:{api-key}
```

## Basic usage

1. Setup your Python environment
2. Create your project directory
3. Configure the file `config.ini` to fit your problem (see [`config.yml`](https://github.com/ArcticSnow/TopoPyScale_examples/blob/main/ex1_norway_finse/config_spatial.yml) for an example)
4. Run TopoPyScale

```python
import pandas as pd
from TopoPyScale import topoclass as tc
from matplotlib import pyplot as plt

# ========= STEP 1 ==========
# Load Configuration
config_file = './config.yml'
mp = tc.Topoclass(config_file)
# Compute parameters of the DEM (slope, aspect, sky view factor)
mp.compute_dem_param()

# ========== STEP 2 ===========
# Extract DEM parameters for points of interest (centroids or physical points)

mp.extract_topo_param()

# ----- Option 1:
# Compute clustering of the input DEM and extract cluster centroids
#mp.extract_dem_cluster_param()
# plot clusters
#mp.toposub.plot_clusters_map()
# plot sky view factor
#mp.toposub.plot_clusters_map(var='svf', cmap=plt.cm.viridis)

# ------ Option 2:
# inidicate in the config file the .csv file containing a list of point coordinates (!!! must same coordinate system as DEM !!!)
#mp.extract_pts_param(method='linear',index_col=0)

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

TopoClass will create a file structure in the project folder (see below). TopoPyScale assumes you have a DEM in GeoTiFF, and a set of climate data in netcdf (following ERA5 variable conventions). 
TopoPyScale can easier segment the DEM using clustering (e.g. K-mean), or a list of predefined point coordinates in `pts_list.csv` can be provided. Make sure all parameters in `config.ini` are correct.
```
my_project/
    ├── inputs/
        ├── dem/ 
            ├── my_dem.tif
            └── pts_list.csv  (optional)
        └── climate/
            ├── PLEV*.nc
            └── SURF*.nc
    ├── outputs/
    └── config.ini
```
