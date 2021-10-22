# TopoPyScale
Pure Python version of Toposcale packaged as a Pypi library. Toposcale is an original idea of Joel Fiddes to perform topography-based downscaling of climate data to the hillslope scale.

**References:**
- 

Contributor to this current version (2021)
- Simon Filhol
- Joel Fiddes
- Kristoffer Aalstad

## TODO

- class `topoclass`:
    - [x] create project folder structure if not existing
    - [x] read config file
    - [x] get ERA5 data
    - [ ] write save output in the various format version
    - [ ] fetch other dataset routines
    - [ ] implement routine to fecth DEM automatically from ArcticDEM/ASTER
    - [ ] add plotting routines.
- `topo_sub.py`:
  - [ ]look into alternative clustering method DBscan
  - [x] implement minibatch kmean for large DEM
- [x] simplify `fetch_era5.py` routines. Code seems over complex
- [ ] write download routine for CORDEX climate projection
- `topo_scale.py`:
  - [ ] do it all
- `fetch_era5.py`:
  - [x] swap `parallel` for [`from multiprocessing import Pool`](https://docs.python.org/3/library/multiprocessing.html)  when launch downloads simulstaneously
  - [ ] figure out in which case to use other function tpmm and other? how to integrate them?   

- **Documentation**:
  - [ ] Complete `README.md` file
  - [ ] create small examples from a small dataset avail in the library
  - [ ] If readme is not enough, then make a ReadTheDoc 
  - [ ] make sure all functions and class have explicit docstring

## Design

1. Inputs
    - Climate data from reanalysis (ERA5, etc)
    - Climate data from future projections (CORDEX)
    - DEM from local source, or fetch from public repository: SRTM, ArcticDEM
2. Run TopoScale
    - compute derived values (from DEM)
    - toposcale (k-mean clustering)
    - interpolation (bilinear, inverse square dist.) (interpolation optional only if output as gridded data)
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
    - random forest temporal gap filling

Validation (4) and Gap filling (4) are future implementation.

## Installation

```bash
conda create -n downscaling
conda activate downscaling
conda install ipython numpy pandas xarray matplotlib netcdf4 ipykernel scikit-learn rasterio gdal
pip install cdsapi
pip install h5netcdf
pip install topocalc
pip install pvlib
pip install elevation

cd github  # navigate to where you want to clone TopoPyScale
git clone git@github.com:ArcticSnow/TopoPyScale.git
pip install -e TopoPyScale    #install a development version

#----------------------------------------------------------
#            OPTIONAL: if using jupyter lab

# if you want to add this Python kernel to your jupyter lab
python -m ipykernel install --user --name downscaling
```

Then you need to setup your `cdsapi` with the Copernicus API key system. Follow [this tutorial](https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key) after creating an account with [Copernicus](https://cds.climate.copernicus.eu/). On Linux, create a file `nano ~/.cdsapirc` with inside:

```
url: https://cds.climate.copernicus.eu/api/v2
key: {uid}:{api-key}
```

## Basic usage

1. Setup your Python environment
2. Create your project directory
3. Configure the file `config.ini` to fit your problem
4. Run TopoPyScale

```python
from TopoPyScale import topoclass as tc
from matplotlib import pyplot as plt

config_file = './config.ini'
mp = tc.Topoclass(config_file)

mp.compute_dem_param()
# Compute clustering of the input DEM
mp.clustering_dem()

# plot clusters
mp.toposub.plot_clusters_map()
# plot sky view factor
mp.toposub.plot_clusters_map(var='svf', cmap=plt.cm.viridis)

# compute solar geometry and horizon angles
mp.compute_solar_geometry()
mp.compute_horizon()


mp.toposcale()
mp.to_netcdf()
```

## Principle of TopoPyScale
