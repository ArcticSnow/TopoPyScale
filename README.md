# TopoPyScale
New version of Toposcale packaged as a Pypi library. Toposcale is an original idea of Joel Fiddes. 

**References:**
- 

Contributor to this current version (2021)
- Simon Filhol
- Joel Fiddes
- Kristoffer Aalstad

## Design

1. Inputs
    - Climate data from ERA5, CLIM
    - DEM from local source, SRTM, ArcticDEM
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
4. Validation toolset
    - validation to local observation timeseries
    - plotting
5. Gap filling algorithm
    - random forest temporal gap filling
    
## Installation

1. 
```bash
conda create -n downscaling
conda activate downscaling
conda install ipython numpy pandas xarray matplotlib netcdf4
pip install h5netcdf
pip install topocalc
pip install pvlib

cd github  # navigate to where you want to clone TopoPyScale
git clone git@github.com:ArcticSnow/TopoPyScale.git
pip install -e TopoPyScale    #install a development version
```

## Basic usage

## Principle of TopoScale