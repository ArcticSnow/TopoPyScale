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
    - Climate data from reanalysis (ERA5, etc)
    - Climate data from future projections ()
    - DEM from local source, or fetch from public repository: SRTM, ArcticDEM
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
conda install ipython numpy pandas xarray matplotlib netcdf4 ipykernel
pip install cdsapi
pip install h5netcdf
pip install topocalc
pip install pvlib

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

## Principle of TopoScale