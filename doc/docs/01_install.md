# Installation

## Python Environment Preparation
`TopoPyScale` is tested for Python 3.9. You may create a new virtual environment using conda prior to installation.


**Option 1 (recommended):**
```bash
conda create -n downscaling python ipython
conda activate downscaling

# Recomended way to install dependencies:
conda install -c conda-forge xarray matplotlib scikit-learn pandas numpy netcdf4 h5netcdf rasterio pyproj dask geopandas

# install forked version of Topocalc compatible with Python >3.9 (tested with 3.13)
pip install git+https://github.com/ArcticSnow/topocalc
```


## Latest Release Installation

```bash
pip install topopyscale
```

TopoPyScale has been tested on the OS:
- [x] Linux Ubuntu
- [x] MacOS
- [ ] Windows

## Setting up `cdsapi`

Then you need to setup your `cdsapi` with the Copernicus API key system. Follow [this tutorial](https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key) after creating an account with [Copernicus](https://cds.climate.copernicus.eu/). On Linux, create a file `nano ~/.cdsapirc` with inside:

```
url: https://cds.climate.copernicus.eu/api/v2
key: {uid}:{api-key}
```

## Setting up Google Cloud authentification

In order to use the routine to download ERA5 data from the [Google Cloud Storage](https://console.cloud.google.com/marketplace/product/bigquery-public-data/arco-era5?invt=AbudXg&project=era5access), you will need to setup your system with the `google-cloud-cli`. Follow Google's instructions: https://cloud.google.com/sdk/docs/install


## Development version Installation

Go to [Development page](./08_Development.md)

## Install Pvlib with SPA C-implementation
By default, TopoPyScale uses the Python Numpy implementatino of Pvlib to compute solar position. The Pvlib library offers multiple method to compute the solar position, with one using a C-implementation. Because of licensing, the C files are not distributed within the Pvlib library. This requries a special manipulation to execute:

1. Clone the Pvlib repository from github: `git clone git@github.com:pvlib/pvlib-python.git`
2. uninstall the pvlib library previously installed: `pip uninstall pvlib`
3. Go download the `spa.c` and `spa.h` from https://midcdmz.nrel.gov/spa/#register
4. Place the files into the folder: `pvlib-python/pvlib/spa_c_files/`
5. then :
```sh
# Install cython:
pip install cython
cd pvlib-python/pvlib/spa_c_files/
python setup.py build_ext --inplace
cd ../../..
pip install .

# test if installation is working correctly:
pyhton pvlib-python/pvlib/spa_c_files/spa_py_example.py
```

## Install FSM2oshd

TopoPyScale incorporates an export option for the snow-canopy model FSM2oshd. This function requires elements that are outside the current scope of TopoPyScale. Missing elements are all the forest structural parameters that require the model CANRAD as well as a number of additional external dataset about forest properties. Nevertheless, the installation of FSM2oshd is as follow:
```sh
git clone git@github.com:ArcticSnow/FSM2oshd.git
cd FSM2oshd
sh compile_oshd_txt.sh
mv FSM_OSHD path/to/my/project
```
