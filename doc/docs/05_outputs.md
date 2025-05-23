# Output Formats

`TopoPyScale` includes a number of functions to export the downscaled timeseries to be compatible with some established energy and mass balance land-surface models for snow and permafrost physics, or hydrology. All these functions are available into the file `topo_export.py` and are integrated to the class `topoclass`. `TopoPyScale` also exports the topographical parameters it relies on to compute corrections in the file `ds_param.nc`, with the corresponding list of points (or clusters) in the file `df_centroids.pckl`. 

## Downscaled Timeseries

### Netcdf files
It is possible to export netcdf files for a list of given variables. Otherwise, by default all downscaled timeseries are available in `outputs/`

### FSM Format
`TopoPyScale` includes tools to interact and work with [FSM](https://github.com/RichardEssery/FSM). Those are available in the file `topo_sim.py`.

### Cryogrid Format
[Cryogrid](https://github.com/CryoGrid/CryoGrid) is a model used for simulating the thermal regime of the ground, particularly applied to permafrost and hydrological application.

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


