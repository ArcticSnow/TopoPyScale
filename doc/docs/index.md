# Welcome to `TopoPyScale` Documentation



<figure markdown>
  ![](./logo.png){ width="250" }
</figure>

`TopoPyScale` is a downscaling toolbox for global and regional climate model datasets, particularly relevant to mountain ranges, and hillslopes. 

- **Source Code Github Repository**: [https://github.com/ArcticSnow/TopoPyScale](https://github.com/ArcticSnow/TopoPyScale)
- **Examples Repository**: [https://github.com/ArcticSnow/TopoPyScale_examples](https://github.com/ArcticSnow/TopoPyScale_examples)
- **Documentation Repository**: [https://github.com/ArcticSnow/TopoPyScale_Documentation](https://github.com/ArcticSnow/TopoPyScale_Documentation)

If you are here to use `TopoPyScale`, then head to the [Quick Start](./02_quickstart.md) page. Further configuration setup are explained in detail

## General Concept

`TopoPyScale` uses both climate model data and Digital Elevation Models (DEM) for correcting atmospheric state variables (e.g. temperature, pressure, humidity, *etc*). `TopoPyScale` provides tools to interpolate and correct such variables to be relevant locally given a topographical context. 

<figure markdown>
  ![](./temperature_comparison.png){ width="1600" }
  <figcaption>Comparison of the surface temperature on January 1, 2020 at 12:00 UTC over the Southern Carpathians, Romania, in between ERA5 Surface product and the downscaled result using `TopoPyScale`. The DEM was segmented in 2000 clusters.</figcaption>
</figure>

The most basic requirements of `TopoPyScale` is a DEM used to defined the spatial domain of interest as well as compute a number of morphometrics, and configuration file defining the temporal period, the downscaling methods and other parameters. In its current version, `TopoPyScale` includes the `topoclass` class that wraps all functionalities for ease of use. It automatically fetches data from the [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) repositories (Pressure and Surface levels). Other climate data sources can be added. Based on the high resolution (30-100m) DEM and the climate data, methods in the `topoclass` will compute, correct and interpolate variables need to force specialized land surface models.

`TopoPyScale` includes a number of export format inter-operable with specialized energy and mass balance land surface models like [CRYOGRID](https://github.com/CryoGrid/CryoGridCommunity_source), [CROCUS](http://bibliotheque.meteo.fr/exl-php/cadcgp.php?CMD=CHERCHE&MODELE=vues/mf_-_internet_recherche_avancee_anonyme/tpl-r.html&WHERE_IS_DOC_REF_LIT=DOC00019133&&TABLE=ILS_DOC), [SNOWPACK](https://www.slf.ch/en/services-and-products/snowpack.html), [FSM](https://github.com/RichardEssery/FSM), [Snowmodel](https://srs.fs.usda.gov/pubs/26319), [GEOTOP](http://geotopmodel.github.io/geotop/) or [MuSa](https://github.com/ealonsogzl/MuSA).

Downscaled variable includes:

- 2m air temperature
- 2m air humidity
- 2m air pressure
- 10m wind speed and direction
- Surface incoming shortwave radiation
- Surface incoming longwave radiation
- Precipitation (possibility to partition snow and rain)

## Quick Installation

### Release Installation
To install the latest release, in a Python 3.9 virtual environment simply use `pip`. It is adviced to first install dependencies using conda. More detailed installation instructions [here](./01_install.md).

```bash
pip install topopyscale
```

As of now, `TopoPyScale` uses the Copernicus `cdsapi` to download data. For this to work, you will need to setup the Copernicus API key in your system. Follow [this tutorial](https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key) after creating an account with Copernicus. 

On Linux, create a file `nano ~/.cdsapirc` with inside:
```text
url: https://cds.climate.copernicus.eu/api/v2
key: {uid}:{api-key}
```

## How to Cite

You are invited to cite `TopoPyScale` when using it with the following: 

Filhol S., Fiddes J., Aalstad K., (2023). TopoPyScale: A Python Package for Hillslope Climate Downscaling. Journal of Open Source Software, 8(86), 5059, [https://doi.org/10.21105/joss.05059](https://doi.org/10.21105/joss.05059)

or in bibtex:
```bibtex
@article{Filhol2023, doi = {10.21105/joss.05059}, url = {https://doi.org/10.21105/joss.05059}, year = {2023}, publisher = {The Open Journal}, volume = {8}, number = {86}, pages = {5059}, author = {Simon Filhol and Joel Fiddes and Kristoffer Aalstad}, title = {TopoPyScale: A Python Package for Hillslope Climate Downscaling}, journal = {Journal of Open Source Software} } 

```

## Contribution

We welcome, and are pleased for any new contribution to this downscaling toolbox. So if you have suggestions, correction and addition to the current code, please come join us on [GitHub](https://github.com/ArcticSnow/TopoPyScale) and talk to us on the [Discussion](https://github.com/ArcticSnow/TopoPyScale/discussions) page.

## Funding and Support

`TopoPyScale` is currently developed by people at:

- [University of Oslo](https://www.mn.uio.no/geo/english/), Norway
- SLF, Switzerland
