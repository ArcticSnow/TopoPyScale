# Dataset Sources

TopoPyScale relies on two main types of data: climate and topographical data. These data can be freely available for your region and we list here a number of public and open access data repository relevant to TopoPyScale.

## Climate data

### ERA5

ERA5 comes in two parts:

- [Hourly land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview)
- [Hourly pressure levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)

#### From Copernicus repository

Copernicus whom generate the ERA5 data is also a portal to download the data. TopoPyScale has a routine to download the data given identification to the cds system has been done.

#### From Google Cloud Storage repository

Google Cloud Storage maintains a publicly accessible repository of the ERA5 data. TopoPyScale includes a routine to download data from there given authentification to Google Cloud has been done has explained by Google: https://cloud.google.com/sdk/docs/install


## Digital Elevation models

### SRTM (global)
[SRTM](https://en.wikipedia.org/wiki/Shuttle_Radar_Topography_Mission) is a global product produced by [NASA](https://www.earthdata.nasa.gov/sensors/srtm). There are 2 basic products available, the 30m and 90m resolution. Useful links to download tiles:

- https://search.earthdata.nasa.gov/search?q=SRTM
- https://www2.jpl.nasa.gov/srtm/
- https://dwtkns.com/srtm30m/

### ArcticDEM (global/Arctic)
[ArcticDEM](https://www.pgc.umn.edu/data/arcticdem/) provides DEMs for the global Arctic region at a fine spatial resolution.

### Copernicus DEM
The European Copernicus program produces a series of digital elevation product based on a number of data sources. These dataset are:
- 10m DEM over Europe including overseas territories: [](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model#anchor)
- Global 30m and 90m DEM: [](https://sentinels.copernicus.eu/web/sentinel/-/copernicus-dem-new-direct-data-download-access)

Example downloading the Global 30m Copernicus public DEM for part of the Alps:
```python
from TopoPyScale import fetch_dem as fd

# extent: [East, West, South, North] boundaries.
fd.fetch_copernicus_dem(extent=[7,10,45,47], product='COP-DEM_GLO-30-DGED/2023_1', n_download_threads=5)
```
**Warning**: it may happen the server hangs and the routine will need to be relaunched.
The data are downloaded as `.tar` archive files for each tile. The script will unpack the archive. Then `gdal_merge` can be used to merge all the tiles into one single DEM with the command:

```bash
# Create a text file listing all tiles to merge
ls Copernicus_DSM_*.tif > tile_list.txt

# Merge tiles into one single DEM with -9999 as NaN
gdal_merge.py -init -9999 -o Copernicus_DSM_merged.tif --optfile tile_list.txt
```


## Observation Dataset

- [WMO](https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-surface-land?tab=overview) stations (Global)
- Norwegian meteorological Institute [Frost API](https://frost.met.no/index.html) (Norway)
- [NOAA Hourly](https://www.ncei.noaa.gov/maps/hourly/) (Global)
