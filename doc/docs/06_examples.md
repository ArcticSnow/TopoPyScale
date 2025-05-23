# Examples

We present here a set of three downscaling project examples: 1) Finse Norway, 2) Retezat, Romania, and 3) Davos, Switzerland. Jupyter Notebook are available for Example 1: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ArcticSnow/TopoPyScale_examples/HEAD). The two other examples will require your own Python VE setup. 

Each example contains the basic required `config.yml` file and its associated Python pipeline script to perform a downscaling. They perform various task to examplify some of the capabilities of `TopoPyscale`.  

**WARNING 1:** the examples require to download the climate data from ERA5 data portal [CDS Copernicus](https://cds.climate.copernicus.eu/). Depending on the amount of data request and the servers' load, it may take up to few days to acquire the complete dataset. To perform the examples, the `cdsapi` credentials must be set as indicated in the [install](./01_instal.md) section. 

**WARNING 2:** as TopoPyScale can have a large amount of verbose output for large projects, we recommand running `TopoPyScale` in a simple IPython console.

## Getting the Example Material

To run the examples you will need to download the project materials (about 32MB) containing the Digital Elevation models for each example as well as the configuration files. Those are available on Github as a dedicated [repository](https://github.com/ArcticSnow/TopoPyScale_examples) and is likely to evolve in the future alongside to the main [TopoPyScale](https://github.com/ArcticSnow/TopoPyScale) repository.

**Option 1:**
```bash
git clone https://github.com/ArcticSnow/TopoPyScale_examples
```

**Option 2**
```bash
wget https://github.com/ArcticSnow/TopoPyScale_examples/archive/refs/tags/v0.1.1.zip
unzip v0.1.1.zip
cd TopoPyScale_examples-0.1.1/
```

Access to the examples from the terminal:
```
# For Finse example (ex1)
cd ex1_norway_finse/

# For Retezat Mountain example (ex2)
cd ex2_romania_retezat/

# For Davos example (ex3)
cd ex3_switzerland_davos/
```

## Finse, Norway

### Site Description
Finse is located at 1200m above sea level in Southern Norway. The site is is equipped of a large set of instruments for weather, snow, carbon fluxes and others. A full description of instrumentation available on [UiO website](https://www.mn.uio.no/geo/english/research/groups/latice/infrastructure/).This example can be run for specific point (weather station) or in spatialized way (TopoSUB). 


### How to run the example

**Online Binder Notebook**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ArcticSnow/TopoPyScale_examples/HEAD)

**Otherwise run locally:**

1. Open an iPython console for the VE in which `TopoPyScale` is installed
2. run the following commands in the console. This will initialize a `topoclass` object and download (if necessary) the climate data. **WARNING** this may take many hours during which the console must be active. You may use `screen` for run the console in the background.
```python
from TopoPyScale import topoclass as tc

config_file = './config_spatial.yml'
mp = tc.Topoclass(config_file)
mp.get_era5()
```
3. Derive all topographical morphometrics
```python
mp.compute_dem_param()
mp.extract_topo_param()
mp.compute_solar_geometry()
mp.compute_horizon()
```
4. Execute the downscaling and store to file in a format compatible with [`Cryogrid-Community`](https://github.com/CryoGrid/CryoGridCommunity_source) model or [Cryogrid.jl](https://github.com/CryoGrid/CryoGrid.jl).
```python
mp.downscale_climate()
mp.to_cryogrid()
```
5. Plot the results
```python
mp.plot.map_variable(time_step=1, var='t')

import matplotlib.pyplot as plt
plt.show()
```

This example stores the downscaled climate in timeseries. There is one timeseries per cluster label. The function `plot.map_variable()` maps the cluster timeseries to the pixels part of each cluster. All results are stored in the folder `./outputs`.

This example can be run to downscale climate to a single point (*e.g.* a weather station).

```python
config_file = './config_point.yml'
mq = tc.Topoclass(config_file)
mq.get_era5()
mq.compute_dem_param()
mq.extract_topo_param()
mq.compute_solar_geometry()
mq.compute_horizon()
mq.downscale_climate()
mq.to_cryogrid()
```

To do so, we use a `.csv` file located in the folder `./inputs/dem/`. 
```csv
Name,stn_number,latitude,longitude,x,y
Finsevatne,SN25830,60.5938,7.527,419320.867306002,6718447.86246835
Fet-I-Eidfjord,SN49800,60.4085,7.2798,405243.856317655,6698143.36494597
```
This file requires the field `x` and `y`. The EPSG projection code of the `x` and `y` coordinates must be indicated in the file config_point.yml

## Retezat Range, Romania

This example perform climate downscaling using the TopoSUB routine for spatial downscaling. Retezat is a mountain range in the Romanian Sourthern Carpathians.

```python
from TopoPyScale import topoclass as tc

config_file = './config.yml'

# create a topopyscale python object
mp = tc.Topoclass(config_file)
mp.get_era5()

# Downscaling preparation routines
mp.compute_dem_param()
mp.extract_topo_param()
mp.compute_solar_geometry()
mp.compute_horizon()

# Downscaling routine
mp.downscale_climate()
```

## Davos, Switzerland

This example demonstrates `TopoPyScale` for the area of Davos. It downscales climate, and performs simulation with the snow model [FSM](https://github.com/RichardEssery/FSM/tree/master). 

### Site Description
Davos is located in South-Eastern Switzerland. 

### Prerequisites to this example: install [FSM](https://github.com/RichardEssery/FSM/tree/master)

The first step is to compile FSM from the source code (instruction for Linux). For this you'll need gfortran installed.
```bash
# if gfortran is not installed
sudo apt-get install gfortran

# then:
git clone git@github.com:RichardEssery/FSM.git
cd FSM
sh compil.sh 
cp FSM my/TopoPyScale/projet/
```


### How to run the example

In an IPython console run the following.
```python
import pandas as pd
from TopoPyScale import topoclass as tc
from matplotlib import pyplot as plt
from TopoPyScale import topo_sim as sim


# ========= STEP 1 ==========
# Load Configuration and download ERA5 data
config_file = './config.ini'
mp = tc.Topoclass(config_file)
mp.get_era5()

# Compute parameters of the DEM (slope, aspect, sky view factor)
mp.compute_dem_param()

# ========== STEP 2 ===========
# Extract DEM parameters for points of interest (centroids or physical points)

# ----- Option 1:
# Compute clustering of the input DEM and extract cluster centroids
mp.extract_topo_param()
# plot clusters
mp.toposub.plot_clusters_map()
mp.toposub.write_landform()

# plot sky view factor
# mp.toposub.plot_clusters_map(var='svf', cmap=plt.cm.viridis)

# ========= STEP 3 ==========
# compute solar geometry and horizon angles
mp.compute_solar_geometry()
mp.compute_horizon()

# ========= STEP 4 ==========
# Perform the downscaling
mp.downscale_climate()

# ========= STEP 5 ==========
# Export output to desired format
# mp.to_netcdf()
mp.to_fsm()

# ========= STEP 6 ===========
# Simulate FSM
for i in range(mp.config.sampling.toposub.n_clusters):
    nsim = "{:0>2}".format(i)
    sim.fsm_nlst(31, "./outputs/FSM_pt_"+ nsim +".txt", 24)
    sim.fsm_sim("./fsm_sims/nlst_FSM_pt_"+ nsim +".txt", "./FSM")

# extract GST results(8)
df = sim.agg_by_var_fsm(8)

# extraxt timeseries average
df_mean = sim.timeseries_means_period(df, mp.config.project.start, mp.config.project.end)

# map to domain grid
sim.topo_map(df_mean)

```

Outputs of FSM simulations are stored in `fsm_sims/`. Columns of the text file are: 

| Column# | Variable | Units  | Description       |
|---------|----------|--------|-------------------|
| 0       | year     | years  | Year              |
| 1       | month    | months | Month of the year |
| 2       | day      | days   | Day of the month  |
| 3       | hour     | hours  | Hour of the day   |
| 4       | alb      | -      | Effective albedo  |
| 5       | Rof      | kg m<sup>-2</sup> | Cumulated runoff from snow    |
| 6       | snd      | m      | Average snow depth                       |
| 7       | SWE      | kg m<sup>-2</sup> | Average snow water equivalent |
| 8       | Tsf      | &deg;C | Average surface temperature              |
| 9       | Tsl      | &deg;C | Average soil temperature at 20 cm depth  |