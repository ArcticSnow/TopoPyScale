---
title: 'TopoPyScale: A Python Package for Hillslope Climate Downscaling'
tags:
  - Python
  - climate science
  - downscaling
  - xarray
  - dem segmentation
  - snow
  - permafrost
authors:
  - name: Simon Filhol
    orcid: 0000-0003-1282-7307
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Joel Fiddes
    orcid: 0000-0003-2870-6972
    equal-contrib: true
    affiliation: 2
  - name: Kristoffer Aalstad
    orcid: 0000-0002-2475-3731
    affiliation: 1
affiliations:
 - name: University of Oslo, Norway
   index: 1
 - name: WSL Institute for Snow and Avalanche Research SLF, Switzerland
   index: 2
date: 8 November 2022
bibliography: paper.bib
---

# Summary

Global climate reanalyses and model projections are available  worldwide for the past and coming century. However, model grids remain too coarse to be directly relevant at the hillslope scale [@fanetal2019], requiring adapted downscaling tools to account for the effects of local topography. Mountain regions are experiencing accelerated warming with the cryosphere rapidly responding to climate change. To understand and study geomorphological, hydrological, and glaciological changes, we need tools to downscale climatic timeseries for the basic atmospheric state variables used to solve surface energy and mass balance in process-based models. Advanced dynamical downscaling methods exist, though they come with a high computational cost and complex technical setup [@10.3389/feart.2022.789332]. `TopoPyScale` uses a pragmatic approach to downscaling by minimizing complexity, reducing computational cost, simplifying interoperability with land surface models, while retaining physical coherence and allowing the primary drivers of land surface-atmosphere interaction to be considered. The toolbox is designed to be flexible in its usage and development. 

# Statement of need

`TopoPyScale` is a community supported open-source Python package for performing climate downscaling following the initial work of @fiddesTopoSCALEDownscalingGridded2014 and @fiddesTopoSUBToolEfficient2012. It is designed as a toolbox combining computationally efficient methods to downscale climate reanalysis and projections [@hersbachERA5GlobalReanalysis2020]. Atmospheric climate models are run at coarse spatial resolution (25 km for ERA5) missing the heterogeneity imposed by the topography of mountain ranges on state variables. \autoref{fig:temp_comp} shows an example of a downscaled temperature field for January 1, 2020 at 12:00 computed with `TopoPyScale` using ERA5 data as input. `TopoPyScale` allows one to reconstruct the variability of temperatures observed in between valleys and mountain tops within one single ERA5 gridcell. This method is now used in a number of studies [@esurf-2022-39, @hess-23-4717-2019] to study geophysical processes (cite something here, Martin et al?), geomorphological dynamics of permafrost (*e.g.* @esurf-2022-39),the hydrology of mountain catchments (*e.g.* @hess-23-4717-2019), mountain glaciers (*e.g.* @tc-2021-380) and downscaling hillslope scale climate projections (*e.g.* @gmd-15-1753-2022). The ease of use as well as the low computational cost help scientists to quickly obtain hillslope-scale atmospheric forcing data that is representative for their study domain.

![Comparison of the 2 m air temperature on January 1, 2020 at 12:00 UTC over the Southern Carpathians, Romania, in between ERA5 Surface product and the downscaled result using TopoPyScale. The DEM was segmented using 2000 clusters.\label{fig:temp_comp}](temperature_comparison.png)

# Toolbox methods and structure

`TopoPyScale` is built on keystone pythonic libraries like [pandas](https://pandas.pydata.org/) [@reback2020pandas], [xarray](https://docs.xarray.dev/en/stable/) [@hoyer2017xarray] with parallelization thanks to [dask](https://docs.dask.org/en/stable/) [@dask_library], [topocalc](https://github.com/USDA-ARS-NWRC/topocalc) and [pvlib](https://pvlib-python.readthedocs.io/en/stable/index.html) [@Holmgren2018] for computation of metrics related to topography and solar position, [scikit-learn](https://scikit-learn.org/stable/) [@scikit-learn] for the clustering algorithm, [rasterio](https://rasterio.readthedocs.io/en/latest/index.html) [@gillies_2019] and [pyproj](https://pyproj4.github.io/pyproj/stable/) for handling geospatial data.

![Workflow of TopoPyScale processing pipeline.\label{fig:figure2}](figure2.png)

`TopoPyScale` consists of a set of tools to be run in a processing pipeline with its main structure following the diagram of \autoref{fig:figure2}. First, it takes a Digital Elevation Model (DEM), and the climate data as inputs. The climate data must include air temperature, pressure, specific humidity and meridional and zonal wind components from atmospheric pressure levels ranging from mean sea level to above the highest elevation of the DEM. It must include surface variables such as incoming shortwave radiation, incoming longwave radiation, and precipitation. The next step is to compute morphometrics based on terrain such as slope, aspect, sky view factor, etc. Then the downscaling can be run in two modes, 1) *point*, or 2) *TopoSUB* downscaling. *Point* downscaling is used for a list of specific points (e.g. weather station location), for which climate is downscaled only at the coordinates of the given points. The *TopoSUB* approach [@fiddesTopoSUBToolEfficient2012] is instead executed for semi-distributed spatial downscaling (\autoref{fig:figure2}). The DEM is clustered based on morphometrics. *Toposcale*, the downscaling routine, is then run only for the centroid of each cluster. This method efficiently abstracts the DEM into a limited number of representative points as opposed to all the DEM grid cells, with potentially several orders of magnitude of computation effort saved.Toposcale uses 3D spatial interpolation and geometrical corrections to downscale the atmospheric state variables. The results can then be exported in a number of formats that are able to force specialized land-surface models for snow, permafrost or hydrology via readily extendable plugins. Currently, this includes the models Cryogrid [@gmd-2022-127], Crocus [@gmd-5-773-2012], SNOWPACK [@BARTELT2002123], Snowmodel [@ADistributedSnowEvolutionModelingSystemSnowModel], Geotop [@gmd-7-2831-2014], or the data assimilation toolkit MuSa [alonso-gonzalezMuSAMultiscaleSnow2022]. Finally, 1D results can be mapped to the domain DEM in order to generate spatial results.  `TopoPyScale` also includes a toolbox to perform snow simulations with [FSM](https://github.com/RichardEssery/FSM) [@gmd-8-3867-2015], and data assimilation of fractional snow-covered area [@hess-23-4717-2019].

# Working examples

TBW

# Acknowledgements

SF was funded from ClimaLand, an EEA/EU collaboration grant between Norway and Romania 2014-2021, project code RO-NO-2019-0415,1290 contract no. 30/2020 contract no. 30/2020. JF is funded by the Swiss National Science Foundation (grant no. 179130). KA was funded by ESA Permafrost_CCI (https://climate.esa.int/en/projects/permafrost/) and the Spot-On project (Research Council of Norway #301552), and acknowledges support from the LATICE strategic research area at the University of Oslo.


# References
