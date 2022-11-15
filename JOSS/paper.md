---
title: 'TopoPyScale: A Python package for hillslope climate downscaling'
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

Climate reanalysis or forecast models are available worldwide for extended periods of time for the twentieth and twenty first century.They are often spatially too course to be relevant to mountainous regions. Mountain regions are experiencing the fastest change with a cryosphere rapidly responding to climate change. To understand and study geomorphological, hydrological, glaciological changes, we need tools to downscale climatic timeseries for the basic atmospheric state variables used to solve energy and water mass fluxes. Advanced physically based methods exist, though they come with high computational cost and complex technical setup **(REF)**. `TopoPyScale` uses a pragmatic approach to downscaling by minimizing complexity, reducing computational cost, simplifying interaoperability with land surface models, while retaining physical coherence. The toolbox is designed to be flexible and incorporate future innovation. 

# Statement of need

`TopoPyScale` is a Python package for performing climate downscaling. It is designed as a toolbox combining computationally efficient methods to downscale climate reanalysis and forecast model products (e.g. [@hersbachERA5GlobalReanalysis2020:2020]). 
Climate models are run at coarse spatial resolution (9km for ERA5) missing the heterogeniety imposed by the topography of mountain ranges on state variables. 
\autoref{fig:temp_comp} shows an example of downscaled temperature field of 



![Comparison of the surface temperature on January 1, 2020 at 12:00 UTC over the Southern Carpathians, Romania, in between ERA5 Surface product and the downscaled result using TopoPyScale. The DEM was segmented in 2000 clusters.\label{fig:temp_comp}](temperature_comparison.png)
and referenced from text using .



# Toolbox methods and structure

![Structure overview of TopoPyScale processing pipeline.\label{fig:figure2}](figure2.png)
- insert here diagram figure
- describe topo_sub
- interapporabitlity with independant land-surface models



# Acknowledgements

Financial support for this project came from ClimaLand, an EEA/EU collaboration grant between Norway and Romania 2014-2021, project code RO-NO-2019-0415,1290 contract no. 30/2020. 
**(JOEL ADD your financial support)**
# References
