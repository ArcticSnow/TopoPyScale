# FAQ

## I am new to Python, how can I use `TopoPyScale` for my project?

`TopoPyScale` is a library building on top of a number of advanced Python libraries such as `xarray`, `scikit-learn`, `pandas`, `pyproj` *et al.*, and data types such as Digital Elevation Models, and climate datasets (e.g. rasters and netcdf files, clustering method, interpolation). In case you never came across these libraries or types of data, we advice you to first get familiarized with their respective concepts. 

`TopoPyScale` is designed for facilitating users to downscale climate while enabling a flexible structure for research and further development. We are a team of scientists supporting the development of the project aside to our respective duties. Its evolution is driven by our needs. 

## I get an `Error` message, why is the example not working?

If reading the error message does not give you clear pointers to solving the problem, please first review your setup with the following questions:

- Is `TopoPyScale` installed in a Python 3.9/3.10 virtual environment?
- Have its dependencies been installed via `conda` as recommended in the [installation instructions](./01_install.md)?
- Is the file structure following the documenation recommendation (i.e. one dedicated folder for a downscaling project)? It is good practice that the project directory is seperated from the package directory.
- Have you checked once more the config file? Might be worth one more look ;) 
- Are the dates right? 
- Is the project path correct?
- is the config file containing all fields according to the example in the [documentation](https://topopyscale.readthedocs.io/en/latest/3_configurationFile/)? If you observe a field is missing, please let us know about it.
- is the DEM  file correct?
- are the projection EPSG codes correct?
- lastly, check indents are fine as YAML is indent sensitive! 

You may also have a look at the github issues currently [open](https://github.com/ArcticSnow/TopoPyScale/issues?q=is%3Aopen+is%3Aissue) or [closed](https://github.com/ArcticSnow/TopoPyScale/issues?q=is%3Aissue+is%3Aclosed).


## I have suggestions of improvements, how can I bring them up?
Please, meet us on Github on the source code repository. A number of improvements and new features have been [identified](https://github.com/ArcticSnow/TopoPyScale/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement+). If your suggestion is not yet within this list, you may either interact via specific [Issues](https://github.com/ArcticSnow/TopoPyScale/issues), or through the [Discussion](https://github.com/ArcticSnow/TopoPyScale/discussions) page. **Note:** you'll need a Github account.

## How to load an existing project without having to process the downscaling again?

Given you have all outputs file available in `/outputs`, then, you may load all necessary files of a downscaling project as follow:

```python
from TopoPyScale import topoclass as tc

config_file = './config.yml'
mp = tc.Topoclass(config_file)
mp.load_project()
```

## How to cite `TopoPyScale`?
If you use `TopoPyScale` in your project, please cite the software and the methods it relies on. We're a team of scientists volunteering in developing the toolbox. We're glad you found value in the tool we are building. 

1. To cite the software, 
    - Filhol S., Fiddes J., Aalstad K., (2023). TopoPyScale: A Python Package for Hillslope Climate Downscaling. Journal of Open Source Software, 8(86), 5059, (https://doi.org/10.21105/joss.05059)[https://doi.org/10.21105/joss.05059]
or in bibtex:
```bibtex
@article{Filhol2023, doi = {10.21105/joss.05059}, url = {https://doi.org/10.21105/joss.05059}, year = {2023}, publisher = {The Open Journal}, volume = {8}, number = {86}, pages = {5059}, author = {Simon Filhol and Joel Fiddes and Kristoffer Aalstad}, title = {TopoPyScale: A Python Package for Hillslope Climate Downscaling}, journal = {Journal of Open Source Software} } 

```
2. To cite the method on which `TopoPyScale` relies on, we invite you to look at:
	- Fiddes, J. and Gruber, S.: TopoSCALE v.1.0: downscaling gridded climate data in complex terrain, Geosci. Model Dev., 7, 387–405, https://doi.org/10.5194/gmd-7-387-2014, 2014.
	- Fiddes, J. and Gruber, S.: TopoSUB: a tool for efficient large area numerical modelling in complex topography at sub-grid scales, Geosci. Model Dev., 5, 1245–1257, https://doi.org/10.5194/gmd-5-1245-2012, 2012.
