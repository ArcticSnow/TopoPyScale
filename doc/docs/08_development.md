# Development

## Development Version Installation

```bash

cd github  # navigate to where you want to clone TopoPyScale
git clone git@github.com:ArcticSnow/TopoPyScale.git
pip install -e TopoPyScale    #install a development version, remove the -e for normal install

#----------------------------------------------------------
#            OPTIONAL: if using jupyter lab
# add this new Python kernel to your jupyter lab PATH
python -m ipykernel install --user --name downscaling

# OPTIONAL: To be able to compile the documentation locally
pip install lazydocs
git clone git@github.com:ArcticSnow/TopoPyScale_Documentation.git
```

## Issues and Forum

We keep track of all issues, bugs and ideas in the Github Issue [page](https://github.com/ArcticSnow/TopoPyScale/issues). So please bring specific problem as an issue and potentially get help. You may also want to browse open issues. Issues tagged **good first issue** are issues that should be easy fix and good to start tinckering with `TopoPyScale` code.


More elaborated discussion are taking place in the Github Discussion [page](https://github.com/ArcticSnow/TopoPyScale/discussions). This works like a forum.


## Documentation
### Function and Class Docstrings

First when writing new function and class, please follow the *Google docstring* format that is compatible with `lazydocs`. Here is an example of the format:
```python
def fsm_nlst(nconfig, metfile, nave):
    """
    Function to generate namelist parameter file that is required to run the FSM model.
    https://github.com/RichardEssery/FSM

    Args:
        nconfig (int): which FSm configuration to run (integer 1-31)
        metfile (str): path to input tscale file (relative as Fortran fails with long strings (max 21 chars?))
        nave (int): number of forcing steps to average output over eg if forcing is hourly and output required is daily then nave = 24
    
    Returns:
        NULL (writes out namelist text file which configures a single FSM run)

    Notes:
        constraint is that Fortran fails with long strings (max?)
        definition: https://github.com/RichardEssery/FSM/blob/master/nlst_CdP_0506.txt
    """


    print('my_func')
```

### Update the API documentation

Run this command from the TopoPyScale virtual environment with correct paths:

```bash
lazydocs --output-path="path/to/TopoPyScale/doc/docs" --overview-file="README.md" --src-base-url="https://github.com/ArcticSnow/TopoPyScale" path/to/TopoPyScale
```


## New Release and Pypi Version (for the code maintainers)


From the terminal run the following
```sh
# First, make sure twine is installed and you have your Pypi credentials
pip install twine

# Run from the relevant VE
python setup.py sdist
twine upload dist/* --verbose
```

### Push TopoPyScale
```sh
twine upload --config-file ~.pypirc --repository TopoPyScale dist/* --verbose
```
