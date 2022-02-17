"""
Implementation of orographic correction for precipitation field following the XX method.

S. Filhol, December 2021

WARNING: in development, NOT READY

References:
    - https://journals.ametsoc.org/view/journals/atsc/61/12/1520-0469_2004_061_1377_altoop_2.0.co_2.xml
    -

TODO:
- read Smith and Barstad 2004, https://journals.ametsoc.org/view/journals/atsc/61/12/1520-0469_2004_061_1377_altoop_2.0.co_2.xml
- read Thomas paper as well as Aurora
- talk to Andy beaucse of: https://github.com/pism/LinearTheoryOrographicPrecipitation
- Thomas' Matlab implementation from ERA5: https://github.com/TVSchuler/Sval_Imp_matlab/blob/0088e31877334428cddcb4ff9b87bcb3973c4bbb/precipitation/LT_matlab_2016_09_16.m#L126
- For validation, we could check against ERA5 Land, or CARRA in Svalbard
"""

import numpy as np
import matplotlib.pyplot as plt


