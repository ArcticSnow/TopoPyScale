"""
Data assimilation methods including ensemble simulation, satelite data retrival and processing and data assimilation algorithms
J. Fiddes, March 2022

TODO:
    - method to mosaic (gdal) multiple tiles before cropping

"""
import os
import glob
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy
import rasterio as rio
from osgeo import gdal
import osr
from scipy.interpolate import interp1d
from TopoPyScale import topo_sim as sim
from datetime import datetime
import xarray as xr


# import gdal


def lognormDraws_kris(n, med, s):
    """
    Function to generate n random draws from a log normal distribution

    Args:
        n: integer number of draws to make
        med: median of distribrition
        s: standard deviation of distribution

    Returns:
        Vector of draws
    """
    sig = np.sqrt(np.log(1 + (s ** 2 / med ** 2)))
    draws = np.exp(np.log(med) + sig * np.random.randn(n))
    return draws


def normDraws(n, m, s):
    """
    Function to generate n random draws from a normal distribution

    Args:
        n: integer number of draws to make
        m: mean of distribrition
        s: standard deviation of distribution

    Returns:
        Vector of draws
    """
    draws = np.random.normal(m, s, n)
    return draws


def ensemble_pars_gen(n):
    """
    Function to generate perturbabion parameters to create an ensemble of meteorological forcings

    Args:
        n: size of the ensemble (integer)

    Returns:
        array of perturbation factors of dimension n
    """

    pbias = lognormDraws_kris(n, 1, 0.5)
    tbias = normDraws(n, 1, 0.01)  # this is for K only, we convert C to K then multiply bias then K to C (roughly +/- 4 degrees)
    swbias = normDraws(n, 1, 0.2)
    lwbias = normDraws(n, 1, 0.2)

    # uncorrelated
    perturb_uncor = pd.DataFrame({'pbias': pbias, 'tbias': tbias, 'swbias': swbias, 'lwbias': lwbias})

    cordat = np.array([1, -0.1, 0.5, -0.1, -0.1, 1, -0.3, 0.3, 0.5, -0.3, 1, 0.6, -0.1, 0.3, 0.6,
                       1])  # from Navari 2016 (not bad when compared to data), auto gen from all meteo in base run.
    # https://stats.stackexchange.com/questions/82261/generating-correlated-distributions-with-a-certain-mean-and-standard-deviation
    cormat = cordat.reshape(4, 4)

    # P,sw,lw,ta X P,sw,lw,ta  as table 1 navari 2016 https://tc.copernicus.org/articles/10/103/2016/tc-10-103-2016.pdf
    # array([[1., -0.1, 0.5, -0.1],
    #        [-0.1, 1., -0.3, 0.3],
    #        [0.5, -0.3, 1., 0.6],
    #        [-0.1, 0.3, 0.6, 1.]])

    # precip vals (pos 2) are the expected means sd of the lognorm distribution. we compute the mead sd of norm distribution below and substitute.
    sd_vec = [0.2, 0.2, 0.1, 0.005]  # p, sw, lw, ta Navari, how to base this on the data?
    mean_vec = np.array([0, 1, 1, 1])  # p, sw, lw, ta

    mean = mean_vec
    # empirical = true gives exact correlations in result eg https://stats.stackexchange.com/questions/82261/generating-correlated-distributions-with-a-certain-mean-and-standard-deviation
    # res = mvrnorm(n=N, m=mean_vec, Sigma=covmat, empirical=TRUE)

    # check for posite definiteness of covariance matrix
    # stopifnot(eigen(covmat)$values > 0 )

    # convert cor2cov matrix: https://stackoverflow.com/questions/39843162/conversion-between-covariance-matrix-and-correlation-matrix/39843276#39843276
    covmat = cormat * sd_vec

    cov = covmat  # diagonal covariance
    # from here https://stackoverflow.com/questions/31666270/python-analog-of-mvrnorms-empirical-setting-when-generating-a-multivariate-dist?rq=1
    res = np.random.multivariate_normal(mean, cov, n).T

    # lognorm transform on P
    plogN = np.exp(res[0,])
    tbias = res[3]  # this is for K only, we convert C to K then multiply bias then K to C
    pbias = plogN
    swbias = res[1,]
    lwbias = res[2,]
    perturb = pd.DataFrame({'pbias': pbias, 'tbias': tbias, 'swbias': swbias, 'lwbias': lwbias})
    perturb.to_csv("ensemble.csv")
    # df.plot(kind="density", subplots=True)
    # check distributions of random draws an correlations between variables
    pd.plotting.scatter_matrix(perturb, alpha=0.2)
    plt.show()
    return perturb, perturb_uncor


def ensemble_meteo_gen(ds, perturb, N_ens, ensemb_type):
    """
    Function to perturb downscaled meteorological timeseries

    Args:
        ds: downscaled_pts
        perturb: matrix of perturbation parameters
        N_ens: ensemble number (integer)
        ensemb_type: string describing which meteo variables to perturb: "T" = Tair, "P" = Precip, "S" = Swin, "L" = Lwin

    Returns:
        ds:

    """

    # create independent copy of ds to export
    ds_perturb = copy.deepcopy(ds)
    if ensemb_type == "T":
        ds_perturb["t"] = ds.t * perturb.tbias[N_ens]
    if ensemb_type == "TP":
        ds_perturb["t"] = ds.t * perturb.tbias[N_ens]
        ds_perturb["tp"] = ds.tp * perturb.pbias[N_ens]
    if ensemb_type == "TPS":
        ds_perturb["t"] = ds.t * perturb.tbias[N_ens]
        ds_perturb["tp"] = ds.tp * perturb.pbias[N_ens]
        ds_perturb["SW"] = ds.SW * perturb.swbias[N_ens]
    if ensemb_type == "TPSL":
        ds_perturb["t"] = ds.t * perturb.tbias[N_ens]
        ds_perturb["tp"] = ds.tp * perturb.pbias[N_ens]
        ds_perturb["SW"] = ds.SW * perturb.swbias[N_ens]
        ds_perturb["LW"] = ds.LW * perturb.lwbias[N_ens]

    return ds_perturb


def construct_HX(wdir, startDA, endDA):
    """
    Function to generate HX matrix of ensembleSamples x day

    Args:
        wdir: working directory
        DAyear: year of data assimilation yyyy
        startDA: start of data assimilation window  yyyy-mm-dd
        endDA: end of data assimilation window yyyy-mm-dd

    Returns:
        ensembleRes: 2D pandas dataframe with dims cols (samples * ensembles) and rows (days / simulation timesteps)

    """
    a = glob.glob(wdir + "/fsm_sims/sim_ENS*")

    # Natural sort to get correct order ensemb * samples
    def natural_sort(l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)
    file_list = natural_sort(a)

    # compute dates index for da we always extract 31 March to Jun 30 for DA -
    # this may of course not always be appropriate
    df = pd.read_csv(file_list[0], delim_whitespace=True, parse_dates=[[0, 1, 2]], header=None)
    startIndex = df[df.iloc[:, 0] == startDA].index.values
    endIndex = df[df.iloc[:, 0] == endDA].index.values

    # if len(startIndex) == 0:
    #     sys.exit("daYear does not exist in simulation period")
    # if len(endIndex) == 0:
    #     sys.exit("daYear does not exist in simulation period")

    # construct a list as long as samples x ensembles (sampleEnsembles)
    data = []
    for file_path in file_list:
        # column 7 (index 6) is snow water equiv
        data.append(np.genfromtxt(file_path, usecols=6)[int(startIndex):int(endIndex) + 1])

    myarray = np.asarray(data)  # sampleEnsembles x days
    ensembRes = pd.DataFrame(myarray.transpose())

    mydates = df.iloc[:, 0][int(startIndex):int(endIndex) + 1]

    return ensembRes, mydates

    # data = []
    # for file_path in file_list:
    #     # column 6 (index 5) is snow height
    #     data.append(np.genfromtxt(file_path, usecols=5)[int(startIndex):int(endIndex)])
    #
    # myarray = np.asarray(data)
    # df = pd.DataFrame(myarray.transpose())
    #
    # df.to_csv(wd + '/ensemble/ensembResHS' + str(ipad) + '.csv', index=False, header=False)
    #
    # # get open - loop results, but only once when ensembleN=0
    # if int(ensembleN) == 1:
    #
    #     a = glob.glob(wd + "/fsm_sims/*")
    #     file_listOL = natural_sort(a)
    #
    #     df = pd.read_csv(file_listOL[0], delim_whitespace=True,
    #                      parse_dates=[[0, 1, 2]], header=None)
    #     startIndex = df[df.iloc[:, 0] == str(daYear - 1) + "-09-01"].index.values
    #     endIndex = df[df.iloc[:, 0] == str(daYear) + "-09-01"].index.values
    #     time = df.iloc[int(startIndex):int(endIndex), 0]
    #
    #     data = []
    #     for file_path in file_listOL:
    #         data.append(np.genfromtxt(file_path, usecols=6)[int(startIndex):int(endIndex)])
    #
    #     myarray = np.asarray(data)
    #     df2 = pd.DataFrame(myarray.transpose())
    #
    #     # 2d rep of 3d ensemble in batches where ensemble changes slowest
    #     df2.to_csv(wd + '/ensemble/openloopRes.csv', index=False, header=False)
    #     time.to_csv(wd + '/ensemble/ensemble_timestamps.csv', index=False, header=False)


def modis_tile(latmax, latmin, lonmax, lonmin):
    """
    Function to retrieve MODIS tile indexes based on a lon/lat bbox

    Args:


    Returns:


    """

    # find modis tiles based on longlat

    os.system("wget --no-check-certificate https://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt")

    # first seven rows contain header information
    # bottom 3 rows are not data
    data = np.genfromtxt('sn_bound_10deg.txt',
                         skip_header=7,
                         skip_footer=3)
    in_tile = False
    i = 0
    while (not in_tile):
        in_tile = latmax >= data[i, 4] and latmax <= data[i, 5] and lonmax >= data[i, 2] and lonmax <= data[i, 3]
        i += 1
    vertmax = data[i - 1, 0]
    horizmax = data[i - 1, 1]
    print('max Vertical Tile:', vertmax, 'max Horizontal Tile:', horizmax)

    in_tile = False
    i = 0
    while (not in_tile):
        in_tile = latmin >= data[i, 4] and latmin <= data[i, 5] and lonmin >= data[i, 2] and lonmin <= data[i, 3]
        i += 1
    vertmin = data[i - 1, 0]
    horizmin = data[i - 1, 1]
    print('min Vertical Tile:', vertmin, 'min Horizontal Tile:', horizmin)

    return vertmax, horizmax, vertmin, horizmin


def get_modis_wrapper(vertmax, horizmax, vertmin, horizmin, startDate, endDate):
    """
    Function to iterate through several MODIS tile downloads.

    Args:


    Returns:


    """


def pymodis_download(wdir, vert, horiz, STARTDATE, ENDDATE):
    # http://www.pymodis.org/scripts/modis_download.html

    USER = "jfiddes"  # os.getenv('user')
    PWD = "sT0kkang!"  # os.getenv('pwd')
    MODPATH = wdir + "/modis/"
    HOST = "https://n5eil01u.ecs.nsidc.org/"
    FOLDER = "MOST"
    PRODUCT = "MOD10A1F.061"
    RAWPATH = MODPATH + "/raw/"
    os.makedirs(MODPATH, exist_ok=True)
    os.makedirs(RAWPATH, exist_ok=True)
    os.makedirs(MODPATH + "/transformed/", exist_ok=True)

    hreg = ("h" + f"{horiz:02d}")  # re.compile("h2[4]")
    vreg = ("v" + f"{vert:02d}")  # re.compile("v0[5]")
    TILE = hreg + vreg
    print("Downloading " + product + " from " + STARTDATE + " to " + ENDDATE)
    os.system("modis_download.py -U " +
              USER +
              " -P " + PWD +
              " -u " + HOST +
              " -p mail@pymodis.com "
              " -s " + FOLDER +
              " -p " + PRODUCT +
              " -f " + STARTDATE +
              " -e " + ENDDATE +
              " -t " + TILE + " " +
              RAWPATH)


def projFromLandform(pathtolandform):
    src = gdal.Open(pathtolandform, gdal.GA_ReadOnly)
    proj = osr.SpatialReference(wkt=src.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY', 1)

    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    bbox = [ulx, lry, lrx, uly]

    return epsg, bbox


def process_modis(wdir, epsg, bbox, layer=0):
    """
    Function to convert hdf to tiff, extract layer, and reproject

    Args:
        modpath: path to 'modis' directory
        layer: layer to extract from hdf
        epsg: reprojected target epsg code
        bbox: output bounds as list [minX, minY, maxX, maxY] in target

    Returns:
        Spatially subsetted reprojected tiffs of data layer specified (still NDSI)


        Docs: https://nsidc.org/sites/nsidc.org/files/technical-references/C6.1_MODIS_Snow_User_Guide.pdf
    """

    MODPATH = wdir + "/modis/"
    RAWPATH = MODPATH + "/raw/"

    # list all downloaded hdf files
    files = glob.glob(RAWPATH + "/*.hdf")

    # extract
    for in_file in files:
        out_file = MODPATH + "/transformed/" + os.path.basename(in_file).replace('.hdf', '.tif')

        # open dataset
        dataset = gdal.Open(in_file, gdal.GA_ReadOnly)

        # extract subset layer [default = 0, this is The cloud-gap-filled (CGF) daily snow cover tiled product.]
        subdataset = gdal.Open(dataset.GetSubDatasets()[layer][0], gdal.GA_ReadOnly)

        # gdalwarp to longlat
        # kwargs = {'format': 'GTiff', 'ouputBounds': str(bbox), 'dstSRS': 'EPSG:'+ str(epsg)}
        # ds = gdal.Warp(destNameOrDestDS=out_file, srcDSOrSrcDSTab=subdataset, **kwargs)
        ds = gdal.Warp(destNameOrDestDS=out_file,
                       srcDSOrSrcDSTab=subdataset,
                       format='GTiff',
                       outputBounds=bbox,
                       outputBoundsSRS='EPSG:' + str(epsg),
                       dstSRS='EPSG:' + str(epsg))
        del ds

    return print(str(len(files)) + " MODIS files processed")


def extract_fsca_timeseries(wdir, plot=True):
    """
    Function to extract mean NDSI from domain and converts to fSCA using the empirical formula based on landsat
    https://modis-snow-ice.gsfc.nasa.gov/uploads/C6_MODIS_Snow_User_Guide.pdf pp.22
    Args:
        wdir: working directory
        plot: booleen whether to plot dataframe

    Returns:
        df_sort: dataframe of julian dates and fSCA values
    """

    files = glob.glob(wdir + "/modis/transformed/*.tif")
    data = []
    juldates = []
    for file in files:
        with rio.open(file) as src:
            # Read as numpy array
            array = src.read()
            profile = src.profile

            mymean = np.mean(array)
            juldate = file.split(".")[1].split("A")[1]

        data.append(mymean)
        juldates.append(juldate)

    # convert to fSCA
    fsca = (-0.01 + (1.45 * np.array(data)))
    fsca[fsca <= 0] = 0
    fsca[fsca > 100] = 100

    d = {'dates': juldates, 'fSCA': fsca}
    df = pd.DataFrame(data=d)
    df_sort = df.sort_values("dates")
    # add datetime object
    df_sort["mydates"] = [pd.to_datetime(e[:4]) + pd.to_timedelta(int(e[4:]) - 1, unit='D') for e in df_sort.dates]

    df_sort_reindex = df_sort.reset_index(drop=True)
    if plot:
        df_sort_reindex.plot(x="mydates", y="fSCA", kind="scatter")
        plt.show()

    return df_sort_reindex

    # import datetime
    # datetime.datetime.strptime(df_sort.dates, '%Y%j').date()
    # df_sort["mydates"]


# def get_modis(vert, horiz, startYear, endYear):
#     """
#     Function to retrieve MODIS tile indexes based on a lon/lat bbox
#
#     Args:
#
#
#     Returns:
#
#
#     """
#
#     # require MODIS downloaded
#     # https://bpostance.github.io/posts/processing-large-spatial-datasets.md/
#
#     # format h,v
#
#     import os
#     import re
#     import time
#     import numpy as np
#     import rasterio as rio
#     from osgeo import gdal
#     import requests
#     from bs4 import BeautifulSoup # pip install beautifulsoup4
#     import dotenv # pip install python-dotenv
#     dotenv.load_dotenv(dotenv.find_dotenv())
#
#     # params fo LP DAAC
#     # you will require an EarthData login
#     host = fr'https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061'
#     host = fr'https://n5eil01u.ecs.nsidc.org/MOST/MOD10A1F.061/'
#     login = "jfiddes" # os.getenv('user')
#     password = "sT0kkang!" # os.getenv('pwd')
#     modpath = "/home/joel/sim/topoPyscale_davos/modis/"
#     os.makedirs(modpath, exist_ok=True)
#     os.makedirs(modpath + "/raw/", exist_ok=True)
#     os.makedirs(modpath + "/transformed/", exist_ok=True)
#
#     # list folders
#     r = requests.get(host, verify=True, stream=True,auth=(login,password))
#     soup = BeautifulSoup(r.text, "html.parser")
#     folders = list()
#     for link in soup.findAll('a', attrs={'href': re.compile("\d{4}.\d{2}.\d{2}/")}):
#         folders.append(link.get('href'))
#     print(f"{len(folders)} folders found")
#
#     # subset folders by date
#     startDate = str(startYear) + ".09.01/"
#     endDate = str(endYear) + ".01.31/"
#
#     s = folders.index(startDate)
#     e = folders.index(endDate)
#
#
#     # list files in folder
#     from timeit import default_timer as timer
#     from datetime import timedelta
#
#     start = timer()
#
#     # 30 sec per 10 days
#     for f in folders[1:5]:
#         file_list = list()
#         folder_url = f"{host}/{f}"
#         r = requests.get(folder_url, verify=True, stream=True, auth=(login,password))
#         soup = BeautifulSoup(r.text, "html.parser")
#         for link in soup.findAll('a', attrs={'href': re.compile(".hdf$")}):
#             file_list.append(link.get('href'))
#     print(f"{len(file_list)} files found in folder n")
#
#     end = timer()
#     print(timedelta(seconds = end - start))
#
#     # USA tiles only
#     # use modis grid to slice only continents / areas of interest
#     # https://modis-land.gsfc.nasa.gov/MODLAND_grid.html
#     hreg = re.compile("h" + f"{horiz:02d}") # re.compile("h2[4]")
#     vreg = re.compile("v" + f"{vert:02d}" )# re.compile("v0[5]")
#     usa_files = list()
#     for fl in file_list:
#         h = hreg.search(fl)
#         if h:
#             v = vreg.search(h.string)
#             if v:
#                 usa_files.append(v.string)
#     print(f"{len(usa_files)} USA tiles found")
#
#     # download file from folder
#     # implement 25 file timeout to avoid rate limit
#     exceptions = list()
#     for f in folders[:1]:
#         for e, fl in enumerate(usa_files[:]):
#             if (e+1) % 10 == 0:
#                 print('sleep')
#                 time.sleep(5)
#             else:
#                 print(fl, e+1)
#                 try:
#                     file_url = f"{host}/{f}/{fl}"
#                     r = requests.get(file_url, verify=True, stream=True,auth=(login,password))
#                     open(modpath + "/raw/" + fl, "wb").write(r.content)
#                 except Exception as error:
#                     print(error)
#                     exceptions.append(fl)
#
#     for fl in usa_files[:]:
#         in_file = modpath + "raw/" + fl
#         out_file = modpath + "/transformed/" + fl.replace('.hdf', '.tif')
#
#         # open dataset
#         dataset = gdal.Open(in_file, gdal.GA_ReadOnly)
#         subdataset = gdal.Open(dataset.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
#
#         # gdalwarp
#         kwargs = {'format': 'GTiff', 'dstSRS': 'EPSG:4326'}
#         ds = gdal.Warp(destNameOrDestDS=out_file, srcDSOrSrcDSTab=subdataset, **kwargs)
#         del ds

def particle_batch_smoother(Y, HX, R):
    # particle batch smoothert
    No = len(Y)
    Rinv = np.array([(1 / R)] * No)
    mat = np.ones((HX.shape[0], HX.shape[1]))

    # reshape Y to have dims of mat
    Y1 = np.reshape(np.array(Y), (HX.shape[0], 1))

    # Calculate the likelihood.
    Inn = np.kron(mat, Y1) - HX

    # with %*% gives NA
    EObj = Rinv.dot(Inn ** 2)  # [1 x Ne] ensemble objective function.

    LH = np.exp(-0.5 * EObj)  # Scaled likelihood.

    # Calculate the posterior weights as the normalized likelihood.
    w = LH / sum(LH)


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


def EnKA(prior, obs, pred, alpha, R):
    """
    EnKA: Implmentation of the Ensemble Kalman Analysis
    Inputs:
        prior: Prior ensemble matrix (n x N array)
        obs: Observation vector (m x 1 array)
        pred: Predicted observation ensemble matrix (m x N array)
        alpha: Observation error inflation coefficient (scalar)
        R: Observation error covariance matrix (m x m array, m x 1 array, or scalar)
    Outputs:
        post: Posterior ensemble matrix (n x N array)
    Dimensions:
        N is the number of ensemble members, n is the number of state variables and/or
        parameters, and m is the number of boservations.

    For now, we have impelemnted the classic (stochastic) version of the Ensemble
    Kalman analysis that involves perturbing the observations. Deterministic variants
    (Sakov and Oke, 2008) also exist that may be worth looking into. This analysis
    step can be used both for filtering, i.e. Ensemble Kalman filter, when observations
    are assimilated sequentially or smoothing, i.e. Ensemble (Kalman) smoother,
    when also be carried out in a multiple data assimilation (i.e. iterative) approach.

    Since the author rusty in python it is possible that this code can be improved
    considerably. Below are also some examples of further developments that may be
    worth looking into:

    To do/try list:
    (i) Using a Bessel correction (Ne-1 normalization) for sample covariances.
    (ii) Perturbing the predicted observations rather than the observations
        following van Leeuwen (2020, doi: 10.1002/qj.3819) which is the formally
        when defining the covariances matrices.
    (iii) Looking into Sect. 4.5.3. in Evensen (2019, doi: 10.1007/s10596-019-9819-z)
    (iv) Using a deterministic rather than stochastic analysis step.
    (v) Augmenting with artificial momentum to get some inertia.
    (vi) Better understand/build on the underlying tempering/annealing idea in ES-MDA.
    (vii) Versions that may scale better for large n such as those working in the
        ensemble subspace (see e.g. formulations in Evensen, 2003, Ocean Dynamics).

    Code by: K. Aalstad (10.12.2020) based on an earlier Matlab version.
    """

    # Dimensions.
    m = np.size(obs)  # Number of obs
    n = np.shape(prior)[0]  # Tentative number of state vars and/or parameters
    if np.size(prior) == n:  # If prior is 1D correct the above.
        N = n  # Number of ens. members.
        n = 1  # Number of state vars and/or parameters.
    else:  # If prior is 2D then the lenght of the second dim is the ensemble size
        N = np.shape(prior)[1]

    # Checks on the observation error covariance matrix.
    try:
        if np.shape(R)[0] == m and np.shape(R)[1] == m:
            pass
            # Do nothing if specified as a matrix.
    except:
        if np.size(R) == 1:
            R = R * np.identity(m)
        elif np.size(R) == m:
            R = np.diag(R)
            # Convert to matrix if specified as a vector.
        else:
            raise Exception('R must be a scalar, m x 1 vector, or m x m matrix.')

    # pdb.set_trace()
    # Anomaly calculations.
    mprior = np.mean(prior, -1)  # Prior ensemble mean
    if n == 1:
        A = prior - mprior
    else:
        A = prior - mprior[:, None]
    mpred = np.mean(pred, -1)  # Prior predicted obs ensemble mean
    if m == 1:
        B = pred - mpred
    else:
        B = pred - mpred[:, None]
    Bt = B.T  # Tranposed -"-

    # Perturbed observations.
    Y = np.outer(obs, np.ones(N)) + np.sqrt(alpha) * np.random.randn(m, N)
    # Y=np.outer(obs,np.ones(N))

    # Covariance matrices.
    C_AB = A @ Bt  # Prior-predicted obs covariance matrix multiplied by N (n x m)
    C_BB = B @ Bt  # Predicted obs covariance matrix multiplied by N (m x m)
    aR = (N * alpha) * R  # Scaled observation error covariance matrix (m x m)

    # Analysis step
    if n == 1 and m == 1:  # Scalar case
        K = C_AB * (np.linalg.inv(C_BB + aR))
        inno = Y - pred
        post = prior + K * inno
    else:
        K = C_AB @ (np.linalg.inv(C_BB + aR))  # Kalman gain (n x m)
        inno = Y - pred  # Innovation (m x N)
        post = prior + K @ inno  # Posterior (n x N)

    return post


def PBS(obs, pred, R):
    """
    PBS: Implmentation of the Particle Batch Smoother
    Inputs:
        obs: Observation vector (m x 1 array)
        pred: Predicted observation ensemble matrix (m x N array)
        R: Observation error covariance 'matrix' (m x 1 array, or scalar)
    Outputs:
        w: Posterior weights (N x 1 array)
    Dimensions:
        N is the number of ensemble members and m is the number of boservations.

    Here we have implemented the particle batch smoother, which is a batch-smoother
    version of the particle filter (i.e. a particle filter without resampling),
    described in Margulis et al. (2015, doi: 10.1175/JHM-D-14-0177.1). As such, this
    routine can also be used for particle filtering with sequential data assimilation.
    This scheme is obtained by using a particle (mixture of Dirac delta functions)
    representation of the prior and applying this directly to Bayes theorem. In other
    words, it is just an application of importance sampling with the prior as the
    proposal density (importance distribution). It is also the same as the Generalized
    Likelihood Uncertainty Estimation (GLUE) technique (with a formal Gaussian likelihood)
    which is widely used in hydrology.

    This particular scheme assumes that the observation errors are additive Gaussian
    white noise (uncorrelated in time and space). The "logsumexp" trick is used to
    deal with potential numerical issues with floating point operations that occur
    when dealing with ratios of likelihoods that vary by many orders of magnitude.

    To do/try list:
        (i) Implement an option for correlated observation errors if R is specified
        as a matrix (this would slow down this function).
        (ii) Consier other likelihoods and observation models.
        (iii) Look into iterative versions of importance sampling.


    Code by: K. Aalstad (14.12.2020) based on an earlier Matlab version.
    """
    # Dimensions.
    m = np.size(obs)  # Number of obs
    N = np.shape(pred)[-1]

    # Checks on the observation error covariance matrix.
    if np.size(R) == 1:
        R = R * np.ones(m)
    elif np.size(R) == m:
        pass
    else:
        raise Exception('R must be a scalar, m x 1 vector.')

    # pdb.set_trace()
    # Residual and log-likelihood
    if m == 1:
        residual = obs - pred
        llh = -0.5 * ((residual ** 2) * (1 / R))
    else:
        residual = obs[:, None] - pred
        # pdb.set_trace()
        llh = -0.5 * ((1 / R) @ (residual ** 2))

    # Log of normalizing constant
    # A properly scaled version of this could be output for model comparison.
    lZ = logsumexp(llh)  # from scipy.special import logsumexp

    # Weights
    logw = llh - lZ  # Log of posterior weights
    w = np.exp(logw)  # Posterior weights
    if np.size(w) == N and np.round(np.sum(w), 10) == 1:
        pass
    else:
        raise Exception('Something went wrong with the PBS, try pdb.set_trace()')

    return w


def da_plots(HX1, HX2, W, mydates, myobs):
    """
    Function to extract mean fSSCA from domain

    Args:
        HX1:
        HX2:
        weights: weights output from PBS (dims 1 x NENS)
        dates: datetime object
        myobs: timeseries of observtions

    Returns:
        plot
    """

    # HX1
    #
    # weights = W
    # HX = HX3D_weighted_mean

    def percentile_plot(prior, wfill, percentile):
        ndays = prior.shape[0]
        med_post = []
        for days in range(ndays):
            mu = prior[days, ]
            d = {'mu': mu, 'wfill': wfill}
            df = pd.DataFrame(d)
            dfOrder = df.sort_values('mu')
            y_interp = interp1d(np.cumsum(dfOrder.wfill), dfOrder.mu, fill_value="extrapolate")
            med = y_interp(percentile)
            med_post.append(med)

        return np.array(med_post)

    # for prior all ensemble members considered equally likely therefore weights are all
    weights = [1/HX1.shape[1]] * HX1.shape[1]
    prior_med = percentile_plot(HX1, weights, 0.5)
    prior_low = percentile_plot(HX1, weights, 0.1)
    prior_high = percentile_plot(HX1, weights, 0.8)

    # for posterior weights are given by the output of PBS, W
    post_med = percentile_plot(HX1, W, 0.5)
    post_low = percentile_plot(HX1, W, 0.1)
    post_high = percentile_plot(HX1, W, 0.8)

    plt.subplot(1, 2, 1)
    plt.fill_between(mydates, prior_low, prior_high, alpha=0.2)
    plt.plot(mydates, prior_med)
    plt.fill_between(mydates, post_low, post_high, alpha=0.2)
    plt.plot(mydates, post_med)
    plt.plot(mydates, myobs)
    plt.xlabel("Date")
    plt.ylabel("fSCA")
    plt.legend()

    # for prior all ensemble members considered equally likely therefore weights are all
    weights = [1/HX1.shape[1]] *  HX1.shape[1]
    prior_med = percentile_plot(HX2, weights, 0.5)
    prior_low= percentile_plot(HX2, weights, 0.1)
    prior_high = percentile_plot(HX2, weights, 0.9)

    # for posterior weights are given by the output of PBS, W
    post_med = percentile_plot(HX2, W, 0.5)
    post_low = percentile_plot(HX2, W, 0.1)
    post_high = percentile_plot(HX2, W, 0.9)

    plt.subplot(1, 2, 2)
    plt.fill_between(mydates, prior_low, prior_high, alpha=0.2)
    plt.plot(mydates, prior_med)
    plt.fill_between(mydates, post_low, post_high, alpha=0.2)
    plt.plot(mydates, post_med)
    plt.xlabel("Date")
    plt.ylabel("HS (m)")
    plt.legend()
    plt.show()


def fsca_plots(wdir, plotDay, df_mean):
    """
    Function to plot side by side observed fSCA 500m [0-1] and modelled SCA 30 m [0,1]

    Args:
        wdir: sim directory
        plotDay: date to plot as YYYY-mm-dd
        df_mean: timeseries mean object given by sim.timeseries_means_period()

    Returns:
        plot

    Usage:
        df = sim.agg_by_var_fsm(6)
        df2 = sim.agg_by_var_fsm_ensemble(6, W)
        df_mean = sim.timeseries_means_period(df, "2018-06-01", "2018-06-01")
        da.fsca_plots("/home/joel/sim/topoPyscale_davos/", "2018150", df_mean)

    TODO:
        - create modelled fSCA option from agreggated 30m to 500m pixels
        - accept datetime object instead of julday.
    """
    # convert data YYYY-mm-dd to YYYYddd
    dt = datetime.strptime(plotDay, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    year = dt.year
    juldate = str(year)+str(doy)


    modis_file = glob.glob(wdir + "modis/transformed/MOD10A1F.A" + juldate + "*")

    src = rio.open(modis_file[0])
    fig, (axrgb, axhist) = plt.subplots(1, 2, figsize=(14,7))

    rio.plot.show(src, ax=axrgb,vmin=0, vmax=100)
    cbar1 = plt.imshow(src.read(1), cmap='viridis', vmin=0, vmax=100)
    fig.colorbar(cbar1, ax=axrgb)

    nclust = len(df_mean)
    lookup = np.arange(nclust, dtype=np.uint16)

    for i in range(0, nclust):
        lookup[i] = df_mean[i]

    with rasterio.open('landform.tif') as src:
        # Read as numpy array
        array = src.read()
        profile = src.profile

        # Reclassify in a single operation using broadcasting
        array = lookup[array]
        # binary map
        array[array <= 0] = 0
        array[array > 0] = 1


    # rasterio.plot.show(array, cmap='viridis')
    # plt.show()

    with rio.open('output_raster.tif', 'w', **profile) as dst:
        # Write to disk
        dst.write(array.astype(rasterio.int16))

    src1 = rio.open("output_raster.tif")
    rio.plot.show(src1, ax=axhist)
    cbar2 = plt.imshow(src1.read(1), cmap='viridis')
    fig.colorbar(cbar2, ax=axhist)
    plt.show()


# def spatial_plots(df_mean, plot=False):
#     """
#     Function to plot side by side observed fSCA 500m [0-1] and modelled SCA 30 m [0,1]
#
#     Args:
#         df_mean: timeseries mean object given by sim.timeseries_means_period()
#         plot: genrate plot? logical
#
#     Returns:
#         plot
#
#     Usage:
#         df = sim.agg_by_var_fsm(6)
#         df2 = sim.agg_by_var_fsm_ensemble(6, W)
#         df_mean = sim.timeseries_means_period(df, "2018-06-01", "2018-06-01")
#         da.fsca_plots("/home/joel/sim/topoPyscale_davos/", "2018150", df_mean)
#
#     TODO:
#         - create modelled fSCA option from agreggated 30m to 500m pixels
#     """
#
#
#     nclust = len(df_mean)        wdir: sim directory
#         plotDay: date to plot as YYYY-mm-dd
#     lookup = np.arange(nclust, dtype=np.uint16)
#
#     for i in range(0, nclust):
#         lookup[i] = df_mean[i]
#
#     with rasterio.open('landform.tif') as src:
#         # Read as numpy array
#         array = src.read()
#         profile = src.profile
#
#         # Reclassify in a single operation using broadcasting
#         array = lookup[array]
#         # binary map
#         array[array <= 0] = 0
#         array[array > 0] = 1
#
#
#     # rasterio.plot.show(array, cmap='viridis')
#     # plt.show()
#
#     with rio.open('output_raster.tif', 'w', **profile) as dst:
#         # Write to disk
#         dst.write(array.astype(rasterio.int16))
#
#     if plot:
#         src1 = rio.open("output_raster.tif")
#         rio.plot.show(src1, ax=axhist)
#         cbar2 = plt.imshow(src1.read(1), cmap='viridis')
#         fig.colorbar(cbar2, ax=axhist)
#         plt.show()

def da_compare_plot(wdir, plotDay, df_mean_open, df_mean_da):
    """
    Function to plot side by side observed fSCA 500m [0-1] and modelled SCA 30 m [0,1]

    Args:
        wdir: sim directory
        plotDay: date to plot as YYYY-mm-dd
        df_mean: timeseries mean object given by sim.timeseries_means_period()

    Returns:
        plot

    Usage:
        df = sim.agg_by_var_fsm(6)
        df2 = sim.agg_by_var_fsm_ensemble(6, W)
        df_mean = sim.timeseries_means_period(df, "2018-06-01", "2018-06-01")
        da.fsca_plots("/home/joel/sim/topoPyscale_davos/", "2018150", df_mean)

    TODO:
        - create modelled fSCA option from agreggated 30m to 500m pixels
        - accept datetime object instead of julday.
    """
    # set up plot axis
    fig, ([axmod1, axopen], [axmod2, axda]) = plt.subplots(2, 2, figsize=(14, 7))

    # convert data YYYY-mm-dd to YYYYddd
    dt = datetime.strptime(plotDay, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    year = dt.year
    juldate = str(year)+str(doy)

    # plot modis
    modis_file = glob.glob(wdir + "modis/transformed/MOD10A1F.A" + juldate + "*")
    src = rio.open(modis_file[0])
    rio.plot.show(src, ax=axmod1,vmin=0, vmax=100)
    cbar1 = plt.imshow(src.read(1), cmap='viridis', vmin=0, vmax=100)
    fig.colorbar(cbar1, ax=axmod1)

    # compute and plot open loop
    nclust = len(df_mean_open)
    lookup = np.arange(nclust, dtype=np.uint16)

    for i in range(0, nclust):
        lookup[i] = df_mean_open[i]

    with rasterio.open('landform.tif') as src:
        # Read as numpy array
        array = src.read()
        profile = src.profile

        # Reclassify in a single operation using broadcasting
        array = lookup[array]
        # binary map
        #array[array <= 0] = 0
        #array[array > 0] = 1


    # rasterio.plot.show(array, cmap='viridis')
    # plt.show()

    with rio.open('output_raster.tif', 'w', **profile) as dst:
        # Write to disk
        dst.write(array.astype(rasterio.int16))

    src1 = rio.open("output_raster.tif")
    rio.plot.show(src1, ax=axopen)
    cbar2 = plt.imshow(src1.read(1), cmap='viridis')
    fig.colorbar(cbar2, ax=axopen)

    # plot modis again
    src = rio.open(modis_file[0])
    rio.plot.show(src, ax=axmod2,vmin=0, vmax=100)
    cbar1 = plt.imshow(src.read(1), cmap='viridis', vmin=0, vmax=100)
    fig.colorbar(cbar1, ax=axmod2)

    # compute and plot DA
    nclust = len(df_mean_da)
    lookup = np.arange(nclust, dtype=np.uint16)

    for i in range(0, nclust):
        lookup[i] = df_mean_da[i]

    with rasterio.open('landform.tif') as src:
        # Read as numpy array
        array = src.read()
        profile = src.profile

        # Reclassify in a single operation using broadcasting
        array = lookup[array]
        # binary map
        #array[array <= 0] = 0
        #array[array > 0] = 1


    # rasterio.plot.show(array, cmap='viridis')
    # plt.show()

    with rio.open('output_raster.tif', 'w', **profile) as dst:
        # Write to disk
        dst.write(array.astype(rasterio.int16))

    src1 = rio.open("output_raster.tif")
    rio.plot.show(src1, ax=axda)
    cbar2 = plt.imshow(src1.read(1), cmap='viridis')
    fig.colorbar(cbar2, ax=axda)
    plt.show()

def build_modis_cube(wdir):
    geotiff_list_unsorted = glob.glob(wdir + "modis/transformed/MOD10A1F.A*")
    # Natural sort to get correct order ensemb * samples
    def natural_sort(l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)
    geotiff_list = natural_sort(geotiff_list_unsorted)

    # Create variable used for time axis
    mydates = [lname.split('.A')[1].split(".h")[0] for lname in geotiff_list]
    time_var = xr.Variable('time', mydates)
    # Load in and concatenate all individual GeoTIFFs
    geotiffs_da = xr.concat([xr.open_rasterio(i) for i in geotiff_list], dim=time_var)
    # Covert our xarray.DataArray into a xarray.Dataset
    geotiffs_ds = geotiffs_da.to_dataset('band')
    # Rename the variable to a more useful name
    ds = geotiffs_ds.rename({1: 'fSCA'})
    arr = ds.to_array()

    # tuples of coordinates
    x, y = np.meshgrid(range(len(arr.x)), range(len(arr.y)))

    myobs = []
    for index in range(len(x.flatten() )):
        xindex = x.flatten()[index]
        yindex = y.flatten()[index]
        obs = arr[0, :, range(len(arr.y))[yindex] ,range(len(arr.x))[xindex]]
        myobs.append(obs.values)

    # obs array of Npixels x time
    obs_array = np.array(myobs).shape


def getModisbbox(Npixel, arr):
    """
    Function to find obtain bbox of modis pixel CRS is projected UTM

    Args:
        Npixel: pixel nuber as integer of 0 - (xdim x ydim) (the flattened grid) eg for a 10 x 10 grid Npixel would have a range of 0-100
        arr: xarray DataArray with dimensions "x" and "y" and attribute "res"

    Returns:
        bbox: bbox of pixel as list [xmax,xmin, ymax, ymin]



    """

    xres = arr.res[0] # resolution in x
    yres = arr.res[1] # resolution in y

    x, y = np.meshgrid(arr.x, arr.y)

    centreX = x.flatten()[Npixel] # pixel N centre x coord
    centreY = y.flatten()[Npixel] # pixel N centre y coord
    xmax = float(centreX + xres/2)
    xmin = float(centreX - xres/2)
    ymax = float(centreY + yres/2)
    ymin = float(centreY - yres/2)
    bbox = [xmax,xmin, ymax, ymin]
    return bbox

def getSamples_inModisPixel(map_path):
    """
    Identify samples that exist within a single bbox and proportianal cover. This can be used to construct a modelled fSCA in a MODIS pixel

    Args:
        map_path: path to landform.tif

    Returns:
        list of samples and frequency

    """

    # # define proj
    # proj = pyproj.Transformer.from_crs(int(epsg_in), int(epsg_out), always_xy=True)
    # # project
    # x2, y2 = proj.transform(x_coord, y_coord)
    # open map
    dataset = rasterio.open(map_path)
    # get pixel x+y of the coordinate
    py, px = dataset.index(x2, y2)
    # create 1x1px window of the pixel
    window = rasterio.windows.Window(px - 1 // 2, py - 1 // 2, 1, 1)
    # read rgb values of the window
    clip = dataset.read(window=window)
    return (int(clip))

