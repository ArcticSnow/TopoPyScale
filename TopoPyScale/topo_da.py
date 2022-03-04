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
import rasterio
from osgeo import gdal
import osr
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
    tbias = normDraws(n, 1,
                      0.01)  # this is for K only, we convert C to K then multiply bias then K to C (roughly +/- 4 degrees)
    swbias = normDraws(n, 1, 0.2)
    lwbias = normDraws(n, 1, 0.2)

    # uncorrelated
    df_uncor = [pbias, tbias, swbias, lwbias]

    cordat = np.array([1, -0.1, 0.5, -0.1, -0.1, 1, -0.3, 0.3, 0.5, -0.3, 1, 0.6, -0.1, 0.3, 0.6, 1])  # from Navari 2016 (not bad when compared to data), auto gen from all meteo in base run.
    # https://stats.stackexchange.com/questions/82261/generating-correlated-distributions-with-a-certain-mean-and-standard-deviation
    cormat = cordat.reshape(4, 4)

    # P,sw,lw,ta X P,sw,lw,ta  as table 1 navari 2016 https://tc.copernicus.org/articles/10/103/2016/tc-10-103-2016.pdf
    # array([[1., -0.1, 0.5, -0.1],
    #        [-0.1, 1., -0.3, 0.3],
    #        [0.5, -0.3, 1., 0.6],
    #        [-0.1, 0.3, 0.6, 1.]])

    # precip vals (pos 2) are the expected means sd of the lognorm distribution. we compute the mead sd of norm distribution below and substitute.
    sd_vec = [0.5, 0.2, 0.1, 0.005]  # p, sw, lw, ta Navari, how to base this on the data?
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
    plogN = np.exp(res[0, ])
    tbias = res[3]  # this is for K only, we convert C to K then multiply bias then K to C
    pbias = plogN
    swbias = res[1, ]
    lwbias = res[2, ]
    perturb = pd.DataFrame({'pbias': pbias, 'tbias': tbias , 'swbias': swbias , 'lwbias': lwbias})
    perturb.to_csv("ensemble.csv")
    # df.plot(kind="density", subplots=True)
    # check distributions of random draws an correlations between variables
    pd.plotting.scatter_matrix(perturb, alpha=0.2)
    plt.show()
    return perturb

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


def construct_HX(wdir, daYear):
    """
    Function to generate HX matrix of ensembleSamples x day

    Args:
        wdir:
        daYear:

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
    startIndex = df[df.iloc[:, 0] == str(daYear) + "-03-1"].index.values
    endIndex = df[df.iloc[:, 0] == str(daYear) + "-07-31"].index.values

    # if len(startIndex) == 0:
    #     sys.exit("daYear does not exist in simulation period")
    # if len(endIndex) == 0:
    #     sys.exit("daYear does not exist in simulation period")

    # construct a list as long as samples x ensembles (sampleEnsembles)
    data = []
    for file_path in file_list:
        # column 7 (index 6) is snow water equiv
        data.append(np.genfromtxt(file_path, usecols = 6)[int(startIndex):int(endIndex)+1])

    myarray = np.asarray(data)  # sampleEnsembles x days
    ensembRes = pd.DataFrame(myarray.transpose())

    mydates = df.iloc[:, 0] [int(startIndex):int(endIndex) +1 ]

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
                         skip_header = 7,
                         skip_footer = 3)
    in_tile = False
    i = 0
    while(not in_tile):
        in_tile = latmax >= data[i, 4] and latmax <= data[i, 5] and lonmax >= data[i, 2] and lonmax <= data[i, 3]
        i += 1
    vertmax = data[i-1, 0]
    horizmax = data[i-1, 1]
    print('max Vertical Tile:', vertmax, 'max Horizontal Tile:', horizmax)

    in_tile = False
    i = 0
    while(not in_tile):
        in_tile = latmin >= data[i, 4] and latmin <= data[i, 5] and lonmin >= data[i, 2] and lonmin <= data[i, 3]
        i += 1
    vertmin = data[i-1, 0]
    horizmin = data[i-1, 1]
    print('min Vertical Tile:', vertmin, 'min Horizontal Tile:', horizmin)

    return vertmax, horizmax, vertmin, horizmin




def get_modis_wrapper(vertmax, horizmax, vertmin, horizmin, startDate, endDate):
    """
    Function to iterate through several MODIS tile downloads.

    Args:


    Returns:


    """

def pymodis_download(wdir, vert, horiz, startYear, endYear):

    # http://www.pymodis.org/scripts/modis_download.html

    USER = "jfiddes" # os.getenv('user')
    PWD = "sT0kkang!" # os.getenv('pwd')
    MODPATH = wdir + "/modis/"
    HOST = "https://n5eil01u.ecs.nsidc.org/"
    FOLDER = "MOST"
    PRODUCT = "MOD10A1F.061"
    STARTDATE = str(startYear) + "-03-01"
    ENDDATE = str(endYear) + "-07-31"
    RAWPATH = MODPATH + "/raw/"
    os.makedirs(MODPATH, exist_ok=True)
    os.makedirs(RAWPATH, exist_ok=True)
    os.makedirs(MODPATH + "/transformed/", exist_ok=True)

    hreg = ("h" + f"{horiz:02d}") # re.compile("h2[4]")
    vreg =("v" + f"{vert:02d}" )# re.compile("v0[5]")
    TILE = hreg+vreg

    os.system("modis_download.py -U " +
              USER +
              " -P " + PWD +
              " -u " +  HOST +
              " -p mail@pymodis.com "
              " -s " +  FOLDER +
              " -p " +  PRODUCT +
              " -f " +  STARTDATE +
              " -e " +  ENDDATE +
              " -t " +  TILE +" "+
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
    """


    MODPATH = wdir+"/modis/"
    RAWPATH = MODPATH + "/raw/"

    # list all downloaded hdf files
    files = glob.glob(RAWPATH + "/*.hdf")

    # extract
    for in_file in files:

        out_file = MODPATH + "/transformed/" + os.path.basename(in_file).replace('.hdf', '.tif')

        # open dataset
        dataset = gdal.Open(in_file, gdal.GA_ReadOnly)

        # extract subset layer [0]
        subdataset = gdal.Open(dataset.GetSubDatasets()[layer][0], gdal.GA_ReadOnly)


        # gdalwarp to longlat
        #kwargs = {'format': 'GTiff', 'ouputBounds': str(bbox), 'dstSRS': 'EPSG:'+ str(epsg)}
        #ds = gdal.Warp(destNameOrDestDS=out_file, srcDSOrSrcDSTab=subdataset, **kwargs)
        ds = gdal.Warp(destNameOrDestDS=out_file,
                  srcDSOrSrcDSTab=subdataset,
                  format='GTiff',
                  outputBounds=bbox,
                  outputBoundsSRS='EPSG:' + str(epsg),
                  dstSRS='EPSG:'+ str(epsg))
        del ds

    return print(str(len(files)) + " MODIS files processed")

def extract_sca_timeseries(wdir, plot=True):

    """
    Function to extract mean fSSCA from domain

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
        with rasterio.open(file) as src:
            # Read as numpy array
            array = src.read()
            profile = src.profile

            mymean = np.mean(array)
            juldate = file.split(".")[1].split("A")[1]

        data.append(mymean)
        juldates.append(juldate)

    d = {'dates': juldates, 'fSCA': data}
    df = pd.DataFrame(data=d)
    df_sort = df.sort_values("dates")
    if plot:
        df_sort.plot(x = "dates", y = "fSCA", kind = "scatter")
        plt.show()

    return df_sort

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


