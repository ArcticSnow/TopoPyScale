import pandas as pd
import pyproj
import rasterio
import linecache
from matplotlib import pyplot as plt
import numpy as np
import glob
from TopoPyScale import topo_da as da

def FsmMetParser(file, freq="1h", resample=False):
    '''
    Parses FSM forcing files from toposcale sims
    '''

    df = pd.read_csv(file, delim_whitespace=True,
                     header=None, index_col='datetime',
                     parse_dates={'datetime': [0, 1, 2, 3]},
                     date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d %H'))

    df.columns = ['ISWR', 'ILWR', 'Sf', 'Rf', 'TA', 'RH', 'VW', 'P']

    if resample == "TRUE":
        df = df.resample(freq).apply(resample_func)

    return (df)


def FsmSnowParser(file, freq="1H", resample=False):
    '''
    parses FSM output fuiles
    '''

    df = pd.read_csv(file, delim_whitespace=True, parse_dates=[[0, 1, 2]], header=None)
    df.set_index(df.iloc[:, 0], inplace=True)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.columns = ['albedo', 'Rof', 'HS', 'SWE', 'TS10', 'TS50']

    if resample == "TRUE":
        df = df.resample(freq).apply(resample_func)

    return (df)


def FsmPlot(df):
    df.plot(subplots=True)
    plt.title("Sample=" + str(sampleN) + " Elev=" + str(np.round(lp.elevation[sampleN])))
    plt.show()

def FsmPlot_ensemble(wdir, simvar, sampleN):
    """
    Function to plot ensemble results for a given variable and point.

    Args:
        wdir: full path to sim directory
        simvar: variable to plot (str)
        sampleN: sample number to plot
    Returns:
        sampleN: sample number

    Example:
        FsmPlot_ensemble('/home/joel/sim/topoPyscale_davos/', "HS", 0)

    """
    sampleNpad = "{:0>2}".format(sampleN)
    myensemble = glob.glob(wdir + "/fsm_sims/sim_ENS"+"*"+"_FSM_pt_"+str(sampleNpad)+".txt")
    lp = pd.read_csv(wdir+"/listpoints.csv")


    ax = plt.subplot(1, 1, 1)
    for ensembleN in myensemble:
        df = FsmSnowParser(ensembleN)
        plt.plot(df[simvar], alpha=0.3)

    plt.title("Sample="+str(sampleN) + " Elev="+str(np.round(lp.elevation[sampleN])))
    plt.show()

def SmetParser(file, doresample=True, freq='1H', NA_val=-999):
    '''
    val_file = full path to a smet
    resample = "TRUE" or "FALSE"
    freq = '1D' '1H' etc

    '''
    resample_func = "mean"
    # find statrt of data
    lookup = '[DATA]'
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                skip = num

    df = pd.read_csv(file, delim_whitespace=True, index_col='datetime',
                     parse_dates={'datetime': [0]}, header=None, skiprows=skip, na_values=NA_val)

    # get col names
    lookup = 'fields'
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                fields = num


    myfields = linecache.getline(file, fields)
    colnames = myfields.split("= ")[1]
    cnames = colnames.split(" ")

    df.columns = cnames[1:]

    if doresample:
        df = df.resample(freq).apply(resample_func)

    return (df)


def getCoordinatePixel(map_path, x_coord, y_coord, epsg_in, epsg_out):
    """
    Function to find which sample a point exists in. Ca accept any combination of projections of the map and points.

    Args:
        map_path: full path to map
        x_coord: x coord of point
        y_coord: y coord of point
        epsg_in: input proj epsg code (lonlat: 4326)
        epsg_out: ouput proj epsg code

    Returns:
        sampleN: sample number

    Example:
        # get epsg of map
        from TopoPyScale import topo_da as da
        epsg_out, bbxox = da.projFromLandform(map_path)

        getCoordinatePixel("/home/joel/sim/topoPyscale_davos/landform.tif", 9.80933, 46.82945, 4326, 32632)

    """
    # define proj
    proj = pyproj.Transformer.from_crs(int(epsg_in), int(epsg_out), always_xy=True)
    # project
    x2, y2 = proj.transform(x_coord, y_coord)
    # open map
    dataset = rasterio.open(map_path)
    # get pixel x+y of the coordinate
    py, px = dataset.index(x2, y2)
    # create 1x1px window of the pixel
    window = rasterio.windows.Window(px - 1 // 2, py - 1 // 2, 1, 1)
    # read rgb values of the window
    clip = dataset.read(window=window)
    return (int(clip))


def val_plots():
    # script from topo_utils (validation stuff)
    map_path = "/home/joel/sim/topoPyscale_davos/landform.tif"
    epsg_out, bbxox = da.projFromLandform(map_path)
    sample = getCoordinatePixel(map_path, 9.80933, 46.82945, 4326, epsg_out)
    filename = "/home/joel/sim/topoPyscale_davos/outputs/FSM_pt_" + str(sample-1)+ ".txt"
    df = FsmMetParser(filename, freq="1h", resample=False)
    wfj = SmetParser("/home/joel/data/wfj_optimal/WFJ_optimaldataset_v8.smet")

    start_date = df.index[0]
    end_date = df.index[-1]

    #greater than the start date and smaller than the end date
    mask = (wfj.index >= start_date) & (wfj.index <= end_date)
    wfj = wfj.loc[mask]


    def plot_xyline(ax):
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax = plt.subplot(2, 2, 1)
    plt.scatter(df.TA, wfj.TA, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)


    ax = plt.subplot(2, 2, 2)
    plt.scatter(df.ISWR, wfj.ISWR, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)


    ax = plt.subplot(2, 2, 3)
    plt.scatter(df.RH, wfj.RH*100, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)

    ax = plt.subplot(2, 2, 4)
    plt.scatter(df.ILWR, wfj.ILWR, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)
    plt.show()

    myfile = "/home/joel/sim/topoPyscale_davos/outputs/FSM_pt_00.txt"
    df = FsmMetParser(myfile)
    FsmPlot(df)

    myfile = "/home/joel/sim/topoPyscale_davos/fsm_sims/sim_ENS1_FSM_pt_00.txt"
    df = FsmSnowParser(myfile)
    FsmPlot(df)




