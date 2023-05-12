"""
Methods to generate required simulation files and run simulations of various models using tscale forcing
J. Fiddes, February 2022

TODO:

"""
import os, re
import glob
import re
import pandas as pd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from matplotlib import pyplot as plt
import xarray as xr
#from TopoPyScale import topo_da as da
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import datetime as dt
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



        Example nlst:
            &config
            /
            &drive
              met_file = 'data/met_CdP_0506.txt'
              zT = 1.5
              zvar = .FALSE.
            /
            &params
            /
            &initial
              Tsoil = 282.98 284.17 284.70 284.70
            /
            &outputs
              out_file = 'out_CdP_0506.txt'
            /
    """

    METEOFILENAME = os.path.split(metfile)[1].split('.txt')[0]
    if not os.path.exists('./fsm_sims'):
        os.mkdir('./fsm_sims')

    f = open('./fsm_sims/nlst_' + METEOFILENAME + '.txt', 'w')
    f.write('&config' + '\n')
    f.write('  nconfig = ' + str(nconfig) + '\n')
    f.write('/' + '\n')
    f.write('&drive' + '\n')
    f.write('  met_file = ' + "'" + metfile + "'" + '\n')
    f.write('  zT = 1.5' + '\n')
    f.write('  zvar = .FALSE.' + '\n')
    f.write('/' + '\n')
    f.write('&params' + '\n')
    f.write('/' + '\n')
    f.write('&initial' + '\n')
    f.write('  Tsoil = 282.98 284.17 284.70 284.70' + '\n')
    f.write('/' + '\n')
    f.write('&outputs' + '\n')
    f.write('  out_file = ' + "'./fsm_sims/sim_" + METEOFILENAME + ".txt'" + '\n')
    f.write('  Nave = ' + str(int(nave)) + '\n')
    f.write('/' + '\n')
    f.close()


def txt2ds(fname):
    '''
    Function to read a single FSM text file output as a xarray dataset
    Args:
        fname (str): filename

    Returns:
        xarray dataset of dimension (time, point_id)
    '''
    df = read_pt_fsm(fname)
    point_id = int( re.findall(r'\d+', fname.split('/')[-1])[-1])
    print(f'---> Reading FSM data for point_id = {point_id}')

    ds = xr.Dataset({
        "albedo": (['time'], df.albedo.values),
        "runoff":  (['time'], df.runoff.values),
        "snd":  (['time'], df.snd.values),
        "swe":  (['time'], df.swe.values),
        "t_surface":  (['time'], df.t_surface.values),
        "t_soil":  (['time'], df.t_soil.values),
        },
        coords={
            "point_id": point_id,
            "time": df.index,
            "reference_time": pd.Timestamp(df.index[0])
        })

    return ds

def to_dataset(fname_pattern='sim_FSM_pt*.txt', fsm_path = "./fsm_sims/"):
    '''
    Function to read FSM outputs of one simulation into a single dataset.
    Args:
        fname_pattern:
        fsm_path:

    Returns:

    '''
    fnames = glob.glob(fsm_path + fname_pattern)
    fnames.sort()

    ds = xr.concat([txt2ds(fname) for fname in fnames],'point_id')

    ds.albedo.attrs = {'units':'ratio', 'standard_name':'albedo', 'long_name':'Surface Effective Albedo', '_FillValue': -9999999.0}
    ds.runoff.attrs = {'units':'kg m-2', 'standard_name':'runoff', 'long_name':'Cumulated runoff from snow', '_FillValue': -9999999.0}
    ds.snd.attrs = {'units':'m', 'standard_name':'snd', 'long_name':'Average snow depth', '_FillValue': -9999999.0}
    ds.swe.attrs = {'units':'kg m-2', 'standard_name':'swe', 'long_name':'Average snow water equivalent', '_FillValue': -9999999.0}
    ds.t_surface.attrs = {'units':'°C', 'standard_name':'t_surface', 'long_name':'Average surface temperature', '_FillValue': -9999999.0}
    ds.t_surface.attrs = {'units':'°C', 'standard_name':'t_soil', 'long_name':'Average soil temperature at 20 cm depth', '_FillValue': -9999999.0}
    ds.attrs = {'title':'FSM simulation outputs',
                'source': 'Data downscaled with TopoPyScale and simulated with FSM',
                'date_created':dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

    return ds


def read_pt_fsm(fname):
    '''
    Function to load FSM simulation output into a pandas dataframe
    Args:
        fname (str): path to simulation file to open

    Returns:
        pandas dataframe
    '''
    fsm = pd.read_csv(fname,
                delim_whitespace=True,
                header=0,
                index_col='time',
                parse_dates = {'time': [0, 1, 2, 3]},
                date_format = '%Y %m %d %H.000')
    fsm.columns = ['albedo', 'runoff', 'snd', 'swe', 't_surface', 't_soil']
    return fsm


def fsm_sim(nlstfile, fsm_exec):
    """
    Function to simulate the FSM model
    https://github.com/RichardEssery/FSM

    Args:
        nlstfile (int): which FSm configuration to run (integer 1-31)
        fsm_exec (str): path to input tscale file (relative as Fortran fails with long strings (max 21 chars?))
    
    Returns: 
        NULL (FSM simulation file written to disk)
    """
    print('Simulation done: ' + nlstfile)
    os.system(fsm_exec + ' < ' + nlstfile)
    os.remove(nlstfile)





def agg_by_var_fsm(ncol=None, var='gst', fsm_path = "./fsm_sims"):
    """
    Function to make single variable multi cluster files as preprocessing step before spatialisation. This is much more efficient than looping over individual simulation files per cluster.
    For V variables , C clusters and T timesteps this turns C individual files of dimensions V x T into V individual files of dimensions C x T.

    Currently written for FSM files but could be generalised to other models.

    Args:
        ncol (int): column number of variable to extract
    Returns: 
        NULL ( file written to disk)

    ncol:
        4 = rof
        5 = hs
        6 = swe
        7 = gst

    """

    # find all simulation files and natural sort https://en.wikipedia.org/wiki/Natural_sort_order
    a = glob.glob(fsm_path+"/sim_FSM_pt*")


    def natural_sort(l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    fsm_columns = {'alb':4,
                   'rof':5,
                   'snd':6,
                   'swe':7,
                   'gst':8.,
                   'tsl':9}

    if ncol is None and var is not None:
        if var.lower() in ['alb', 'rof', 'snd', 'swe', 'gst', 'tsl']:
            ncol = int(fsm_columns.get(var))
        else:
            print("indicate ncol or var within ['alb', 'rof', 'snd', 'swe', 'gst', 'tsl']")

    file_list = natural_sort(a)

    mydf = pd.read_csv(file_list[0], delim_whitespace=True, parse_dates=[[0, 1, 2]], header=None)
    mydates = mydf.iloc[:, 0]

    # can do temp subset here
    # startIndex = df[df.iloc[:,0]==str(daYear-1)+"-09-01"].index.values
    # endIndex = df[df.iloc[:,0]==str(daYear)+"-09-01"].index.values

    # all values
    startIndex = 0
    endIndex = mydf.shape[0]

    # efficient way to parse multifile
    data = []
    for file_path in file_list:

        data.append(np.genfromtxt(file_path, usecols=ncol)[int(startIndex):int(endIndex)])

    myarray = np.asarray(data)  # samples x days
    df = pd.DataFrame(myarray.transpose())


    # add timestamp
    df.insert(0, 'Datetime', mydates)
    df = df.set_index("Datetime")
    print(f'Variable {list(fsm_columns)[ncol-4]} extracted')
    return df
    # df.to_csv('./fsm_sims/'+ varname +'.csv', index=False, header=True)


def agg_by_var_fsm_ensemble(ncol=None, var='gst', W=1):
    """
    Function to make single variable multi cluster files as preprocessing step before spatialisation. This is much more efficient than looping over individual simulation files per cluster.
    For V variables , C clusters and T timesteps this turns C individual files of dimensions V x T into V individual files of dimensions C x T.

    Currently written for FSM files but could be generalised to other models.

    Args:
        ncol (int): column number of variable to extract
        W ():
    Returns: 
        NULL ( file written to disk)

    ncol:
        4 = rof
        5 = hs
        6 = swe
        7 = gst

    """

    # find all simulation files and natural sort https://en.wikipedia.org/wiki/Natural_sort_order
    a = glob.glob("./fsm_sims/sim_ENS*_FSM_pt*")
    fsm_columns = {'alb':4,
                   'rof':5,
                   'snd':6,
                   'swe':7,
                   'gst':8.,
                   'tsl':9}
    if ncol is None and var is not None:
        if var.lower() in ['alb', 'rof', 'snd', 'swe', 'gst', 'tsl']:
            ncol = int(fsm_columns.get(var))
        else:
            print("indicate ncol or var within ['alb', 'rof', 'snd', 'swe', 'gst', 'tsl']")


    def natural_sort(l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    file_list = natural_sort(a)

    mydf = pd.read_csv(file_list[0], delim_whitespace=True, parse_dates=[[0, 1, 2]], header=None)
    mydates = mydf.iloc[:, 0]

    # can do temp subset here
    # startIndex = df[df.iloc[:,0]==str(daYear-1)+"-09-01"].index.values
    # endIndex = df[df.iloc[:,0]==str(daYear)+"-09-01"].index.values

    # all values
    startIndex = 0
    endIndex = mydf.shape[0]

    # efficient way to parse multifile
    data = []
    for file_path in file_list:
        data.append(np.genfromtxt(file_path, usecols=ncol)[int(startIndex):int(endIndex)])

    myarray = np.asarray(data)  # samples x days
    df = pd.DataFrame(myarray.transpose())

    # add timestamp
    df.insert(0, 'Datetime', mydates)
    df = df.set_index("Datetime")

    Nensembles = int(len(W))
    Nsamples = int(len(a)/Nensembles)

    dfreshape = np.array(df).reshape(len(mydates), Nensembles, Nsamples).transpose(0,2,1) # is this order correct?
    # dfreshape= np.array(df).reshape(365, 20, 50)# .transpose(0,2,1) # is this order correct?
    dfreshape_weight = dfreshape*W
    dfreshape_weight_sum = dfreshape_weight.sum(2)

    df = pd.DataFrame(dfreshape_weight_sum)
    # add timestamp
    df.insert(0, 'Datetime', mydates)
    df = df.set_index("Datetime")

    print(f'Variable {list(fsm_columns)[ncol-4]} extracted')
    return df
    # df.to_csv('./fsm_sims/'+ varname +'.csv', index=False, header=True)


def timeseries_means_period(df, start_date, end_date):
    """
    Function to extract results vectors from simulation results. This can be entire time period some subset or sing day.

    Args:
        df (dataframe): results df
        start_date (str): start date of average (can be used to extract single day) '2020-01-01'
        end_date (str): end date of average (can be used to extract single day) '2020-01-03'

    Returns:
        dataframe: averaged dataframe
    """

    # df = pd.read_csv(results_file, parse_dates=True, index_col=0)
    df_mean = df[start_date:end_date].mean()
    return df_mean

    # filename = os.path.split(results_file)[1].split('.csv')[0]

    # pd.Series(df_mean).to_csv(
    #     "./fsm_sims/av_" +
    #     filename +
    #     "_" +
    #     start_date +"_"+
    #     end_date +
    #     ".csv",
    #     header=False,
    #     float_format='%.3f',
    #     index=False)



def topo_map(df_mean,  mydtype, outname="outputmap.tif"):
    """
    Function to map results to toposub clusters generating map results.

    Args:
        df_mean (dataframe): an array of values to map to dem same length as number of toposub clusters
        mydtype: HS (maybeGST) needs "float32" other vars can use "int16" to save space

    Here
    's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron'
    s answer:
    https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy
    """

    # Build a "lookup array" where the index is the original value and the value
    # is the reclassified value.  Setting all of the reclassified values is cheap
    # because the memory is only allocated once for the lookup array.
    nclust = len(df_mean)
    lookup = np.arange(nclust, dtype=mydtype)

    for i in range(0, nclust):
        lookup[i] = df_mean[i]

    with rasterio.open('outputs/landform.tif') as src:
        # Read as numpy array
        array = src.read()
        profile = src.profile
        profile["dtype"] = mydtype

        # Reclassify in a single operation using broadcasting
        array = lookup[array]

    # rasterio.plot.show(array, cmap='viridis')
    # plt.show()

    with rasterio.open(outname, 'w', **profile) as dst:
        # Write to disk
        dst.write(array.astype(profile["dtype"] ))

    src = rasterio.open(outname)
    plt.imshow(src.read(1), cmap='viridis')
    plt.colorbar()
    plt.show()

def topo_map_headless(df_mean,  mydtype, outname="outputmap.tif"):
    """
    Headless server version of Function to map results to toposub clusters generating map results.

    Args:
        df_mean (dataframe): an array of values to map to dem same length as number of toposub clusters
        mydtype: HS (maybeGST) needs "float32" other vars can use "int16" to save space

    Here
    's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron'
    s answer:
    https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy
    """

    # Build a "lookup array" where the index is the original value and the value
    # is the reclassified value.  Setting all of the reclassified values is cheap
    # because the memory is only allocated once for the lookup array.
    nclust = len(df_mean)
    lookup = np.arange(nclust, dtype=mydtype)

    for i in range(0, nclust):
        lookup[i] = df_mean[i]

    with rasterio.open('outputs/landform.tif') as src:
        # Read as numpy array
        array = src.read()
        profile = src.profile
        profile["dtype"] = mydtype

        # Reclassify in a single operation using broadcasting
        array = lookup[array]

    # rasterio.plot.show(array, cmap='viridis')
    # plt.show()

    with rasterio.open(outname, 'w', **profile) as dst:
        # Write to disk
        dst.write(array.astype(profile["dtype"] ))


def topo_map_forcing(ds_var, n_decimals=2, dtype='float32', new_res=None):
    """
    Function to map forcing to toposub clusters generating gridded forcings

    Args:
        ds_var: single variable of ds eg. mp.downscaled_pts.t
        n_decimals (int): number of decimal to round vairable. default 2
        dtype (str): dtype to export raster. default 'float32'
        new_res (float): optional parameter to resample output to (in units of projection

    Return:
        grid_stack: stack of grids with dimension Time x Y x X

    Here
    's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron'
    s answer:
    https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy
    """

    # Build a "lookup array" where the index is the original value and the value
    # is the reclassified value.  Setting all of the reclassified values is cheap
    # because the memory is only allocated once for the lookup array.
    nclust = ds_var.shape[0]
    lookup = np.arange(nclust, dtype=dtype)

    # replicate looup through timedimens (dims Time X sample )
    lookup2D = np.tile(lookup, (ds_var.shape[1], 1))

    for i in range(0, nclust):
        lookup2D[:, i] = ds_var[i, :]

    from osgeo import gdal
    inputFile = "outputs/landform.tif"
    outputFile = "outputs/landform_newres.tif"

    # check if new_res is declared - if so resample landform and therefore output
    if new_res is not None:
        xres = new_res
        yres = new_res
        resample_alg = gdal.GRA_NearestNeighbour

        ds = gdal.Warp(destNameOrDestDS=outputFile,
                       srcDSOrSrcDSTab=inputFile,
                       format='GTiff',
                       xRes=xres,
                       yRes=yres,
                       resampleAlg=resample_alg)
        del ds
        landformfile = "outputs/landform_newres.tif"
    else:
        landformfile = "outputs/landform.tif"


    with rasterio.open(landformfile) as src:

        # new res
        # upscale_factor = src.res[0] / new_res
        # # resample data to target shape
        # array = src.read(
        #     out_shape=(
        #         src.count,
        #         int(src.height * upscale_factor),
        #         int(src.width * upscale_factor)
        #     ),
        #     resampling=Resampling.nearest
        # )

    # return coords of resampled grid here (this does not preserve dimensions perfectly (can be 1pix out))
        array = src.read()
        min_E, min_N, max_E, max_N = src.bounds
        lons = np.arange(min_E, max_E, src.res[0])
        lats = np.arange(min_N, max_N, src.res[1])
        lats = lats[::-1]

    array2 = lookup2D.transpose()[array]  # Reclassify in a single operation using broadcasting
    try:
        grid_stack = np.round(array2.squeeze().transpose(2, 0, 1) , n_decimals)# transpose to Time x Y x X
    except: # the case where we gen annual grids (no time dimension)
        grid_stack = np.round(array2.squeeze(), n_decimals)  # transpose to Time x Y x X
    return grid_stack, lats, lons

    # # rasterio.plot.show(array, cmap='viridis')

    # # plt.show()
    #
    # with rasterio.open('output_raster.tif', 'w', **profile) as dst:
    #     # Write to disk
    #     dst.write(array.astype(rasterio.int16))
    #
    # src = rasterio.open("output_raster.tif")
    # plt.imshow(src.read(1), cmap='viridis')
    # plt.colorbar()
    # plt.show()



def write_ncdf(wdir, grid_stack, var, units, longname, mytime, lats, lons, mydtype):
    # https://www.earthinversion.com/utilities/Writing-NetCDF4-Data-using-Python/

    # # coords
    # with rasterio.open("landform.tif") as src:
    #     # bounding box of image
    #     min_E, min_N, max_E, max_N = src.bounds
    #     # resolution of image
    #     res = src.res
    # lons = np.arange(min_E, max_E, res[0])
    # lats = np.arange(min_N, max_N, res[1])
    # lats = lats[::-1]

    try:
        ds = xr.Dataset(
             {var: (("Time", "northing", "easting"), grid_stack)},
             coords={
                 "Time":  mytime.data,
                 "northing": lats,
                 "easting": lons

             },
         )
    except:
        ds = xr.Dataset(
            {var: (("northing", "easting"), grid_stack)},
            coords={
                "northing": lats,
                "easting": lons

            },
        )

    ds.attrs["units"] = units  # add epsg here
    if var == "ta" or var=="tas" or var=="TA":
        ds.to_netcdf( wdir + "/outputs/"+str(mytime[0].values).split("-")[0]+str(mytime[0].values).split("-")[1]+".nc",
                      mode="w",
                      encoding={var: {"dtype": mydtype, 'zlib': True, 'complevel': 5}},
                      engine='h5netcdf')
    else:
        ds.to_netcdf( wdir + "/outputs/"+str(mytime[0].values).split("-")[0]+str(mytime[0].values).split("-")[1]+".nc",
                      mode="a",
                      encoding={var: {"dtype": mydtype, 'zlib': True, 'complevel': 5}},
                      engine='h5netcdf')


def agg_stats(df ):
    # generate aggregated stats from the simulation
    # USAGE:
    # HS = sim.agg_by_var_fsm(6)
    # SWE = sim.agg_by_var_fsm(5)
    # HSdf = agg_stats(HS)
    # SWEdf = agg_stats(SWE)
    # pddf = pd.concat([HSdf, SWEdf], axis=1)
    # avSnow.to_csv('avSnow.csv')
    #
    # from matplotlib import pyplot as plt
    # from matplotlib.backends.backend_pdf import PdfPages
    #
    # with PdfPages("plots2.pdf") as pdf:
    #         pddf.plot( subplots=True)
    #         plt.legend()
    #         pdf.savefig()  # saves the current figure into a pdf page
    #         plt.close()

    lp = pd.read_csv('listpoints.csv')

    weighteddf = df * lp.members
    dfagg = np.sum(weighteddf, 1)/ np.sum(lp.members)
    return (dfagg)

def climatology(ncol, fsm_path):
    HS = agg_by_var_fsm(ncol, fsm_path)
    HSdf = agg_stats(HS)
    
    # Group the DataFrame by day of the year and calculate the mean
    HSdf_daily_median = HSdf.groupby(HSdf.index.dayofyear).median()
    # Group the DataFrame by day of the year and calculate the quantiles
    HSdf_daily_quantiles = HSdf.groupby(HSdf.index.dayofyear).quantile([0.05, 0.95])

    return HSdf_daily_median, HSdf_daily_quantiles



def climatology_plot(mytitle, HSdf_daily_median, HSdf_daily_quantiles,  HSdf_realtime=None, plot_show=False):

    with PdfPages("snow_tracker.pdf") as pdf:

        df_shifted = HSdf_daily_median#.shift(-244)
        df_quantiles_shifted = HSdf_daily_quantiles#.shift(-244)

        # Concatenate the shifted time series such that the last 122 days are first, followed by the first 244 days - water year
        df_shifted0 = pd.concat([df_shifted[244:], df_shifted[:244]])
        df_quantiles_shifted1 = pd.concat([df_quantiles_shifted.xs(0.05, level=1)[244:], 
                                          df_quantiles_shifted.xs(0.05, level=1)[:244]],  axis=0)

        df_quantiles_shifted2 = pd.concat([df_quantiles_shifted.xs(0.95, level=1)[244:], 
                                          df_quantiles_shifted.xs(0.95, level=1)[:244]], axis=0)

        #s.reset_index(drop=True)   
        df_shifted0.reset_index(drop=True, inplace=True)   

        # Plot the timeseries and the quantiles
        plt.figure(figsize=(10, 6))
        plt.plot(df_shifted0, label='Long-term Median')
        plt.fill_between(df_shifted0.index, 
                         df_quantiles_shifted1, 
                         df_quantiles_shifted2, 
                         alpha=0.2, label='Long-term 5% - 95% Quantiles')

        # plot realtime data
        if HSdf_realtime is not None:
            # get water year
            thisYear = datetime.now().year
            thisMonth = datetime.now().month

            wy_start = {9,10,11,12}
            wy_end = {1, 2, 3, 4, 5,6,7,8}

            if thisMonth in wy_start:
                thisWaterYear = thisYear
            else:
                thisWaterYear = thisYear-1

            plt.plot(HSdf_realtime.values, label = "Water year " + str(thisWaterYear) )
        
        # plot fSCA (working on this)
        #plt.plot(fsca.DOY.astype(int)+(366-244)    , fsca.fSCA, label="fSCA MODIS (0-100%) ") # doy starts at jan 1 for modis data, here it is 1 sept so we shift by 122 days

        plt.xlabel('Day of the year')
        plt.ylabel('Height of snow (cm)')
        plt.title("Snow height " +mytitle+ " (Starting September 1st)")
        plt.legend()
        if plot_show ==True:
            plt.show()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()



# usage with a climate dir and realtime dir
# clim_dir = '/home/joel/sim/tscale_projects/naryn4'    
# realtime_dir = "/home/joel/mnt/aws/sim/naryn4"

# os.chdir(realtime_dir)
# ncol= 7 # aws fsm produces contant "hour" column
# HS = sim.agg_by_var_fsm(ncol)
# HS_realtime = sim.agg_stats(HS)

# os.chdir(clim_dir)
# ncol=ncol-1
# median, quantiles = sim.climatology(ncol)
# mytitle = clim_dir.split("/")[-1] 
# sim.climatology_plot(mytitle, median, quantiles, HS_realtime)

# # get modis
# # download routine
# # get domain parameters from landform
# # epsg, bbox = da.projFromLandform(wdir + "/landform.tif")

# # # process modis data
# # da.process_modis(wdir, epsg, bbox)

# fsca = da.extract_fsca_timeseries(clim_dir, plot=True)







