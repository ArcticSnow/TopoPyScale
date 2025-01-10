"""
Methods to generate required simulation files and run simulations of various models using tscale forcing
J. Fiddes, February 2022

TODO:

"""
import os, re
import sys
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
from TopoPyScale import topo_utils as tu
from TopoPyScale import topo_export as te

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


def to_netcdf(fname_fsm_sim, complevel=9):
    '''
    Function to convert a single FSM simulation output file (.txt) to a compressed netcdf file (.nc)

    Args:
        fname_fsm_sim(str): filename to convert from txt to nc
        complevel (int): Compression level. 1-9

    Returns:
        NULL (FSM simulation file written to disk)
    '''
    ver_dict = tu.get_versionning()

    ds = txt2ds(fname_fsm_sim)
    ds.albedo.attrs = {'units':'ratio', 'standard_name':'albedo', 'long_name':'Surface Effective Albedo', '_FillValue': -9999999.0}
    ds.runoff.attrs = {'units':'kg m-2', 'standard_name':'runoff', 'long_name':'Cumulated runoff from snow', '_FillValue': -9999999.0}
    ds.snd.attrs = {'units':'m', 'standard_name':'snd', 'long_name':'Average snow depth', '_FillValue': -9999999.0}
    ds.swe.attrs = {'units':'kg m-2', 'standard_name':'swe', 'long_name':'Average snow water equivalent', '_FillValue': -9999999.0}
    ds.t_surface.attrs = {'units':'°C', 'standard_name':'t_surface', 'long_name':'Average surface temperature', '_FillValue': -9999999.0}
    ds.t_surface.attrs = {'units':'°C', 'standard_name':'t_soil', 'long_name':'Average soil temperature at 20 cm depth', '_FillValue': -9999999.0}
    ds.attrs = {'title':'FSM simulation outputs',
                'source': 'Data downscaled with TopoPyScale and simulated with FSM',
                'package_version':ver_dict.get('package_version'),
                'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                'url_FSM': 'https://github.com/RichardEssery/FSM',
                'git_commit': ver_dict.get('git_commit'),
                'date_created':dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    fout = f"{fname_fsm_sim[:-4]}.nc"
    te.to_netcdf(ds, fout, complevel=complevel)
    print(f"File {fout} saved")


def to_netcdf_parallel(fsm_sims='./fsm_sims/sim_FSM*.txt',
                       n_core=6,
                       complevel=9,
                       delete_txt_files=False):
    '''
    Function to convert FSM simulation output (.txt files) to compressed netcdf. This function parallelize jobs on n_core

    Args:
        fsm_sims (str or list): file pattern or list of file output of FSM. Note that must be in short relative path!. Default = 'fsm_sims/sim_FSM_pt*.txt'
        n_core (int): number of cores. Default = 6:
        complevel (int): Compression level. 1-9
        delete_txt_files (bool): delete simulation txt files or not. Default=True

    Returns:
        NULL (FSM simulation file written to disk)
    '''
    print('---> Run FSM simulation in parallel')
    # 1. create all nlsit_files
    if isinstance(fsm_sims, str):
        flist = glob.glob(fsm_sims)
    elif isinstance(fsm_sims, list):
        flist = fsm_sims
    else:
        print('ERROR: fsm_input must either be a list of file path or a string of file pattern')
        return

    fun_param = zip(flist, [complevel]*len(flist))
    tu.multicore_pooling(to_netcdf, fun_param, n_core)
    print('---> All FSM simulations stored as netcdf')

    if delete_txt_files:
        print('---> Removing FSM simulation .txt file')
        for file in flist:
            os.remove(file)



def to_dataset(fname_pattern='sim_FSM_pt*.txt', fsm_path = "./fsm_sims/"):
    '''
    Function to read FSM outputs of one simulation into a single dataset.
    Args:
        fname_pattern:
        fsm_path:

    Returns:
        dataset (xarray)
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

def _run_fsm(fsm_exec, nlstfile):
    '''
    function to execute FSM
    Args:
        fsm_exec (str): path to FSM executable
        nlstfile (str): path to FSM simulation config file

    Returns:
        NULL (FSM simulation file written to disk)
    '''
    os.system(fsm_exec + ' < ' + nlstfile)
    print('Simulation done: ' + nlstfile)

def fsm_sim_parallel(fsm_input='outputs/FSM_pt*.txt',
                     fsm_nconfig=31,
                     fsm_nave=24,  # for daily output
                     fsm_exec='./FSM',
                     n_core=6,
                     n_thread=100,
                     delete_nlst_files=True):
    '''
    Function to run parallelised simulations of FSM

    Args:
        fsm_input (str or list): file pattern or list of file input to FSM. Note that must be in short relative path!. Default = 'outputs/FSM_pt*.txt'
        fsm_nconfig (int):  FSM configuration number. See FSM README.md: https://github.com/RichardEssery/FSM. Default = 31
        fsm_nave (int): number of timestep to average for outputs. e.g. If input is hourly and fsm_nave=24, outputs will be daily. Default = 24
        fsm_exec (str): path to FSM executable. Default = './FSM'
        n_core (int): number of cores. Default = 6
        n_thread (int): number of threads when creating simulation configuration files. Default=100
        delete_nlst_files (bool): delete simulation configuration files or not. Default=True

    Returns:
        NULL (FSM simulation file written to disk)
    '''
    print('---> Run FSM simulation in parallel')
    # 1. create all nlsit_files
    if isinstance(fsm_input, str):
        met_flist = glob.glob(fsm_input)
    elif isinstance(fsm_input, list):
        met_list = fsm_input
    else:
        print('ERROR: fsm_input must either be a list of file path or a string of file pattern')
        return

    fun_param = zip([fsm_nconfig]*len(met_flist), met_flist, [fsm_nave]*len(met_flist) )
    tu.multithread_pooling(fsm_nlst, fun_param, n_thread)
    print('nlst files ready')

    # 2. execute FSM on multicore
    nlst_flist = glob.glob('fsm_sims/nlst_*.txt')
    fun_param = zip([fsm_exec]*len(nlst_flist), nlst_flist)
    tu.multicore_pooling(_run_fsm, fun_param, n_core)
    print('---> FSM simulations finished.')

    # 3. remove nlist files
    if delete_nlst_files:
        print('---> Removing FSM simulation config file')
        for file in nlst_flist:
            os.remove(file)


def fsm_sim(nlstfile, fsm_exec, delete_nlst_files=True):
    """
    Function to simulate the FSM model
    https://github.com/RichardEssery/FSM

    Args:
        nlstfile (int): which FSm configuration to run (integer 1-31)
        fsm_exec (str): path to input tscale file (relative as Fortran fails with long strings (max 21 chars?))
    
    Returns: 
        NULL (FSM simulation file written to disk)
    """

    os.system(fsm_exec + ' < ' + nlstfile)
    if delete_nlst_files:
        print('---> Removing FSM simulation config file')
        for file in nlst_flist:
            os.remove(nlstfile)
    print('Simulation done: ' + nlstfile)





def agg_by_var_fsm( var='snd', fsm_path = "./fsm_sims"):
    """
    Function to make single variable multi cluster files as preprocessing step before spatialisation. This is much more efficient than looping over individual simulation files per cluster.
    For V variables , C clusters and T timesteps this turns C individual files of dimensions V x T into V individual files of dimensions C x T.

    Currently written for FSM files but could be generalised to other models.

    Args:
        var (str): column name of variable to extract, one of: alb, rof, hs, swe, gst, gt50
                alb - albedo
                rof - runoff
                snd - snow height (m)
                swe - snow water equivalent (mm)
                gst - ground surface temp (10cm depth) degC
                gt50 - ground temperature (50cm depth) degC

        fsm_path (str): location of simulation files
    Returns: 
        dataframe



    """

    # find all simulation files and natural sort https://en.wikipedia.org/wiki/Natural_sort_order
    a = glob.glob(fsm_path+"/sim_FSM_pt*")

    if len(a) == 0:                                                                                                                                                                                                                           
        sys.exit("ERROR: " +fsm_path + " does not exist or is empty")   


    def natural_sort(l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    fsm_columns = {'alb':-6,
                   'rof':-5,
                   'snd':-4,
                   'swe':-3,
                   'gst':-2,
                   'gt50':-1}


    if var.lower() in ['alb', 'rof', 'snd', 'swe', 'gst', 'gt50']:
        ncol = int(fsm_columns.get(var))
    else:
        print("indicate ncol or var within ['alb', 'rof', 'snd', 'swe', 'gst', 'gt50']")

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

    print(f'Variable {var} extracted')
    return df
    # df.to_csv('./fsm_sims/'+ varname +'.csv', index=False, header=True)


def agg_by_var_fsm_ensemble( var='snd', W=1):
    """
    Function to make single variable multi cluster files as preprocessing step before spatialisation. This is much more efficient than looping over individual simulation files per cluster.
    For V variables , C clusters and T timesteps this turns C individual files of dimensions V x T into V individual files of dimensions C x T.

    Currently written for FSM files but could be generalised to other models.

    Args:
        var (str): column name of variable to extract, one of: alb, rof, hs, swe, gst, gt50
                alb - albedo
                rof - runoff
                hs - snow height (m)
                swe - snow water equivalent (mm)
                gst - ground surface temp (10cm depth) degC
                gt50 - ground temperature (50cm depth) degC

        W - weight vector from PBS
    Returns: 
        dataframe

    ncol:
        4 = rof
        5 = hs
        6 = swe
        7 = gst

    """

    # find all simulation files and natural sort https://en.wikipedia.org/wiki/Natural_sort_order
    a = glob.glob("./fsm_sims/sim_ENS*_FSM_pt*")
    fsm_columns = {'alb':-6,
                   'rof':-5,
                   'snd':-4,
                   'swe':-3,
                   'gst':-2,
                   'gt50':-1}

    if var.lower() in ['alb', 'rof', 'snd', 'swe', 'gst', 'gt50']:
        ncol = int(fsm_columns.get(var))
    else:
        print("indicate ncol or var within ['alb', 'rof', 'snd', 'swe', 'gst', 'gt50']")


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

    print(f'Variable {var} extracted')
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

    # return coords of resampled grid here (this does not preserve dimensions perfectly (can be 1pix out)) -FIXED!
        array = src.read()
        min_E, min_N, max_E, max_N = src.bounds
        # lons = np.arange(min_E, max_E, src.res[0])
        # lats = np.arange(min_N, max_N, src.res[1])
        # lats = lats[::-1]
        lons = np.linspace(min_E, max_E, src.width)
        lats = np.linspace(min_N, max_N, src.height)
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


def topo_map_sim(ds_var, n_decimals=2, dtype='float32', new_res=None):
    """
    Function to map sim results to toposub clusters generating gridded results

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
    nclust = ds_var.shape[1]
    lookup = np.arange(nclust, dtype=dtype)

    # replicate looup through timedimens (dims Time X sample )
    lookup2D = np.tile(lookup, (ds_var.shape[0], 1))

    for i in range(0, nclust):
        lookup2D[:, i] = ds_var[i]

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

        # return coords of resampled grid here (this does not preserve dimensions perfectly (can be 1pix out)) -FIXED!
        array = src.read()
        min_E, min_N, max_E, max_N = src.bounds
        # lons = np.arange(min_E, max_E, src.res[0])
        # lats = np.arange(min_N, max_N, src.res[1])
        # lats = lats[::-1]
        lons = np.linspace(min_E, max_E, src.width)
        lats = np.linspace(min_N, max_N, src.height)
        lats = lats[::-1]

    array2 = lookup2D.transpose()[array]  # Reclassify in a single operation using broadcasting
    try:
        grid_stack = np.round(array2.squeeze().transpose(2, 0, 1), n_decimals)  # transpose to Time x Y x X
    except:  # the case where we gen annual grids (no time dimension)
        grid_stack = np.round(array2.squeeze(), n_decimals)  # transpose to Time x Y x X
    return grid_stack, lats, lons



def write_ncdf(wdir, grid_stack, var, units,epsg,res,  mytime, lats, lons, mydtype,newfile, outname=None):
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
                 "Time":  mytime,
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

    ds.attrs["units"] = units 
    ds.attrs["epsg_code"] = epsg # add epsg here
    ds.attrs["resolution_m"] = str(res) # add epsg here

    if newfile == True:
        ds.to_netcdf( wdir + "/outputs/"+str(mytime[0]).split("-")[0]+str(mytime[0]).split("-")[1]+"_"+outname+".nc", mode="w", encoding={var: {"dtype": mydtype, 'zlib': True, 'complevel': 5} })
    else:
        ds.to_netcdf( wdir + "/outputs/"+str(mytime[0]).split("-")[0]+str(mytime[0]).split("-")[1]+"_"+outname+".nc", mode="a", encoding={var: {"dtype": mydtype, 'zlib': True, 'complevel': 5} })
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

def climatology(HSdf, fsm_path):
    # taken this out of this function:
    # HS = agg_by_var_fsm(ncol, fsm_path)
    # HSdf = agg_stats(HS)
    
    # Group the DataFrame by day of the year and calculate the mean
    HSdf_daily_median = HSdf.groupby(HSdf.index.dayofyear).median()
    # Group the DataFrame by day of the year and calculate the quantiles
    HSdf_daily_quantiles = HSdf.groupby(HSdf.index.dayofyear).quantile([0.05, 0.95])

    return HSdf_daily_median, HSdf_daily_quantiles



def climatology_plot(var, mytitle, HSdf_daily_median, HSdf_daily_quantiles,  HSdf_realtime=None, plot_show=False):

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
        if var == 'snd':
            plt.ylabel('Height of snow (cm)')
            plt.title("Basin average snow height " +mytitle+ " (Starting September 1st)")
        if var == 'swe':
            plt.ylabel('Snow water equivalent (mm)')
            plt.title("Basin average snow water equivalent " +mytitle+ " (Starting September 1st)")
        plt.legend()
        if plot_show ==True:
            plt.show()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


def climatology_plot2(mytitle, HSdf_daily_median, HSdf_daily_quantiles, HSdf_realtime=None, plot_show=False):
    with PdfPages("snow_tracker.pdf") as pdf:

        df_shifted = HSdf_daily_median  # .shift(-244)
        df_quantiles_shifted = HSdf_daily_quantiles  # .shift(-244)

        # Concatenate the shifted time series such that the last 122 days are first, followed by the first 244 days - water year
        df_shifted0 = pd.concat([df_shifted[244:], df_shifted[:244]])
        df_quantiles_shifted1 = pd.concat([df_quantiles_shifted.xs(0.05, level=1)[244:],
                                           df_quantiles_shifted.xs(0.05, level=1)[:244]], axis=0)

        df_quantiles_shifted2 = pd.concat([df_quantiles_shifted.xs(0.95, level=1)[244:],
                                           df_quantiles_shifted.xs(0.95, level=1)[:244]], axis=0)

        # s.reset_index(drop=True)
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

            wy_start = {9, 10, 11, 12}
            wy_end = {1, 2, 3, 4, 5, 6, 7, 8}

            if thisMonth in wy_start:
                thisWaterYear = thisYear
            else:
                thisWaterYear = thisYear - 1

            plt.plot(HSdf_realtime.values, label="Water year " + str(thisWaterYear))

        # plot fSCA (working on this)
        # plt.plot(fsca.DOY.astype(int)+(366-244)    , fsca.fSCA, label="fSCA MODIS (0-100%) ") # doy starts at jan 1 for modis data, here it is 1 sept so we shift by 122 days

        plt.xlabel('Day of the year')
        plt.ylabel('Height of snow (cm)')
        plt.title("Snow height " + mytitle + " (Starting September 1st)")
        plt.legend()
        if plot_show == True:
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


def concat_fsm(mydir):
    
    """ 
    A small routine to concatinate fsm results from separate years in a single file. 
    This is mainly needed when a big job is split into years for parallel processing on eg a cluster

    rsync -avz --include="<file_pattern>" --include="*/" --exclude="*" <username>@<remote_server_address>:/path/to/remote/directory/ <local_destination_directory>
    
    """
    #mydir = "/home/joel/sim/tscale_projects/aws_debug/b6/"

    def natural_sort(l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    # find all the differnt files we have (1 per cluster)
    file_paths = glob.glob(mydir +"fsm_clim/sim_2000/fsm_sims/*")
    nclust = len(files)


    def get_file_names(file_paths):
        file_names = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_names.append(file_name)
        return file_names

    names = get_file_names(file_paths)
    filenames = natural_sort(names)

    for myname in filenames:
        a = glob.glob(mydir + "/fsm_clim/sim_*/fsm_sims/" + myname)

        filenames2 = natural_sort(a)
        with open(mydir +"/fsm_clim/" + myname, 'w') as outfile:
            for fname in filenames2:
                with open(fname) as infile:
                    outfile.write(infile.read())

