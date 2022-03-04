"""
Methods to generate required simulation files and run simulations of various models using tscale forcing
J. Fiddes, February 2022

TODO:

"""

import os
import glob
import re
import pandas as pd
import numpy as np
import rasterio
from matplotlib import pyplot as plt


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


def agg_by_var_fsm(ncol):
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
    a = glob.glob("./fsm_sims/sim_FSM_pt*")

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
    if ncol == 4:
        varname = "rof"
    if ncol == 5:
        varname = "hs"
    if ncol == 6:
        varname = "swe"
    if ncol == 7:
        varname = "gst"

    # add timestamp
    df.insert(0, 'Datetime', mydates)
    df = df.set_index("Datetime")
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


def topo_map(df_mean):
    """
    Function to map results to toposub clusters generating map results.

    Args:
        df_mean (dataframe): an array of values to map to dem same length as number of toposub clusters

    Here
    's an approach for arbitrary reclassification of integer rasters that avoids using a million calls to np.where. Rasterio bits taken from @Aaron'
    s answer:
    https://gis.stackexchange.com/questions/163007/raster-reclassify-using-python-gdal-and-numpy
    """

    # Build a "lookup array" where the index is the original value and the value
    # is the reclassified value.  Setting all of the reclassified values is cheap
    # because the memory is only allocated once for the lookup array.
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

    # rasterio.plot.show(array, cmap='viridis')
    # plt.show()

    with rasterio.open('output_raster.tif', 'w', **profile) as dst:
        # Write to disk
        dst.write(array.astype(rasterio.int16))

    src = rasterio.open("output_raster.tif")
    plt.imshow(src.read(1), cmap='viridis')
    plt.colorbar()
    plt.show()




