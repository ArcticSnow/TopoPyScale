'''
Methods to perform FSM2oshd simulations
S. Filhol, September 2023

TODO:

- open and forest namelist
- run open/forest simulation
- combine results based on weights of open vs forest
- recover information of forest cover fraction from namelist file. Python namelist parser package?


'''
from TopoPyScale import topo_utils as tu
from TopoPyScale import topo_export as te
from pathlib import Path
import pandas as pd
import xarray as xr
import linecache
import glob, os, re


def _run_fsm2oshd(fsm_exec, nam_file):
    '''
    function to execute FSM
    Args:
        fsm_exec (str): path to FSM executable
        nam_file (str): path to FSM simulation config file  .nam

    Returns:
        NULL (FSM simulation file written to disk)
    '''
    os.system(fsm_exec + ' ' + nam_file)
    print('Simulation done: ' + nam_file)

    # convert output to netcdf file
    fout = linecache.getline(nam_file, 16).split("'")[1]
    to_netcdf(fout)
    os.remove(fout)

def fsm2oshd_sim_parallel(simulation_path='./fsm_sim,',
                          fsm_forest_nam='fsm__forest*.nam',
                          fsm_open_nam='fsm_sim/fsm__open*.nam',
                          fsm_exec='./FSM_OSHD',
                          n_cores=6,
                          delete_nlst_files=False):
    '''
    Function to run parallelised simulations of FSM

    Args:
        fsm_forest_nam (str): file pattern of namelist file for forest setup. Default = 'fsm_sim/fsm__forest*.nam'
        fsm_open_nam (str): file pattern of namelist file for open setup. Default = 'fsm_sim/fsm__open*.nam'
        fsm_exec (str): path to FSM executable. Default = './FSM'
        n_cores (int): number of cores. Default = 6
        delete_nlst_files (bool): delete simulation configuration files or not. Default=True

    Returns:
        NULL (FSM simulation file written to disk)
    '''
    print('---> Run FSM simulation in parallel')

    # 2. execute FSM on multicore for Forest
    nlst_forest = glob.glob(fsm_forest_nam)
    fun_param = zip([fsm_exec]*len(nlst_forest), nlst_forest)
    tu.multicore_pooling(_run_fsm2oshd, fun_param, n_cores)
    print('---> FSM2oshd forest simulations finished.')

    # 2. execute FSM on multicore for Open
    nlst_open = glob.glob(fsm_open_nam)
    fun_param = zip([fsm_exec]*len(nlst_open), nlst_open)
    tu.multicore_pooling(_run_fsm2oshd, fun_param, n_cores)
    print('---> FSM2oshd open simulations finished.')

    # 3. remove nlist files
    if delete_nlst_files:
        print('---> Removing FSM simulation config files')
        for file in nlst_forest:
            os.remove(file)
        for file in nlst_open:
            os.remove(file)



def txt2ds(fname):
    '''
    Function to read a single FSM text file output as a xarray dataset
    Args:
        fname (str): filename

    Returns:
        xarray dataset of dimension (time, point_id)
    '''
    df = read_pt_fsm2oshd(fname)
    point_id = int( re.findall(r'\d+', fname.split('/')[-1])[-1])
    print(f'---> Reading FSM data for point_id = {point_id}')
    ds = xr.Dataset({
        "sd": (['time'], df.sd.values),
        "scf":  (['time'], df.scf.values),
        "swe":  (['time'], df.swe.values),
        "t_surface":  (['time'], df.tsurf.values),
        "t_soil":  (['time'], df.tsoil.values),
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
    ds.sd.attrs = {'units':'m', 'standard_name':'sd', 'long_name':'Average snow depth', '_FillValue': -9999999.0}
    ds.scf.attrs = {'units':'%', 'standard_name':'scf', 'long_name':'Average snow cover fraction', '_FillValue': -9999999.0}
    ds.swe.attrs = {'units':'kg m-2', 'standard_name':'swe', 'long_name':'Average snow water equivalent', '_FillValue': -9999999.0}
    ds.t_surface.attrs = {'units':'°C', 'standard_name':'t_surface', 'long_name':'Average surface temperature', '_FillValue': -9999999.0}
    ds.t_soil.attrs = {'units':'°C', 'standard_name':'t_soil', 'long_name':'Average soil temperature at 20 cm depth', '_FillValue': -9999999.0}
    ds.attrs = {'title':'FSM2oshd simulation outputs',
                'source': 'Data downscaled with TopoPyScale and simulated with FSM',
                'package_TopoPyScale_version':ver_dict.get('package_version'),
                'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                'url_FSM': 'https://github.com/ArcticSnow/FSM2oshd',
                'git_commit': ver_dict.get('git_commit'),
                'date_created':dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    fout = f"{fname_fsm_sim[:-4]}.nc"
    te.to_netcdf(ds, fout, complevel=complevel)
    print(f"File {fout} saved")



def combine_open_forest(df_forest,
                        fout_forest='fsm_sim/fsm_out_forest.nc',
                        fout_open='fsm_sim/fsm_out_open.nc'):
    '''
    Function to compute weighted average of forest and open simulations
    Args:
        df_forest: dataframe look up table for proportion of forested pixels vs open in a given cluster
        fout_forest: filename of netcdf file with forest simulation output
        fout_open: filename of netcdf file with open simulation output

    Returns:

    '''
    dsf = xr.open_dataset(fname)
    dso = xr.open_dataset(fout_open)
    point_id = dsf.point_id.values
    ds = dsf * df_forest.proportion_with_forest[point_id] + dso * (1-df_forest.proportion_with_forest[point_id])

    return ds

def read_pt_fsm2oshd(fname):
    df = pd.read_csv(fname, delim_whitespace=True, header=None,names=['year', 'month', 'day', 'hour', 'sd', 'scf', 'swe', 'tsurf','tsoil'])
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('time')
    return df