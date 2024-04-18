'''
Methods to perform FSM2oshd simulations
S. Filhol, September 2023

TODO:

- open and forest namelist
- run open/forest simulation
- combine results based on weights of open vs forest
- recover information of forest cover fraction from namelist file. Python namelist parser package?


'''
import pdb

from TopoPyScale import topo_utils as tu
from TopoPyScale import topo_export as te
from pathlib import Path
import datetime as dt
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
                          fsm_open_nam='fsm__open*.nam',
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
    p = Path(simulation_path)
    # 2. execute FSM on multicore for Forest
    nlst_forest = glob.glob(str(p / fsm_forest_nam))
    fun_param = zip([fsm_exec]*len(nlst_forest), nlst_forest)
    tu.multicore_pooling(_run_fsm2oshd, fun_param, n_cores)
    print('---> FSM2oshd forest simulations finished.')

    # 2. execute FSM on multicore for Open
    nlst_open = glob.glob(str(p / fsm_open_nam))
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
        xarray dataset of dimension (time, point_name)
    '''
    df = read_pt_fsm2oshd(fname)
    point_ind = int( re.findall(r'\d+', fname.split('/')[-1])[-1])
    print(f'---> Reading FSM data for point_name = {point_ind}')
    ds = xr.Dataset({
        "sd": (['time'], df.sd.values),
        "scf":  (['time'], df.scf.values),
        "swe":  (['time'], df.swe.values),
        "tsurf":  (['time'], df.tsurf.values),
        "tsoil_1":  (['time'], df.tsoil_1.values),
        "tsoil_2":  (['time'], df.tsoil_2.values),
        "tsoil_3":  (['time'], df.tsoil_3.values),
        "tsoil_4":  (['time'], df.tsoil_4.values)
        },
        coords={
            "point_ind": point_ind,
            "time": df.index,
            "reference_time": pd.Timestamp(df.index[0])
        })

    return ds

def output_to_ds(fname_pattern, complevel=9, remove_file=False, fout='fsm_outputs'):
    flist = glob.glob(fname_pattern)
    flist.sort()

    def _to_ds(fname, fout='fsm_outputs', complevel=9, remove_file=False, n_digits=3):
        ds = xr.open_dataset(fname)
        point_ind = ds.point_ind.values
        fname_out = f'{fout}_{str(point_ind).zfill(n_digits)}.nc'
        te.to_netcdf(ds, fname_out)
        if remove_file:
            os.remove(fname)
        print(f'--> File {fname_out} saved')

    fun_param = zip(flist,
                    ['fsm_output_open'] * len(flist),
                    flist_open,)
    tu.multicore_pooling(_to_ds, fun_param, n_cores)
    print('---> All fsm outputs stored as netcdf')


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
    ds.tsurf.attrs = {'units':'°C', 'standard_name':'t_surface', 'long_name':'Average surface temperature', '_FillValue': -9999999.0}
    ds.tsoil_1.attrs = {'units':'°C', 'standard_name':'tsoil_1', 'long_name':'Average soil temperature at 5 cm depth', '_FillValue': -9999999.0}
    ds.tsoil_2.attrs = {'units':'°C', 'standard_name':'tsoil_2', 'long_name':'Average soil temperature at 15 cm depth', '_FillValue': -9999999.0}
    ds.tsoil_3.attrs = {'units':'°C', 'standard_name':'tsoil_3', 'long_name':'Average soil temperature at 35 cm depth', '_FillValue': -9999999.0}
    ds.tsoil_4.attrs = {'units':'°C', 'standard_name':'tsoil_4', 'long_name':'Average soil temperature at 75 cm depth', '_FillValue': -9999999.0}
    ds.attrs = {'title':'FSM2oshd simulation outputs',
                'source': 'Data downscaled with TopoPyScale and simulated with FSM',
                'package_TopoPyScale_version':ver_dict.get('package_version'),
                'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                'url_FSM': 'https://github.com/ArcticSnow/FSM2oshd',
                'git_commit': ver_dict.get('git_commit'),
                'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    fout = f"{fname_fsm_sim[:-4]}.nc"
    te.to_netcdf(ds, fout, complevel=complevel)
    print(f"File {fout} saved")



def _combine_open_forest(fname_df_forest='fsm_sim/df_forest.pckle',
                         fname_forest='fsm_sim/fsm_out_forest.nc',
                         fname_open='fsm_sim/fsm_out_open.nc',
                         remove_forest_open_file=False,
                         fout=None,
                         n_digits=3):
    '''
    Function to compute weighted average of forest and open simulations from single simulation output files
    Args:
        df_forest: dataframe look up table for proportion of forested pixels vs open in a given cluster
        fname_forest: filename of netcdf file with forest simulation output
        fname_open: filename of netcdf file with open simulation output

    Returns:

    '''
    df_forest = pd.read_pickle(fname_df_forest)
    dsf = xr.open_dataset(fname_forest)
    dso = xr.open_dataset(fname_open)
    point_ind = dsf.point_ind.values

    coeff = df_forest.proportion_with_forest.iloc[point_ind]
    if coeff <= 0.1:
        coeff = 0
    ds = dsf * coeff + dso * (1-coeff)

    fname_out = f'{fout}_{str(point_ind).zfill(n_digits)}.nc'
    te.to_netcdf(ds, fname_out)
    if remove_forest_open_file:
        os.remove(fname_open)
        os.remove(fname_forest)
    print(f'--> File {fname_out} saved')

def aggregate_all_open_forest(fname_df_forest,
                              fname_forest_outputs='fsm_sim/fsm_outputs_fores_*.nc',
                              fname_open_outputs='fsm_sim/fsm_outputs_open_*.nc',
                              fout='fsm_sim/fsm_outputs',
                              n_cores=6,
                              n_digits=3,
                              remove_tmp_outputs=True):

    flist_forest = glob.glob(fname_forest_outputs)
    flist_open = glob.glob(fname_open_outputs)
    flist_forest.sort()
    flist_open.sort()

    fun_param = zip([fname_df_forest] * len(flist_forest),
                    flist_forest,
                    flist_open,
                    [remove_tmp_outputs] * len(flist_forest),
                    [fout] * len(flist_forest),
                    [n_digits] * len(flist_forest))
    tu.multicore_pooling(_combine_open_forest, fun_param, n_cores)
    print('---> All forest and open outputs combined')



def read_pt_fsm2oshd(fname):
    df = pd.read_csv(fname, delim_whitespace=True, header=None, names=['year', 'month', 'day', 'hour', 'sd', 'scf', 'swe', 'tsurf','tsoil_1','tsoil_2','tsoil_3','tsoil_4'])
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('time')
    return df

def read_pt_met_fsm2oshd(fname):
    df = pd.read_csv(fname, delim_whitespace=True, header=None, names=['year', 'month', 'day', 'hour', 'SWdir', 'SWdif', 'LW', 'Sf', 'Rf', 'Ta', 'RH', 'Ua', 'Ps', 'Sf24h', 'Tvt'])
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('time')
    return df