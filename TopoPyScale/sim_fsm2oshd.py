'''
Methods to perform FSM2oshd simulations
S. Filhol, September 2023

Workflow:
1. topo_export.to_fsm2oshd() writes, for each point, a met file and namelists for the 'forest' and 'open' modes:
   <simulation_path>/<fname_format>_forest_<ind>.nam, ..._open_<ind>.nam, outputs ..._outputs_<mode>_<ind>.txt
2. fsm2oshd_sim_parallel() runs FSM2oshd on all namelists and converts each text output to netcdf
3. aggregate_all_open_forest() combines the forest and open outputs of each point, weighted by the forest proportion
4. read_fsm_ds_with_mask() reads the combined outputs as one dataset
'''
import datetime as dt
import glob
import os
import re
import shlex
import shutil
import subprocess
import warnings
from pathlib import Path

import pandas as pd
import xarray as xr

from TopoPyScale import topo_export as te
from TopoPyScale import topo_utils as tu
import logging

logger = logging.getLogger(__name__)

N_BITS = 15  # packing precision of FSM netcdf outputs (int16): resolution = range / 32767
FSM_OUTPUT_COLUMNS = ['year', 'month', 'day', 'hour', 'sd', 'scf', 'swe', 'tsurf',
                      'tsoil_1', 'tsoil_2', 'tsoil_3', 'tsoil_4']
FSM_MET_COLUMNS = ['year', 'month', 'day', 'hour', 'SWdir', 'SWdif', 'LW', 'Sf', 'Rf', 'Ta', 'RH', 'Ua', 'Ps',
                   'Sf24h', 'Tvt']
OUTPUT_ATTRS = {
    'sd': {'units': 'm', 'standard_name': 'sd', 'long_name': 'Average snow depth'},
    'scf': {'units': '%', 'standard_name': 'scf', 'long_name': 'Average snow cover fraction'},
    'swe': {'units': 'kg m-2', 'standard_name': 'swe', 'long_name': 'Average snow water equivalent'},
    'tsurf': {'units': '°C', 'standard_name': 't_surface', 'long_name': 'Average surface temperature'},
    'tsoil_1': {'units': '°C', 'standard_name': 'tsoil_1', 'long_name': 'Average soil temperature at 5 cm depth'},
    'tsoil_2': {'units': '°C', 'standard_name': 'tsoil_2', 'long_name': 'Average soil temperature at 15 cm depth'},
    'tsoil_3': {'units': '°C', 'standard_name': 'tsoil_3', 'long_name': 'Average soil temperature at 35 cm depth'},
    'tsoil_4': {'units': '°C', 'standard_name': 'tsoil_4', 'long_name': 'Average soil temperature at 75 cm depth'},
}


def _point_index(fname):
    """Point index: last group of digits of the file name (e.g. fsm__outputs_forest_012.nc -> 12)."""
    digits = re.findall(r'\d+', Path(fname).name)
    if not digits:
        raise ValueError(f'No point index in file name {fname}')
    return int(digits[-1])


def namelist_output_file(nam_file):
    """Name of the output file declared in a FSM2oshd namelist (out_file = '...')."""
    with open(nam_file) as f:
        match = re.search(r"out_file\s*=\s*'([^']+)'", f.read())
    if match is None:
        raise ValueError(f'No out_file in namelist {nam_file}')
    return match.group(1)


def read_fsm_ds_with_mask(path):
    '''
    Read FSM netcdf outputs (file pattern) as one dataset along point_ind, with a dummy point -9999 for the
    masked area.
    '''
    ds = tu.open_netcdf_stack(sorted(glob.glob(path)), concat_dim='point_ind')
    tp = ds.isel(point_ind=0).copy()
    tp['point_ind'] = -9999
    for var in list(tp.keys()):
        tp[var] = tp[var] * 0 - 9999

    ds_complete = xr.concat([tp, ds], dim='point_ind')
    return ds_complete


def _run_fsm2oshd(fsm_exec, nam_file, keep_txt=False):
    '''
    Run FSM2oshd on one namelist and convert its text output to netcdf.

    Args:
        fsm_exec (str): FSM executable (may include arguments, e.g. a wrapper command)
        nam_file (str): path to FSM simulation config file .nam
        keep_txt (bool): keep the text output after conversion

    Returns:
        tuple: (nam_file, error message or None)
    '''
    proc = subprocess.run(shlex.split(fsm_exec) + [str(nam_file)], capture_output=True, text=True)
    fout = namelist_output_file(nam_file)
    if proc.returncode != 0 or not os.path.isfile(fout):
        msg = (proc.stderr or proc.stdout or '').strip()[-500:]
        return str(nam_file), f'exit code {proc.returncode}, output {fout} {"found" if os.path.isfile(fout) else "missing"}: {msg}'
    logger.info('Simulation done: ' + str(nam_file))
    to_netcdf(fout)
    if not keep_txt:
        os.remove(fout)
    return str(nam_file), None


def fsm2oshd_sim_parallel(simulation_path='./fsm_sim',
                          fsm_forest_nam='fsm__forest*.nam',
                          fsm_open_nam='fsm__open*.nam',
                          fsm_exec='./FSM_OSHD',
                          n_cores=6,
                          delete_nlst_files=False,
                          keep_txt=False):
    '''
    Function to run parallelised simulations of FSM

    Args:
        simulation_path (str): folder of the namelists. Default './fsm_sim'
        fsm_forest_nam (str): file pattern of namelist file for forest setup. Default = 'fsm__forest*.nam'
        fsm_open_nam (str): file pattern of namelist file for open setup. Default = 'fsm__open*.nam'. If None, then only performs forest simulation
        fsm_exec (str): path to FSM executable. Default = './FSM_OSHD'
        n_cores (int): number of cores. Default = 6
        delete_nlst_files (bool): delete simulation configuration files or not. Default=False
        keep_txt (bool): keep the text outputs of FSM after their conversion to netcdf. Default=False

    Returns:
        list: namelists whose simulation failed (empty if all succeeded). A warning lists them.
    '''
    exe = shlex.split(fsm_exec)[0]
    if shutil.which(exe) is None and not os.path.isfile(exe):
        raise FileNotFoundError(f'FSM executable {exe} not found')

    logger.info('---> Run FSM simulation in parallel')
    p = Path(simulation_path)
    patterns = [fsm_forest_nam] + ([fsm_open_nam] if fsm_open_nam is not None else [])
    failed, nlst_all = [], []
    for pattern in patterns:
        nlst = sorted(glob.glob(str(p / pattern)))
        if not nlst:
            warnings.warn(f'No namelist matching {p / pattern}')
            continue
        nlst_all += nlst
        results = tu.multicore_pooling(_run_fsm2oshd, zip([fsm_exec] * len(nlst), nlst, [keep_txt] * len(nlst)),
                                       n_cores)
        failed += [(nam, err) for nam, err in (results or []) if err is not None]
        logger.info(f'---> FSM2oshd simulations {pattern} finished.')

    if failed:
        warnings.warn(f'{len(failed)} FSM2oshd simulations failed:\n' +
                      '\n'.join(f'  {nam}: {err}' for nam, err in failed))

    if delete_nlst_files:
        logger.info('---> Removing FSM simulation config files')
        for file in nlst_all:
            if file not in [nam for nam, _ in failed]:
                os.remove(file)
    return [nam for nam, _ in failed]


def txt2ds(fname):
    '''
    Function to read a single FSM text file output as a xarray dataset
    Args:
        fname (str): filename

    Returns:
        xarray dataset of dimension (time), with the point index as coordinate point_ind
    '''
    df = read_pt_fsm2oshd(fname)
    point_ind = _point_index(fname)
    logger.info(f'---> Reading FSM data for point_ind = {point_ind}')
    ds = xr.Dataset({var: (['time'], df[var].values) for var in OUTPUT_ATTRS},
                    coords={
                        "point_ind": point_ind,
                        "time": df.index,
                        "reference_time": pd.Timestamp(df.index[0])
                    })
    return ds


def _rename_output(fname, fout='fsm_outputs', n_digits=3, remove_file=False, complevel=9):
    with xr.open_dataset(fname) as ds:
        ds = ds.load()
    point_ind = int(ds.point_ind.values)
    fname_out = f'{fout}_{str(point_ind).zfill(n_digits)}.nc'
    te.to_netcdf(ds, fname_out, complevel=complevel, n_bits=N_BITS)
    if remove_file and os.path.abspath(fname) != os.path.abspath(fname_out):
        os.remove(fname)
    logger.info(f'--> File {fname_out} saved')
    return fname_out


def output_to_ds(fname_pattern, complevel=9, remove_file=False, fout='fsm_outputs', n_digits=3, n_cores=1):
    '''
    Copy FSM netcdf outputs to files named {fout}_{point_ind}.nc

    Args:
        fname_pattern (str): pattern of the FSM netcdf outputs
        complevel (int): compression level
        remove_file (bool): remove the original files
        fout (str): prefix of the output files
        n_digits (int): number of digits of the point index in the file names
        n_cores (int): number of cores

    Returns:
        list: written files
    '''
    flist = sorted(glob.glob(fname_pattern))
    n = len(flist)
    out = tu.multicore_pooling(_rename_output, zip(flist, [fout] * n, [n_digits] * n, [remove_file] * n,
                                                   [complevel] * n), n_cores)
    logger.info('---> All fsm outputs stored as netcdf')
    return out


def to_netcdf(fname_fsm_sim, complevel=9):
    '''
    Function to convert a single FSM simulation output file (.txt) to a compressed netcdf file (.nc)

    Args:
        fname_fsm_sim(str): filename to convert from txt to nc
        complevel (int): Compression level. 1-9

    Returns:
        str: name of the netcdf file
    '''
    ver_dict = tu.get_versionning()

    ds = txt2ds(fname_fsm_sim)
    for var, attrs in OUTPUT_ATTRS.items():
        ds[var].attrs = attrs
    ds.attrs = {'title': 'FSM2oshd simulation outputs',
                'source': 'Data downscaled with TopoPyScale and simulated with FSM',
                'package_TopoPyScale_version': ver_dict.get('package_version'),
                'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                'url_FSM': 'https://github.com/ArcticSnow/FSM2oshd',
                'git_commit': ver_dict.get('git_commit'),
                'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    fout = f"{str(fname_fsm_sim)[:-4]}.nc"
    te.to_netcdf(ds, fout, complevel=complevel, n_bits=N_BITS)
    logger.info(f"File {fout} saved")
    return fout


def _combine_open_forest(fname_df_forest='fsm_sim/df_forest.pckl',
                         fname_forest='fsm_sim/fsm__outputs_forest_000.nc',
                         fname_open='fsm_sim/fsm__outputs_open_000.nc',
                         remove_forest_open_file=False,
                         fout='fsm_sim/fsm_outputs',
                         n_digits=3,
                         forest_threshold=0.1):
    '''
    Weighted average of the forest and open simulations of one point:
    proportion_with_forest * forest + (1 - proportion_with_forest) * open.

    Args:
        fname_df_forest (str): pickle of the forest table written by topo_export.to_fsm2oshd (one row per point,
            in the order of the points, i.e. of the index in the file names)
        fname_forest, fname_open (str): netcdf outputs of the forest and open simulations of the same point
        forest_threshold (float): forest proportions at or below this value are set to 0 (open only)

    Returns:
        str: written file
    '''
    df_forest = pd.read_pickle(fname_df_forest)
    with xr.open_dataset(fname_forest) as dsf, xr.open_dataset(fname_open) as dso:
        dsf, dso = dsf.load(), dso.load()
    point_ind = int(dsf.point_ind.values)
    if int(dso.point_ind.values) != point_ind:
        raise ValueError(f'{fname_forest} and {fname_open} are not outputs of the same point')

    coeff = float(df_forest.proportion_with_forest.iloc[point_ind])
    if coeff <= forest_threshold:
        coeff = 0.
    ds = dsf * coeff + dso * (1 - coeff)
    ds.attrs = dict(dsf.attrs, forest_proportion=coeff)

    fname_out = f'{fout}_{str(point_ind).zfill(n_digits)}.nc'
    te.to_netcdf(ds, fname_out, n_bits=N_BITS)
    if remove_forest_open_file:
        os.remove(fname_open)
        os.remove(fname_forest)
    logger.info(f'--> File {fname_out} saved')
    return fname_out


def aggregate_all_open_forest(fname_df_forest='fsm_sim/df_forest.pckl',
                              fname_forest_outputs='fsm_sim/fsm__outputs_forest_*.nc',
                              fname_open_outputs='fsm_sim/fsm__outputs_open_*.nc',
                              fout='fsm_sim/fsm_outputs',
                              n_cores=6,
                              n_digits=3,
                              remove_tmp_outputs=True,
                              forest_threshold=0.1):
    '''
    Combine the forest and open outputs of all points (paired by point index, not by file order).

    Returns:
        list: written files
    '''
    forest = {_point_index(f): f for f in glob.glob(fname_forest_outputs)}
    open_ = {_point_index(f): f for f in glob.glob(fname_open_outputs)}
    unpaired = sorted(set(forest) ^ set(open_))
    if unpaired:
        warnings.warn(f'Points without both forest and open outputs are skipped: {unpaired}')
    points = sorted(set(forest) & set(open_))
    n = len(points)
    out = tu.multicore_pooling(_combine_open_forest,
                               zip([fname_df_forest] * n,
                                   [forest[i] for i in points],
                                   [open_[i] for i in points],
                                   [remove_tmp_outputs] * n,
                                   [fout] * n,
                                   [n_digits] * n,
                                   [forest_threshold] * n),
                               n_cores)
    logger.info('---> All forest and open outputs combined')
    return out


def read_pt_fsm2oshd(fname):
    df = pd.read_csv(fname, sep=r'\s+', header=None, names=FSM_OUTPUT_COLUMNS)
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('time')
    return df


def read_pt_met_fsm2oshd(fname):
    df = pd.read_csv(fname, sep=r'\s+', header=None, names=FSM_MET_COLUMNS)
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('time')
    return df
