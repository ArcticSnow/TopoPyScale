'''
Methods to perform FSM2oshd simulations
S. Filhol, September 2023

TODO:

- open and forest namelist
- run open/forest simulation
- combine results based on weights of open vs forest
- recover information of forest cover fraction from namelist file. Python namelist parser package?


'''

from pathlib import Path
import glob, os


def _run_fsm2oshd(fsm_exec, nlstfile):
    '''
    function to execute FSM
    Args:
        fsm_exec (str): path to FSM executable
        nlstfile (str): path to FSM simulation config file

    Returns:
        NULL (FSM simulation file written to disk)
    '''
    os.system(fsm_exec + ' ' + nlstfile)
    print('Simulation done: ' + nlstfile)

def fsm2oshd_sim_parallel(fsm_forest_nam='fsm_sim/fsm__forest*.nam',
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

def combine_outputs(df_centroids, fname='fsm_sim/fsm__ou*.txt'):
    
    return