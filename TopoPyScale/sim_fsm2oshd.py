'''
Methods to perform FSM2oshd simulations
S. Filhol, September 2023

TODO:
- [ ] Update run fsm function:
    - open and forest namelist
    - run open/forest simulation
    - combine results based on weights of open vs forest
    - If forest cover < 10% or threshold run only open
    - recover information of forest cover fraction from namelist file. Python namelist parser package?


'''

from pathlib import Path
import glob, os


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