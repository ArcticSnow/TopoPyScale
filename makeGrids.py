from TopoPyScale import topo_sim as sim
import glob

a = glob.glob(wd + "/fsm_sims/sim_FSM_pt_*.txt")

# finishedsims = [os.path.split(i)[0] for i in a]
# myfiles = [glob.glob(i+"/out*") for i in finishedsims]
# flat_list = [item for sublist in myfiles for item in sublist]
# file_list = sorted(flat_list)

# Natural sort to get correct order ensemb * samples
import re

def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

file_list = natural_sort(a)



data = []
for file_path in file_list:
     #column 7 (index 6) is snow water equiv
     #data.append(np.genfromtxt(file_path, usecols=6)[int(startIndex):int(endIndex)])
     data.append(np.genfromtxt(file_path, usecols=6))

myarray = np.asarray(data)  # samples x days
df = pd.DataFrame(myarray.transpose())


startTime = "2000-01-01"
endTime = "2022-12-31"

mytime = np.arange(np.datetime64(startTime), np.datetime64(endTime)+  np.timedelta64(1, "D"),np.timedelta64(1, "D"))


import importlib
importlib.reload(sim)
grid_stack, lats, lons = sim.topo_map_forcing( myarray, 1, "float32")
sim.write_ncdf(wd, grid_stack, "tas", "mm", "snow_water_equiv", dt, lats, lons, "float32")