"""
Toposcale class definition

S. Filhol, September 2021

project/
    config.ini
    -> input/
        -> dem/
        -> climate/
    -> output/

"""
import glob
import os
import pdb
#import configparser
import sys
import shutil
from munch import DefaultMunch
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from TopoPyScale import fetch_era5 as fe
from TopoPyScale import topo_param as tp
from TopoPyScale import topo_sub as ts
from TopoPyScale import fetch_dem as fd
from TopoPyScale import solar_geom as sg
from TopoPyScale import topo_scale as ta
from TopoPyScale import topo_export as te
from TopoPyScale import topo_plot as tpl
from TopoPyScale import topo_obs as tpo






class Topoclass(object):
    """
    A python class to bring the typical use-case of toposcale in a user friendly object
    """

    def __init__(self, config_file):
        
        try:
            with open(config_file, 'r') as f:
                self.config = DefaultMunch.fromYAML(f)
        except IOError:
                print('ERROR: config file does not exist. Check path.')

        if self.config.outputs.file.clean_outputs:
            # remove outputs directory because if results already exist this causes concat of netcdf files
            try:
                shutil.rmtree(self.config.project.directory + '/outputs/')
                print('---> Output directory cleaned')
            except:
                os.makedirs('/'.join((self.config.project.directory, 'outputs/')))

        # remove output fsm directory
        if self.config.outputs.file.clean_FSM:
            try:
                shutil.rmtree(self.config.project.directory + '/fsm_sims/')
                print('---> FSM directory cleaned')
            except:
                print("---> no FSM directory to clean")

            # remove output fsm directory
            try:
                shutil.rmtree(self.config.project.directory + '/ensemble/')
                print("---> Ensemble directory cleaned")
            except:
                print("---> no ensemble directory to clean")
        # check if tree directory exists. If not create it
        if not os.path.exists('/'.join((self.config.project.directory, 'inputs/'))):
            os.makedirs('/'.join((self.config.project.directory, 'inputs/')))
        if not os.path.exists('/'.join((self.config.project.directory, 'outputs/'))):
            os.makedirs('/'.join((self.config.project.directory, 'outputs/')))
        
        self.config.climate.path = self.config.project.directory + 'inputs/climate/'
        if not os.path.exists('/'.join((self.config.project.directory, 'inputs/climate/'))):
            os.makedirs('/'.join((self.config.project.directory, 'inputs/climate')))
        if not os.path.exists('/'.join((self.config.project.directory, 'inputs/climate/tmp/'))):
            os.makedirs('/'.join((self.config.project.directory, 'inputs/climate/tmp')))
        if not os.path.exists('/'.join((self.config.project.directory, 'outputs/tmp/'))):
            os.makedirs('/'.join((self.config.project.directory, 'outputs/tmp')))
        if not os.path.exists('/'.join((self.config.project.directory, 'outputs/downscaled'))):
            os.makedirs('/'.join((self.config.project.directory, 'outputs/downscaled')))

        self.config.dem.path = self.config.project.directory + '/inputs/dem/'
        if not os.path.exists('/'.join((self.config.project.directory, 'inputs/dem/'))):
            os.makedirs('/'.join((self.config.project.directory, 'inputs/dem')))

        self.config.dem.filepath = self.config.dem.path + self.config.dem.file
        if not os.path.isfile(self.config.dem.filepath):
            
            if self.config.project.extent is not None:
                self.config.project.extent = dict(zip(['latN', 'latS', 'lonW', 'lonE'], self.config.project.extent))
            else:
                print('ERROR: no extent provided. Must follow this format: [latN, latS, lonW, lonE]')
                sys.exit()

            fd.fetch_dem(self.config.dem.path, self.config.project.extent, self.config.dem.epsg, self.config.dem.file)            
        else:
            print('\n---> DEM file found')


        if self.config.project.extent is not None:
            extent_NSWE = self.config.project.extent
            self.config.project.extent = dict(zip(['latN', 'latS', 'lonW', 'lonE'], extent_NSWE))
        else:
            # little routine extracting lat/lon extent from DEM
            self.config.project.extent = tp.get_extent_latlon(self.config.dem.filepath, self.config.dem.epsg)

        print(self.config.project.extent)
        print('Project lat/lon extent:\n')
        print('\t------------------------------')
        print('\t|        North:{}          |\n\t|West:{}          East:{}|\n\t|        South:{}          |'.format(np.round(self.config.project.extent.get('latN'),1),
                                                        np.round(self.config.project.extent.get('lonW'),1),
                                                          np.round(self.config.project.extent.get('lonE'),1),
                                                          np.round(self.config.project.extent.get('latS'),1)))
        print('\t------------------------------')

        self.toposub = self.Toposub()
        self.plot = Plotting()
        self.ds_solar = None
        self.da_horizon = None

        self.toposub.dem_path = self.config.dem.filepath
        self.toposub.project_directory = self.config.project.directory

        if self.config.project.climate.lower() == 'era5':
            self.get_era5()

        if self.config.project.split.IO:
            self.time_splitter = self.TimeSplitter(self.config.project.start,
                                                   self.config.project.end,
                                                   self.config.project.split.time,
                                                   self.config.outputs.file.ds_solar,
                                                   self.config.outputs.file.downscaled_pt)


    def load_project(self):
        '''
        Function to load pre-existing TopoPyScale project saved in files
        '''
        if os.path.isfile(self.config.project.directory + 'outputs/' + self.config.outputs.file.df_centroids):
            self.toposub.df_centroids = pd.read_pickle(self.config.project.directory + 'outputs/' + self.config.outputs.file.df_centroids)
            print(f'---> Centroids file {self.config.outputs.file.df_centroids} exists and loaded')
        else:
            print(f'-> WARNING: Centroid file {self.config.outputs.file.df_centroids} not found')

        if os.path.isfile(self.config.project.directory + 'outputs/' + self.config.outputs.file.ds_param):
            self.toposub.ds_param = xr.open_dataset(self.config.project.directory + 'outputs/' + self.config.outputs.file.ds_param)
            print(f'---> DEM parameter file {self.config.outputs.file.ds_param} exists and loaded')
        else:
            print(f'-> WARNING: DEM parameter file {self.config.outputs.file.ds_param} not found')

        if os.path.isfile(self.config.project.directory + 'outputs/' + self.config.outputs.file.ds_solar):
            self.ds_solar = xr.open_dataset(self.config.project.directory + 'outputs/' + self.config.outputs.file.ds_solar)
            print(f'---> Solar file {self.config.outputs.file.ds_solar} exists and loaded')
        else:
            print(f'-> WARNING: Solar file {self.config.outputs.file.ds_solar} not found')

        if os.path.isfile(self.config.project.directory + 'outputs/' + self.config.outputs.file.da_horizon):
            self.da_horizon = xr.open_dataarray(self.config.project.directory+ 'outputs/' + self.config.outputs.file.da_horizon)
            print(f'---> Horizon file {self.config.outputs.file.da_horizon} exists and loaded')
        else:
            print(f'-> WARNING: Horizon file {self.config.outputs.file.da_horizon} not found')

        flist = glob.glob(f'{self.config.project.directory}outputs/downscaled/{self.config.outputs.file.downscaled_pt}')
        if len(flist) > 0 :
            print('---> Loading downscaled points \n ...')
            self.downscaled_pts = xr.open_mfdataset(f'{self.config.project.directory}outputs/downscaled/{self.config.outputs.file.downscaled_pt}',
                                                          concat_dim='point_id',
                                                          combine='nested',
                                                          parallel=True)
            print(f'---> Downscaled point files {self.config.outputs.file.ds_param} exists and loaded')
        else:
            print(f'-> WARNING: Downscale point files {self.config.outputs.file.downscaled_pt} not found')

    class Toposub:
        """
        Sub-Class to initialize variables to store TopoSub variables
        """
        def __init__(self):
            self.dem_path = None
            self.project_directory = None
            self.ds_param = None
            self.df_centroids = None
            self.kmeans_obj = None
            self.scaler = None

        def plot_clusters_map(self, var='cluster_labels', cmap=plt.cm.hsv, figsize=(14, 10)):
            ts.plot_center_clusters(self.dem_path, self.ds_param, self.df_centroids, var=var, cmap=cmap, figsize=figsize)

        def write_landform(self):
            ts.write_landform(self.dem_path, self.ds_param, self.project_directory)

    def compute_dem_param(self):
        if os.path.isfile(self.config.project.directory + 'outputs/'+ self.config.outputs.file.ds_param):
            self.toposub.ds_param = xr.open_dataset(self.config.project.directory + 'outputs/'+ self.config.outputs.file.ds_param)
            print(f'---> DEM parameter file {self.config.outputs.file.ds_param} exists and loaded')
        else:
            self.toposub.ds_param = tp.compute_dem_param(self.config.dem.filepath,
                                                         fname=self.config.outputs.file.ds_param,
                                                         project_directory=self.config.project.directory)

    def extract_pts_param(self, method='nearest', **kwargs):
        """
        Function to use a list point as input rather than cluster centroids from DEM segmentation (topo_sub.py/self.clustering_dem()).

        Args:
            df (dataFrame):
            method (str): method of sampling
            **kwargs: pd.read_csv() parameters
        Returns:
        """
        self.toposub.df_centroids = pd.read_csv(self.config.project.directory+ 'inputs/dem/' + self.config.sampling.points.csv_file, **kwargs)
        self.toposub.df_centroids['point_id'] = self.toposub.df_centroids.index.astype(int)
        self.toposub.df_centroids = tp.extract_pts_param(self.toposub.df_centroids, self.toposub.ds_param, method=method)

    def extract_topo_cluster_param(self):
        """
        Function to segment a DEM in clusters and retain only the centroids of each cluster.

        :return:

        TODO:
        - try to migrate most code of this functino to topo_sub.py as a function itself segment_topo()
        """
        df_param = ts.ds_to_indexed_dataframe(self.toposub.ds_param)
        df_scaled, self.toposub.scaler = ts.scale_df(df_param)
        if self.config.sampling.toposub.clustering_method.lower() == 'kmean':
            self.toposub.df_centroids, self.toposub.kmeans_obj, df_param['cluster_labels'] = ts.kmeans_clustering(
                df_scaled,
                self.config.sampling.toposub.n_clusters,
                seed=self.config.sampling.toposub.random_seed)
        elif self.config.sampling.toposub.clustering_method.lower() == 'minibatchkmean':
            self.toposub.df_centroids, self.toposub.kmeans_obj, df_param['cluster_labels'] = ts.minibatch_kmeans_clustering(
                df_scaled,
                self.config.sampling.toposub.n_clusters,
                self.config.project.CPU_cores,
                seed=self.config.sampling.toposub.random_seed)
        else:
            print('ERROR: {} clustering method not available'.format(self.config.sampling.toposub.clustering_method))
        self.toposub.df_centroids = ts.inverse_scale_df(self.toposub.df_centroids, self.toposub.scaler)
        self.toposub.df_centroids['point_id'] = self.toposub.df_centroids.index.astype(int)
        self.toposub.ds_param['cluster_labels'] = (["y", "x"], np.reshape(df_param.cluster_labels.values, self.toposub.ds_param.slope.shape))

        # update file
        fname = self.config.project.directory + 'outputs/'+ self.config.outputs.file.ds_param
        te.to_netcdf(self.toposub.ds_param, fname=fname)

        # update plotting class variable
        self.plot.ds_param = self.toposub.ds_param

    def extract_topo_param(self):
        """
        Function to select which 
        """
        if  os.path.isfile(self.config.project.directory + 'outputs/'+ self.config.outputs.file.df_centroids):
            self.toposub.df_centroids = pd.read_pickle(self.config.project.directory + 'outputs/'+ self.config.outputs.file.df_centroids)
            print(f'---> Centroids file {self.config.outputs.file.df_centroids} exists and loaded')
        else:
            if self.config.sampling.method in ['points', 'point'] :
                self.extract_pts_param()
            elif self.config.sampling.method == 'toposub':
                self.extract_topo_cluster_param()
            elif self.config.sampling.method == 'both':

                # implement the case one wann run both toposub and a list of points
                print('ERROR: method not yet implemented')
                
            else:
                print('ERROR: Extraction method not available')

            # Store dataframe to pickle
            self.toposub.df_centroids.to_pickle(self.config.project.directory + 'outputs/'+ self.config.outputs.file.df_centroids)
            print(f'---> Centroids file {self.config.outputs.file.df_centroids} saved')

    class TimeSplitter():
        def __init__(self, start, end, period, ds_solar_fname, downscale_fname):
            self.start = start
            self.end = end
            self.period = period

            s = pd.date_range(self.start, self.end, freq=f'{self.period}YS')
            self.start_list = list(s.astype(str))
            self.end_list = list((s-pd.Timedelta('1D')).astype(str))[1:]
            self.end_list.append(end.strftime('%Y-%m-%d'))

            self.ds_solar_flist = self._flist_generator(ds_solar_fname)
            self.downscaled_flist = self._flist_generator(downscale_fname)

        def _flist_generator(self, fname):
            # function to creat list of filename
            fcom = fname.split('.')
            flist = []
            for i, start in enumerate(self.start_list):
                flist.append(f'{fcom[0]}_{start}_{self.end_list[i]}.{fcom[1]}')
            return flist


    def compute_solar_geometry(self):

        if self.config.project.split.IO:
            for i, start in enumerate(self.time_splitter.start_list):
                end = self.time_splitter.end_list[i]
                fname = self.config.project.directory + 'outputs/' + self.time_splitter.ds_solar_flist[i]

                if os.path.isfile(fname):
                    self.ds_solar = xr.open_dataset(fname, chunks='auto', engine='h5netcdf')
                    print(f'---> Solar file {self.time_splitter.ds_solar_flist[i]} exists and loaded')
                else:
                    self.ds_solar = sg.get_solar_geom(self.toposub.df_centroids,
                                                      start,
                                                      end,
                                                      self.config.climate[self.config.project.climate].timestep,
                                                      str(self.config.dem.epsg),
                                                      self.config.project.CPU_cores,
                                                      self.time_splitter.ds_solar_flist[i],
                                                      self.config.project.directory)

        else:
            fname = self.config.project.directory + 'outputs/'+ self.config.outputs.file.ds_solar
            if os.path.isfile(fname):
                self.ds_solar = xr.open_dataset(fname, chunks='auto', engine='h5netcdf')
                print(f'---> Solar file {self.config.outputs.file.ds_solar} exists and loaded')
            else:
                self.ds_solar = sg.get_solar_geom(self.toposub.df_centroids,
                                                  self.config.project.start,
                                                  self.config.project.end,
                                                  self.config.climate[self.config.project.climate].timestep,
                                                  str(self.config.dem.epsg),
                                                  self.config.project.CPU_cores,
                                                  self.config.outputs.file.ds_solar,
                                                  self.config.project.directory)

    def compute_horizon(self):
        """
        Function to compute horizon angle and sample values for list of points
        """
        fname = self.config.project.directory + 'outputs/'+ self.config.outputs.file.da_horizon
        if os.path.isfile(fname):
            self.da_horizon = xr.open_dataarray(fname, chunks='auto', engine='h5netcdf')
            print(f'---> Horizon file {self.config.outputs.file.da_horizon} exists and loaded')
        else:
            self.da_horizon = tp.compute_horizon(self.config.dem.filepath,
                                                 self.config.dem.horizon_increments,
                                                 self.config.project.CPU_cores,
                                                 self.config.outputs.file.da_horizon,
                                                 self.config.project.directory)
        tgt_x = tp.xr.DataArray(self.toposub.df_centroids.x.values, dims="points")
        tgt_y = tp.xr.DataArray(self.toposub.df_centroids.y.values, dims="points")
        for az in self.da_horizon.azimuth.values:
            self.toposub.df_centroids['hori_azi_'+str(az)] = self.da_horizon.sel(x=tgt_x,
                                                                            y=tgt_y,
                                                                            azimuth=az,
                                                                            method='nearest').values.flatten()
        # update df_centroid file with horizons angle
        self.toposub.df_centroids.to_pickle(self.config.project.directory + 'outputs/'+ self.config.outputs.file.df_centroids)
        print(f'---> Centroids file {self.config.outputs.file.df_centroids} updated with horizons')

    def downscale_climate(self):
        if self.config.project.split.IO:
            for i, start in enumerate(self.time_splitter.start_list):
                print()
                end = self.time_splitter.end_list[i]
                fname = self.time_splitter.downscaled_flist[i]

                self.ds_solar = None
                self.ds_solar = xr.open_dataset(self.config.project.directory + 'outputs/' + self.time_splitter.ds_solar_flist[i])

                ta.downscale_climate(self.config.project.directory,
                                     self.toposub.df_centroids,
                                     self.da_horizon,
                                     self.ds_solar,
                                     self.config.dem.epsg,
                                     start,
                                     end,
                                     self.config.toposcale.interpolation_method,
                                     self.config.toposcale.LW_terrain_contribution,
                                     self.config.climate[self.config.project.climate].timestep,
                                     fname)

            # Concatenate time-splitted outputs along time-dimension
            n_digits = len(str(self.toposub.df_centroids.index.max()))
            ds_list = []
            out_path_list = []
            tmp_path_list = []
            for pt_id in self.toposub.df_centroids.point_id.values:
                print(f'Concatenating point {pt_id}')
                num = str(pt_id).zfill(n_digits)
                f_pattern = f'{self.config.outputs.file.downscaled_pt.split("*")[0]}{num}*'
                tmp_path_list.append(f_pattern)
                #pdb.set_trace()
                ds = xr.open_mfdataset(f'{self.config.project.directory}outputs/downscaled/{f_pattern}')
                ds_list.append(ds)
                file = f'{self.config.project.directory}outputs/downscaled/{self.config.outputs.file.downscaled_pt.split("*")[0]}{num}.nc'
                out_path_list.append(file)
                ds = None
            xr.save_mfdataset(ds_list, out_path_list, engine='h5netcdf')

            # Delete time slice files.
            for fpat in self.time_splitter.downscaled_flist:
                flist = glob.glob(f'{self.config.project.directory}outputs/downscaled/{fpat}')
                for file in flist:
                    os.remove(file)

        else:
            ta.downscale_climate(self.config.project.directory,
                                 self.toposub.df_centroids,
                                 self.da_horizon,
                                 self.ds_solar,
                                 self.config.dem.epsg,
                                 self.config.project.start,
                                 self.config.project.end,
                                 self.config.toposcale.interpolation_method,
                                 self.config.toposcale.LW_terrain_contribution,
                                 self.config.climate[self.config.project.climate].timestep,
                                 self.config.outputs.file.downscaled_pt)

        self.downscaled_pts = ta.read_downscaled(self.config.project.directory + 'outputs/downscaled/' + self.config.outputs.file.downscaled_pt)
        # update plotting class variables
        self.plot.ds_down = self.downscaled_pts

    def get_era5(self):
        """
        Funtion to call fetching of ERA5 data
        TODO:
        - merge monthly data into one file (cdo?)- this creates massive slow down!
        """
        lonW = self.config.project.extent.get('lonW') - 0.4
        lonE = self.config.project.extent.get('lonE') + 0.4
        latN = self.config.project.extent.get('latN') + 0.4
        latS = self.config.project.extent.get('latS') - 0.4

        # retreive ERA5 surface data
        fe.retrieve_era5(
            self.config.climate[self.config.project.climate].product,
            self.config.project.start,
            self.config.project.end,
            self.config.climate.path,
            latN, latS, lonE, lonW,
            self.config.climate[self.config.project.climate].timestep,
            self.config.climate[self.config.project.climate].download_threads,
            surf_plev='surf'
            )
        # retrieve era5 plevels
        fe.retrieve_era5(
            self.config.climate[self.config.project.climate].product,
            self.config.project.start,
            self.config.project.end,
            self.config.climate.path,
            latN, latS, lonE, lonW, 
            self.config.climate[self.config.project.climate].timestep,
            self.config.climate[self.config.project.climate].download_threads,
            surf_plev='plev',
            plevels=self.config.climate[self.config.project.climate].plevels,
            )

    def get_WMO_observations(self):
        """
        Function to download and parse in-situ data from WMO database
        """
        lonW = self.config.project.extent.get('lonW') - 0.4
        lonE = self.config.project.extent.get('lonE') + 0.4
        latN = self.config.project.extent.get('latN') + 0.4
        latS = self.config.project.extent.get('latS') - 0.4

        df = pd.DataFrame()
        df['dates'] = pd.date_range(self.config.project.start,
                                    self.config.project.end, freq='M')
        df['month'] = df.dates.dt.month
        df['year'] = df.dates.dt.year

        bbox = [latS, lonW, latN, lonE]
        tpo.fetch_WMO_insitu_observations(list(df.year.unique()),
                                          list(df.month.unique()))
        tpo.parse_WMO_insitu_observations()

    def to_cryogrid(self, fname_format='Cryogrid_pt_*.nc', precip_partition='continuous'):
        """
        wrapper function to export toposcale output to cryosgrid format from TopoClass
        
        Args:
            fname_format (str): filename format. point_id is inserted where * is

        """
        path = self.config.project.directory + 'outputs/'
        if 'cluster:labels' in self.toposub.ds_param.keys():
            label_map = True
            da_label = self.toposub.ds_param.cluster_labels
        else:
            label_map = False
            da_label = None
        te.to_cryogrid(self.downscaled_pts,
                       self.toposub.df_centroids,
                       fname_format=fname_format,
                       path=path,
                       label_map=label_map,
                       snow_partition_method=precip_partition,
                       da_label=da_label,
                       climate_dataset_name=self.config.project.climate,
                       project_author=self.config.project.authors)
        
    def to_fsm(self, fname_format='FSM_pt_*.txt'):
        """
        function to export toposcale output to FSM format
        """
        te.to_fsm(self.downscaled_pts, f'{self.config.project.directory}outputs/' + fname_format)

    def to_crocus(self, fname_format='CROCUS_pt_*.nc', scale_precip=1):
        """
        function to export toposcale output to crocus format .nc. This functions saves one file per point_id
        
        Args:
            fout_format (str): filename format. point_id is inserted where * is
            scale_precip(float): scaling factor to apply on precipitation. Default is 1
        """
        te.to_crocus(self.downscaled_pts,
                     self.toposub.df_centroids,
                     fname_format=f'{self.config.project.directory}outputs/' + fname_format,
                     scale_precip=scale_precip,
                     climate_dataset_name=self.config.project.climate,
                     project_author=self.config.project.authors)
    
    def to_snowmodel(self, fname_format='Snowmodel_stn_*.csv'):
        """
        function to export toposcale output to snowmodel format .ascii, for single station standard

            fout_format: str, filename format. point_id is inserted where * is
        """
        te.to_micromet_single_station(self.downscaled_pts,
                                      self.toposub.df_centroids,
                                      fname_format=f'{self.config.project.directory}outputs/' + fname_format,
                                      na_values=-9999,
                                      headers=False)
    
    def to_netcdf(self, file_out='output.nc', variables=None):
        """
        function to export toposcale output to one single generic netcdf format, compressed

        Args:
            file_out (str): name of export file
            variables (list str): list of variable to export. Default exports all variables
        """

        te.to_netcdf(self.downscaled_pts[variables], f'{self.config.project.directory}outputs/' + file_out, variables)
        print('---> File {} saved'.format(file_out))

    def to_snowpack(self, fname_format='smet_pt_*.smet'):
        """
        function to export toposcale output to FSM format
        """
        te.to_snowpack(self.downscaled_pts, f'{self.config.project.directory}outputs/' + fname_format)

    def to_geotop(self, fname_format='meteo_*.txt'):
        """
        function to export toposcale output to FSM format
        """
        te.to_geotop(self.downscaled_pts, f'{self.config.project.directory}outputs/' + fname_format)

    def to_musa(self, 
                fname_met='musa_met.nc', 
                fname_labels='musa_labels.nc'):
        """
        function to export TopoPyScale output in a format compatible with MuSa
        MuSa: https://github.com/ealonsogzl/MuSA

        Args:
            fname: filename of the netcdf 
        """

        te.to_musa(ds=self.downscaled_pts,
            df_pts=self.toposub.df_centroids,
            da_label=self.toposub.ds_param.cluster_labels,
            fname_met=fname_met,
            fname_labels=fname_labels,
            path=self.config.project.directory + 'outputs/',
            climate_dataset_name=self.config.project.climate,
            project_authors=self.config.project.authors
            )
        print('---> File ready for MuSa {} saved'.format(fname_met))

class Plotting:
    '''
    Sub-Class with plotting functions for topoclass object
    '''
    def __int__(self):
        self.ds_param = None
        self.ds_down = None

    def map_variable(self,
                     time_step=1,
                     var='t',
                     cmap=plt.cm.RdBu_r,
                     hillshade=True,
                     **kwargs):

        # add logic to check ds_down and ds_param exist.
        tpl.map_unclustered(self.ds_down,
                     self.ds_param,
                     time_step=time_step,
                     var=var,
                     cmap=cmap,
                     hillshade=hillshade,
                     **kwargs)
    def map_terrain(self, var='elevation', hillshade=True, **kwargs):
        tpl.map_terrain(self.ds_param,
                        var=var,
                        hillshade=hillshade,
                        **kwargs
                        )

    def map_center_clusters(self,
                            background_var='elevation',
                            cmap=plt.cm.viridis,
                            hillshade=True,
                            **kwargs):
        print('to be implemented')

    def cluster_map(self):
        print('To be implemented')

    def timeseries(self):
        print('To be implemented')

    def solar_geom(self):
        print('To be implemented')

    def horizon(self):
        print('To be implemented')

