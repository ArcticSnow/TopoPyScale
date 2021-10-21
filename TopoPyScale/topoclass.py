'''
Toposcale class definition
S. Filhol, September 2021

project/
    config.ini
    -> input/
        -> dem/
        -> met_forcing/
    -> output/

'''
import os
#import configparser
import sys

from configobj import ConfigObj
import matplotlib.pyplot as plt
from TopoPyScale import fetch_era5 as fe
from TopoPyScale import topo_param as tp
from TopoPyScale import topo_sub as ts
from TopoPyScale import fetch_dem as fd
from TopoPyScale import solar_geom as sg


class Topoclass(object):
    
    def __init__(self, config_file):
        
        self.config = self.Config(config_file)
        self.toposub = self.Toposub()

        self.solar_ds = None
        self.horizon_da = None

        if self.config.forcing_dataset.lower() == 'era5':
            self.get_era5()

        if not os.path.isfile(self.config.dem_path):
            fd.fetch_dem(self.config.project_dir, self.config.extent, self.config.dem_file)
        else:
            print('\n---> DEM file found')
            self.toposub.dem_path = self.config.dem_path

    class Toposub:
        '''
        Class to initialize variables to store TopoSub variables
        '''
        def __init__(self):
            self.dem_path = None
            self.df_param = None
            self.df_centroids = None
            self.kmeans_obj = None
            self.scaler = None

        def plot_clusters_map(self, var='cluster_labels', cmap=plt.cm.hsv):
            ts.plot_center_clusters(self.dem_path, self.df_param, self.df_centroids, var=var, cmap=cmap)

    def compute_dem_param(self):
        self.toposub.df_param = tp.compute_dem_param(self.config.dem_path)

    def clustering_dem(self):
        '''
        Function to compute DEM parameters, and cluster DEM in n_clusters
        :return:
        '''
        df_scaled, self.toposub.scaler = ts.scale_df(self.toposub.df_param)
        if self.config.clustering_method.lower() == 'kmean':
            self.toposub.df_centroids, self.toposub.kmeans_obj, self.toposub.df_param['cluster_labels'] = ts.kmeans_clustering(df_scaled, self.config.n_clusters)
        elif self.config.clustering_method.lower() == 'minibatchkmean':
            self.toposub.df_centroids, self.toposub.kmeans_obj, self.toposub.df_param['cluster_labels'] = ts.minibatch_kmeans_clustering(df_scaled, self.config.n_clusters, self.config.n_cores)
        else:
            print('ERROR: {} clustering method not available'.format(self.config.clustering_method))
        self.toposub.df_centroids = ts.inverse_scale_df(self.toposub.df_centroids, self.toposub.scaler)

    def compute_solar_geometry(self):
        self.solar_ds = sg.get_solar_geom(self.toposub.df_centroids, self.config.start_date, self.config.end_date, self.config.time_step, self.config.dem_epsg)

    def compute_horizon(self):
        self.horizon_da = tp.compute_horizon(self.config.dem_path, self.config.horizon_az_inc)

    class Config:
        '''
        Class to contain all config.ini parameters
        '''
        def __init__(self, config_file):
            self.file_config = config_file 
            
            # parse configuration file into config class
            self._parse_config_file()
            
            # check if tree directory exists. If not create it
            if not os.path.exists('/'.join((self.project_dir, 'inputs/'))):
                os.makedirs('/'.join((self.project_dir, 'inputs/')))
            if not os.path.exists('/'.join((self.project_dir, 'inputs/forcings/'))):
                os.makedirs('/'.join((self.project_dir, 'inputs/forcings')))
            if not os.path.exists('/'.join((self.project_dir, 'inputs/dem/'))):
                os.makedirs('/'.join((self.project_dir, 'inputs/dem/')))
            if not os.path.exists('/'.join((self.project_dir, 'outputs/'))):
                os.makedirs('/'.join((self.project_dir, 'outputs/')))
                
        def _parse_config_file(self):
            '''
            Function to parse config file .ini into a python class
            '''
            try:
                conf = ConfigObj(self.file_config)
            except IOError:
                print('ERROR: config file does not exist. Check path.')

            self.project_dir = conf['main']['project_dir']
            self.project_description = conf['main']['project_description']
            self.project_name = conf['main']['project_name']
            self.project_author = conf['main']['project_authors']
            
            self.start_date = conf['main']['start_date']
            self.end_date = conf['main']['end_date']
            self.extent = {'latN': conf['main'].as_float('latN'),
                           'latS': conf['main'].as_float('latS'),
                           'lonW': conf['main'].as_float('lonW'),
                           'lonE': conf['main'].as_float('lonE')}
            
            self.forcing_dataset = conf['forcing'].get('dataset')
            if self.forcing_dataset.lower() == 'era5':
                self.forcing_era5_product = conf['forcing']['era5_product']
                self.forcing_n_threads = conf['forcing'].as_int('n_threads_download')
            self.n_cores = conf['forcing'].as_int('n_cores')
                
            self.time_step = conf['forcing']['time_step']
            self.plevels = conf['forcing']['plevels']
            
            self.dem_file = conf['forcing'].get('dem_file')
            self.dem_epsg = conf['forcing'].get('dem_EPSG')
            self.dem_path = self.project_dir + 'inputs/dem/' + self.dem_file
            self.horizon_az_inc = conf['forcing'].as_int('horizon_az_inc')

            self.n_clusters = conf['toposcale'].as_int('n_clusters')
            self.clustering_method = conf['toposcale']['clustering_method']
            self.interp_method = conf['toposcale']['interpolation_method']
            
    def get_era5(self):
        '''
        Funtion to call fetching of ERA5 data
        TODO:
        - merge monthly data into one file (cdo?)
        '''
        lonW = self.config.extent.get('lonW') - 0.25
        lonE = self.config.extent.get('lonE') + 0.25
        latN = self.config.extent.get('latN') + 0.25
        latS = self.config.extent.get('latS') - 0.25

        # retreive ERA5 surface data
        fe.retrieve_era5(
            self.config.forcing_era5_product,
            self.config.start_date,
            self.config.end_date,
            self.config.project_dir + 'inputs/forcings/',
            latN, latS, lonE, lonW,
            self.config.time_step,
            self.config.forcing_n_threads,
            surf_plev='surf'
            )
        # retrieve era5 plevels
        fe.retrieve_era5(
            self.config.forcing_era5_product,
            self.config.start_date,
            self.config.end_date,
            self.config.project_dir + 'inputs/forcings/',
            latN, latS, lonE, lonW, 
            self.config.time_step,
            self.config.forcing_n_threads,
            surf_plev='plev',
            plevels=self.config.plevels,
            )
            

    
    def to_cryogrid(self):
        '''
        function to export toposcale output to cryosgrid format .mat
        '''
        
    def to_fsm(self):
        '''
        function to export toposcale output to FSM format
        '''
        
    def to_crocus(self):
        '''
        function to export toposcale output to crocus format .nc
        '''

    
    def to_snowmodel(self):
        '''
        function to export toposcale output to snowmodel format .ascii, for single station standard
        '''
    
    def to_netcdf(self):
        '''
        function to export toposcale output to generic netcdf format
        '''