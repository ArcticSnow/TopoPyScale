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
import xarray as xr
import glob, os
import configparser
import rasterio
import fetch_era5 as fe
import pandas as pd


class toposcale(object):
    
    def __init__(self, config_file):
        
        self.config = self.config(config_file)
        
        if self.config.download_dataset:
            if self.config.dataset.lower() == 'era5':
                self.get_era5()
        
        if self.config.dem_file is None:
            #write code to fetch dem from SRTM, Arctic DEM or other public repository
            if (self.config.dem_dataset.lower() == 'srtm') and self.config.dem_download:
                
                import elevation
                # use STRM DEM for extent of interest
                print('ERROR: Fetching SRTM DEM not yet available')
                self.config.dem_file = 'dem_30m_SRTM.tif'
                elevation.clip(bounds=(self.config.lonW, 
                                       self.config.latS, 
                                       self.config.lonE, 
                                       self.config.latN),
                               output= self.config.project_dir + '/input/dem/' + self.config.dem_file )
                self.dem = rasterio.open(self.config.dem_file)
                
            elif (self.config.dem_dataset.lower() == 'arcticdem') and self.config.dem_download:
                # write code to pull ArcticDEM for extent of interest
                print('ERROR: Fetching ArcticDEM not yet available.\n\nThere are libraries to download DEM independently of TopoPyScale: \n-https://github.com/samapriya/ArcticDEM-Batch-Pipeline')
                self.dem = None
                
        else:
            self.dem = rasterio.open(self.config.dem_file)

    def compute_DEM_parameters(self):
        # add here code to compute slope, aspect, horizon angle, solar zenith and azimuth
        # store all in a pandas dataframe. X,Y,Z,slope,aspect. What about for ha,sz,sa?
        from topocalc import gradient
        from topocalc import viewf
        
        self.dem_param = pd.DataFrame()
        
        dx, dy = self.dem.read_transform()[1], -self.dem.read_transform()[-1]
        dem_arr = self.dem.read()
        if dem_arr.shape.__len__() == 3:
            dem_arr = dem_arr[0,:,:]
        slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
        self.dem_param['elev'] = dem_arr.flatten()
        self.dem_param['slope'] = slope.flatten()
        self.dem_param['aspect'] = aspect.flatten()
        slope, aspect = None, None
        
        svf = viewf.viewf(dem_arr,dx)[0]
        self.dem_param['svf'] = svf.flatten()
        svf, slope, aspect = None, None, None
        dem_arr = None
        
        
    class config():
        def __init__(self, config_file):
            self.file_config = config_file 
            
            # parse configuration file into config class
            self._parse_config_file()
            
            # check if tree directory exists. If not create it
            if not os.path.exists('/'.join(self.project_dir, 'inputs/')):
                os.makedirs('/'.join(self.project_dir, 'inputs/'))
            if not os.path.exists('/'.join(self.project_dir, 'inputs/forcings/')):
                os.makedirs('/'.join(self.project_dir, 'inputs/forcings'))
            if not os.path.exists('/'.join(self.project_dir, 'inputs/dem/')):
                os.makedirs('/'.join(self.project_dir, 'inputs/dem/'))
            if not os.path.exists('/'.join(self.project_dir, 'outputs/')):
                os.makedirs('/'.join(self.project_dir, 'outputs/'))                       
                
        def _parse_config_file(self):
            '''
            Function to parse config file .ini into a python class
            '''
            conf = configparser.ConfigParser(allow_no_value=True)
            conf.read(self.file_config)
            
            self.project_dir = conf.get('main', 'project_dir')
            self.project_description = conf.get('main', 'project_description')
            self.project_name = conf.get('main', 'project_name')
            self.project_author = conf.get('main', 'project_authors')
            
            self.start_date = conf.get('main', 'start_date')
            self.end_date = conf.get('main', 'end_date')
            self.extent = {'latN': conf.getfloat('main','latN'),
                           'latS': conf.getfloat('main','latS'),
                           'lonW': conf.getfloat('main','lonW'),
                           'lonE': conf.getfloat('main','lonE')}
            
            self.forcing_dataset = conf.get('forcing', 'dataset')
            self.download_dataset = conf.getbool('forcing', 'download_dataset')
            if self.forcing_dataset is 'era5':
                self.forcing_era5_product = conf.get('forcing', 'era5_product')
            self.number_cores = conf.getint('forcing','number_cores')
                
            self.time_step = conf.getint('forcing','time_step')
            self.plevels = conf.getint('forcing', 'plevels')
            
            self.dem_file = conf.get('forcing','dem_file')
            self.dem_dataset = conf.get('forcing','dem_dataset')
            self.dem_download = conf.getbool('forcing','dem_download')
            
            
    def get_era5(self):
        # write code to fetch data from era5
        lonW = self.config.extent.get('lonW') - 0.25
        lonE = self.config.extent.get('lonE') + 0.25
        latN = self.config.extent.get('latN') + 0.25
        latS = self.config.extent.get('latS') - 0.25

        # retreive ERA5 surface data
        fe.retrieve_era5_surf(
            self.config.forcing_era5_product,
            self.config.start_date, 
            self.config.end_date,
            self.config.project_dir + 'inputs/forcings/', 
            latN,latS,lonE,lonW,
            self.config.time_step, self.config.number_cores
            )
        # retrieve era5 plevels
        fe.retrieve_era5_plev(
            self.config.forcing_era5_product,
            self.config.start_date, 
            self.config.end_date,
            self.config.project_dir + 'inputs/forcings/', 
            latN, latS, lonE, lonW, 
            self.config.time_step, self.config.plevels,
            self.config.number_cores
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