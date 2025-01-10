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
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr
from munch import DefaultMunch
from sklearn.preprocessing import StandardScaler

from TopoPyScale import fetch_dem as fd
from TopoPyScale import fetch_era5 as fe
from TopoPyScale import solar_geom as sg
from TopoPyScale import topo_export as te
from TopoPyScale import topo_obs as tpo
from TopoPyScale import topo_param as tp
from TopoPyScale import topo_plot as tpl
from TopoPyScale import topo_scale as ta
from TopoPyScale import topo_sub as ts


class Topoclass(object):
    """
    A python class to bring the typical use-case of toposcale in a user friendly object
    """

    def __init__(self, config_file):

        try:
            with open(config_file, 'r') as f:
                self.config = DefaultMunch.fromYAML(f)

            if self.config.project.directory in [None, {}]:
                self.config.project.directory = os.getcwd() + '/'

        except IOError:
            print(
                f'ERROR: config file does not exist. \n\t Current file path: {config_file}\n\t Current working directory: {os.getcwd()}')

        if self.config.outputs.directory in [None, {}]:
            self.config.outputs.directory = 'outputs'

        self.config.outputs.path = Path(self.config.project.directory, self.config.outputs.directory)
        self.config.outputs.tmp_path = self.config.outputs.path / 'tmp'
        self.config.outputs.downscaled = self.config.outputs.path / 'downscaled'

        if self.config.outputs.file.clean_outputs:
            # remove outputs directory because if results already exist this causes concat of netcdf files
            try:
                shutil.rmtree(self.config.outputs.path)
                print('---> Output directory cleaned')
            except:
                os.makedirs(self.config.outputs.path)

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

        # climate path
        if os.path.isabs(self.config.climate[self.config.project.climate].path):
            self.config.climate.path = self.config.climate[self.config.project.climate].path

        else:
            self.config.climate.path = '/'.join((self.config.project.directory, 'inputs/climate/'))
        self.config.climate.tmp_path = '/'.join((self.config.climate.path, 'tmp'))

        # check if tree directory exists. If not create it
        os.makedirs(self.config.climate.path, exist_ok=True)
        os.makedirs(self.config.climate.tmp_path, exist_ok=True)
        os.makedirs(self.config.outputs.path, exist_ok=True)
        os.makedirs(self.config.outputs.tmp_path, exist_ok=True)
        os.makedirs(self.config.outputs.downscaled, exist_ok=True)

        if not self.config.dem.path:
            self.config.dem.path = self.config.project.directory + '/inputs/dem/'
        os.makedirs(self.config.dem.path, exist_ok=True)

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
        print('\t|        North:{}          |\n\t|West:{}          East:{}|\n\t|        South:{}          |'.format(
            np.round(self.config.project.extent.get('latN'), 1),
            np.round(self.config.project.extent.get('lonW'), 1),
            np.round(self.config.project.extent.get('lonE'), 1),
            np.round(self.config.project.extent.get('latS'), 1)))
        print('\t------------------------------')

        self.toposub = self.Toposub()
        self.plot = Plotting()
        self.ds_solar = None
        self.da_horizon = None

        self.toposub.dem_path = self.config.dem.filepath
        self.toposub.project_directory = self.config.project.directory

        if self.config.project.climate.lower() == 'era5':
            print("era5 download no longer automatically done in class decalaration - please declare <mp.get_era5()> in run.py file.")

        if self.config.project.climate.lower() == 'ifs_forecast':
            self.get_ifs_forecast()        

        if self.config.project.split.IO:
            self.time_splitter = self.TimeSplitter(self.config.project.start,
                                                   self.config.project.end,
                                                   self.config.project.split.time,
                                                   self.config.outputs.file.ds_solar,
                                                   self.config.outputs.file.downscaled_pt)
        if self.config.dem.solar_position_method is None:
            self.config.dem.solar_position_method = 'nrel_numpy'

    def load_project(self):
        '''
        Function to load pre-existing TopoPyScale project saved in files
        '''
        if (self.config.outputs.path / self.config.outputs.file.df_centroids).is_file():
            self.toposub.df_centroids = pd.read_pickle(
                self.config.outputs.path / self.config.outputs.file.df_centroids)
            print(f'---> Centroids file {self.config.outputs.file.df_centroids} exists and loaded')
        else:
            print(f'-> WARNING: Centroid file {self.config.outputs.file.df_centroids} not found')

        if (self.config.outputs.path / self.config.outputs.file.ds_param).is_file():
            self.toposub.ds_param = xr.open_dataset(
                self.config.outputs.path / self.config.outputs.file.ds_param)
            print(f'---> DEM parameter file {self.config.outputs.file.ds_param} exists and loaded')
        else:
            print(f'-> WARNING: DEM parameter file {self.config.outputs.file.ds_param} not found')

        if (self.config.outputs.path / self.config.outputs.file.ds_solar).is_file():
            self.ds_solar = xr.open_dataset(
                self.config.outputs.path / self.config.outputs.file.ds_solar)
            print(f'---> Solar file {self.config.outputs.file.ds_solar} exists and loaded')
        else:
            print(f'-> WARNING: Solar file {self.config.outputs.file.ds_solar} not found')

        if (self.config.outputs.path / self.config.outputs.file.da_horizon).is_file():
            self.da_horizon = xr.open_dataarray(
                self.config.outputs.path / self.config.outputs.file.da_horizon)
            print(f'---> Horizon file {self.config.outputs.file.da_horizon} exists and loaded')
        else:
            print(f'-> WARNING: Horizon file {self.config.outputs.file.da_horizon} not found')

        flist = glob.glob(f'{self.config.outputs.downscaled}/{self.config.outputs.file.downscaled_pt}')
        if len(flist) > 0:
            print('---> Loading downscaled points \n ...')
            self.downscaled_pts = xr.open_mfdataset(
                f'{self.config.outputs.downscaled}/{self.config.outputs.file.downscaled_pt}',
                concat_dim='point_name',
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
            ts.plot_center_clusters(self.dem_path, self.ds_param, self.df_centroids, var=var, cmap=cmap,
                                    figsize=figsize)

        def write_landform(self):
            ts.write_landform(self.dem_path, self.ds_param, self.project_directory)

    def compute_dem_param(self):
        self.toposub.ds_param = tp.compute_dem_param(self.config.dem.filepath,
                                                     fname=self.config.outputs.file.ds_param,
                                                     project_directory=Path(self.config.project.directory),
                                                     output_folder=self.config.outputs.path)

    def search_optimum_number_of_clusters(self,
                                          cluster_range=np.arange(100, 1000, 200),
                                          scaler_type=StandardScaler(),
                                          plot=True):
        '''
        Function to test what would be an appropriate number of clusters
        Args:
            cluster_range (int array): numpy array or list of number of clusters to compute scores.

        Returns:

        '''
        print('---> Testing number of clusters')
        print(f'Computing scores for {cluster_range} clusters')
        df_param = ts.ds_to_indexed_dataframe(self.toposub.ds_param)
        print(f'Variables used in clustering: {list(df_param.columns.values)}')

        df_param = ts.ds_to_indexed_dataframe(self.toposub.ds_param)
        df_nclusters = ts.search_number_of_clusters(df_param,
                                                    method=self.config.sampling.toposub.clustering_method,
                                                    cluster_range=cluster_range,
                                                    features=self.config.sampling.toposub.clustering_features,
                                                    scaler_type=scaler_type,
                                                    plot=plot)

        # print to console stats about
        print(df_nclusters)

        return df_nclusters

    def extract_pts_param(self, method='nearest', **kwargs):
        """
        Function to use a list point as input rather than cluster centroids from DEM segmentation (topo_sub.py/self.clustering_dem()).

        Args:
            df (dataFrame):
            method (str): method of sampling
            **kwargs: pd.read_csv() parameters
        Returns:
        """
        if not os.path.isabs(self.config.sampling.points.csv_file):
            self.config.sampling.points.csv_file = self.config.project.directory + 'inputs/dem/' + self.config.sampling.points.csv_file
        df_centroids = pd.read_csv(self.config.sampling.points.csv_file, **kwargs)
        if self.config.sampling.points.name_column:
            df_centroids['point_name'] = df_centroids[self.config.sampling.points.name_column].astype(str)
        else:
            df_centroids['point_name'] = df_centroids.index + 1
            n_digits = len(str(df_centroids.point_name.max()))
            df_centroids['point_name'] = df_centroids.point_name.astype(str).str.zfill(n_digits)

        df_centroids['point_ind'] = df_centroids.index + 1
        df_centroids = tp.extract_pts_param(df_centroids, self.toposub.ds_param,
                                            method=method)

        self.toposub.df_centroids = df_centroids


    def extract_grid_param(self):
        return

    def extract_topo_cluster_param(self):
        """
        Function to segment a DEM in clusters and retain only the centroids of each cluster.

        :return:

        TODO:
        - try to migrate most code of this function to topo_sub.py as a function itself segment_topo()
        """
        # ----------handle input----------
        mask_file = self.config.sampling.toposub.clustering_mask
        groups_file = self.config.sampling.toposub.clustering_groups

        # ----------proceed----------

        # read df param
        df_param = ts.ds_to_indexed_dataframe(self.toposub.ds_param)

        # add mask and cluster feature
        if mask_file in [None, {}]:
            mask = [True] * len(df_param)
        else:
            if not os.path.isabs(mask_file):
                mask_file = Path(self.config.project.directory, mask_file)
            # read mask TIFF
            ds_mask = rio.open_rasterio(mask_file).to_dataset('band').rename({1: 'mask'})

            # check if bounds and resolution match
            dem = self.toposub.ds_param
            if not ds_mask.rio.bounds() == dem.rio.bounds() or not ds_mask.rio.resolution() == dem.rio.resolution():
                raise ValueError(
                    'The GeoTIFFS of the DEM and the MASK need to habe the same bounds/resolution. Please check.')
            print(f'---> Only consider grid cells inside mask ({Path(mask_file).name})')

            # get mask
            mask = ts.ds_to_indexed_dataframe(ds_mask)['mask'] == 1

        # add cluster groups. Groups can be landcover classes for instance
        if groups_file in [None, {}]:
            split_clustering = False
            df_param['cluster_group'] = 1
            groups = [1]
        else:
            split_clustering = True
            if not os.path.isabs(groups_file):
                groups_file = Path(self.config.project.directory, groups_file)

            # read cluster TIFF
            ds_group = rio.open_rasterio(groups_file).to_dataset('band').rename({1: 'group'})

            # check if bounds and resolution match
            dem = self.toposub.ds_param
            if not ds_group.rio.bounds() == dem.rio.bounds() or not ds_group.rio.resolution() == dem.rio.resolution():
                raise ValueError(
                    'The GeoTIFFS of the DEM and the MASK must have the same bounds/resolution. Please check.')

            # add cluster group
            df_param['cluster_group'] = ts.ds_to_indexed_dataframe(ds_group)['group'].astype(int)

            groups = sorted(df_param.cluster_group.unique())
            print(f'---> Split clustering into {len(groups)} groups.')

        # ----------cluster per group----------
        self.toposub.df_centroids = pd.DataFrame()
        total_clusters = 0
        for group in groups:
            if split_clustering:
                print(f'cluster group: {group}')
            subset_mask = mask & (df_param.cluster_group == group)
            df_subset = df_param[subset_mask]

            # derive number of clusters
            relative_cover = np.count_nonzero(subset_mask) / np.count_nonzero(mask)  # relative area of the group
            n_clusters = int(np.ceil(self.config.sampling.toposub.n_clusters * relative_cover))

            df_scaled, scaler = ts.scale_df(df_subset, features=self.config.sampling.toposub.clustering_features)
            if self.config.sampling.toposub.clustering_method.lower() == 'kmean':
                df_centroids, _, cluster_labels = ts.kmeans_clustering(
                    df_scaled,
                    features=self.config.sampling.toposub.clustering_features,
                    n_clusters=n_clusters,
                    seed=self.config.sampling.toposub.random_seed)
            elif self.config.sampling.toposub.clustering_method.lower() == 'minibatchkmean':
                df_centroids, _, cluster_labels = ts.minibatch_kmeans_clustering(
                    df_scaled,
                    n_clusters=n_clusters,
                    features=self.config.sampling.toposub.clustering_features,
                    n_cores=self.config.project.CPU_cores,
                    seed=self.config.sampling.toposub.random_seed)
            else:
                raise ValueError(
                    'ERROR: {} clustering method not available'.format(self.config.sampling.toposub.clustering_method))

            df_centroids = ts.inverse_scale_df(df_centroids, scaler,
                                               features=self.config.sampling.toposub.clustering_features)

            # re-number cluster labels
            df_param.loc[subset_mask, 'cluster_labels'] = cluster_labels + total_clusters
            df_centroids.index = df_centroids.index + total_clusters

            # merge df centroids
            self.toposub.df_centroids = pd.concat([self.toposub.df_centroids, df_centroids])

            # update total clusters
            total_clusters += n_clusters

        if not split_clustering:
            # drop cluster_group
            df_param = df_param.drop(columns=['cluster_group'])
        else:
            print(f'-----> Created in total {total_clusters} clusters in {len(groups)} groups.')

        # logic to add variables not used as clustering predictors into df_centroids
        feature_list = self.config.sampling.toposub.clustering_features.keys()
        flist = list(feature_list)
        flist.append('cluster_labels')
        if len(df_param.columns) > len(flist):
            tmp = df_param.groupby('cluster_labels').mean()
            for var in df_param.drop(flist, axis=1).columns:
                self.toposub.df_centroids[var] = tmp[var]
        self.toposub.df_centroids['cluster_labels'] = tmp.index.values.astype(int)
        n_digits = len(str(self.toposub.df_centroids.cluster_labels.max()))
        self.toposub.df_centroids['point_name'] = self.toposub.df_centroids.cluster_labels.astype(int).astype(str).str.zfill(n_digits)
        self.toposub.df_centroids['point_ind'] = self.toposub.df_centroids.cluster_labels.astype(int)
        if df_param.cluster_labels.isnull().values.any():
            df_param['point_name'] = '-9999'
            df_param['point_name'].loc[~df_param.cluster_labels.isnull()] = df_param.cluster_labels.loc[~df_param.cluster_labels.isnull()].astype(int).astype(str).str.zfill(n_digits)
            df_param['point_ind'] = df_param.point_name.astype(int)
        else:
            df_param['point_name'] = df_param.cluster_labels.astype(int).astype(str).str.zfill(n_digits)
            df_param['point_ind'] = df_param.cluster_labels.astype(int)

        # Build the final cluster map
        self.toposub.ds_param['cluster_labels'] = (["y", "x"], np.reshape(df_param.point_name.values, self.toposub.ds_param.slope.shape))
        self.toposub.ds_param['point_name'] = (["y", "x"], np.reshape(df_param.point_name.values, self.toposub.ds_param.slope.shape))
        self.toposub.ds_param['point_ind'] = (["y", "x"], np.reshape(df_param.point_ind.values, self.toposub.ds_param.slope.shape))

        # update file
        fname = self.config.outputs.path / self.config.outputs.file.ds_param
        te.to_netcdf(self.toposub.ds_param, fname=fname)

        # update plotting class variable
        self.plot.ds_param = self.toposub.ds_param

    def extract_topo_param(self):
        """
        Function to select which
        """
        if (self.config.outputs.path / self.config.outputs.file.df_centroids).is_file():
            self.toposub.df_centroids = pd.read_pickle(
                self.config.outputs.path / self.config.outputs.file.df_centroids)
            print(f'---> Centroids file {self.config.outputs.file.df_centroids} exists and loaded')
        else:
            if self.config.sampling.method.lower() in ['points', 'point']:
                self.extract_pts_param()
                # if self.config.sampling.points.ID_col:
                #     self.config.sampling.pt_names = list(
                #         self.toposub.df_centroids[self.config.sampling.points.ID_col])
            elif self.config.sampling.method.lower() in ['toposub', 'cluster', 'clusters']:
                self.extract_topo_cluster_param()
            elif self.config.sampling.method == 'grid':
                self.toposub.df_centroids = ts.ds_to_indexed_dataframe(self.toposub.ds_param)
                self.toposub.df_centroids['point_name'] = self.toposub.df_centroids.index

            else:
                raise ValueError('ERROR: Extraction method not available')

            self.toposub.df_centroids['lon'], self.toposub.df_centroids['lat'] = tp.convert_epsg_pts(
                self.toposub.df_centroids.x,
                self.toposub.df_centroids.y,
                self.config.dem.epsg,
                4326)

            # Store dataframe to pickle
            self.toposub.df_centroids.to_pickle(
                self.config.outputs.path / self.config.outputs.file.df_centroids)
            print(f'---> Centroids file {self.config.outputs.file.df_centroids} saved')

    class TimeSplitter():
        def __init__(self, start, end, period, ds_solar_fname, downscale_fname):
            self.start = start
            self.end = end
            self.period = period

            s = pd.date_range(self.start, self.end, freq=f'{self.period}YS')
            self.start_list = list(s.astype(str))
            self.end_list = list((s - pd.Timedelta('1D')).astype(str))[1:]
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
                fname = self.config.outputs.path / self.time_splitter.ds_solar_flist[i]

                if fname.is_file():
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
                                                      self.config.outputs.path,
                                                      method_solar=self.config.solar_position_method)

        else:
            fname = self.config.outputs.path / self.config.outputs.file.ds_solar
            if fname.is_file():
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
                                                  self.config.outputs.path)

    def compute_horizon(self):
        """
        Function to compute horizon angle and sample values for list of points
        """
        fname = self.config.outputs.path / self.config.outputs.file.da_horizon
        if fname.is_file():
            self.da_horizon = xr.open_dataarray(fname, engine='h5netcdf')
            print(f'---> Horizon file {self.config.outputs.file.da_horizon} exists and loaded')
        else:
            self.da_horizon = tp.compute_horizon(self.config.dem.filepath,
                                                 self.config.dem.horizon_increments,
                                                 self.config.project.CPU_cores,
                                                 self.config.outputs.file.da_horizon,
                                                 self.config.outputs.path)
        tgt_x = tp.xr.DataArray(self.toposub.df_centroids.x.values, dims="points")
        tgt_y = tp.xr.DataArray(self.toposub.df_centroids.y.values, dims="points")

        # In case horizon angles are computed after clustering
        if self.toposub.df_centroids is not None:
            for az in self.da_horizon.azimuth.values:
                self.toposub.df_centroids['hori_azi_' + str(az)] = self.da_horizon.sel(x=tgt_x,
                                                                                       y=tgt_y,
                                                                                       azimuth=az,
                                                                                       method='nearest').values.flatten()
            # update df_centroid file with horizons angle
            self.toposub.df_centroids.to_pickle(
                self.config.outputs.path / self.config.outputs.file.df_centroids)
            print(f'---> Centroids file {self.config.outputs.file.df_centroids} updated with horizons')

    def downscale_climate(self):
        downscaled_dir = self.config.outputs.downscaled
        f_pattern = self.config.outputs.file.downscaled_pt
        output_folder = self.config.outputs.path.name

        if '*' not in f_pattern:
            raise ValueError(
                f'ERROR: The filepattern for the downscaled files does need to have a * in the name. You provided {f_pattern}')

        # clean directory from files with the same downscaled output file pattern (so they get replaced)
        existing_files = sorted(downscaled_dir.glob(f_pattern))
        for file in existing_files:
            file.unlink()
            print(f'existing file {file.name} removed.')

        if self.config.project.split.IO:
            for i, start in enumerate(self.time_splitter.start_list):
                print()
                end = self.time_splitter.end_list[i]
                fname = self.time_splitter.downscaled_flist[i]

                self.ds_solar = None
                self.ds_solar = xr.open_dataset(
                    self.config.outputs.path / self.time_splitter.ds_solar_flist[i])

                ta.downscale_climate(self.config.project.directory,
                                     self.config.climate.path,
                                     self.config.outputs.path,
                                     self.toposub.df_centroids,
                                     self.da_horizon,
                                     self.ds_solar,
                                     self.config.dem.epsg,
                                     start,
                                     end,
                                     self.config.climate[self.config.project.climate].timestep,
                                     self.config.toposcale.interpolation_method,
                                     self.config.toposcale.LW_terrain_contribution,
                                     self.config.climate.precip_lapse_rate,
                                     fname,
                                     self.config.project.CPU_cores)

            for pt_id in self.toposub.df_centroids.point_name.values:
                print(f'Concatenating point {pt_id}')
                filename = Path(f_pattern.replace('*', pt_id))
                flist = sorted([file for file in downscaled_dir.rglob(f'{filename.stem}_*')])
                ds_list = [xr.open_dataset(file, engine='h5netcdf') for file in flist]

                fout = Path(downscaled_dir, filename)
                ds = xr.concat(ds_list, dim='time')
                ds.to_netcdf(fout, engine='h5netcdf')
                del ds

            # Delete time slice files.
            for fpat in self.time_splitter.downscaled_flist:
                flist = glob.glob(f'{self.config.outputs.downscaled}/{fpat}')
                for file in flist:
                    os.remove(file)

            # concatenate ds solar
            print('concatenating solar files')
            out_solar_name = Path(self.config.outputs.file.ds_solar)
            solar_pattern = f'{out_solar_name.stem}_*{out_solar_name.suffix}'
            solar_flist = sorted(self.config.outputs.path.glob(solar_pattern))
            ds_solar_list = [xr.open_dataset(file, engine='h5netcdf') for file in solar_flist]
            fout = Path(self.config.outputs.path, out_solar_name)
            ds = xr.concat(ds_solar_list, dim='time')
            ds.to_netcdf(fout, engine='h5netcdf')
            [f.unlink() for f in solar_flist]  # delete tmp solar files
            print('solar files concatenated')
            del ds

        else:
            ta.downscale_climate(self.config.project.directory,
                                 self.config.climate.path,
                                 self.config.outputs.path,
                                 self.toposub.df_centroids,
                                 self.da_horizon,
                                 self.ds_solar,
                                 self.config.dem.epsg,
                                 self.config.project.start,
                                 self.config.project.end,
                                 self.config.climate[self.config.project.climate].timestep,
                                 self.config.toposcale.interpolation_method,
                                 self.config.toposcale.LW_terrain_contribution,
                                 self.config.climate.precip_lapse_rate,
                                 self.config.outputs.file.downscaled_pt,
                                 self.config.project.CPU_cores)

        self.downscaled_pts = ta.read_downscaled(f'{downscaled_dir}/{f_pattern}')
        # update plotting class variables
        self.plot.ds_down = self.downscaled_pts

        # delete tmp directories
        if self.config.clean_up.delete_tmp_dirs:
            shutil.rmtree(self.config.outputs.tmp_path, ignore_errors=True)
            shutil.rmtree(self.config.climate.tmp_path, ignore_errors=True)

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

        # if keyword exists in config set realtime to value (True/False)
        if self.config.climate[self.config.project.climate].realtime:
            realtime = self.config.climate[self.config.project.climate].realtime
        # else set realtime to False
        else:
            realtime = False

        if realtime:  # make sure end date is correct
            lastdate = fe.return_last_fullday()
            # 5th day will always be incomplete so we got to last fullday
            self.config.project.end = self.config.project.end.replace(year=lastdate.year, month=lastdate.month,
                                                                      day=lastdate.day)

            # remove existing results (temporary results are ok and updated but cannot append to final files eg .outputs/downscaled/down_pt_0.nc)
            try:
                shutil.rmtree(self.config.outputs.path / "downscaled")
                print('---> Downscaled directory cleaned')
                os.makedirs(self.config.outputs.path / "downscaled")
            except:
                os.makedirs(self.config.outputs.path / "downscaled")
                
        realtime = False # always false now as redownload of current month handled elsewhere ? key required only to dynamically set config.project.end
        output_format = self.config.climate[self.config.project.climate].output_format

        # if keyword exists in config set out_format to value or default to netcdf
        if self.config.climate[self.config.project.climate].output_format:
            output_format = self.config.climate[self.config.project.climate].output_format
        # else set realtime to False
        else:
            output_format = 'netcdf'


        # retreive ERA5 surface data
        fe.retrieve_era5(
            self.config.climate[self.config.project.climate].product,
            self.config.project.start,
            self.config.project.end,
            self.config.climate.path,
            latN, latS, lonE, lonW,
            self.config.climate[self.config.project.climate].timestep,
            self.config.climate[self.config.project.climate].download_threads,
            surf_plev='surf',
            realtime=realtime,
            output_format=output_format
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
            realtime=realtime,
            output_format=output_format
        )


    def get_era5_snowmapper(self, surf_plev, lastday):
        """
        Funtion to call fetching of ERA5 data
        TODO:
        - merge monthly data into one file (cdo?)- this creates massive slow down!
        """
        lonW = self.config.project.extent.get('lonW') - 0.4
        lonE = self.config.project.extent.get('lonE') + 0.4
        latN = self.config.project.extent.get('latN') + 0.4
        latS = self.config.project.extent.get('latS') - 0.4


        # if keyword exists in config set out_format to value or default to netcdf
        if self.config.climate[self.config.project.climate].output_format:
            output_format = self.config.climate[self.config.project.climate].output_format
        # else set realtime to False
        else:
            output_format = 'netcdf'

        if surf_plev == 'surf':
            # retreive ERA5 surface data
            fe.era5_request_surf_snowmapper(
                lastday,
                latN, latS, lonE, lonW,
                self.config.climate.path,
                output_format=output_format
            )

        if surf_plev == 'plev':            
            # retrieve era5 plevels
            fe.era5_request_plev_snowmapper(
                lastday,
                latN, latS, lonE, lonW,
                self.config.climate.path,
                plevels=self.config.climate[self.config.project.climate].plevels,
                output_format=output_format
            )

    def get_lastday(self):
        lastday = fe.return_last_fullday()
        return(lastday)

    def process_SURF_file(self, wdir):
        fe.process_SURF_file(wdir)

    def remap_netcdf(self, filepath):
        fe.remap_CDSbeta(filepath)

    def get_ifs_forecast(self):
        # run openData script here
        print("doing forecast stuff here")
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
        for year in df.year.unique():
            # API only accept one year request
            tpo.fetch_WMO_insitu_observations(years=year,
                                              months=list(df.month.unique().astype(str)),
                                              bbox=bbox)
        tpo.parse_WMO_insitu_observations()

    def to_cryogrid(self, fname_format='Cryogrid_pt_*.nc', precip_partition='continuous'):
        """
        wrapper function to export toposcale output to cryosgrid format from TopoClass
        
        Args:
            fname_format (str): filename format. point_name is inserted where * is

        """
        path = self.config.outputs.path
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
        te.to_fsm(self.downscaled_pts, self.config.outputs.path /  fname_format)

    def to_crocus(self, fname_format='CROCUS_pt_*.nc', scale_precip=1):
        """
        function to export toposcale output to crocus format .nc. This functions saves one file per point_name
        
        Args:
            fout_format (str): filename format. point_name is inserted where * is
            scale_precip(float): scaling factor to apply on precipitation. Default is 1
        """
        te.to_crocus(self.downscaled_pts,
                     self.toposub.df_centroids,
                     fname_format=self.config.outputs.path / fname_format,
                     scale_precip=scale_precip,
                     climate_dataset_name=self.config.project.climate,
                     project_author=self.config.project.authors)

    def to_surfex(self, 
                  fname_format='FORCING*.nc',
                  scale_precip=1, 
                  year_start=None, 
                  year_end=None):
        """
        function to export toposcale output to surfex format .nc. This functions saves one file per point_name
        
        Args:
            fout_format (str): filename format. point_name is inserted where * is
            scale_precip(float): scaling factor to apply on precipitation. Default is 1
        """
        if year_start is None or year_end is None:
            raise ValueError("Arguments year_start and year_end are required.")
        
        te.to_surfex(self.downscaled_pts,
                     self.toposub.df_centroids,
                     fname_format=self.config.outputs.path / fname_format,
                     scale_precip=scale_precip,
                     climate_dataset_name=self.config.project.climate,
                     project_author=self.config.project.authors,
                     snow_partition_method='continuous',
                     year_start=year_start,
                     year_end=year_end)

    def to_snowmodel(self, fname_format='Snowmodel_stn_*.csv'):
        """
        function to export toposcale output to snowmodel format .ascii, for single station standard

            fout_format: str, filename format. point_name is inserted where * is
        """
        te.to_micromet_single_station(self.downscaled_pts,
                                      self.toposub.df_centroids,
                                      fname_format=self.config.outputs.path / fname_format,
                                      na_values=-9999,
                                      headers=False)

    def to_netcdf(self, file_out='output.nc', variables=None):
        """
        function to export toposcale output to one single generic netcdf format, compressed

        Args:
            file_out (str): name of export file
            variables (list str): list of variable to export. Default exports all variables
        """
        if variables is None:
            if (type(self.config.outputs.variables) is not list) or (self.config.outputs.variables is None):
                variables = list(self.downscaled_pts.keys())
            else:
                variables = self.config.outputs.variables

        out_path = Path(self.config.outputs.path, file_out)

        te.to_netcdf(self.downscaled_pts[variables], out_path, variables)
        print('---> File {} saved'.format(file_out))

    def to_snowpack(self, fname_format='smet_pt_*.smet'):
        """
        function to export toposcale output to FSM format
        """
        te.to_snowpack(self.downscaled_pts, self.config.outputs.path / fname_format)

    def to_geotop(self, fname_format='meteo_*.txt'):
        """
        function to export toposcale output to FSM format
        """
        te.to_geotop(self.downscaled_pts, self.config.outputs.path / fname_format)

    def to_musa(self,
                fname_met='musa_met.nc',
                fname_labels='musa_labels.nc'):
        """
        function to export TopoPyScale output in a format compatible with MuSa
        MuSa: https://github.com/ealonsogzl/MuSA

        Args:
            fname: filename of the netcdf 
        """
        try:
            labels = self.toposub.ds_param.cluster_labels
        except Exception:
            labels = None

        te.to_musa(ds=self.downscaled_pts,
                   df_pts=self.toposub.df_centroids,
                   da_label=labels,
                   fname_met=fname_met,
                   fname_labels=fname_labels,
                   path=self.config.outputs.path,
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
                     var='t',
                     time_step=1,
                     time=None,
                     cmap=plt.cm.RdBu_r,
                     hillshade=True,
                     **kwargs):
        # add logic to check ds_down and ds_param exist.
        tpl.map_variable(self.ds_down,
                         self.ds_param,
                         time_step=time_step,
                         time=time,
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

    def map_clusters(self, **kwargs):
        tp.map_clusters(self.ds_down,
                        self.toposub.ds_param,
                        **kwargs)

    def timeseries(self):
        print('To be implemented')

    def solar_geom(self):
        print('To be implemented')

    def horizon(self):
        print('To be implemented')
