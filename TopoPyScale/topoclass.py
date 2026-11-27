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
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from munch import DefaultMunch
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from TopoPyScale import fetch_dem as fd
from TopoPyScale import fetch_era5 as fe
from TopoPyScale import era5_store as es
from TopoPyScale import log as tps_log
from TopoPyScale import config_check
import yaml
from datetime import datetime
from TopoPyScale import solar_geom as sg
from TopoPyScale import topo_export as te
from TopoPyScale import topo_obs as tpo
from TopoPyScale import topo_param as tp
from TopoPyScale import topo_plot as tpl
from TopoPyScale import topo_sub as ts
from TopoPyScale import topo_scale_zarr as tz
from TopoPyScale import surface_effect as se
from TopoPyScale import toposcale_core as core
from TopoPyScale import topo_utils as tu
from TopoPyScale import topo_checks as tc
import logging

logger = logging.getLogger(__name__)

class Topoclass(object):
    """
    A python class to bring the typical use-case of toposcale in a user friendly object
    """

    def __init__(self, config_file):

        if not Path(config_file).is_file():
            raise FileNotFoundError(f'Config file {config_file} not found (current directory: {os.getcwd()})')
        with open(config_file, 'r') as f:
            raw = yaml.safe_load(f)
        self.config = DefaultMunch.fromDict(raw)
        self._config_messages = config_check.check_config(raw)
        self.config.project.directory = Path(self.config.project.directory or os.getcwd())

        if self.config.outputs.directory in [None, {}]:
            self.config.outputs.directory = 'outputs'

        self.config.outputs.path = Path(self.config.project.directory, self.config.outputs.directory)
        self.config.outputs.tmp_path = self.config.outputs.path / 'tmp'
        self.config.outputs.downscaled = self.config.outputs.path / 'downscaled'

        if self.config.outputs.file.clean_outputs:
            # remove outputs directory because if results already exist this causes concat of netcdf files
            shutil.rmtree(self.config.outputs.path, ignore_errors=True)
            logger.info('---> Output directory cleaned')

        # remove output fsm directory
        if self.config.outputs.file.clean_FSM:
            for folder in ['fsm_sims', 'ensemble']:
                path = self.config.project.directory / folder
                if path.exists():
                    shutil.rmtree(path)
                    logger.info(f'---> {folder} directory cleaned')


        # climate path
        if os.path.isabs(self.config.climate[self.config.project.climate].path):
            # modify pth handling with Path
            self.config.climate.path = Path(self.config.climate[self.config.project.climate].path)

        else:
            #self.config.climate.path = self.config.project.directory / 'inputs' / 'climate'
            self.config.climate.path = self.config.project.directory / Path(self.config.climate[self.config.project.climate].path)
            self.config.climate.tmp_path = self.config.climate.path / 'tmp'
            # check if tree directory exists. If not create it
            self.config.climate.path.mkdir(parents=True, exist_ok=True)
            self.config.climate.tmp_path.mkdir(parents=True, exist_ok=True)

        # pandas and the ERA5 request use lower-case frequencies ('1h', not '1H')
        climate_source = self.config.climate[self.config.project.climate]
        if isinstance(climate_source.timestep, str):
            climate_source.timestep = climate_source.timestep.lower()

        self.config.outputs.path.mkdir(parents=True, exist_ok=True)
        self.config.outputs.tmp_path.mkdir(parents=True, exist_ok=True)
        self.config.outputs.downscaled.mkdir(parents=True, exist_ok=True)
        self.log_file = self._setup_logging()
        for m in self._config_messages:   # once the log is set up
            logger.warning(f'config: {m}')

        if not self.config.dem.path:
            self.config.dem.path = self.config.project.directory / 'inputs' / 'dem'
        else:
            # relative to the project directory, absolute paths are kept as they are
            self.config.dem.path = Path(self.config.project.directory, self.config.dem.path)
        self.config.dem.path.mkdir(parents=True, exist_ok=True)

        self.config.dem.filepath = self.config.dem.path / self.config.dem.file
        if not self.config.dem.filepath.exists():

            if self.config.project.extent:
                self.config.project.extent = dict(zip(['latN', 'latS', 'lonW', 'lonE'], self.config.project.extent))
            else:
                raise ValueError(f'DEM {self.config.dem.filepath} not found, and no project.extent to download '
                                 f'one: give extent as [latN, latS, lonW, lonE]')

            fd.fetch_dem(str(self.config.dem.path), self.config.project.extent, self.config.dem.epsg, self.config.dem.file)
        else:
            logger.info('\n---> DEM file found')

        if self.config.project.extent:
            extent_NSWE = self.config.project.extent
            self.config.project.extent = dict(zip(['latN', 'latS', 'lonW', 'lonE'], extent_NSWE))
        else:
            # little routine extracting lat/lon extent from DEM
            self.config.project.extent = tp.get_extent_latlon(self.config.dem.filepath, self.config.dem.epsg)

        logger.info(self.config.project.extent)
        logger.info('Project lat/lon extent:\n')
        logger.info('\t------------------------------')
        logger.info('\t|        North:{}          |\n\t|West:{}          East:{}|\n\t|        South:{}          |'.format(
            np.round(self.config.project.extent.get('latN'), 1),
            np.round(self.config.project.extent.get('lonW'), 1),
            np.round(self.config.project.extent.get('lonE'), 1),
            np.round(self.config.project.extent.get('latS'), 1)))
        logger.info('\t------------------------------')

        self.toposub = self.Toposub()
        self.plot = Plotting()
        self.ds_solar = None
        self.da_horizon = None

        self.toposub.dem_path = self.config.dem.filepath
        self.toposub.project_directory = self.config.project.directory

        if self.config.project.climate.lower() == 'era5':
            self.era5_obj = fe.FetchERA5(config=self.config)
            logger.info("era5 download no longer automatically done in class decalaration - please declare <mp.get_era5()> in run.py file.")

        if self.config.project.climate.lower() == 'ifs_forecast':
            self.get_ifs_forecast()

        if not self.config.dem.solar_position_method:
            self.config.dem.solar_position_method = 'nrel_numpy'

    def load_project(self):
        '''
        Function to load pre-existing TopoPyScale project saved in files
        '''
        if (self.config.outputs.path / self.config.outputs.file.df_centroids).is_file():
            self.toposub.df_centroids = pd.read_pickle(
                self.config.outputs.path / self.config.outputs.file.df_centroids)
            logger.info(f'---> Centroids file {self.config.outputs.file.df_centroids} exists and loaded')
        else:
            logger.warning(f'Centroid file {self.config.outputs.file.df_centroids} not found')

        if (self.config.outputs.path / self.config.outputs.file.ds_param).is_file():
            self.toposub.ds_param = xr.open_dataset(
                self.config.outputs.path / self.config.outputs.file.ds_param)
            logger.info(f'---> DEM parameter file {self.config.outputs.file.ds_param} exists and loaded')
        else:
            logger.warning(f'DEM parameter file {self.config.outputs.file.ds_param} not found')

        if (self.config.outputs.path / self.config.outputs.file.ds_solar).is_file():
            self.ds_solar = xr.open_dataset(
                self.config.outputs.path / self.config.outputs.file.ds_solar)
            logger.info(f'---> Solar file {self.config.outputs.file.ds_solar} exists and loaded')

        if (self.config.outputs.path / self.config.outputs.file.da_horizon).is_file():
            self.da_horizon = xr.open_dataarray(
                self.config.outputs.path / self.config.outputs.file.da_horizon)
            logger.info(f'---> Horizon file {self.config.outputs.file.da_horizon} exists and loaded')
        else:
            logger.warning(f'Horizon file {self.config.outputs.file.da_horizon} not found')

        store = self.config.outputs.downscaled / self._downscaled_store_name()
        if store.exists():
            self.downscaled_pts = tu.open_downscaled_store(store)
            logger.info(f'---> Downscaled points {store.name} exists and loaded')
        else:
            logger.warning(f'Downscaled points {store.name} not found')

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
                                                     project_directory=self.config.project.directory,
                                                     output_folder=self.config.outputs.path,
                                                     n_processes=self.config.project.parallelization.setting.multicore.CPU_cores)

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
        logger.info('---> Testing number of clusters')
        logger.info(f'Computing scores for {cluster_range} clusters')
        df_param = ts.ds_to_indexed_dataframe(self.toposub.ds_param)
        logger.info(f'Variables used in clustering: {list(df_param.columns.values)}')

        df_param = ts.ds_to_indexed_dataframe(self.toposub.ds_param)
        df_nclusters = ts.search_number_of_clusters(df_param,
                                                    method=self.config.sampling.toposub.clustering_method,
                                                    cluster_range=cluster_range,
                                                    features=self.config.sampling.toposub.clustering_features,
                                                    scaler_type=scaler_type,
                                                    plot=plot)

        # print to console stats about
        logger.info(df_nclusters)

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
            self.config.sampling.points.csv_file = self.config.project.directory / 'inputs'/'dem' / self.config.sampling.points.csv_file
        df_centroids = pd.read_csv(str(self.config.sampling.points.csv_file), **kwargs)
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

        # variables of a previous clustering (ds_param.nc read from disk) are replaced, not used as cluster attributes
        self.toposub.ds_param = self.toposub.ds_param.drop_vars(
            ['cluster_labels', 'point_name', 'point_ind', 'mask', 'cluster_group'], errors='ignore')

        # read df param. Rows in map order (y, x), whatever the order of the dimensions of ds_param (it differs
        # between a dataset built from the DEM and one read from ds_param.nc): the mask, the groups and the cluster
        # maps below rely on it
        df_param = ts.ds_to_indexed_dataframe(self.toposub.ds_param, dim_order=['y', 'x'])

        # Auto-mask nodata pixels (e.g. from reprojected DEMs with non-rectangular valid regions)
        valid_elev = ~df_param['elevation'].isna()
        n_nodata = (~valid_elev).sum()
        if n_nodata > 0:
            logger.info(f'---> Auto-masking {n_nodata} nodata pixels from DEM')

        # add mask and cluster feature
        if mask_file in [None, {}]:
            mask = valid_elev
        else:
            if not os.path.isabs(mask_file):
                mask_file = Path(self.config.project.directory, mask_file)
                if not mask_file.exists():
                    raise ValueError(f'File {mask_file} does not exist.')

            # read mask TIFF
            ds_mask = xr.open_dataset(mask_file, engine='rasterio').band_data.isel(band=0).drop_vars('band').to_dataset()
            ds_mask = ds_mask.rename({'band_data': 'mask'})
            self.toposub.ds_param['mask'] = ds_mask.mask.drop_vars('spatial_ref')

            # check if bounds and resolution match
            if not ds_mask.rio.bounds() == self.toposub.ds_param.rio.bounds() or not ds_mask.rio.resolution() == self.toposub.ds_param.rio.resolution():

                str_bounds = f"mask bounds: {ds_mask.rio.bounds()} \t|\t dem bounds: {self.toposub.ds_param.rio.bounds()}"
                str_res = f"mask resolution: {ds_mask.rio.resolution()} \t|\t dem resolution: {self.toposub.ds_param.rio.resolution()}"

                raise ValueError(
                    f'The GeoTIFFS of the DEM and the MASK need to have the same bounds/resolution. \n{str_bounds}\n{str_res}')
            logger.info(f'---> Only consider grid cells inside mask ({Path(mask_file).name})')

            # get mask (combine with nodata auto-mask)
            mask = (ts.ds_to_indexed_dataframe(ds_mask, dim_order=['y', 'x'])['mask'] == 1) & valid_elev

        # add cluster groups. Groups can be landcover classes for instance
        if groups_file in [None, {}]:
            split_clustering = False
            df_param['cluster_group'] = 1
            groups = [1]
        else:
            split_clustering = True
            if not os.path.isabs(groups_file):
                groups_file = Path(self.config.project.directory, groups_file)
                if not groups_file.exists():
                    raise ValueError(f'File {groups_file} does not exist.')

            # read cluster TIFF
            ds_group = xr.open_dataset(groups_file, engine='rasterio').band_data.isel(band=0).drop_vars('band').to_dataset()
            ds_group = ds_group.rename({'band_data': 'group'})

            # check if bounds and resolution match
            if not ds_group.rio.bounds() == self.toposub.ds_param.rio.bounds() or not ds_group.rio.resolution() == self.toposub.ds_param.rio.resolution():
                raise ValueError(
                    'The GeoTIFFS of the DEM and the MASK must have the same bounds/resolution. Please check.')

            # add cluster group
            df_param['cluster_group'] = ts.ds_to_indexed_dataframe(ds_group, dim_order=['y', 'x'])['group'].astype(int)

            # create group dataframe
            gr = df_param[mask].groupby('cluster_group').slope.count()
            df_group = pd.DataFrame(gr).rename(columns={'slope':'nPix'})

            group_weights_file = self.config.sampling.toposub.clustering_group_weights
            if group_weights_file not in [None, {}]:
                group_weights_file = Path(self.config.project.directory, group_weights_file)

            if group_weights_file not in [None, {}] and group_weights_file.is_file():
                gw = pd.read_csv(group_weights_file)
                wcol = 'weight' if 'weight' in gw.columns else 'weights'
                if not np.isclose(gw[wcol].sum(), 1):
                    raise ValueError(f'The group weights in {group_weights_file.name} must sum to 1 (sum={gw[wcol].sum()})')
                missing = set(df_group.index) - set(gw.group)
                if missing:
                    raise ValueError(f'Groups {sorted(missing)} have no weight in {group_weights_file.name}')
                df_group['weights'] = gw.set_index('group')[wcol].loc[df_group.index].values
            else:
                df_group['weights'] = df_group.nPix / df_group.nPix.sum()
            df_group['n_clusters'] = np.ceil(self.config.sampling.toposub.n_clusters * df_group.weights).astype(int)
            groups = list(df_group.index)

            logger.info(f'---> Split clustering into {len(groups)} groups.')

        # ----------cluster per group----------
        self.toposub.df_centroids = pd.DataFrame()
        i_clusters = 0

        for group in groups:
            if split_clustering:
                logger.info(f'cluster group: {group}')
                n_clusters = int(df_group.loc[group].n_clusters)
            else:
                n_clusters = self.config.sampling.toposub.n_clusters
            subset_mask = mask & (df_param.cluster_group == group)

            # add here check in case a group bleed over the mask. 
            if subset_mask.sum() > 0:
                df_subset = df_param[subset_mask]

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
                        n_cores=self.config.project.parallelization.setting.multicore.CPU_cores,
                        seed=self.config.sampling.toposub.random_seed)
                else:
                    raise ValueError(
                        'ERROR: {} clustering method not available'.format(self.config.sampling.toposub.clustering_method))

                df_centroids = ts.inverse_scale_df(df_centroids, scaler,
                                                   features=self.config.sampling.toposub.clustering_features)

                # re-number cluster labels
                df_param.loc[subset_mask, 'cluster_labels'] = cluster_labels + i_clusters
                df_centroids.index = df_centroids.index + i_clusters

                # merge df centroids
                self.toposub.df_centroids = pd.concat([self.toposub.df_centroids, df_centroids])

                # update total clusters
                i_clusters += n_clusters
            else:
                logger.info(f"---> Group {group} is not fully within masked area")

        if not split_clustering:
            # drop cluster_group
            df_param = df_param.drop(columns=['cluster_group'])
        else:
            logger.info(f'-----> Created in total {i_clusters} clusters in {len(groups)} groups.')

        # logic to add variables not used as clustering predictors into df_centroids
        feature_list = self.config.sampling.toposub.clustering_features.keys()
        flist = list(feature_list)
        flist.append('cluster_labels')
        if len(df_param.columns) > len(flist):
            tmp = df_param.groupby('cluster_labels').mean()
            if 'aspect' in tmp.columns:
                # aspect is an angle in [0, 2pi) with 0 = south: use the circular mean. An arithmetic mean gives ~pi
                # (north-facing) for south-facing clusters made of pixels on both sides of due south.
                circ = pd.DataFrame({'s': np.sin(df_param.aspect), 'c': np.cos(df_param.aspect),
                                     'cluster_labels': df_param.cluster_labels}).groupby('cluster_labels').mean()
                tmp['aspect'] = np.mod(np.arctan2(circ.s, circ.c), 2 * np.pi)
            for var in df_param.drop(flist, axis=1).columns:
                self.toposub.df_centroids[var] = tmp[var]
        self.toposub.df_centroids['cluster_labels'] = tmp.index.values.astype(int)
        n_digits = len(str(self.toposub.df_centroids.cluster_labels.max()))
        self.toposub.df_centroids['point_name'] = self.toposub.df_centroids.cluster_labels.astype(int).astype(str).str.zfill(n_digits)
        self.toposub.df_centroids['point_ind'] = self.toposub.df_centroids.cluster_labels.astype(int)
        if df_param.cluster_labels.isnull().values.any():
            valid = ~df_param.cluster_labels.isnull()
            df_param['point_name'] = '-9999'
            # .loc on the frame: chained assignment (df[col].loc[...] = ...) does nothing with pandas >= 3
            df_param.loc[valid, 'point_name'] = df_param.cluster_labels[valid].astype(int).astype(str).str.zfill(n_digits)
            df_param['point_ind'] = df_param.point_name.astype(int)
        else:
            df_param['point_name'] = df_param.cluster_labels.astype(int).astype(str).str.zfill(n_digits)
            df_param['point_ind'] = df_param.cluster_labels.astype(int)

        # Build the final cluster map
        shape = (self.toposub.ds_param.sizes['y'], self.toposub.ds_param.sizes['x'])
        self.toposub.ds_param['cluster_labels'] = (["y", "x"], np.reshape(df_param.point_name.to_numpy(), shape, order='C'))
        self.toposub.ds_param['point_name'] = (["y", "x"], np.reshape(df_param.point_name.to_numpy(), shape, order='C'))
        self.toposub.ds_param['point_ind'] = (["y", "x"], np.reshape(df_param.point_ind.to_numpy(), shape, order='C'))
        if split_clustering:
            self.toposub.ds_param['cluster_group'] = (["y", "x"], np.reshape(df_param.cluster_group.to_numpy(), shape, order='C'))

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
            logger.info(f'---> Centroids file {self.config.outputs.file.df_centroids} exists and loaded')
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
            logger.info(f'---> Centroids file {self.config.outputs.file.df_centroids} saved')

    def _accumulation_window(self, default=None):
        """
        Radiation accumulation window of the climate forcing: climate.<source>.accumulation_window, else the hours the
        accumulated fields of the climate store cover per time step (topo_scale_zarr.accumulation), else default.
        """
        window = self.config.climate[self.config.project.climate].accumulation_window
        if window or default is None:
            return window
        try:
            with xr.open_zarr(str(self._climate_store())) as era:
                tstep_h = int(pd.Timedelta(self.config.climate[self.config.project.climate].timestep or '1h')
                              / pd.Timedelta('1h'))
                return f'{tz.accumulation(era.time.values, era.attrs, tstep_h)[1]}h'
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return default

    @staticmethod
    def _warn_old_solar_file(ds_solar, fname):
        if 'solar_geometry_time_offset' not in ds_solar.attrs:
            logger.warning(f'{fname} was computed at the time stamps, not at the centre of the radiation '
                  'accumulation window (SW then lags the sun by ~30 min). Delete it to recompute it.')
        elif 'solar_constant' not in ds_solar.attrs:
            logger.warning(f'{fname} was computed by an older version (SWtoa at the window centre, without the '
                  'Earth-Sun distance, angles quantized to 10 bits). Delete it to recompute it.')

    def _solar_geometry(self, start, end, fname):
        return sg.get_solar_geom(self.toposub.df_centroids,
                                 start,
                                 end,
                                 self.config.climate[self.config.project.climate].timestep,
                                 sr_epsg=str(self.config.dem.epsg),
                                 fname=fname,
                                 project_ouput=self.config.outputs.path,
                                 method_solar=self.config.dem.solar_position_method or 'nrel_numpy',
                                 accumulation_window=self._accumulation_window(default='1h'),
                                 solar_constant=self.config.toposcale.solar_constant or sg.S0_DEFAULT,
                                 n_substeps=self.config.toposcale.solar_substeps or 6)

    def compute_solar_geometry(self, save=None):
        """
        Solar geometry of the points for the project period, saved to outputs/ds_solar.nc (or loaded if the file
        exists) and kept as self.ds_solar.

        The downscaling does not need it: it computes the solar geometry of each block of points and time steps when
        it downscales the block. The file is only written when save is True or toposcale.save_solar_geometry is
        true, for plots and analysis; it holds (points x time steps) values and can be large.

        Args:
            save (bool): write (or load) the file. Default: toposcale.save_solar_geometry (false)
        """
        if save is None:
            save = bool(self.config.toposcale.save_solar_geometry)
        if not save:
            logger.info('---> Solar geometry: computed block by block during the downscaling '
                        '(set toposcale.save_solar_geometry: true to also save ds_solar.nc)')
            return
        fname = self.config.outputs.path / self.config.outputs.file.ds_solar
        if fname.is_file():
            self.ds_solar = xr.open_dataset(fname, chunks='auto', engine='h5netcdf')
            logger.info(f'---> Solar file {self.config.outputs.file.ds_solar} exists and loaded')
            self._warn_old_solar_file(self.ds_solar, fname.name)
        else:
            self.ds_solar = self._solar_geometry(self.config.project.start, self.config.project.end,
                                                 self.config.outputs.file.ds_solar)

    def compute_horizon(self):
        """
        Horizon angles of the points (df_centroids), every dem.horizon_increments degrees, along straight rays
        from each point (topo_param.compute_horizon_points). Saved to outputs/da_horizon.nc and as the columns
        hori_azi_<azimuth> of df_centroids. An existing horizon file is loaded if it holds these points (names and
        positions), and recomputed otherwise.
        """
        df = self.toposub.df_centroids
        if df is None:
            raise ValueError('Horizons are computed at the points: run extract_topo_param() first')
        fname = self.config.outputs.path / self.config.outputs.file.da_horizon
        da = None
        if fname.is_file():
            da = xr.open_dataarray(fname, engine='h5netcdf').load()
            da.close()
            if tu.horizon_matches_points(da, df):
                logger.info(f'---> Horizon file {self.config.outputs.file.da_horizon} exists and loaded')
            else:
                logger.info(f'---> Horizon file {self.config.outputs.file.da_horizon} holds other points: recomputing')
                da = None
        if da is None:
            da = tp.compute_horizon_points(self.config.dem.filepath, df, self.config.dem.horizon_increments,
                                           fname=self.config.outputs.file.da_horizon,
                                           output_directory=self.config.outputs.path,
                                           n_threads=self.config.project.parallelization.setting.multicore.CPU_cores)
        self.da_horizon = da

        hp = tu.horizon_at_points(da, df)
        for az in hp.azimuth.values:
            df['hori_azi_' + str(az)] = hp.sel(azimuth=az).values
        # update df_centroid file with horizons angle
        df.to_pickle(self.config.outputs.path / self.config.outputs.file.df_centroids)
        logger.info(f'---> Centroids file {self.config.outputs.file.df_centroids} updated with horizons')

    def downscale_climate(self):
        """
        Downscale the climate forcing to the points, into the zarr store outputs/downscaled/<zarr_store>
        (topo_scale_zarr.ClimateDownscaler).
        """
        self.downscaled_pts = self._downscale_zarr(self.config.outputs.downscaled)
        # update plotting class variables
        self.plot.ds_down = self.downscaled_pts

        # remove tmp directories
        if self.config.clean_up.rm_tmp_dirs:
            self._remove_tmp_dirs()

        # optional physics-consistency checks (config: toposcale.physics_checks: true | 'warning' | 'error')
        checks = self.config.toposcale.physics_checks
        if checks:
            self.check_physics(strict=checks if checks in ['warning', 'error'] else False)

    def _remove_tmp_dirs(self):
        # an absolute climate path (shared archive) has no tmp directory of the project
        for tmp in [self.config.outputs.tmp_path, self.config.climate.tmp_path]:
            if tmp not in [None, {}]:
                shutil.rmtree(tmp, ignore_errors=True)

    def _downscaled_store_name(self):
        return self.config.outputs.file.zarr_store or 'down.zarr'

    def _climate_store(self):
        """
        Path of the climate zarr store. A store written by get_era5() must cover the project period without gaps.
        Otherwise the store is converted from the netcdf files of the climate folder when missing, or when it does
        not cover the project period.
        """
        name = self.config.climate[self.config.project.climate].zarr_store or 'ERA5.zarr'
        store = self.config.climate.path / name
        if store.exists():
            gaps = es.store_gaps(store, self.config.project.start, self.config.project.end)
            if gaps:
                raise ValueError(f'{name} lacks {", ".join(gaps)} of the project period. Run mp.get_era5() again: '
                                 f'it fetches only what is missing')
            if gaps is not None:
                return store
        nc_dir = fe.climate_netcdf_dir(self.config.climate.path)
        rebuild = not store.exists()
        if store.exists() and list(nc_dir.glob('PLEV_*.nc')):
            # rebuild a store that does not cover the project period, e.g. after downloading more years
            times = pd.DatetimeIndex(xr.open_zarr(str(store)).time.values)
            start = pd.to_datetime(self.config.project.start)
            end = pd.to_datetime(self.config.project.end) + pd.Timedelta('1D') - pd.Timedelta('1h')
            rebuild = times.min() > start or times.max() < end
            if rebuild:
                logger.info(f'---> {name} covers {times.min()} to {times.max()}, not the project period: rebuilding it')
        if rebuild:
            logger.info(f'---> Converting the netcdf files of {nc_dir} to {name}')
            fe.netcdf_to_zarr(path_to_netcdfs=nc_dir, zarrout=store)
        return store

    def _downscale_zarr(self, downscaled_dir):
        """Downscale all points from the climate zarr store into one zarr store of downscaled points."""
        variables = self.config.outputs.variables
        df = self.toposub.df_centroids
        surface = se.setup(df, self.config.toposcale.surface_effect, self.config.dem.filepath, self.config.dem.epsg)
        if surface is not None and surface.get('weight') is not None:
            df.to_pickle(self.config.outputs.path / self.config.outputs.file.df_centroids)
        cda = tz.ClimateDownscaler(
            era5_zarr_path=self._climate_store(),
            output_path=downscaled_dir,
            df_centroids=self.toposub.df_centroids,
            da_horizon=self.da_horizon,
            ds_solar=None,
            target_EPSG=self.config.dem.epsg,
            start_date=self.config.project.start,
            end_date=self.config.project.end,
            tstep=self.config.climate[self.config.project.climate].timestep,
            varout=list(variables) if isinstance(variables, (list, tuple)) and len(variables) else None,
            interp_method=self.config.toposcale.interpolation_method,
            lw_terrain_flag=self.config.toposcale.LW_terrain_contribution,
            precip_lapse_rate_flag=self.config.climate.precip_lapse_rate,
            store_name=self._downscaled_store_name(),
            precip_lapse_rate_max_factor=self.config.climate.precip_lapse_rate_max_factor or core.PRECIP_LAPSE_MAX_FACTOR,
            point_block=self.config.toposcale.point_block or 100,
            time_block=self.config.toposcale.time_block,
            surface_effect=surface,
            shading_substeps=self.config.toposcale.solar_substeps or None,
            accumulation_window=self._accumulation_window(),
            solar_substeps=self.config.toposcale.solar_substeps or 6,
            solar_method=self.config.dem.solar_position_method or 'nrel_numpy',
            solar_constant=self.config.toposcale.solar_constant or sg.S0_DEFAULT,
            direct_diffuse=self.config.toposcale.direct_diffuse or 'empirical')
        n_core = self.config.project.parallelization.setting.multicore.CPU_cores or 1
        return cda.downscale(n_core=n_core)

    def check_physics(self, strict=False, snow_partition_method=None, fname='physics_checks.csv', points=None):
        """
        Run physics-consistency checks on the downscaled outputs (see topo_checks.py) and save the report to csv.

        Args:
            strict (bool or str): False to only report; True or 'error' to raise on errors; 'warning' to raise on
                warnings too
            snow_partition_method (str or list): partition method(s) to probe, e.g. 'continuous'
            fname (str): report file name, written to the outputs directory
            points (list): optional subset of point_name to check

        Returns:
            dataframe: report with one row per check and variable
        """
        ds_down = getattr(self, 'downscaled_pts', None)
        if ds_down is None:
            ds_down = tu.open_downscaled_store(self.config.outputs.downscaled / self._downscaled_store_name())
        report = tc.run_all_checks(ds_down,
                                   ds_solar=self.ds_solar,
                                   df_centroids=self.toposub.df_centroids,
                                   ds_param=self.toposub.ds_param,
                                   snow_partition_method=snow_partition_method,
                                   points=points,
                                   epsg=getattr(self.config.dem, 'epsg', None))
        report.to_csv(self.config.outputs.path / fname, index=False)
        logger.info(tc.summarize(report, min_status='warning'))
        logger.info(f'---> Physics-consistency report saved to {fname}')
        if strict:
            tc.raise_on_failure(report, level='error' if strict is True else strict)
        return report

    def get_era5(self):
        # to maintain backward compatibility and simplicity of use.
        self.era5_obj.go_fetch(surf_plev='all')


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

    def unzip_file(self, filepath):
        fe.unzip_file(filepath)

    def remap_netcdf(self, filepath, file_type):
        fe.remap_CDSbeta(filepath, file_type)

    def get_ifs_forecast(self):
        # run openData script here
        logger.info("doing forecast stuff here")

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


    def _setup_logging(self):
        """
        Messages to the terminal (project.verbose: True by default, False for warnings and errors only, or a level
        name such as 'debug') and to a log file (project.log_file: True by default for
        outputs/logs/topopyscale_<date>.log, a path, or False). Python warnings go to both.
        """
        verbose = self.config.project.verbose
        verbose = True if verbose is None else verbose
        log_file = self.config.project.log_file
        if log_file is None or log_file is True:
            log_file = self.config.outputs.path / 'logs' / f'topopyscale_{datetime.now():%Y%m%d_%H%M%S}.log'
        elif log_file:
            log_file = Path(log_file)
            if not log_file.is_absolute():
                log_file = self.config.project.directory / log_file
        else:
            log_file = None
        log_file = tps_log.setup(verbose=verbose, logfile=log_file, capture_warnings=True)
        if log_file:
            logger.info(f'---> Log written to {log_file}')
        return log_file

    def _export_jobs(self):
        """Processes writing export files: project.parallelization CPU_cores, 1 if not set."""
        try:
            return self.config.project.parallelization.setting.multicore.CPU_cores or 1
        except AttributeError:
            return 1

    def to_cryogrid(self, fname_format='Cryogrid_pt_*.nc', precip_partition='continuous'):
        """
        wrapper function to export toposcale output to cryosgrid format from TopoClass
        
        Args:
            fname_format (str): filename format. point_name is inserted where * is

        """
        path = self.config.outputs.path
        if 'cluster_labels' in self.toposub.ds_param.keys():
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
                       project_author=self.config.project.authors,
                       n_jobs=self._export_jobs())

    def to_fsm(self, fname_format='FSM_pt_*.txt'):
        """
        function to export toposcale output to FSM format, one file per point, written by CPU_cores processes
        """
        te.to_fsm(self.downscaled_pts, self.config.outputs.path / fname_format, n_jobs=self._export_jobs())

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
                     project_author=self.config.project.authors,
                     n_jobs=self._export_jobs())

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
                                      headers=False,
                                      n_jobs=self._export_jobs())

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
        logger.info('---> File {} saved'.format(file_out))

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
                layout='grid',
                path=None,
                bbox=None,
                stride=1,
                cell_mask=None,
                start=None,
                end=None):
        """
        function to export TopoPyScale output as forcing for MuSA
        MuSA: https://github.com/ealonsogzl/MuSA

        Writes daily forcing files, mask.nc, and DEM.nc ('grid') or points_meta.csv ('points') to path. See
        topo_export.to_musa() for the layouts, files, units and the matching MuSA config.py settings.

        Args:
            layout (str): 'grid' (default): one MuSA cell per DEM cell, var(time, northing, easting) on the DEM grid,
                each cell with the forcing of its TopoSUB cluster. Needs the toposub sampling method.
                'points': one MuSA cell per point or cluster, var(time, y, x) with y = point id and x = [0].
            path (str): output folder. Default: outputs/musa_<layout>
            bbox (list): [xmin, xmax, ymin, ymax] in DEM coordinates, to crop the DEM for layout 'grid'
            stride (int): keep every stride-th DEM cell for layout 'grid'
            cell_mask (array): boolean (y, x) array of the DEM cells to simulate for layout 'grid'
            start (str): first timestep to export. Default: first downscaled timestep
            end (str): last timestep to export. Default: last downscaled timestep

        Returns:
            dict: grid shape, number of cells to simulate, and date_ini, date_end and dt for MuSA's config.py
        """
        if path is None:
            path = self.config.outputs.path / f'musa_{layout}'

        return te.to_musa(ds=self.downscaled_pts,
                          df_pts=self.toposub.df_centroids,
                          ds_param=self.toposub.ds_param,
                          path=path,
                          layout=layout,
                          bbox=bbox,
                          stride=stride,
                          cell_mask=cell_mask,
                          start=start,
                          end=end,
                          n_cores=self.config.project.parallelization.setting.multicore.CPU_cores,
                          climate_dataset_name=self.config.project.climate,
                          project_authors=self.config.project.authors
                          )


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
        logger.info('to be implemented')

    def map_clusters(self, **kwargs):
        tp.map_clusters(self.ds_down,
                        self.toposub.ds_param,
                        **kwargs)

    def timeseries(self):
        logger.info('To be implemented')

    def solar_geom(self):
        logger.info('To be implemented')

    def horizon(self):
        logger.info('To be implemented')
