"""
Methods to fetch DEM from various public repository

TODO:
- [ ] SRTM
- [ ] ArcticDEM
- [ ] ASTER dem
- [ ] Norwegian DEM
- [x] Copernicus DEM

"""

import sys, os, tarfile
from pathlib import Path
import requests
from xml.dom.minidom import parse, parseString
import pandas as pd
import matplotlib.pyplot as plt
from TopoPyScale import topo_utils as tu

import warnings, os
warnings.filterwarnings('ignore')

class copernicus_dem():
    """---> Class to download Publicly available Copernicus DEM products
    
    Args:
        directory (str): directory where to store downloads
    """

    def __init__(self, directory='.', product=None, n_download_threads=1):
        self.directory = Path(directory)
        self.product = product
        self.n_download_threads = n_download_threads
        self.sub = None
        self.df_downloaded = None
        self.dem_file_extension = {'COP-DEM_GLO-30-DGED/2023_1':'DEM.tif',
                                        'COP-DEM_GLO-90-DGED/2023_1':'DEM.tif',
                                        'COP-DEM_GLO-30-DTED/2023_1':'DEM.dt2',
                                        'COP-DEM_GLO-90-DTED/2023_1':'DEM.dt1'}
        self.copernicus_public_url = 'https://prism-dem-open.copernicus.eu/pd-desk-open-access/publicDemURLs'

        print("""
Class to download Publicly available Copernicus DEM products. 

Data are directly available via HTTPS requests at https://prism-dem-open.copernicus.eu/pd-desk-open-access/publicDemURLs

Further online Resources:
    - Sentinel Online blog post: https://sentinels.copernicus.eu/web/sentinel/-/copernicus-dem-new-direct-data-download-access
    - Airbus data description: https://spacedata.copernicus.eu/documents/20123/122407/GEO1988-CopernicusDEM-SPE-002_ProductHandbook_I5.0+%281%29.pdf
    - tutorial: https://spacedata.copernicus.eu/documents/20123/121286/Copernicus+DEM+Open+HTTPS+Access.pdf
""")
        

    def fetch_copernicus_dem_product_list(self):
        '''
        Function to fetch the list of Copernicus DEM products available to download.

        '''
        self.r_products = requests.get(self.copernicus_public_url)
        products_xml = parseString(self.r_products.content)
        products = products_xml.getElementsByTagName('datasetId')

        self.product_list = []
        print('---> List of product available:')
        for i, d in enumerate(products):
            self.product_list.append(d.firstChild.data)
            print(f'\t [{i}]-> {d.firstChild.data}')
        if self.product is None:
            self._product_ind = int(input('Which product to download?'))
            
    def request_tile_list(self, fname_tile=None):
        """Function to download list of tiles of a given Copernicus product

        Args:
            fname_tile (str, optional): _description_. Defaults to 'tiles.csv'.
        """
        if self.product is None:
            self.product = self.product_list[self._product_ind]
        if fname_tile is None:
            fname_tile = 'tar_list_' + self.product.replace('/', '__') + '.csv'
            
        # Add code to check first if fname_file does not exist already
        if not os.path.isfile(self.directory / fname_tile):

            print(f'---> Requesting list of tiles for {self.product}')

            headers = {'accept': 'csv',}
            url = f"{self.copernicus_public_url}/{self.product.replace('/', '__')}"
            r = requests.get(url, headers=headers)
            
            with open(fname_tile, 'w') as file:
                file.write(r.content.decode("utf-8"))

            print(f'---> All tile URL written to {fname_tile}')

        # open csv file containing list of tar file urls
        df = pd.read_csv(self.directory / fname_tile, header=None)
        df.columns=['url']

        # extract latitude and longitude of tiles from filename
        df['lat_str'] = df.url.apply(lambda x: x[-18:-12])
        df['lon_str'] = df.url.apply(lambda x: x[-11:-4])

        # Convert latitude and longitude strings to float
        df['hemisphere'] = df.lat_str.apply(lambda x: x[:1])
        df['lat'] = df.lat_str.apply(lambda x: float(x.split('_')[0][1:]) + float(x.split('_')[1]))
        df.lat.loc[df.hemisphere=='S'] = -df.lat.loc[df.hemisphere=='S']

        df['east'] = df.lon_str.apply(lambda x: x[:1])
        df['lon'] = df.lon_str.apply(lambda x: float(x.split('_')[0][1:]) + float(x.split('_')[1]))
        df.lon.loc[df.east=='W'] = -df.lon.loc[df.east=='W']

        self.df_tile_list = df

    def _check_file_downloaded(self):
        '''
        Function to check which files have been downloaded. It will update self.sub dataframe

        '''
        # Loop to check if file exist and has been downloaded.
        downloaded_list = []
        tar_list = []
        for i, row in self.sub.iterrows():
            tar_file = self.directory / row.url.split('/')[-1]
            tar_list.append(tar_file)

            if (not os.path.isfile(tar_file)) | (os.path.getsize(tar_file) < 100):
                downloaded_list.append(False)
            else:
                print(f'-> File {tar_file} already exists')
                downloaded_list.append(True)

        self.sub['downloaded'] = downloaded_list
        self.sub['tar_file'] = tar_list

    def download_tiles_in_extent(self, extent=[22,28,44,45]):
        """Function to download tiles falling into the extent requested

        Args:
            extent (list, int): [east, west, south, north] longitudes and latitudes of the extent of interest. Defaults to [22,28,44,45].

        Returns:
            dataframe: dataframes of the tile properties (url, lat, lon, etc.)
        """
        import urllib.request as ur
        
        self.sub = self.df_tile_list.loc[(self.df_tile_list.lon>=extent[0]) & (self.df_tile_list.lon<=extent[1]) & (self.df_tile_list.lat>=extent[2]) & (self.df_tile_list.lat<=extent[3])]

        def _download_single_tile(url, tar_file):
            print(f'---> Downloading {url}')
            ur.urlretrieve(url, tar_file)

        _check_file_downloaded()
        while self.sub.downloaded.sum()>0:
            tar_download = sub.tar_file.loc[~sub.downloaded].values
            url_download = sub.url.loc[~sub.downloaded].values

            # Parallelize download of tiles
            fun_param = zip(url_download, tar_download)
            tu.multithread_pooling(_download_single_tile, fun_param, n_threads=self.n_download_threads)
            _check_file_downloaded()

        print("---> All tiles downloaded")
        sub['downloaded'] = True
        self.df_downloaded = sub
        

    def extract_all_tar(self, dem_extension='DEM.dt2'):
        '''
        Function to extract DEM file from Copernicus tar file
        Args:
            dem_extension: extension of the DEM raster file. for instance: .dt2, .tif, .dt1

        '''
        
        for i, row in self.df_downloaded.iterrows():
            tar_file = Path(row.url).name
            tar = tarfile.open(tar_file, "r")
            file = None
            #print(f'---> Content of tar archive: \n{tar.getmembers()}\n')
            for tarinfo in tar.getmembers():
                if tarinfo.name[-7:] == dem_extension:
                    file = tarinfo.name
                    file_ext = tar.extractfile(file)
                    with open((self.directory / file).name, 'wb') as outfile:
                        outfile.write(file_ext.read())

                file = None
            
        print('\n---> READ DATA LICENSING IN TAR FILE <---')
        print('--->        eula_F.pdf               <---\n')
        
        
def fetch_copernicus_dem(directory='.', 
                        extent=[23,24,44,45], 
                        product='COP-DEM_GLO-90-DTED/2023_1', 
                        file_extension=None,
                        n_download_threads=1):
    """Routine to download Copernicus DEM product for a given extent

    Args:
        directory (str, optional): _description_. Defaults to '.'.
        extent (list of int, optional): extent in [east, west, south, north] in lat/lon degrees. Defaults to [23,24,44,45].
        product (str, optional): name of product to download. Default is 'COP-DEM_GLO-90-DTED/2023_1'
        file_extension (str, optional): last 7 characters of the file to extract from the tar archive. for DEM, 'DEM.tif' for the DGED products
        n_download_threads (int): number of download threads
    """
    dn = copernicus_dem(directory=directory, product=product, n_download_threads=n_download_threads)
    dn.request_tile_list()
    dn.download_tiles_in_extent(extent=extent)
    if file_extension is None:
        if dn.product in dn.dem_file_extension.keys():
            file_extension = dn.dem_file_extension.get(dn.product)
        else:
            raise ValueError('File extension of product not known. Add new extension to copernicus_dem.dem_file_extension dictionnary')
    dn.extract_all_tar(dem_extension=file_extension)

    return dn
    
    

def fetch_dem(dem_dir, extent, dem_epsg, dem_file, dem_resol=None):
    """
    Function to fetch DEM data from SRTM and potentially other sources

    Args:
        dem_dir (str): path to dem folder
        extent (dict): list of spatial extent in lat-lon [latN, latS, lonW, lonE]
        epsg (int): epsg projection code
        dem_file (str): filename of the downloaded DEM. must be myfile.tif
    """

    ymax = extent.get('latN')
    ymin = extent.get('latS')
    xmin = extent.get('lonW')
    xmax = extent.get('lonE')

    ans = input("\n---> Do you want to downlaod DEM from a repository?\n\t(1) SRTM1,\n\t(2) SRTM3,\n\t(3) ArcticDEM,\n\t(4) ASTER,\n\t(5) Exit\n")

    if ans == '1':



        # use STRM DEM for extent of interest, buffer arg "margin" enbles us to crop projected DEM back to a rectangle defined by extentNSWE (projected)
        cmd_1 = 'eio --product SRTM1 clip -o {} --bounds {} {} {} {} --margin {}'.format(dem_dir + 'dem_SRTM1.tif',
                                                                                         xmin,
                                                                                         ymin,
                                                                                         xmax,
                                                                                         ymax,
                                                                                      0.2)
        print('>===== command to download DEM  from SRTM1 ====<\n')
        # os.system('eio clean')
        print(cmd_1)
        try:
            os.system(cmd_1)
        except RuntimeError:
            return RuntimeError
        # os.system('eio clean')



        # target_epsg = input("---> provide target EPSG (default: 32632):") or '32645'
        # crop to extent defined by "-te <xmin ymin xmax ymax>" flag to ensure rectangulatr output with no NAs. -te_srs  states the epsg of crop parameeters (WGS84)
        if dem_resol is not None:
            res = dem_resol
        else:
            res = 30
        cmd_2 = 'gdalwarp -tr '+str(res) + ' ' + str(res) +  ' -r bilinear -s_srs epsg:4326 -t_srs epsg:{} -te_srs epsg:{} -te {} {} {} {} {} {}'.format(dem_epsg,
                                                                                                                4326,
                                                                                                                 xmin,
                                                                                                                 ymin,
                                                                                                                 xmax,
                                                                                                                 ymax,
                                                                                               dem_dir + 'dem_SRTM1.tif',
                                                                                               dem_dir  + dem_file,
                                                                                                               )
        # as cmd-2 but without crop
        # cmd_3 = 'gdalwarp -tr 30 30 -r bilinear -s_srs epsg:4326 -t_srs epsg:{} {} {}'.format(dem_epsg,
        #                                                                                        dem_dir + 'inputs/dem/dem_SRTM1.tif',
        #                                                                                        dem_dir + 'inputs/dem/dem_SRTM1_proj.tif'
        #                                                                                                        )
        print(cmd_2)
        os.system(cmd_2)
        # print(cmd_3)
        # os.system(cmd_3)
        # sys.exit()

    elif ans == '2':
        # use STRM DEM for extent of interest

        cmd_1 = 'eio --product SRTM3 clip -o {} --bounds {} {} {} {} --margin {}'.format(dem_dir + 'dem_SRTM3.tif' ,
                                                                                         xmin,
                                                                                         ymin,
                                                                                         xmax,
                                                                                         ymax,
                                                                                         0.2)
        print('>===== command to download DEM  from SRTM3 ====<')
        print('eio clean')
        os.system(cmd_1)
        print('eio clean')
        #target_epsg = input("---> provide target EPSG (default: 32632):") or '32632'

        if dem_resol is not None:
            res = dem_resol
        else:
            res = 90
        cmd_2 = 'gdalwarp -tr ' +str(res) + ' ' + str(res) +   ' -r bilinear -s_srs epsg:4326 -t_srs epsg:{} -te_srs epsg:{} -te {} {} {} {} {} {}'.format(dem_epsg,
                                                                                                                4326,
                                                                                                                 xmin,
                                                                                                                 ymin,
                                                                                                                 xmax,
                                                                                                                 ymax,
                                                                                               dem_dir + 'dem_SRTM3.tif',
                                                                                               dem_dir + dem_file,
                                                                                                               )

        os.system(cmd_2)
        # print('\n>========== Another option =======<')
        # print('Download manually tiles from: https://dwtkns.com/srtm30m/')
        # sys.exit('---> EXIT: run those commands and update dem_file in config.ini')

    elif ans == '3':
        sys.exit('WARNING: fetch ArcticDEM functionality not available')
    elif ans == '4':
        print('WARNING: fetch ASTER DEM functionality not available')
        print('Please visit https://lpdaacsvc.cr.usgs.gov/appeears/task/area to manually download a DEM')
        sys.exit()
    else:
        sys.exit("ERROR: dem file '{}' not existing".format(dem_file))
