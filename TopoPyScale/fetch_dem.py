"""
Methods to fetch DEM from various public repository

TODO:
- [ ] SRTM
- [ ] ArcticDEM
- [ ] ASTER dem
- [ ] Norwegian DEM

"""

import sys
import os

def fetch_dem(dem_dir, extent, dem_epsg, dem_file):
    """
    Function to fetch DEM data from SRTM and potentially other sources

    Args:
        dem_dir (str): path to dem folder
        extent (list): list of spatial extent in lat-lon [latN, latS, lonW, lonE]
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
        cmd_2 = 'gdalwarp -tr 30 30 -r bilinear -s_srs epsg:4326 -t_srs epsg:{} -te_srs epsg:{} -te {} {} {} {} {} {}'.format(dem_epsg,
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

        cmd_2 = 'gdalwarp -tr 90 90 -r bilinear -s_srs epsg:4326 -t_srs epsg:{} -te_srs epsg:{} -te {} {} {} {} {} {}'.format(dem_epsg,
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
