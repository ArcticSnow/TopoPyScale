'''
Methods to fetch DEM from various public repository

TODO:
- [ ] SRTM
- [ ] ArcticDEM
- [ ] ASTER dem
- [ ] Norwegian DEM

'''

import sys

def fetch_dem(project_dir, extent, dem_file):

    ans = input("\n---> Do you want to downlaod DEM from a repository?\n\t(1) SRTM1,\n\t(2) SRTM3,\n\t(3) ArcticDEM,\n\t(4) ASTER,\n\t(5) Exit\n")

    if ans == '1':
        # use STRM DEM for extent of interest
        cmd_1 = 'eio --product SRTM1 clip -o {} --bounds {} {} {} {} --margin {}'.format(project_dir + 'inputs/dem/dem_SRTM1.tif',
                                                                                      extent.get('lonW'),
                                                                                      extent.get('latS'),
                                                                                      extent.get('lonE'),
                                                                                      extent.get('latN'),
                                                                                      0.25)
        print('>===== command to download DEM  from SRTM1 ====<\n')
        print('eio clean')
        print(cmd_1)
        print('eio clean')
        target_epsg = input("---> provide target EPSG (default: 32632):") or '32632'
        cmd_2 = 'gdal_warp -tr 30 30 -r bilinear -s_srs epsg:4326 -t_srs epsg:{} {} {}'.format(target_epsg,
                                                                                               project_dir + 'inputs/dem/dem_SRTM1.tif',
                                                                                               project_dir + 'inputs/dem/dem_SRTM1_proj.tif',)
        print(cmd_2)
        sys.exit()

    elif ans == '2':
        # use STRM DEM for extent of interest
        dem_file = 'dem_SRTM3.tif'
        cmd_1 = 'eio --product SRTM3 clip -o {} --bounds {} {} {} {} --margin {}'.format(project_dir + 'inputs/dem/' + dem_file,
                                                                                         extent.get('lonW'),
                                                                                         extent.get('latS'),
                                                                                         extent.get('lonE'),
                                                                                         extent.get('latN'),
                                                                                         0.25)
        print('>===== command to download DEM  from SRTM3 ====<')
        print('eio clean')
        print(cmd_1)
        print('eio clean')
        target_epsg = input("---> provide target EPSG (default: 32632):") or '32632'
        cmd_2 = 'gdal_warp -tr 30 30 -r bilinear -s_srs epsg:4326 -t_srs epsg:{} {} {}'.format(target_epsg,
                                                                                               project_dir + 'inputs/dem/dem_SRTM3.tif',
                                                                                               project_dir + 'inputs/dem/dem_SRTM3_proj.tif',)
        print(cmd_2)
        sys.exit('---> EXIT: run those commands and update dem_file in config.ini')

    elif ans == '3':
        sys.exit('WARNING: fetch ArcticDEM functionality not available')
    elif ans == '4':
        print('WARNING: fetch ASTER DEM functionality not available')
        sys.exit()
    else:
        sys.exit("ERROR: dem file '{}' not existing".format(dem_file))
