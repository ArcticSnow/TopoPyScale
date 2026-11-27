"""
Prepare the terrain of the clustering sketches: three mountain watersheds around Piz Kesch (Albula Alps, Switzerland),
written to data/kesch.nc and data/kesch_streams.json. make_cluster_sketches.py reads these files.

The DEM is the SRTM 1 arc-second tile of the Davos example of TopoPyScale_examples
(ex3_switzerland_davos/inputs/dem/dem_SRTM1.tif; SRTM: NASA/USGS, public domain):

    python prepare_kesch.py path/to/dem_SRTM1.tif

Steps:
    - reproject a 42 km x 57 km area to UTM 32N at 100 m (mean of the SRTM cells);
    - flow directions and upstream areas (pyflwdir, after filling the depressions);
    - the three watersheds upstream of three outlets, the groups raster of the sketches (1, 2, 3; 0 outside);
    - slope, aspect and sky view factor with the functions of TopoPyScale (topo_param), as compute_dem_param does;
    - crop to the watersheds, with a margin of 8 cells;
    - stream lines (upstream area > 0.4 km2) in the crop.

Needs rasterio, pyflwdir, xarray and netCDF4, and TopoPyScale itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyflwdir
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from TopoPyScale.topo_param import gradient_d8, sky_view_factor

DATA = Path(__file__).parent / 'data'
RES = 100.
X0, Y1, NX, NY = 548000., 5212000., 420, 570                   # UTM 32N (EPSG:32632), upper left corner
OUTLETS = [(563050., 5165950.), (571950., 5159450.), (570650., 5158650.)]   # watersheds 1, 2, 3
STREAM_KM2 = 0.4
MARGIN = 8


def main(src):
    tr = from_origin(X0, Y1, RES, RES)
    dem = np.full((NY, NX), np.nan, 'float32')
    with rasterio.open(src) as r:
        reproject(rasterio.band(r, 1), dem, dst_transform=tr, dst_crs='EPSG:32632', resampling=Resampling.average,
                  src_nodata=r.nodata, dst_nodata=np.nan)
    assert not np.isnan(dem).any()

    flw = pyflwdir.from_dem(dem, nodata=np.nan, transform=tr, latlon=False)
    upa = flw.upstream_area('km2')
    rows, cols = rasterio.transform.rowcol(tr, *zip(*OUTLETS))
    idxs = np.ravel_multi_index((rows, cols), dem.shape)
    basin = flw.basins(idxs=idxs, ids=np.arange(1, len(OUTLETS) + 1)).astype('uint8')
    for k, i in enumerate(idxs, 1):
        print(f'watershed {k}: {(basin == k).sum() * RES ** 2 / 1e6:.2f} km2, upstream area at the outlet '
              f'{upa.flat[i]:.2f} km2, {dem[basin == k].min():.0f}-{dem[basin == k].max():.0f} m')

    slope, aspect = gradient_d8(dem, RES, -RES)                  # as compute_dem_param: dy < 0 for north up
    svf = sky_view_factor(dem.astype('float64'), RES, -RES)

    rr, cc = np.where(basin > 0)
    r0, r1 = rr.min() - MARGIN, rr.max() + MARGIN + 1
    c0, c1 = cc.min() - MARGIN, cc.max() + MARGIN + 1
    win = (slice(r0, r1), slice(c0, c1))
    x = X0 + (np.arange(c0, c1) + 0.5) * RES
    y = Y1 - (np.arange(r0, r1) + 0.5) * RES
    ds = xr.Dataset(
        {'elevation': (('y', 'x'), dem[win], {'units': 'm'}),
         'slope': (('y', 'x'), slope[win].astype('float32'), {'units': 'rad'}),
         'aspect': (('y', 'x'), np.deg2rad(aspect[win]).astype('float32'),
                    {'units': 'rad', 'convention': '0 = south, pi/2 = east'}),
         'svf': (('y', 'x'), svf[win].astype('float32'), {'units': '1'}),
         'watershed': (('y', 'x'), basin[win], {'long_name': 'groups raster: watershed 1, 2, 3; 0 outside'})},
        coords={'x': x, 'y': y},
        attrs={'crs': 'EPSG:32632', 'source': 'SRTM 1 arc-second (NASA/USGS, public domain), mean over 100 m cells',
               'script': 'doc/sketches/prepare_kesch.py'})
    DATA.mkdir(exist_ok=True)
    enc = {v: {'zlib': True, 'complevel': 9} for v in ds.data_vars}
    for v in ['elevation', 'slope', 'aspect', 'svf']:
        enc[v]['least_significant_digit'] = {'elevation': 1, 'slope': 3, 'aspect': 3, 'svf': 3}[v]
    ds.to_netcdf(DATA / 'kesch.nc', encoding=enc)

    # stream lines in cell coordinates of the crop (column, row of the cell centres)
    mask = np.zeros(dem.shape, bool)
    mask[win] = upa[win] > STREAM_KM2
    feats = flw.streams(mask=mask, xs=np.broadcast_to(np.arange(NX) + 0.5 - c0, dem.shape),
                        ys=np.broadcast_to(np.arange(NY)[:, None] + 0.5 - r0, dem.shape))
    inside = np.ones(basin[win].shape, bool)
    lines = []
    for f in feats:                                              # keep the parts inside the crop
        run = []
        for px, py in f['geometry']['coordinates']:
            i, j = int(np.floor(py)), int(np.floor(px))
            if 0 <= i < inside.shape[0] and 0 <= j < inside.shape[1] and inside[i, j]:
                run.append([round(px, 2), round(py, 2)])
            else:
                if len(run) > 1:
                    lines.append(run)
                run = []
        if len(run) > 1:
            lines.append(run)
    (DATA / 'kesch_streams.json').write_text(json.dumps(lines, separators=(',', ':')))
    print(f'{ds.sizes["y"]} x {ds.sizes["x"]} cells, {len(lines)} stream segments')


if __name__ == '__main__':
    main(sys.argv[1])
