# Sketches of the documentation

The figures of the science pages (`doc/docs/img/science/*.svg`) are drawn by hand in the clean SVG files of `src/`,
then turned into hand-drawn sketches by `roughen.js`. Edit the files of `src/`, never the generated ones, then run:

```sh
cd doc/sketches
npm install roughjs@4 jsdom@26
node roughen.js
```

- Lines, rectangles, circles, ellipses and paths are redrawn with [rough.js](https://roughjs.com) (MIT licence),
  with a fixed seed per element, so the output does not change between runs.
  Their coordinates are rounded to 0.1 px. Images (`<image>`) are kept as they are.
- Arrowheads (`marker-end`, `marker-start` on `<line>`) become two hand-drawn strokes in the colour of the marker.
- Text is set in Patrick Hand (Patrick Wagesreiter, SIL Open Font License 1.1, see `fonts/OFL.txt`), embedded in
  each SVG as a subset (`fonts/patrick-hand-subset.woff2`: printable ASCII and `°·½×…−±²³`). Greek letters, arrows
  and mathematical symbols, which the font lacks, are set in a math font.
- Text is enlarged by `SCALE` (1.35): the handwriting font is narrower than the sans-serif of the sources. Check that
  labels still fit after an edit.

The clustering figures (`clustering.svg`, `groups.svg`) are computed on real terrain: three catchments around Piz
Kesch (Switzerland), from the SRTM 1 arc-second DEM of the Davos example (SRTM: NASA/USGS, public domain).

- `prepare_kesch.py` reprojects the DEM to 100 m, delineates the catchments from the flow directions (pyflwdir),
  computes slope, aspect and sky view factor with `TopoPyScale.topo_param`, and writes `data/kesch.nc` and
  `data/kesch_streams.json`. Run it only to change the terrain:

    ```sh
    PYTHONPATH=../.. python prepare_kesch.py path/to/TopoPyScale_examples/ex3_switzerland_davos/inputs/dem/dem_SRTM1.tif
    ```

- `make_cluster_sketches.py` clusters the cells as TopoPyScale does, over the whole DEM, inside the mask, and per
  catchment, and writes the sources in `src/`. The maps of clusters are embedded as PNG images. It needs numpy, scipy,
  xarray, pillow, scikit-learn, scikit-image, rasterio and shapely:

    ```sh
    python make_cluster_sketches.py && node roughen.js
    ```

To check a figure, open it in a browser: the embedded font is a web font, which some command-line renderers ignore.
