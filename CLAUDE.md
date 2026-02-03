# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TopoPyScale is a Python package for topography-based downscaling of climate data to hillslope scale. It processes ERA5 reanalysis data and digital elevation models (DEMs) to produce meteorological forcing for land surface models like Cryogrid, FSM, CROCUS, Snowmodel, and others.

## Development Commands

```bash
# Install in development mode
pip install -e .

# Install dependencies via conda
conda env create -f environment.yml

# Lint (syntax errors and undefined names only)
flake8 . --select=E9,F63,F7,F82 --show-source

# Run example test (requires cloning TopoPyScale_examples)
git clone https://github.com/ArcticSnow/TopoPyScale_examples.git
cd TopoPyScale_examples/ex1_norway_finse/
python pipeline_test.py

# Build documentation locally
pip install mkdocs mkdocs-material pygments lazydocs
mkdocs serve  # serves at http://127.0.0.1:8000/

# Update API docs
lazydocs --output-path="doc/docs" --overview-file="README.md" --src-base-url="https://github.com/ArcticSnow/TopoPyScale" TopoPyScale
```

## Architecture

### Core Pipeline Flow

The main entry point is `Topoclass` in `topoclass.py`. A typical workflow:

1. **Configuration**: Load YAML config file → `Topoclass(config_file)`
2. **Data Fetching**: `get_era5()` downloads ERA5 surface and pressure level data
3. **DEM Processing**: `compute_dem_param()` calculates slope, aspect, sky view factor
4. **Spatial Sampling**: `extract_topo_param()` either:
   - Clusters DEM using k-means (`topo_sub.py`) to reduce computation
   - Uses user-provided point coordinates from CSV
5. **Solar Geometry**: `compute_solar_geometry()` and `compute_horizon()` calculate sun position and terrain shading
6. **Downscaling**: `downscale_climate()` interpolates climate variables to points
7. **Export**: `to_cryogrid()`, `to_fsm()`, `to_crocus()`, etc.

### Key Modules

| Module | Purpose |
|--------|---------|
| `topoclass.py` | Main orchestration class wrapping all functionality |
| `fetch_era5.py` | ERA5 data download via CDS API |
| `topo_param.py` | DEM parameter computation (slope, aspect, SVF, horizon) |
| `topo_sub.py` | K-means clustering for DEM segmentation (TopoSUB method) |
| `topo_scale.py` | Core downscaling algorithms (vertical/horizontal interpolation) |
| `topo_scale_zarr.py` | Zarr-based parallel downscaling with Dask support |
| `solar_geom.py` | Solar position calculations using pvlib |
| `topo_export.py` | Export to model-specific formats |
| `meteo_util.py` | Meteorological calculations (vapor pressure, LW radiation, etc.) |

### Data Flow

```
ERA5 (SURF*.nc, PLEV*.nc) + DEM (GeoTIFF)
         ↓
    DEM parameters (slope, aspect, svf)
         ↓
    Clustering OR point extraction → df_centroids
         ↓
    Solar geometry + horizon angles
         ↓
    Downscaling (interpolation + corrections)
         ↓
    Model-specific output formats
```

### Configuration

Projects use YAML config files (see `TopoPyScale_examples` repo). Key sections:
- `project`: extent, dates, climate source
- `dem`: file path, EPSG code
- `sampling`: method (toposub/points), clustering parameters
- `climate.era5`: timestep, pressure levels
- `outputs`: file patterns, variables

### Parallelization

Two methods supported in `topo_scale_zarr.py`:
- `multicore`: Python multiprocessing
- `dask`: Distributed computing with configurable workers

## External Dependencies

- **CDS API**: Requires `~/.cdsapirc` with Copernicus credentials for ERA5 download
- **topocalc**: Forked version for Python 3.9+ compatibility
- **CDO**: Climate Data Operators for netCDF manipulation
