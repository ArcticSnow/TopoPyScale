# Copilot Instructions for TopoPyScale

## Project Overview
TopoPyScale is a Python package for topography-based downscaling of climate data to hillslope scale. It processes climate reanalysis data (ERA5, etc.) and digital elevation models (DEM) to produce outputs for models like Cryogrid and FSM. The codebase is organized as a PyPI library, with main logic in the `TopoPyScale/` directory and documentation in `doc/`.

## Key Components
- **Core Library:** All main modules are in `TopoPyScale/`. Each file typically handles a specific aspect (e.g., `fetch_dem.py` for DEM fetching, `topo_scale.py` for downscaling logic).
- **Documentation:** Markdown docs are in `doc/docs/`. API docs are auto-generated using `lazydocs`.
- **Configuration:** Project uses `pyproject.toml` and `MANIFEST.in` for packaging. Documentation build is managed by `mkdocs.yml`.

## Developer Workflows
- **Testing:** GitHub Actions run tests via `test_topopyscale.yml`. Local test commands are not specified; add/maintain tests in the main package directory.
- **Documentation:**
  - Edit docs in `doc/docs/`.
  - Build locally: `mkdocs serve` (see `doc/README.md`).
  - API docs: Run `lazydocs` as described in `doc/README.md`.
- **Releases:** Create a new branch for features/bugfixes, merge to `main`, then follow release video guide (see `README.md`).

## Project-Specific Patterns
- **Data Flow:**
  - Inputs: Climate data (ERA5, CORDEX), DEM (local or fetched).
  - Processing: DEM-derived values, clustering (k-means), interpolation (bilinear, inverse square distance).
  - Outputs: Cryogrid/FSM formats.
- **Contribution:**
  - Issues/discussions on GitHub.
  - Branching for features/bugfixes is encouraged.
- **Docs Build:** Changes to docs auto-build on ReadTheDocs when committed.

## Integration Points
- **External Data:** Fetches DEM from public repositories (SRTM, ArcticDEM, ASTER).
- **Dependencies:** See `pyproject.toml` and `doc/requirements.txt` for required packages (e.g., `mkdocs`, `lazydocs`).

## Examples
- To fetch DEM: see `fetch_dem.py`.
- To run downscaling: see `topo_scale.py` and related modules.
- To build docs locally:
  ```bash
  pip install mkdocs sphinx_rtd_theme mkdocs-material pygments
  mkdocs serve
  # Open http://127.0.0.1:8000/
  ```
- To update API docs:
  ```bash
  lazydocs --output-path="doc/docs" --overview-file="README.md" --src-base-url="https://github.com/ArcticSnow/TopoPyScale" TopoPyScale
  ```

## References
- Main documentation: https://topopyscale.readthedocs.io
- Release workflow: https://www.youtube.com/watch?v=Ob9llA_QhQY

---
**Feedback:** Please review and suggest updates for any unclear or missing sections.
