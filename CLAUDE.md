# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying **border/boundary effects in spatial graphs** — how truncating a spatial domain distorts graph-theoretic centrality measures (degree, betweenness, closeness, PageRank, clustering). The project develops and benchmarks two correction strategies:

1. **BOSPERRUS**: Fits a parametric curve (piecewise-linear or exponential saturation) of centrality vs. distance-to-border, then uses the fitted model to correct observed values.
2. **SERN** (Spatial Edge-based Random Network): Builds a surrogate null ensemble by sampling random graphs that preserve the empirical edge-length distribution, then subtracts the surrogate median from observed centralities.

## Codebase Structure

```
truncated_graphs/
├── src/                         # Core analysis scripts
│   ├── border_effects_kNN_del.py  # Graph construction utilities + main trunc_graphs() API
│   ├── sern.py                    # SERN surrogate ensemble (parallel via joblib)
│   ├── sphere.py                  # Sphere benchmark (entry point: python sphere.py)
│   ├── figure_2.py                # Figure 2 pipeline (entry point: python figure_2.py)
│   └── archive/                   # Older iterations (not in active use)
├── notebooks/                   # Jupyter notebooks producing paper figures
│   ├── figure1.ipynb, figure2abcd.ipynb, figure2ef.ipynb, figure3.ipynb, figure3_a.ipynb
│   └── data/, cache/, geo_data/ # Notebook-specific cached data
├── results/                     # Output CSVs, organized by figure (figure1/, figure2/, figure3/, figure4/)
├── mibitof_coords/              # Pickled cell coordinate dicts from MIBI-TOF imaging
├── SNAP_data/                   # SNAP social network datasets
├── CD34_data/                   # CD34 tissue imaging data
└── TWOMBLI_data/                # TWOMBLI fiber network data
```

The core dependency is the **bosperrus package** located at `/data/bionets/je30bery/bosperrus-package/` — a local dev install. Scripts reach it via `sys.path.append`. It is not on PyPI.

## bosperrus Package Architecture

`bosperrus/pipeline.py` — `Flow` class: the main pipeline object. Four construction paths:
- `Flow.from_coords(coordinates, distance_fn, measures, graph_type, ...)` — full pipeline
- `Flow.from_coords_and_edgelist(...)` — skip graph construction
- `Flow.from_coords_and_scores(...)` — skip both
- `Flow.from_distances_and_scores(...)` — no coords needed

After construction, call `flow.flow()` to run all fits, then read `flow.fit_quality` (DataFrame) and `flow.observations` (DataFrame with `BOSPERRUS corrected {measure}` columns).

`bosperrus/fit.py` — Fit models (all subclass `Fit`): `ConstantFit`, `PiecewiseLinearFit`, `ExponentialSaturationFit`, `MichaelisMentenFit`. Call `fit_correct()` to fit and apply correction in one step.

`bosperrus/distances.py` — Border distance functions: `distance_to_convex_hull`, `distance_to_rectangular_border`, `distance_to_mask`, `distance_to_pointset`.

`bosperrus/graph_construction.py` — `knn_edges(coords, k)` (directed, asymmetric), `rnn_edges(coords, r)` (undirected), `delaunay_edges(coords)` (undirected), `construct_graph(coords, graph_type, ...)`.

`bosperrus/centrality_measures.py` — `compute_centrality_measures(edge_list, N, measures=...)`.

## Running Scripts

Scripts are run from the `src/` directory (imports use relative `sys.path` additions):

```bash
cd /data/bionets/je30bery/truncated_graphs/src

# Sphere benchmark (100 runs × 3 coord configs × 3 graph types; writes to ../results/figure4/)
python sphere.py

# Figure 2 pipeline (reads mibitof_coords/coords.pickle; writes to ../results/figure2/)
python figure_2.py
```

Notebooks are run from the `notebooks/` directory. Results are cached to CSV in `results/` and re-read on subsequent runs to avoid recomputation.

## Key Parameters (sphere.py)

- `NUMBER_OF_SERNS = 100` — SERN ensemble size
- `N_JOBS = 128` — joblib workers for parallel surrogate generation
- `N_OF_RUNS = 100` — independent outer repetitions
- `OMP_NUM_THREADS = 8` — set via `os.environ`

## Graph Types

| Type | Directedness | Key param | Notes |
|------|-------------|-----------|-------|
| `delaunay` | undirected | — | Scipy `Delaunay`; on sphere: convex hull |
| `knn` | directed, asymmetric | `k` | Not symmetric: (u→v) ≠ (v→u) |
| `rnn` | undirected | `r` | Radius-neighbor; edges as `frozenset` |

Note: kNN edges are `(u, v)` tuples; Delaunay and rNN edges are `frozenset({u, v})`. The `reindex_edges_to_crop()` function in `border_effects_kNN_del.py` converts global node indices to local crop indices.

## Data Sources

- **MIBI-TOF**: Segmentation TIFFs → centroid extraction in `get_mibitof()`. Coordinate dicts pickled to `mibitof_coords/coords.pickle`.
- **Squidpy Visium**: 35 spatial transcriptomics datasets fetched via `sq.datasets.visium()` in `get_squidpy_visium_datasets()`.
- **SNAP**: Geolocated check-in network (Brightkite); used for urban/social network comparison.

## Dependencies

Core: `numpy`, `scipy`, `pandas`, `scikit-learn`, `networkx`, `squidpy`, `tifffile`, `joblib`, `tqdm`, `matplotlib`.

Install bosperrus in editable mode if not already installed:
```bash
pip install -e /data/bionets/je30bery/bosperrus-package/
```
