# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying **border/boundary effects in spatial graphs** — how truncating a spatial domain distorts graph-theoretic centrality measures (degree, betweenness, closeness, PageRank, clustering). The project develops and benchmarks two correction strategies:

1. **BOSPERRUS**: Fits a parametric curve (piecewise-linear or exponential saturation) of centrality vs. distance-to-border, then uses the fitted model to correct observed values.
2. **SERN** (Spatial Edge-based Random Network): Builds a surrogate null ensemble by sampling random graphs that preserve the empirical edge-length distribution, then subtracts the surrogate median from observed centralities.

## Codebase Structure

Each of `results/`, `result_plots/`, and `notebooks/` is organized **one
subfolder per manuscript figure** (`figure1/` … `figure5/`), plus an
`exploratory/` subfolder for work not yet tied to a numbered figure (see
`../STORY.md`'s "File organization convention" for the full rationale —
this replaced an earlier, inconsistent ad hoc numbering where folder/file
names didn't match the manuscript's actual figure order).

`src/` follows the same `figureN/`/`exploratory/` convention. Two exceptions,
both deliberate: `archive/` stays separate from `exploratory/` (pre-existing
"confirmed superseded, not in active use" category, distinct from
"not yet graduated"), and `__init__.py` stays at `src/` root (package
marker — though it's already broken independent of this reorg: it imports
`src.truncated_graphs` and `src.fit`, neither of which exists). Per-figure
shared code was resolved by checking actual cross-file imports rather than
assumed: `border_effects_kNN_del.py` and `sern.py` each turned out to be
used by exactly one script, so they moved in alongside their sole consumer
rather than needing a separate shared location.

```
truncated_graphs/
├── src/
│   ├── figure2/
│   │   ├── edge_effects.py        # Figure 2 (edge truncation effects, Delaunay/kNN);
│   │   │                            writes results/figure2/, result_plots/figure2/
│   │   └── border_effects_kNN_del.py  # Graph construction utilities + trunc_graphs()
│   │                                    API; used only by edge_effects.py
│   ├── figure3/
│   │   ├── compute_fits.py        # Figure 3 data (AIC fits per MIBI-TOF dataset/graph
│   │   │                            type); writes results/figure3/
│   │   └── plot_metrics.py        # Figure 3 plot (panels e-h); reads results/figure3/
│   │                                + ../../fit_palette.json, writes result_plots/figure3/
│   ├── figure4/
│   │   └── micron_comparison.py   # Figure 4 (MICrONS buffer identification); writes
│   │                                 result_plots/figure4/ and ../../../bosperrus/figures/figure4/
│   ├── figure5/
│   │   ├── sphere.py              # Figure 5 (sphere benchmark); writes results/figure5/
│   │   └── sern.py                # SERN surrogate ensemble (parallel via joblib);
│   │                                used only by sphere.py
│   ├── exploratory/                # sf_distance_over_time.py (Brightkite temporal,
│   │                                  misfiled under fig2/ before this reorg — not
│   │                                  figure-2-related), es_border.py, iow_coast.py,
│   │                                  uk_rail_analysis.py (geo/social scratch scripts),
│   │                                  make_pipeline_fig.py (older pipeline-figure
│   │                                  script; writes to a path,
│   │                                  "bosperrus/figures/Fig_pipeline", that matches no
│   │                                  current file — likely superseded by
│   │                                  notebooks/figure1/pipeline_figure.ipynb)
│   ├── figure1/                    # currently empty — no src/ code maps here yet;
│   │                                  figure1's only current producer is the notebook
│   └── archive/                    # Older iterations (not in active use)
├── notebooks/
│   ├── figure1/pipeline_figure.ipynb
│   ├── figure4/                   # "Microns check edge effect bosperrus.ipynb" and
│   │                                 "...original REIMANN.ipynb"
│   ├── figure5/figure5.ipynb, figure5_a.ipynb
│   └── exploratory/                # edge_effect_in_BBP.ipynb (Blue Brain Project
│                                      connectome, not MICrONS), squidpy_example.ipynb,
│                                      timezones_snap_data.ipynb, urbanity_snap_data.ipynb
│                                      — none currently declare a rubric #8 status marker
├── results/                       # figure1(empty) .. figure5(300 CSVs), exploratory
│                                     (empty), archive/figure3_pre_revision (101 CSVs,
│                                     an older non-overlapping sphere.py batch, see
│                                     ../STORY.md Key Result #1)
├── result_plots/                  # figure1(8 pipeline panels) .. figure5, exploratory
├── fit_palette.json                # shared color palette; used by both notebooks/
│                                     and src/figure3/plot_metrics.py — kept at this shared
│                                     root rather than nested under notebooks/
├── mibitof_coords/, SNAP_data/, CD34_data/, TWOMBLI_data/, geo_data/, reimann_data/
│                                   # raw/cached input data — all match .gitignore's
│                                     `*data*` pattern except mibitof_coords/, which
│                                     simply doesn't exist in this checkout at all
│                                     (referenced by compute_fits.py and
│                                     pipeline_figure.ipynb; needs manual provisioning)
```

The core dependency is the **bosperrus package**, published on PyPI and pulled in as a normal dependency via `uv` (see `pyproject.toml` / `uv.lock`). The sibling checkout lives at `/home/woody/iwbn/iwbn007h/bosperrus/bosperrus-package/`. Note: `src/figure2/border_effects_kNN_del.py` and `src/figure5/sern.py` still contain leftover `sys.path.append(...)` hacks pointing at a local checkout, predating the PyPI release — if you edit the package locally and want these two scripts to pick up the change, make sure that path actually resolves, since the PyPI-installed version and local source are not automatically kept in sync. `sern.py`'s hack is a *relative* path (`../../bosperrus-package/`) resolved against whatever the process's cwd is at import time, not against the file's own location — it stays correct only because the documented run convention below keeps `cwd=src/` regardless of which `figureN/` subfolder actually holds the script being invoked.

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

Invoke each script by its `figureN/` sub-path, but keep `src/` itself as
the working directory — every script's internal relative paths
(`../results/...`, `../../results/...`, etc.) assume that cwd, not
`cwd=` their own subfolder:

```bash
cd /home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs/src

# Figure 2 — edge truncation effects (Delaunay/kNN); writes ../results/figure2/, ../result_plots/figure2/
uv run python figure2/edge_effects.py

# Figure 3 data — AIC fits per MIBI-TOF dataset/graph type (reads mibitof_coords/coords.pickle,
# which does not exist in this checkout yet); writes ../results/figure3/
uv run python figure3/compute_fits.py

# Figure 3 plot — reads ../results/figure3/ + ../fit_palette.json (from cwd=src/); writes ../result_plots/figure3/
uv run python figure3/plot_metrics.py

# Figure 4 — MICrONS buffer identification (FN_MAT input path is still a stale,
# nonexistent-here absolute path — see ../STORY.md Key Result #4)
uv run python figure4/micron_comparison.py

# Figure 5 — sphere benchmark (100 runs × 3 coord configs × 3 graph types; writes
# ../results/figure5/; NOT seeded — see ../STORY.md Key Result #1)
uv run python figure5/sphere.py
```

Notebooks are run from their own subdirectory under `notebooks/` (e.g.
`notebooks/figure1/`), which is one level deeper than the old flat
`notebooks/` layout — any `../` relative path inside a notebook now needs
one extra `../` to reach `truncated_graphs/`. Results are cached to CSV in
`results/` and re-read on subsequent runs to avoid recomputation.

## Key Parameters (sphere.py)

- `NUMBER_OF_SERNS = 100` — SERN ensemble size
- `N_JOBS = 128` — joblib workers for parallel surrogate generation
- `N_OF_RUNS = 100` — independent outer repetitions
- `OMP_NUM_THREADS = 8` — set via `os.environ`
- **No RNG seed anywhere** — `sample_uniform_on_unit_sphere`,
  `sample_von_mises_fisher`, and `crop_cap` all use unseeded
  `np.random.default_rng()`/`np.random.choice`. Runs are re-runnable-in-kind
  but not bit-reproducible (found via `results-hygiene`; see `../STORY.md`).

## Graph Types

| Type | Directedness | Key param | Notes |
|------|-------------|-----------|-------|
| `delaunay` | undirected | — | Scipy `Delaunay`; on sphere: convex hull |
| `knn` | directed, asymmetric | `k` | Not symmetric: (u→v) ≠ (v→u) |
| `rnn` | undirected | `r` | Radius-neighbor; edges as `frozenset` |

Note: kNN edges are `(u, v)` tuples; Delaunay and rNN edges are `frozenset({u, v})`. The `reindex_edges_to_crop()` function in `border_effects_kNN_del.py` converts global node indices to local crop indices.

## Data Sources

- **MIBI-TOF**: Segmentation TIFFs → centroid extraction in `get_mibitof()`. Coordinate dicts expected at `mibitof_coords/coords.pickle` — **this directory does not exist in this checkout**; `compute_fits.py` and `pipeline_figure.ipynb` cannot run until it's provisioned.
- **Squidpy Visium**: 35 spatial transcriptomics datasets fetched via `sq.datasets.visium()` in `get_squidpy_visium_datasets()`.
- **SNAP**: Geolocated check-in network (Brightkite); used for urban/social network comparison. Expected at `SNAP_data/` (gitignored via `.gitignore`'s `*data*` pattern — populate locally).
- **MICrONS / Reimann connectome**: expected at `reimann_data/microns_mm3_connectome.h5`; `micron_comparison.py`'s `FN_MAT` still points at a stale, nonexistent `je30bery`-only absolute path (see `../STORY.md` Key Result #4).
- **Blue Brain Project connectome**: expected at `reimann_data/connectome_BBP.h5`, read by the exploratory `edge_effect_in_BBP.ipynb`.

All data-containing directories (`SNAP_data/`, `CD34_data/`, `TWOMBLI_data/`, `geo_data/`, `reimann_data/`) match `.gitignore`'s `*data*` pattern and must be populated per-checkout; `mibitof_coords/` is the one exception — it doesn't match that pattern but is nonetheless absent here.

## Dependencies

Core: `numpy`, `scipy`, `pandas`, `scikit-learn`, `networkx`, `squidpy`, `tifffile`, `joblib`, `tqdm`, `matplotlib`.

`bosperrus` is a normal PyPI dependency here (see `pyproject.toml`), installed via `uv sync`. Only install it editable from the local checkout if you need to test unreleased package changes:
```bash
pip install -e /home/woody/iwbn/iwbn007h/bosperrus/bosperrus-package/
```
