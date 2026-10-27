# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying **border/boundary effects in spatial graphs** — how truncating a spatial domain distorts graph-theoretic centrality measures (degree, betweenness, closeness, PageRank, clustering). The project develops and benchmarks two correction strategies:

1. **BOSPERRUS**: Fits a parametric curve (piecewise-linear or exponential saturation) of centrality vs. distance-to-border, then uses the fitted model to correct observed values.
2. **SERN** (Spatial Edge-based Random Network): Builds a surrogate null ensemble by sampling random graphs that preserve the empirical edge-length distribution, then subtracts the surrogate median from observed centralities.

## Codebase Structure

Organised by **topic = paper section** (restructured 2026-10-26; the old
`figure1/`…`figure5/` numbering is gone, because section order may still
change). The same topic names are used in `src/`, `notebooks/`, `results/`
and `result_plots/`:

| topic | paper section | what |
|---|---|---|
| `overview` | 1 | Figure 1 (boundary-effect examples, pipeline panels) |
| `spatial_omics` | 2 + 3 | Visium HD / Stereo-seq: data-driven analysis buffer (section 2) and RNA-diffusion QC (section 3); one shared pipeline |
| `microns` | 2 (panel) | MICrONS connectome analysis buffer (more code/data in `/home/woody/iwbn/iwbn007h/MICrONS_data/`) |
| `computational_effects` | 4 | MIBI-TOF fits across graph types/measures; edge-truncation root causes |
| `correction_benchmark` | 5 | sphere benchmark (BOSPERRUS additive/multiplicative/log vs. linear/spline detrending vs. SERN), sanity checks, runtime scaling |
| `archive` | — | superseded or removed-from-paper code, notebooks and outputs -- local only, gitignored (untracked 2026-10-27) |

Results notes (what was found, decisions, to-dos) live outside this repo, one
file per topic: `../notes/<topic>.md`, indexed by `../RESULTS.md`. Each
`src/<topic>/README.md` says what the folder is for.

```
truncated_graphs/
├── src/
│   ├── spatial_omics/          st_precompute.py (buffer + BLADE + diffusion per sample; writes
│   │                           results/spatial_omics/st_reroll/, gitignored), blade.py, st_samples.csv
│   ├── microns/                micron_comparison.py (library for the MICrONS notebooks)
│   ├── computational_effects/  compute_fits.py, plot_metrics.py, plot_entropy*.py (MIBI-TOF fits),
│   │                           edge_effects.py + border_effects_kNN_del.py (edge truncation)
│   ├── correction_benchmark/   sphere.py, sern.py, summarize.py, plot_violins.py,
│   │                           sanity_multiplicative.py, plot_delaunay_cap.py, runtime_scaling.py
│   └── archive/                old Visium HD pipeline (visium_hd.py, border_fit.py, figure4/, ...),
│                               old exploratory scripts, older iterations
├── notebooks/
│   ├── overview/               boundary_effects.ipynb (Figure 1 examples), pipeline_figure.ipynb
│   ├── spatial_omics/          ST_demo.ipynb (sec. 2), ST_paired.ipynb (sec. 3), border_sanity_checks_{demo,paired}.ipynb
│   ├── microns/                MICRONS_analysis_buffer.ipynb
│   ├── tutorials/              bosperrus_on_TMA.ipynb
│   └── archive/                exploratory/ (clustering, spatial domains, ...), pre_reroll/, older
├── results/<topic>/            computational_effects/{edge_effects,mibitof_fits}/,
│                               correction_benchmark/{sphere,sanity,runtime_scaling}/ + summary.csv,
│                               spatial_omics/st_reroll/ (gitignored); results/archive/
├── result_plots/<topic>/       same topics; result_plots/archive/
├── jobs/                       <topic>_*.sbatch (SLURM, tinyfat); jobs/archive/; logs in jobs/logs/
├── literature/                 reference PDFs (gitignored, not redistributable)
├── fit_palette.json            shared colour palette
└── mibitof_coords/, data/      inputs (data dirs gitignored via `*data*`)
```

The core dependency is the **bosperrus package** (sibling checkout `../bosperrus-package/`, installed editable into the pixi env via `pyproject.toml`; see Dependencies). `src/computational_effects/border_effects_kNN_del.py` still has a stale `sys.path.append("/data/bionets/je30bery/bosperrus-package")` (harmless, the installed package is used); `src/correction_benchmark/sern.py` appends `../../bosperrus-package/` relative to the cwd, which resolves only with `cwd=src/`.

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

Scripts run with **`src/` as the working directory**, invoked by their topic sub-path; their relative paths (`../results/<topic>/...`) assume that cwd:

```bash
cd /home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs/src
pixi run python computational_effects/compute_fits.py      # MIBI-TOF fits -> ../results/computational_effects/mibitof_fits/ (cached per graph type)
pixi run python computational_effects/plot_metrics.py      # -> ../result_plots/computational_effects/mibitof_fits/
pixi run python computational_effects/edge_effects.py      # -> ../results|result_plots/computational_effects/edge_effects/
pixi run python correction_benchmark/sphere.py --n-runs 1 --seed 0   # one sphere run -> ../results/correction_benchmark/sphere/
pixi run python correction_benchmark/summarize.py          # -> ../results/correction_benchmark/summary.csv
pixi run python correction_benchmark/plot_violins.py       # -> ../result_plots/correction_benchmark/
```

Heavy work goes through SLURM on the tinyfat cluster, never the login node; submit from `truncated_graphs/`:

```bash
sbatch --clusters=tinyfat jobs/correction_benchmark_sphere.sbatch        # 100-seed array (~4 h per task with 400 SERN surrogates)
sbatch --clusters=tinyfat jobs/correction_benchmark_sphere_detrend.sbatch  # same seeds, --no-sern (detrending baselines, ~4 min per task)
sbatch --clusters=tinyfat jobs/correction_benchmark_plot.sbatch          # violins + summary (merges sphere/ + sphere_detrend/)
sbatch --clusters=tinyfat jobs/computational_effects_compute_fits.sbatch
sbatch --clusters=tinyfat --job-name=st_<sample> jobs/spatial_omics_st_precompute.sbatch <sample> [--ablation|--image-mask]
```

Notebooks run from their own folder (`notebooks/<topic>/`), so their paths start with `../../`; execute heavy ones on a compute node with `pixi run jupyter nbconvert --to notebook --execute --inplace <nb>`.

## Key Parameters (correction_benchmark/sphere.py)

- `NUMBER_OF_SERNS = 400` — SERN ensemble size (Rheinwalt et al. 2012 use 1000, but their Fig. 2 shows reliability plateaus after ~400; paper in `literature/`)
- `N_JOBS = -1` — joblib workers for SERN surrogates (all cores of the job)
- `N_OF_RUNS = 100` — outer repetitions; the SLURM array runs one per task with `--seed <task id>`
- `OMP_NUM_THREADS = 8` — set via `os.environ`
- Seeded: `--seed s` seeds the sampler, the crops and SERN's global RNG for run `s` (bit-reproducible per seed).

## Graph Types

| Type | Directedness | Key param | Notes |
|------|-------------|-----------|-------|
| `delaunay` | undirected | — | Scipy `Delaunay`; on sphere: convex hull |
| `knn` | directed, asymmetric | `k` | Not symmetric: (u→v) ≠ (v→u) |
| `rnn` | undirected | `r` | Radius-neighbor; edges as `frozenset` |

Note: kNN edges are `(u, v)` tuples; Delaunay and rNN edges are `frozenset({u, v})`. The `reindex_edges_to_crop()` function in `border_effects_kNN_del.py` converts global node indices to local crop indices.

## Data Sources

- **MIBI-TOF**: Segmentation TIFFs → centroid extraction in `get_mibitof()`. Coordinate dicts at `mibitof_coords/coords.pickle`, provisioned in this checkout; used by `computational_effects/compute_fits.py` and the `overview` notebooks.
- **Squidpy Visium**: 35 spatial transcriptomics datasets fetched via `sq.datasets.visium()` in `get_squidpy_visium_datasets()`.
- **SNAP**: Geolocated check-in network (Brightkite); used for urban/social network comparison. Expected at `SNAP_data/` (gitignored via `.gitignore`'s `*data*` pattern — populate locally).
- **MICrONS / Reimann connectome**: `data/reimann_data/microns_mm3_connectome.h5`, read by the `microns` and `overview` notebooks via `src/microns/micron_comparison.py`.
- **Visium HD demo data**: the public 10x "v1 - Protocol 2.0" datasets with H&E (11; the IF-stained ovarian cancer is downloaded but excluded), binned outputs at 8/16 µm in `/home/woody/iwbn/iwbn007h/visium_hd_demo_data/` (list: `chemistry_protocol2_list.txt`); samples for the pipeline are listed in `src/spatial_omics/st_samples.csv`.
- **Blue Brain Project connectome**: expected at `reimann_data/connectome_BBP.h5`, read by the exploratory `edge_effect_in_BBP.ipynb`.

All data-containing directories (`SNAP_data/`, `CD34_data/`, `TWOMBLI_data/`, `geo_data/`, `reimann_data/`) match `.gitignore`'s `*data*` pattern and must be populated per-checkout; `mibitof_coords/` is the one exception — it doesn't match that pattern, and is provisioned in this checkout.

## Dependencies

Managed via `pixi` (see `pyproject.toml`'s `[tool.pixi.*]` tables and `pixi.lock`), not `uv` — this project mixes conda-forge dependencies (currently just `graph-tool`, which has no usable PyPI wheel) with regular PyPI dependencies (`squidpy`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `networkx`, `tifffile`, `joblib`, `tqdm`, `matplotlib`, ...) in one resolved environment. `requires-python` is capped at `<3.13` — pixi otherwise resolves to the newest available Python satisfying `[project]`'s constraint, and the broader scientific stack (squidpy/scanpy/numba) isn't yet reliably compatible with very new Python versions.

`bosperrus` is installed **editable from the local checkout** (`[tool.pixi.pypi-dependencies] bosperrus = { path = "../bosperrus-package", editable = true }`): every `pixi run` uses whatever branch is currently checked out in `../bosperrus-package/` -- no release, no `PYTHONPATH` needed. Keep it this way until the paper is done (no PyPI version bumps for paper work); pin a released version only at the end. Check which branch is checked out there before runs whose results depend on unmerged package changes (e.g. `RisingPiecewiseLinearFit` on `feature/rising-piecewise`).
