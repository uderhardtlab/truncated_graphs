# truncated_graphs

Experiments and figures for the BOSPERRUS paper (boundary effects in spatial graphs: detection and correction).
The method itself is the `bosperrus` package (sibling checkout `../bosperrus-package/`, installed editable).

## Layout

One folder per topic (= paper section) in each of the following, with the same topic names everywhere:

| folder | contents |
|---|---|
| `src/<topic>/` | analysis scripts; they only compute and write numbers/CSVs to `results/<topic>/` |
| `results/<topic>/` | computed results (CSV/JSON) |
| `figure_plotting/<topic>/` | one notebook per figure; reads `results/`, saves the figure (SVG) next to itself; longer plotting code in `<figure>_helpers.py` alongside |
| `jobs/` | SLURM scripts (`<topic>_*.sbatch`; `figure_plotting.sbatch <notebook>` runs any figure notebook) |
| `tutorials/` | usage example of the package |

Rule of thumb: anything slow, or any number quoted in the paper, is computed in `src/`; a figure notebook may only
do quick, illustrative computations (a single fit or graph, seconds).

## Topics and figures

| topic | paper section | compute (`src/<topic>/`) | figure notebooks (`figure_plotting/<topic>/`) |
|---|---|---|---|
| `overview` | 1 | — | `pipeline_figure.ipynb` → `fig1_*.svg` (pipeline panels on one MIBI-TOF image) |
| `spatial_omics` | 2 (analysis buffer), 3 (RNA diffusion QC) | `st_precompute.py` (+ `blade.py`, `border_criterion.py`, `interior_windows.py`) | `ST_demo.ipynb`, `ST_paired.ipynb`; checks: `border_sanity_checks_{demo,paired}.ipynb` |
| `physically_realized_graphs` | measured networks | `microns_buffer.py`, `jackson_centroids.py`, `jackson2019_analysis.py` | `microns.ipynb`, `meristem.ipynb` |
| `computational_effects` | 4 | `compute_fits.py`, `example_scores.py`, `imc_centrality.py`, `edge_effects.py` | `mibitof_fits.ipynb`, `edge_effects.ipynb` |
| `correction_benchmark` | 5 | `sphere.py` (+ `sern.py`), `summarize.py`, `runtime_scaling.py` | `sphere_benchmark.ipynb` |

Each `src/<topic>/README.md` lists inputs, outputs and SLURM jobs per script.

## Setup and running

```bash
pixi install                                   # conda-forge (graph-tool) + PyPI dependencies, bosperrus editable
cd src && pixi run python correction_benchmark/sphere.py --n-runs 1 --seed 0     # e.g. one sphere run
sbatch --clusters=tinyfat jobs/figure_plotting.sbatch figure_plotting/correction_benchmark/sphere_benchmark.ipynb
```

Heavy computations run on the cluster (`jobs/`), not on the login node.

Lint: `pixi exec ruff check src` (settings in `pyproject.toml`).

## Data

Inputs live in `data/` (gitignored, populate per checkout); `mibitof_coords/coords.pickle` is tracked.
MIBI-TOF segmentations, squidpy example datasets, Visium HD / Stereo-seq demo
data (`src/spatial_omics/st_samples.csv`), the MICrONS mm³ connectome (`data/reimann_data/`) and the
Arabidopsis meristem networks of Jackson et al. 2019 (https://osf.io/mxgrh/, `data/jackson2019/`).
