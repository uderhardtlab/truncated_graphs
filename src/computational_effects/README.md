# computational_effects — paper section 4

How computational boundary effects depend on graph construction and centrality measure.

- `compute_fits.py` → `results/computational_effects/mibitof_fits/` (AIC fits per MIBI-TOF FOV and graph type;
  SLURM: `jobs/computational_effects_compute_fits.sbatch`).
- `example_scores.py` → `results/computational_effects/mibitof_fits/figure3_examples.csv` (per-cell scores for the
  example maps on top of the fits figure).
- `imc_centrality.py` → `results/computational_effects/mibitof_fits/imc_centrality.csv` (per-cell degree/clustering/
  closeness on squidpy's IMC demo FOV, raw and corrected; bottom row of the fits figure; input `data/anndata/imc.h5ad`).
- `edge_effects.py` (+ `border_effects_kNN_del.py`) → `results/computational_effects/edge_effects/`
  (root causes: 𝒢(V′) ≠ 𝒢(V)|V′).

Figures: `figure_plotting/computational_effects/` (`mibitof_fits.ipynb`, `edge_effects.ipynb`).

Results, decisions and to-dos: `../../../notes/computational_effects.md`.
