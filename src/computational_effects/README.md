# computational_effects — paper section 4

How computational boundary effects depend on graph construction and centrality measure.

- `compute_fits.py` → `results/computational_effects/mibitof_fits/` (AIC fits per MIBI-TOF FOV and graph type;
  SLURM: `jobs/computational_effects_compute_fits.sbatch`); plotted by `plot_metrics.py`, `plot_entropy*.py`.
- `edge_effects.py` (+ `border_effects_kNN_del.py`) → `results|result_plots/computational_effects/edge_effects/`
  (root causes: 𝒢(V′) ≠ 𝒢(V)|V′).

Results, decisions and to-dos: `../../../notes/computational_effects.md`.
