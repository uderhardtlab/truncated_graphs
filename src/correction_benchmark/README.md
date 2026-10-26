# correction_benchmark — paper section 5

Sphere ground truth: crop a cap, recompute the graph on the crop, correct, compare with the uncropped graph.

- `sphere.py` (+ `sern.py`) → `results/correction_benchmark/sphere/correlations_seed*.csv`
  (SLURM array: `jobs/correction_benchmark_sphere.sbatch`, one seed per task).
- `summarize.py` → `results/correction_benchmark/summary.csv`; `plot_violins.py` → `result_plots/correction_benchmark/`
  (both: `jobs/correction_benchmark_plot.sbatch`).
- `sanity_multiplicative.py` (correction variants, `jobs/correction_benchmark_sanity.sbatch`),
  `plot_delaunay_cap.py` (Delaunay crop artefact), `runtime_scaling.py` (BOSPERRUS vs. SERN scaling).

Results, decisions and to-dos: `../../../notes/correction_benchmark.md`.
