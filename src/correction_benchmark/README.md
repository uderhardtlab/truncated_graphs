# correction_benchmark — paper section 5

Sphere ground truth: crop a cap, recompute the graph on the crop, correct, compare with the uncropped graph.

- `sphere.py` (+ `sern.py`) → `results/correction_benchmark/sphere/correlations_seed*.csv`
  (SLURM array: `jobs/correction_benchmark_sphere.sbatch`, one seed per task).
- `sphere.py --no-sern` → `results/correction_benchmark/sphere_detrend/` (detrending baselines only, same crops;
  `jobs/correction_benchmark_sphere_detrend.sbatch`), merged with `sphere/` by `load_results.py`.
- `summarize.py` → `results/correction_benchmark/summary.csv` (`jobs/correction_benchmark_summarize.sbatch`).
- `runtime_scaling.py` → `results/correction_benchmark/runtime_scaling/` (BOSPERRUS vs. SERN scaling).

Figure: `figure_plotting/correction_benchmark/sphere_benchmark.ipynb`.

Results, decisions and to-dos: `../../../notes/correction_benchmark.md`.
