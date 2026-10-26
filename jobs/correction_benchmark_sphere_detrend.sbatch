#!/bin/bash
# Detrending baselines for the sphere benchmark without recomputing SERN: sphere.py --no-sern with the same
# seeds as jobs/correction_benchmark_sphere.sbatch (same crops) -> results/correction_benchmark/sphere_detrend/,
# merged with sphere/ by src/correction_benchmark/load_results.py. About 4 min per task (seed-2 test, 2026-10-26).
# Submit from truncated_graphs/:  sbatch --clusters=tinyfat jobs/correction_benchmark_sphere_detrend.sbatch
#SBATCH --job-name=cb-sphere-detrend
#SBATCH --array=0-99
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/correction_benchmark_detrend_%A_%a.out
#SBATCH --error=jobs/logs/correction_benchmark_detrend_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR/src"
pixi run python correction_benchmark/sphere.py --n-runs 1 --seed "$SLURM_ARRAY_TASK_ID" --no-sern \
    --output-dir ../results/correction_benchmark/sphere_detrend
