#!/bin/bash
# Summary table from results/correction_benchmark/sphere/ (after the sphere array); the figure is
# figure_plotting/correction_benchmark/sphere_benchmark.ipynb.
# Submit from truncated_graphs/:  sbatch --clusters=tinyfat jobs/correction_benchmark_summarize.sbatch
#SBATCH --job-name=cb_summarize
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/correction_benchmark_summarize_%j.out
#SBATCH --error=jobs/logs/correction_benchmark_summarize_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
cd src && pixi run python correction_benchmark/summarize.py
