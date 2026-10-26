#!/bin/bash
# Figure 5 violins + summary table from results/correction_benchmark/sphere/ (after the sphere array).
# Submit from truncated_graphs/:  sbatch --clusters=tinyfat jobs/correction_benchmark_plot.sbatch
#SBATCH --job-name=figure5-plot
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/correction_benchmark_plot_%j.out
#SBATCH --error=jobs/logs/correction_benchmark_plot_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
pixi run python src/correction_benchmark/plot_violins.py
cd src && pixi run python correction_benchmark/summarize.py
