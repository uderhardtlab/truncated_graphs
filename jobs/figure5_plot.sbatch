#!/bin/bash
# Figure 5 violins + summary table from results/figure5/ (after the sphere array).
# Submit from truncated_graphs/:  sbatch --clusters=tinyfat jobs/figure5_plot.sbatch
#SBATCH --job-name=figure5-plot
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/figure5_plot_%j.out
#SBATCH --error=jobs/logs/figure5_plot_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
pixi run python src/figure5/plot_violins.py
cd src && pixi run python figure5/summarize.py
