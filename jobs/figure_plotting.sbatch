#!/bin/bash
# Execute one figure notebook in place on a compute node (figures are saved next to it).
# Submit from truncated_graphs/:
#   sbatch --clusters=tinyfat jobs/figure_plotting.sbatch figure_plotting/<topic>/<notebook>.ipynb
#SBATCH --job-name=figure_plotting
#SBATCH --partition=broadwell256
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/figure_plotting_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/$(dirname "$1")"
pixi run --manifest-path "$SLURM_SUBMIT_DIR/pyproject.toml" jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "$(basename "$1")"
