#!/bin/bash
# Illustration of the Delaunay cap (convex-hull lid) artefact in the sphere benchmark.
# Submit from truncated_graphs/:  sbatch --clusters=tinyfat jobs/correction_benchmark_delaunay_cap.sbatch
#SBATCH --job-name=figure5-delcap
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/correction_benchmark_delcap_%j.out
#SBATCH --error=jobs/logs/correction_benchmark_delcap_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/src"
pixi run python correction_benchmark/plot_delaunay_cap.py
