#!/bin/bash
# MICrONS analysis buffer vs. Reimann heuristic -> results/physically_realized_graphs/microns/.
# Needs data/reimann_data/microns_mm3_connectome.h5. Submit from truncated_graphs/:
#   sbatch --clusters=tinyfat jobs/physically_realized_graphs_microns.sbatch
#SBATCH --job-name=microns
#SBATCH --partition=broadwell256
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/physically_realized_graphs_microns_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
pixi run python src/physically_realized_graphs/microns_buffer.py
