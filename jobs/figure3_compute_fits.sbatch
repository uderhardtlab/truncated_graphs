#!/bin/bash
# Figure 3 data: AIC fits per MIBI-TOF dataset/graph type.
# One array task per graph type (8 total), each parallelizing across the
# ~1000 MIBI-TOF datasets internally via compute_fits.py's ProcessPoolExecutor.
# Submit from anywhere with (on tinyfat, no GPU needed):
#   sbatch.tinyfat jobs/figure3_compute_fits.sbatch
#
# Results are cached per graph type in results/figure3/ -- delete a CSV there
# to force that graph type to be recomputed; re-submitting this job otherwise
# just no-ops on any graph type whose CSV already exists.
#SBATCH --job-name=figure3-compute-fits
#SBATCH --clusters=tinyfat
#SBATCH --partition=broadwell256
#SBATCH --array=0-7
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/figure3_%A_%a.out
#SBATCH --error=jobs/logs/figure3_%A_%a.err

set -euo pipefail

GRAPH_TYPES=(delaunay "knn_k=5" "knn_k=10" "knn_k=15" "rnn_r=0.02" "rnn_r=0.03" "rnn_r=0.04" "rnn_r=0.05")
GT="${GRAPH_TYPES[$SLURM_ARRAY_TASK_ID]}"

# $0 points at the staged copy of this script in the compute node's spool
# directory, not its original location -- use SLURM's own submit-dir variable.
cd "$SLURM_SUBMIT_DIR"
# stagger task start-up: simultaneous `pixi run` start-ups race on the env
sleep $((SLURM_ARRAY_TASK_ID * 30))
echo "Computing graph_type=$GT on $(hostname) with $SLURM_CPUS_PER_TASK cpus"
pixi run python src/figure3/compute_fits.py --graph-type "$GT"
