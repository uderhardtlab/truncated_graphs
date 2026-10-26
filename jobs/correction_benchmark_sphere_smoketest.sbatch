#!/bin/bash
# Figure 5 smoke test: validates the sphere.py/sern.py pipeline actually runs
# end-to-end on a real compute node (not the login node), at toy scale, before
# committing to the full 100-task figure5_sphere.sbatch. Writes to a separate
# results/correction_benchmark/smoketest/ directory -- NOT results/correction_benchmark/sphere/ -- so toy-scale
# output never mixes into the real dataset used for the violin plots.
# Submit from anywhere with (on tinyfat, no GPU needed):
#   sbatch.tinyfat jobs/correction_benchmark_sphere_smoketest.sbatch
#SBATCH --job-name=figure5-smoketest
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=jobs/logs/correction_benchmark_smoketest_%A_%a.out
#SBATCH --error=jobs/logs/correction_benchmark_smoketest_%A_%a.err

set -euo pipefail

# $0 points at the staged copy of this script in the compute node's spool
# directory, not its original location -- use SLURM's own submit-dir variable.
cd "$SLURM_SUBMIT_DIR/src"
echo "Smoke test (task $SLURM_ARRAY_TASK_ID) on $(hostname) with $SLURM_CPUS_PER_TASK cpus"
pixi run python correction_benchmark/sphere.py \
    --n-runs 1 \
    --n 300 \
    --n-surrogates 10 \
    --output-dir ../results/correction_benchmark/smoketest
