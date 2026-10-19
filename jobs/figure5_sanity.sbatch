#!/bin/bash
# Sanity check of the BOSPERRUS correction variants (src/figure5/sanity_multiplicative.py) before the
# figure 5 rerun: one seeded sphere per coord config (uniform, kappa=1) x delaunay / delaunay_pruned / knn k=10 / rnn r=0.1.
# Submit from truncated_graphs/:  sbatch --clusters=tinyfat jobs/figure5_sanity.sbatch
#SBATCH --job-name=figure5-sanity
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/figure5_sanity_%j.out
#SBATCH --error=jobs/logs/figure5_sanity_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR/src"
echo "host $(hostname) start $(date -Is)"
/usr/bin/time -v pixi run python figure5/sanity_multiplicative.py
echo "end $(date -Is)"
