#!/bin/bash
# Combined image-mask vs. spot-mask figure across all 12 samples -- see
# src/exploratory/combined_tissue_masks.py's docstring.
#
# Submit from anywhere with (on tinyfat, no GPU needed):
#   sbatch.tinyfat jobs/combined_tissue_masks.sbatch
#SBATCH --job-name=combined-tissue-masks
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=jobs/logs/combined_tissue_masks_%j.out
#SBATCH --error=jobs/logs/combined_tissue_masks_%j.err

set -euo pipefail

REPO_ROOT="/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs"
cd "${REPO_ROOT}/src/exploratory"
echo "Combined tissue-mask figure on $(hostname) with $SLURM_CPUS_PER_TASK cpus"

pixi run --manifest-path "${REPO_ROOT}/pyproject.toml" python combined_tissue_masks.py
