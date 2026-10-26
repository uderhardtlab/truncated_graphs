#!/bin/bash
# Final STOmics ssDNA tissue segmentation (disk=6@small, per-sample method) --
# see src/exploratory/segment_stomics_tissue_final.py's docstring.
#
# Submit from anywhere with (on tinyfat, no GPU needed):
#   sbatch.tinyfat jobs/stomics_tissue_final.sbatch
#SBATCH --job-name=stomics-tissue-final
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=jobs/logs/stomics_tissue_final_%j.out
#SBATCH --error=jobs/logs/stomics_tissue_final_%j.err

set -euo pipefail

REPO_ROOT="/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs"
cd "${REPO_ROOT}/src/exploratory"
echo "Final segmentation on $(hostname) with $SLURM_CPUS_PER_TASK cpus"

pixi run --manifest-path "${REPO_ROOT}/pyproject.toml" python segment_stomics_tissue_final.py
