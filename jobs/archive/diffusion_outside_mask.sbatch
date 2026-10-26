#!/bin/bash
# Fit exp-sat diffusion model outside the image tissue mask, one array task
# per sample (src/figure4/sample_manifest.csv) -- see
# src/exploratory/diffusion_outside_mask.py's docstring.
#
# Submit from anywhere with (on tinyfat, no GPU needed):
#   sbatch.tinyfat jobs/diffusion_outside_mask.sbatch
#
# Memory sized like figure4_border_analysis.sbatch (same largest-Visium-
# sample concern -- here it's the full sparse X matrix, there it was
# n_bins-scaled squidpy neighbor graphs).
#SBATCH --job-name=diffusion-outside-mask
#SBATCH --array=0-11
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=jobs/logs/diffusion_outside_mask_%A_%a.out
#SBATCH --error=jobs/logs/diffusion_outside_mask_%A_%a.err

set -euo pipefail

REPO_ROOT="/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs"
cd "${REPO_ROOT}/src/exploratory"
echo "Diffusion-outside-mask fit for index $SLURM_ARRAY_TASK_ID on $(hostname) with $SLURM_CPUS_PER_TASK cpus"

pixi run --manifest-path "${REPO_ROOT}/pyproject.toml" python diffusion_outside_mask.py --index "${SLURM_ARRAY_TASK_ID}"
