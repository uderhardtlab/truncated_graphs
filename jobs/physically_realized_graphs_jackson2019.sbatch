#!/bin/bash
# Jackson et al. 2019 meristem networks: centroids from the .mgxs stacks, then the analysis
# (results/physically_realized_graphs/jackson2019/; figure: figure_plotting/physically_realized_graphs/meristem.ipynb). Needs data/jackson2019/{WT,ktn1}_Arabidopsis.zip from https://osf.io/mxgrh/.
# Submit from truncated_graphs/:
#   sbatch jobs/physically_realized_graphs_jackson2019.sbatch
#SBATCH --job-name=jackson2019
#SBATCH --clusters=tinyfat
#SBATCH --partition=broadwell256
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/physically_realized_graphs_jackson2019_%j.out
#SBATCH --error=jobs/logs/physically_realized_graphs_jackson2019_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ "${SKIP_CENTROIDS:-0}" = 1 ] || pixi run python src/physically_realized_graphs/jackson_centroids.py
pixi run python src/physically_realized_graphs/jackson2019_analysis.py
