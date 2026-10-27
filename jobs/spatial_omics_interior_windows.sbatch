#!/bin/bash
# Usage (from truncated_graphs/):
#   sbatch jobs/spatial_omics_interior_windows.sbatch [--resolution R] [sample ...]   # default: all visium_demo samples, 8 um
# Interior-window null for the border criterion (src/spatial_omics/interior_windows.py).
#SBATCH --clusters=tinyfat
#SBATCH --partition=broadwell256
#SBATCH --job-name=st_interior_windows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs/results/spatial_omics/st_reroll/logs/%x_%j.out

cd /home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs
pixi run --manifest-path . python -u src/spatial_omics/interior_windows.py "$@"
