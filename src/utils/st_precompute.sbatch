#!/bin/bash
# Usage (from truncated_graphs/):
#   sbatch --job-name=st_<sample> src/utils/st_precompute.sbatch <sample> [--ablation]
# Stagger submissions (~30s): simultaneous `pixi run` startups race on the env.
#SBATCH --clusters=tinyfat
#SBATCH --partition=broadwell256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs/results/exploratory/st_reroll/logs/%x_%j.out

cd /home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs
pixi run --manifest-path . python -u src/utils/st_precompute.py "$@"
