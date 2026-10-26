#!/bin/bash
#SBATCH --job-name=fitcheck_fig3
#SBATCH --clusters=tinyfat
#SBATCH --partition=broadwell256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs/results/exploratory/fit_optimizer_check/logs/%x_%j.out
cd /home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs
pixi run --manifest-path . python -u src/exploratory/fit_optimizer_check/figure3_subset.py
