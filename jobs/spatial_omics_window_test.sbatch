#!/bin/bash
# Border analysis with the per-component interior-window test (bosperrus.interior_window_test) for one sample.
# Usage (from truncated_graphs/):
#   sbatch --job-name=wt_<sample> jobs/spatial_omics_window_test.sbatch <sample> demo|paired
# demo: --image-mask, --in-tissue, --tissue-edge at 8 and 2 um, --blade-image-border; paired: --image-mask.
# Needs the main run's {sample}_diffusion_8um.npz. Stagger submissions (~30s): pixi startups race on the env.
#SBATCH --clusters=tinyfat
#SBATCH --partition=broadwell256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs/results/spatial_omics/st_reroll/logs/%x_%j.out
set -euo pipefail
cd /home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs
run() { pixi run --manifest-path . python -u src/spatial_omics/st_precompute.py "$1" "${@:2}"; }
if [ "$2" = demo ]; then
    run "$1" --image-mask
    run "$1" --in-tissue
    run "$1" --tissue-edge --resolution 8
    run "$1" --tissue-edge --resolution 2
    run "$1" --blade-image-border
else
    run "$1" --image-mask
fi
