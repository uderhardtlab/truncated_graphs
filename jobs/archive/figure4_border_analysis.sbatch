#!/bin/bash
# Figure 4 data: per-sample border-effect analysis (piecewise-linear +
# exponential-saturation fits, BLADE peel-sweep) across all 12 Visium HD /
# STOmics samples in src/figure4/sample_manifest.csv. One array task per
# sample; each writes results/figure4/per_sample/<name>.json.
#
# REPO_ROOT below is hardcoded (not derived from $SLURM_SUBMIT_DIR or $0 --
# SLURM stages this script into a spool directory on the compute node, so $0
# doesn't point at its real location, and $SLURM_SUBMIT_DIR depends on
# whatever directory you happened to run `sbatch` from) since this cluster
# and the one this checkout normally runs on share the same filesystem. If
# you copy this checkout to a cluster that does NOT share this filesystem,
# update REPO_ROOT to that cluster's path, run `pixi install` there once
# (this job assumes it's already done), and update
# src/figure4/sample_manifest.csv's h5ad_path column to wherever you put the
# 12 h5ad files.
#
# Submit from anywhere:
#   sbatch jobs/figure4_border_analysis.sbatch
#
# Resources sized for the largest sample (~1.7M bins, visium demo
# colon_cancer_ff); adjust --mem/--time/--cpus-per-task if your cluster is
# tighter or you know these are generous. No --partition/--account set
# (cluster default) -- add them if your cluster requires one.
#
# process_sample.py skips (doesn't recompute) a sample whose
# results/figure4/per_sample/<name>.json already exists, so re-submitting
# this job after a partial run only fills in what's missing (pass --force to
# process_sample.py below if you want to force recomputation).
#SBATCH --job-name=figure4-border-analysis
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/figure4_%A_%a.out
#SBATCH --error=jobs/logs/figure4_%A_%a.err

set -euo pipefail

REPO_ROOT="/home/woody/iwbn/iwbn007h/bosperrus/truncated_graphs"
cd "${REPO_ROOT}"
echo "Processing manifest row $SLURM_ARRAY_TASK_ID (job $SLURM_ARRAY_JOB_ID) on $(hostname) with $SLURM_CPUS_PER_TASK cpus"

pixi run python src/figure4/process_sample.py \
    --manifest src/figure4/sample_manifest.csv \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --outdir results/figure4/per_sample
