#!/bin/bash
# Figure 5 data: sphere benchmark (border-effect correction: bosperrus vs. SERN),
# both Pearson and Spearman correlations, bigger cap radius only (cap_radii=[2]
# in sphere.py's main() -- the smaller cap_radius=1 crops were dropped).
# N_OF_RUNS (100) split one-per-array-task so all outer repetitions run
# concurrently instead of serially; each task writes its own timestamped CSVs
# to results/figure5/ via sphere.py's --n-runs 1.
#
# First full-scale attempt (job 1440602, cap_radii=[1, 2]) timed out at
# --time=01:00:00 for 99/100 tasks -- the one that finished (task 19) took
# 57:22. Dropping to a single cap radius roughly halves the per-run cost
# (each of the 21 graph configs now does 1 BOSPERRUS+SERN pass instead of 2),
# and --time is bumped well above even the old two-cap-radius runtime for
# real safety margin.
#
# Per-run cost is dominated by BOSPERRUS's own AIC fitting (curve_fit +
# differential_evolution in fit.py, single-threaded -- no `workers=` used)
# rather than the SERN surrogate generation, which completes in well under a
# second regardless of core count. So this deliberately requests few cores
# per task (just headroom for SERN's brief joblib-parallel step) rather than
# a whole node's worth: broadwell512 is `OverSubscribe=EXCLUSIVE` with only 3
# nodes, so it hands each job an entire node no matter what's requested,
# capping concurrency at 3 array tasks regardless of --cpus-per-task. `work`
# is shared (`OverSubscribe=NO`) across 36 nodes, so a small per-task request
# here lets far more of the 100 tasks run concurrently instead of each one
# starving 90%+ of a monopolized node during single-threaded fitting. `work`
# is also this cluster's default partition, so it's left unspecified below.
# Submit from anywhere with (on tinyfat, no GPU needed):
#   sbatch.tinyfat jobs/figure5_sphere.sbatch
#SBATCH --job-name=figure5-sphere
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/figure5_%A_%a.out
#SBATCH --error=jobs/logs/figure5_%A_%a.err

set -euo pipefail

# $0 points at the staged copy of this script in the compute node's spool
# directory, not its original location -- use SLURM's own submit-dir variable.
cd "$SLURM_SUBMIT_DIR/src"
echo "Running 1 sphere.py repetition (task $SLURM_ARRAY_TASK_ID) on $(hostname) with $SLURM_CPUS_PER_TASK cpus"
pixi run python figure5/sphere.py --n-runs 1
