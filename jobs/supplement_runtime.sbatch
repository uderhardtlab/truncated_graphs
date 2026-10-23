#!/bin/bash
# Supplementary runtime/memory scaling: BOSPERRUS vs. SERN (src/supplement/runtime_scaling.py).
# One run per method x graph type x size, each on an exclusive node (128 cores, 512 GB on tinyfat work).
# Index = method (2) x graph (2) x size (7) = 28 tasks.
# Submit from truncated_graphs/:  sbatch --clusters=tinyfat jobs/supplement_runtime.sbatch
#SBATCH --job-name=runtime-scaling
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-27
#SBATCH --time=08:00:00
#SBATCH --output=jobs/logs/runtime_%A_%a.out
#SBATCH --error=jobs/logs/runtime_%A_%a.err

set -uo pipefail
METHODS=(bosperrus sern)
GRAPHS=(knn delaunay)
SIZES=(1000 2000 5000 10000 20000 50000 100000)
i=$SLURM_ARRAY_TASK_ID
method=${METHODS[$((i / 14))]}
graph=${GRAPHS[$(((i / 7) % 2))]}
n=${SIZES[$((i % 7))]}
cd "$SLURM_SUBMIT_DIR/src"
echo "task $i: $method $graph n=$n on $(hostname), start $(date -Is)"
pixi run python supplement/runtime_scaling.py --method "$method" --graph "$graph" --n "$n" "$@"
rc=$?
# exit code next to the result, so OOM kills (137) are visible even if the JSON stayed at "started"
mkdir -p ../results/supplement/runtime_scaling
echo "$rc" > "../results/supplement/runtime_scaling/${method}_${graph}_${n}.exitcode"
echo "end $(date -Is), exit $rc"
