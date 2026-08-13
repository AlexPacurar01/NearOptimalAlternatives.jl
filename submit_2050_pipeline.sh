#!/bin/bash
# ===========================================================================
# Submit the full 2050 Investment pipeline to DHPC (DelftBlue) with the
# correct job dependencies, so the whole thing can be launched with one
# command and then left alone:
#
#   baseline (1 node)                         [Stage 1]
#     -> arms array, 4 tasks in parallel      [Stage 2, on 4 nodes at once:
#                                               continuation, cold_barrier,
#                                               warm_simplex, cold_simplex]
#          -> aggregate (1 node)              [Stage 3]
#
# Run from the directory CONTAINING the project (e.g. $HOME), same
# convention as the other run_dhpc_*.slurm scripts:
#   bash NearOptimalAlternatives.jl/submit_2050_pipeline.sh
#
# Each stage is a separate `sbatch` submission because SLURM dependencies
# (`--dependency=afterok:<jobid>`) can only be set once the earlier job's ID
# is known - that's the reason this is a wrapper script and not one big
# .slurm file. Re-running is safe: it always submits a fresh baseline solve
# (the biggest, most important-to-get-right measurement), and the array/
# aggregate jobs key off that run's own job ID, so nothing collides with a
# previous pipeline's output files.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] submitting baseline job..."
BASE_JOBID=$(sbatch --parsable NearOptimalAlternatives.jl/run_dhpc_2050_baseline.slurm)
echo "      baseline job id: $BASE_JOBID"

# The baseline job's cache file name is predictable ahead of time: it's tagged
# with its own $SLURM_JOB_ID (see run_2050_baseline.jl / BASE_2050_ID).
BASE_CACHE="results/2050_investment/baseline/base_solution_${BASE_JOBID}.bin"

echo "[2/3] submitting arms array (depends on $BASE_JOBID)..."
ARMS_JOBID=$(
  sbatch --parsable \
    --dependency=afterok:"$BASE_JOBID" \
    --export=ALL,BASE_2050_CACHE="$BASE_CACHE" \
    NearOptimalAlternatives.jl/run_dhpc_2050_arms.slurm
)
echo "      arms array job id: $ARMS_JOBID"

echo "[3/3] submitting aggregate job (depends on $ARMS_JOBID)..."
AGG_JOBID=$(
  sbatch --parsable \
    --dependency=afterany:"$ARMS_JOBID" \
    NearOptimalAlternatives.jl/run_dhpc_2050_aggregate.slurm
)
echo "      aggregate job id: $AGG_JOBID"

echo
echo "Pipeline submitted:"
echo "  baseline  $BASE_JOBID  (results/2050_investment/baseline/)"
echo "  arms      $ARMS_JOBID  (4 parallel tasks; results/2050_investment/{continuation,arms}/)"
echo "  aggregate $AGG_JOBID   (results/2050_investment/REPORT.txt + *_all.csv)"
echo
echo "Watch with: squeue --me"
echo "Cache path (if you need to submit arms/aggregate manually later):"
echo "  $BASE_CACHE"
