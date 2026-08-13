#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) SLURM job ARRAY: Stage 2 of the 2050 Investment pipeline.
#
# One ARRAY TASK = one experiment arm, each running in isolation on its own
# node so the scheduler runs them in parallel with no solver contention:
#
#   task 0  continuation-vs-SPORES      (experiment_cont_spores_2050.jl)
#   task 1  algorithm-switch arms, small grid  (experiment_arms_2050.jl, N=5)
#   task 2  algorithm-switch arms, larger grid (experiment_arms_2050.jl, N=10)
#
# All three load the SAME cached baseline (x*/C*) produced by
# run_dhpc_2050_baseline.slurm, so none of them repeats that solve.
#
# Not meant to be submitted directly - use submit_2050_pipeline.sh, which
# submits the baseline job first, then this array with
# --dependency=afterok:<baseline_job_id> and BASE_2050_CACHE exported to the
# actual cache path. Direct submission needs BASE_2050_CACHE set by hand:
#   BASE_2050_CACHE=results/2050_investment/baseline/base_solution_123.bin \
#     sbatch --export=ALL run_dhpc_2050_arms.slurm
# ===========================================================================
#SBATCH --job-name="mga-2050-arms"
#SBATCH --output="mga_2050_arms_%A_%a.out"
#SBATCH --partition=memory
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=15600
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-2

module load julia
module load gurobi/12.0.0

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd NearOptimalAlternatives.jl

(
  flock -x 9
  echo "[$(date)] task $SLURM_ARRAY_TASK_ID: ensuring project is instantiated"
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
) 9>"$HOME/.mga_julia_instantiate.lock"

if [ -z "$BASE_2050_CACHE" ]; then
  echo "ERROR: BASE_2050_CACHE is not set - point it at the baseline job's base_solution_*.bin"
  exit 1
fi

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "[$(date)] task 0 -> continuation-vs-SPORES"
    export CONT_2050_BASE_CACHE="$BASE_2050_CACHE"
    export CONT_2050_BACKEND=gurobi
    export N_DIR=5
    export N_BUDGET=6
    export EPS=0.1
    export CONT_2050_OUT="results/2050_investment/continuation"
    export CONT_2050_ID="$SLURM_ARRAY_JOB_ID"
    srun julia --project=. experiment_cont_spores_2050.jl
    ;;
  1)
    echo "[$(date)] task 1 -> algorithm-switch arms (N=5)"
    export ARMS_2050_BASE_CACHE="$BASE_2050_CACHE"
    export ARMS_2050_BACKEND=gurobi
    export ARMS_2050_NPOINTS="5"
    export ARMS_2050_EPS=0.1
    export ARMS_2050_REPEATS=1              # median-of-3 is likely too expensive at this scale; bump if budget allows
    export ARMS_2050_OUT="results/2050_investment/arms"
    export ARMS_2050_ID="${SLURM_ARRAY_JOB_ID}_n5"
    srun julia --project=. experiment_arms_2050.jl
    ;;
  2)
    echo "[$(date)] task 2 -> algorithm-switch arms (N=10)"
    export ARMS_2050_BASE_CACHE="$BASE_2050_CACHE"
    export ARMS_2050_BACKEND=gurobi
    export ARMS_2050_NPOINTS="10"
    export ARMS_2050_EPS=0.1
    export ARMS_2050_REPEATS=1
    export ARMS_2050_OUT="results/2050_investment/arms"
    export ARMS_2050_ID="${SLURM_ARRAY_JOB_ID}_n10"
    srun julia --project=. experiment_arms_2050.jl
    ;;
esac
echo "[$(date)] task $SLURM_ARRAY_TASK_ID done"
