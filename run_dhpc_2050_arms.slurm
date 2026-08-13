#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) SLURM job ARRAY: Stage 2 of the 2050 Investment pipeline.
#
# One ARRAY TASK = one experiment arm, each running in isolation on its own
# node so the scheduler runs them in parallel with no solver contention:
#
#   task 0  continuation-vs-SPORES  (experiment_cont_spores_2050.jl)
#   task 1  algorithm-switch arms: cold_barrier   (experiment_arms_2050.jl)
#   task 2  algorithm-switch arms: warm_simplex   (experiment_arms_2050.jl)
#   task 3  algorithm-switch arms: cold_simplex   (experiment_arms_2050.jl)
#
# Each arm gets its OWN task/node and its own full time budget - a first
# attempt that ran all three arms sequentially in one task didn't even finish
# a single repeat of the FIRST arm (cold_barrier, N=5) inside 20h, so bundling
# them was actively harmful: it meant a slow/stuck arm (most likely
# cold_simplex - dual simplex on an 11.7M-variable network LP is an unknown
# quantity at this scale) blocked the other two from reporting anything at
# all. Splitting them out means cold_barrier/warm_simplex still report even
# if cold_simplex is intractable here - which is itself a useful finding.
#
# No per-solve Gurobi TimeLimit is set: since each arm is isolated on its own
# node, one slow solve can't block another arm's results the way it could
# when they shared one task, so there's no need to truncate a solve early -
# and the whole point of this run is to measure how long each arm's solves
# actually take, not how they perform under an artificial cap. The only
# remaining bound is this job's own SLURM --time below.
#
# All four load the SAME cached baseline (x*/C*) produced by
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
#SBATCH --time=23:59:00                     # max within the Education-account 24h cap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=15600
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-3

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
    echo "[$(date)] task 1 -> algorithm-switch arms: cold_barrier"
    export ARMS_2050_BASE_CACHE="$BASE_2050_CACHE"
    export ARMS_2050_BACKEND=gurobi
    export ARMS_2050_ARM="cold_barrier"
    export ARMS_2050_NPOINTS="3"
    export ARMS_2050_EPS=0.1
    export ARMS_2050_REPEATS=1              # single sample at this scale - see experiment_arms_2050.jl
    export ARMS_2050_OUT="results/2050_investment/arms"
    export ARMS_2050_ID="${SLURM_ARRAY_JOB_ID}_coldbarrier"
    srun julia --project=. experiment_arms_2050.jl
    ;;
  2)
    echo "[$(date)] task 2 -> algorithm-switch arms: warm_simplex"
    export ARMS_2050_BASE_CACHE="$BASE_2050_CACHE"
    export ARMS_2050_BACKEND=gurobi
    export ARMS_2050_ARM="warm_simplex"
    export ARMS_2050_NPOINTS="3"
    export ARMS_2050_EPS=0.1
    export ARMS_2050_REPEATS=1
    export ARMS_2050_OUT="results/2050_investment/arms"
    export ARMS_2050_ID="${SLURM_ARRAY_JOB_ID}_warmsimplex"
    srun julia --project=. experiment_arms_2050.jl
    ;;
  3)
    echo "[$(date)] task 3 -> algorithm-switch arms: cold_simplex"
    export ARMS_2050_BASE_CACHE="$BASE_2050_CACHE"
    export ARMS_2050_BACKEND=gurobi
    export ARMS_2050_ARM="cold_simplex"
    export ARMS_2050_NPOINTS="3"
    export ARMS_2050_EPS=0.1
    export ARMS_2050_REPEATS=1
    export ARMS_2050_OUT="results/2050_investment/arms"
    export ARMS_2050_ID="${SLURM_ARRAY_JOB_ID}_coldsimplex"
    srun julia --project=. experiment_arms_2050.jl
    ;;
esac
echo "[$(date)] task $SLURM_ARRAY_TASK_ID done"
