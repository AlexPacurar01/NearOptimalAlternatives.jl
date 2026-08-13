#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) job array: RAW (no-repair) first-order MGA runs.
#
# Same as run_dhpc.slurm but with operational recovery OFF and only the
# full-space first-order solvers. This measures the *raw* first-order points
# (no repair barrier solve): their wall-clock and, crucially, their original-
# unit constraint violation (`max_violation`) - i.e. how "tol-feasible" they
# are. Pair with run_dhpc.slurm (repaired) to quantify what the repair costs.
#
# Recovery/warm-start only affect the gradient methods, so the exact methods
# (Spores/HSJ/oracle) and the convergence trace are NOT repeated here - they
# come from run_dhpc.slurm. This run is gradient-only and cheap.
#
# Submit from the directory CONTAINING the project:
#   sbatch NearOptimalAlternatives.jl/run_dhpc_raw.slurm
# ===========================================================================
#SBATCH --job-name="mga-raw"
#SBATCH --output="mga_raw_%A_%a.out"
#SBATCH --partition=memory
#SBATCH --time=08:00:00                      # gradient-only -> much shorter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=15600
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-4

module load julia
module load gurobi/12.0.0

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd NearOptimalAlternatives.jl

# Instantiate the project once; flock serialises the array tasks so they don't
# clash writing the shared depot (idempotent no-op once done).
(
  flock -x 9
  echo "[$(date)] task $SLURM_ARRAY_TASK_ID: ensuring project is instantiated"
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
) 9>"$HOME/.mga_julia_instantiate.lock"

DATASETS=(5h 4h 3h 2h 1h)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

export RUN_SOURCE=tulipa
export RUN_DATASET=$DS
export RUN_BACKEND=gurobi
export RUN_METHODS="alm_lbfgs,penalty,pdhg"   # gradient full-space only
export RUN_NALTS=5
export RUN_REPEATS=3
export RUN_WARMSTART=true
export RUN_RECOVERY=false                      # <-- raw points, no repair
export RUN_TRACE=false                         # trace already in the main run
export RUN_OUT="results/experiments"
export RUN_ID="raw_${DS}_${SLURM_ARRAY_JOB_ID}"

echo "[$(date)] RAW task $SLURM_ARRAY_TASK_ID -> dataset $DS, $SLURM_CPUS_PER_TASK threads"
srun julia --project=. run_experiments.jl
echo "[$(date)] done dataset $DS"
