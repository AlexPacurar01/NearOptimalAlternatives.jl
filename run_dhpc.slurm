#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) SLURM job array for the MGA solver benchmark.
#
# One ARRAY TASK = one dataset, running in isolation on its own cores/memory so
# wall-clock timings are clean. The scheduler runs the datasets in parallel
# across the cluster -> fast overall, with no solver contention.
#
# Submit from the directory that CONTAINS the project (e.g. $HOME), like:
#   mkdir -p NearOptimalAlternatives.jl/results/experiments
#   sbatch NearOptimalAlternatives.jl/run_dhpc.slurm
#
# ONE-TIME SETUP on a login node (after `module load julia gurobi/12.0.0`):
#   cd NearOptimalAlternatives.jl
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# Education account: max 24h wall-clock per task. Watch: squeue --me
# ===========================================================================
#SBATCH --job-name="mga-main"
#SBATCH --output="mga_main_%A_%a.out"       # %A=array job id, %a=array task index
#SBATCH --partition=memory                  # 750GB/1.5TB nodes (largest models)
#SBATCH --time=20:00:00                      # < 24h Education-account limit
#SBATCH --ntasks=1                           # single multithreaded process
#SBATCH --cpus-per-task=16                   # 16 cores (solver + matvec threads)
#SBATCH --mem-per-cpu=15600                  # ~15.6 GB/core (MB) -> ~250 GB total
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-4                          # one task per dataset below

module load julia
module load gurobi/12.0.0

# Match Julia threads to the SLURM core allocation.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd NearOptimalAlternatives.jl

# --- instantiate the project (once) ----------------------------------------
# All array tasks start together, so serialise with flock on a shared-home
# lock file: the first task installs + precompiles the deps, the rest find the
# depot ready and their instantiate/precompile become quick no-ops. Idempotent,
# so safe to keep across resubmits.
(
  flock -x 9
  echo "[$(date)] task $SLURM_ARRAY_TASK_ID: ensuring project is instantiated"
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
) 9>"$HOME/.mga_julia_instantiate.lock"

# --- dataset for this array task -------------------------------------------
DATASETS=(5h 4h 3h 2h 1h)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# --- experiment configuration (override any RUN_* here) --------------------
export RUN_SOURCE=tulipa
export RUN_DATASET=$DS
export RUN_BACKEND=gurobi            # Gurobi 12 barrier (license via the module)
export RUN_METHODS="Spores,HSJ,alm_lbfgs,penalty,pdhg,oracle"
export RUN_NALTS=5
export RUN_REPEATS=3                  # median over repeats for robust timings
export RUN_WARMSTART=true
export RUN_RECOVERY=true              # repaired (exactly feasible) alternatives
export RUN_TRACE=true                 # convergence trace (warm+cold) lives here
export RUN_OUT="results/experiments"
export RUN_ID="main_${DS}_${SLURM_ARRAY_JOB_ID}"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> dataset $DS, $SLURM_CPUS_PER_TASK threads"
srun julia --project=. run_experiments.jl
echo "[$(date)] done dataset $DS"
