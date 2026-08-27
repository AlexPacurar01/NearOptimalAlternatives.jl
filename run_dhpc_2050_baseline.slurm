#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) SLURM job: Stage 1 of the 2050 Investment pipeline.
#
# Single node/task: cold-solves the cost-minimum problem ONCE (a full
# calendar year, 8760 hourly timesteps, no representative-period compression
# - the largest, most memory-heavy model in the study) and caches x*/C* to
# disk. Every later experiment arm (continuation, algorithm-switch) loads
# this cache instead of repeating the solve.
#
# Not meant to be submitted directly - use submit_2050_pipeline.sh, which
# submits this job first and wires its output cache path into the dependent
# array/aggregate jobs. Direct submission works too (e.g. for a standalone
# re-run): sbatch run_dhpc_2050_baseline.slurm
#
# PILOT FIRST, recommended: the barrier method's Newton system (the A*D*A'
# normal-equations factorization, refactored every iteration) is the real
# memory risk at this scale, not the LP itself - and it was never validated
# locally (this machine ran out of RAM just building the JuMP model, before
# any solve was attempted). Run a build-only pass first to see nvars/ncons
# and peak RSS cheaply (minutes, not hours) before trusting this job's
# request on a full barrier solve:
#   sbatch --export=ALL,BASE_2050_BUILD_ONLY=true --time=01:00:00 \
#     run_dhpc_2050_baseline.slurm
#
# Resource request below targets 64 threads - the Education account's
# MaxCPUsPerJob=64 caps this below the reference machine (48 cores / 96
# threads, 384GB RAM, barrier + no crossover) the memory numbers come from.
# Measured peak RSS on that reference machine was ~106GB while building the
# JuMP model (the dominant peak) and
# ~21GB once the barrier solve itself is running (the solver's own reported
# working-set is smaller still, ~7GB), so 120GB is requested here: a bit
# more than the build-phase peak actually needs, not a full-node request.
# compute-p2 is DelftBlue's 48-core/384GB partition, matching that reference
# machine.
# ===========================================================================
#SBATCH --job-name="mga-2050-base"
#SBATCH --output="mga_2050_base_%j.out"
#SBATCH --partition=compute-p2               # 48-core/384GB nodes, matches the reference config
#SBATCH --time=20:00:00                      # < 24h Education-account limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                   # capped by the Education account's MaxCPUsPerJob=64 (below the reference machine's 96 threads)
#SBATCH --mem-per-cpu=1920M                  # 64 * 1920M = 120G: peak (build phase) ~106GB + a bit of headroom
#SBATCH --account=Education-EEMCS-MSc-DSAIT

module load julia
module load gurobi/12.0.0

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd NearOptimalAlternatives.jl

(
  flock -x 9
  echo "[$(date)] ensuring project is instantiated"
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
) 9>"$HOME/.mga_julia_instantiate.lock"

# One-time: convert the raw export if the flat, loadable folder isn't there yet.
if [ ! -d "data/2050_Investment" ]; then
  echo "[$(date)] data/2050_Investment missing - running the schema converter"
  julia --project=. convert_2050_investment.jl
fi

export BASE_2050_BACKEND=gurobi
export BASE_2050_NUMERICFOCUS=3
export BASE_2050_OUT="results/2050_investment/baseline"
export BASE_2050_ID="$SLURM_JOB_ID"
export BASE_2050_BUILD_ONLY="${BASE_2050_BUILD_ONLY:-false}"   # set true via --export for a pilot run

echo "[$(date)] baseline solve starting, $SLURM_CPUS_PER_TASK threads"
srun julia --project=. run_2050_baseline.jl
echo "[$(date)] baseline done -> $BASE_2050_OUT/base_solution_$SLURM_JOB_ID.bin"
