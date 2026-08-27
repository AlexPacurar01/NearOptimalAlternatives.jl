#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) SLURM job ARRAY: 2050 Investment corrector-arms study.
#
# One ARRAY TASK = one arm, each on its own node, each building its OWN
# model and doing its OWN base cost-min solve from scratch - no shared
# cache, no shared state between tasks. This is deliberate: it makes all
# five runs fully independent measurements (no arm's execution can leave
# state that influences another's), at the cost of each task repeating the
# ~106GB-peak model build and one extra barrier solve. That cost is
# accepted here for methodological soundness; see experiment_2050_corrector_arms.jl.
# Every arm uses only the package's public API (generate_alternatives_
# optimization!/generate_alternatives_arclength! plus reconfigure_solver!) -
# no hand-rolled solve loop - so the five are a fair, apples-to-apples
# comparison of the same real code path under different solver configurations.
#
#   task 0  regular            all_barrier_no_crossover (every alternative:
#                               barrier, no crossover - the plain baseline)
#   task 1  exact               all_barrier_with_crossover (barrier+crossover
#                               pre-step, then every alternative stays on
#                               barrier+crossover - the "clean vertex every
#                               time" reference, and measures crossover
#                               overhead directly against task 0's base solve)
#   task 2  warm                crossover_then_warm_primal_simplex (same
#                               pre-step, then generate_alternatives_
#                               optimization!'s reconfigure_solver! hook
#                               switches to PRIMAL simplex for alternatives
#                               2+ - only the objective changes between
#                               alternatives here, the textbook primal case)
#   task 3  continuation        arclength_all_barrier_no_crossover
#                               (generate_alternatives_arclength!, single
#                               direction, barrier, no crossover throughout -
#                               the arclength-side counterpart of task 0)
#   task 4  warm_continuation   arclength_warm_dual_simplex (same pre-step,
#                               generate_alternatives_arclength! with
#                               reconfigure_solver! switching to DUAL simplex
#                               after its first point - only the budget moves
#                               within a direction, the textbook dual case)
#
# Resource request: 48 cores (96 threads via hyperthreading) on compute-p2
# (384GB nodes) - the full reference machine (same one run_dhpc_2050_baseline.slurm
# targets), well within the Education account's MaxCPUsPerJob=64 (cores, not
# threads - confirmed by the sacctmgr output, so no scaling down needed at
# all). Measured peak RSS on that reference machine was ~106GB while building
# the JuMP model (the dominant peak) and
# ~21GB once a barrier solve is running (the solver's own reported working-set
# is much smaller still, ~7GB) - so 120GB is requested here: a bit more than
# the ~106GB build-phase peak actually needs, not the much larger padding a
# full-node request would give.
#
# Standalone job - no dependency on run_dhpc_2050_baseline.slurm and no
# BASE_2050_CACHE needed, unlike run_dhpc_2050_arms.slurm's arms (this study
# intentionally does NOT reuse that cache - see experiment_2050_corrector_arms.jl
# for why). Submit directly:
#   sbatch NearOptimalAlternatives.jl/run_dhpc_2050_corrector_arms.slurm
# ===========================================================================
#SBATCH --job-name="mga-2050-corrector"
#SBATCH --output="mga_2050_corrector_%A_%a.out"
#SBATCH --partition=compute-p2
#SBATCH --time=23:59:00                     # max within the Education-account 24h cap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                  # 48 CORES = the full reference machine, well within the Education account's MaxCPUsPerJob=64 (cores, not threads)
#SBATCH --mem-per-cpu=2560M                 # 48 * 2560M = 120G: peak (build phase) ~106GB + a bit of headroom
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-4

module load julia
module load gurobi/12.0.0

# SLURM_CPUS_PER_TASK is in CORES (48); the reference machine exposes 2
# hardware threads/core via hyperthreading, so the job should see 96 logical
# CPUs - hence *2 here rather than using SLURM_CPUS_PER_TASK directly, which
# would silently leave half the reserved hardware idle. ASSUMPTION: confirm
# this cluster actually exposes both hyperthreads per allocated core to the
# job (e.g. check `nproc` inside a quick interactive/test job) before relying
# on it for a real run - if it doesn't, drop the *2 and use 48 directly.
export JULIA_NUM_THREADS=$((SLURM_CPUS_PER_TASK * 2))

cd NearOptimalAlternatives.jl

(
  flock -x 9
  echo "[$(date)] task $SLURM_ARRAY_TASK_ID: ensuring project is instantiated"
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
) 9>"$HOME/.mga_julia_instantiate.lock"

if [ ! -d "data/2050_Investment" ]; then
  echo "[$(date)] data/2050_Investment missing - running the schema converter"
  julia --project=. convert_2050_investment.jl
fi

# K=15: from the first real run, "regular" averaged ~136 min/alternative and
# "continuation" ~158 min/point (both without crossover) - far slower than the
# ~40 min/alternative estimate this comment used to cite. At that rate 15
# alternatives for regular/exact/warm (~34h) will NOT fit the 24h limit, but
# results are written to disk after every alternative, so a job killed at the
# time limit still keeps whatever it completed (regular/exact/warm reached
# 5/5 or 0/5 respectively in the first run) - this just aims for more points
# than 24h can hold, on purpose, rather than under-using the time budget.
# NBUDGET stays at 5 for continuation/warm_continuation for now (that arm
# already used ~14.75h/24h at n_budget=5; the smaller sample is intentional
# while K is being pushed up on the other arms).
export ARMS2050_K=15
export ARMS2050_NBUDGET=5
export ARMS2050_EPS=0.1
export ARMS2050_NUMERICFOCUS=3
export ARMS2050_OUT="results/2050_investment/corrector_arms"
export ARMS2050_ID="$SLURM_ARRAY_JOB_ID"

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "[$(date)] task 0 -> regular (all_barrier_no_crossover)"
    export ARMS2050_ARM="regular"
    ;;
  1)
    echo "[$(date)] task 1 -> exact (all_barrier_with_crossover)"
    export ARMS2050_ARM="exact"
    ;;
  2)
    echo "[$(date)] task 2 -> warm (crossover_then_warm_primal_simplex)"
    export ARMS2050_ARM="warm"
    ;;
  3)
    echo "[$(date)] task 3 -> continuation (arclength_all_barrier_no_crossover)"
    export ARMS2050_ARM="continuation"
    ;;
  4)
    echo "[$(date)] task 4 -> warm_continuation (arclength_warm_dual_simplex)"
    export ARMS2050_ARM="warm_continuation"
    ;;
esac

srun julia --project=. experiment_2050_corrector_arms.jl
echo "[$(date)] task $SLURM_ARRAY_TASK_ID ($ARMS2050_ARM) done"
