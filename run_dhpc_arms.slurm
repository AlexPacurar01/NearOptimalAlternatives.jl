#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) SLURM job array for the exact-solver warm-start arms
# experiment (experiment_basis_arms.jl): cold barrier vs warm dual-simplex
# basis reuse vs cold dual simplex, per dataset, over the budget sweep.
#
# One ARRAY TASK = one dataset (5h smallest ... 1h largest), isolated on its
# own cores/memory so wall-clock timings and the peak-RSS memory measurement
# (hypothesis H3) are clean.
#
# BEFORE SUBMITTING: the data/1h-5h datasets use the OLD Tulipa schema.
# Make sure Project.toml pins TulipaEnergyModel back to 0.7.0 and re-resolve:
#   julia --project=. -e 'using Pkg; Pkg.add(name="TulipaEnergyModel", version="0.7.0"); Pkg.resolve()'
# (the small-case study temporarily upgraded it to 0.22 for Tiny/Norse).
#
# Submit from the directory that CONTAINS the project (e.g. $HOME):
#   sbatch NearOptimalAlternatives.jl/run_dhpc_arms.slurm
# Pilot first: sbatch --array=0 ... (5h only), check the output, then 0-4.
# ===========================================================================
#SBATCH --job-name="mga-arms"
#SBATCH --output="mga_arms_%A_%a.out"
#SBATCH --partition=memory
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=15600
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-4

module load julia
module load gurobi/12.0.0

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd NearOptimalAlternatives.jl

(
  flock -x 9
  echo "[$(date)] task $SLURM_ARRAY_TASK_ID: ensuring project is instantiated"
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
) 9>"$HOME/.mga_julia_instantiate.lock"

DATASETS=(5h 4h 3h 2h 1h)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

export ARMS_SOURCE=tulipa
export ARMS_DATASET=$DS
export ARMS_BACKEND=gurobi
export ARMS_NPOINTS="5,25"           # sparse + dense grid (H2: density helps warm arm)
export ARMS_EPS=0.1
export ARMS_REPEATS=3                # medians over repeats
export ARMS_OUT="results/basis_arms"
export ARMS_ID="arms_${DS}_${SLURM_ARRAY_JOB_ID}"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> dataset $DS, $SLURM_CPUS_PER_TASK threads"
srun julia --project=. experiment_basis_arms.jl
echo "[$(date)] done dataset $DS  (sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed for H3)"
