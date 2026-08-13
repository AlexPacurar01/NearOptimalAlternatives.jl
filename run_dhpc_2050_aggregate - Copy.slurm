#!/bin/bash
# ===========================================================================
# DelftBlue (DHPC) SLURM job: Stage 3 of the 2050 Investment pipeline.
#
# Lightweight merge pass over whatever the baseline + arms jobs already wrote
# to results/2050_investment/ - see aggregate_2050_results.jl. Not meant to
# be submitted directly; submit_2050_pipeline.sh chains it after the arms
# array via --dependency=afterany so it runs once those finish (successfully
# or not - a partial merge from whichever arms completed is still useful).
# ===========================================================================
#SBATCH --job-name="mga-2050-agg"
#SBATCH --output="mga_2050_agg_%j.out"
#SBATCH --partition=compute-p1              # trivial footprint (4GB/cpu) - DHPC rejects that on
                                             # the memory partition, which is reserved for larger requests
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --account=Education-EEMCS-MSc-DSAIT

module load julia

cd NearOptimalAlternatives.jl

export AGG_2050_DIR="results/2050_investment"
echo "[$(date)] aggregating -> $AGG_2050_DIR"
julia --project=. aggregate_2050_results.jl
echo "[$(date)] done -> $AGG_2050_DIR/REPORT.txt"
