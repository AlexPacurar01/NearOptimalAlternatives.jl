# ===========================================================================
# Stage 3 of the 2050 Investment pipeline: aggregate the independently-
# produced outputs of the baseline + experiment-arm jobs (each ran on its own
# DHPC node, so nothing is merged yet) into one place for analysis.
#
# Concatenates every results/2050_investment/{continuation,arms}/*.csv into
# one combined file per kind, and writes a short headline-numbers report
# (base cost/time, arm speedups vs cold_barrier) alongside them. Does not
# recompute anything - purely a merge/summary pass over what's already on
# disk, safe to (re-)run any time after the arm jobs finish.
#
# Config via environment variables:
#   AGG_2050_DIR   results root (default results/2050_investment)
#
#   julia --project=. aggregate_2050_results.jl
# ===========================================================================
using Printf, Dates

# CSV/DataFrames aren't Project.toml deps in this repo (other scripts
# read/write CSV by hand); the concatenation below is plain line-based I/O so
# no extra dependency is needed.

env(k, d) = get(ENV, k, d)
const ROOT = env("AGG_2050_DIR", joinpath(@__DIR__, "results", "2050_investment"))
const BASEDIR = joinpath(ROOT, "baseline")
const CONTDIR = joinpath(ROOT, "continuation")
const ARMSDIR = joinpath(ROOT, "arms")

"Concatenate every CSV in `dir` matching `glob_prefix` into `out_path`,
keeping only the first header. Dependency-free (plain line I/O)."
function concat_csvs(dir, prefix, out_path)
    isdir(dir) || return 0
    files = sort(filter(f -> startswith(f, prefix) && endswith(f, ".csv"), readdir(dir)))
    isempty(files) && return 0
    header = nothing
    nrows = 0
    open(out_path, "w") do out
        for f in files
            lines = readlines(joinpath(dir, f))
            isempty(lines) && continue
            if header === nothing
                header = lines[1]
                println(out, "source_file,", header)
            end
            for line in lines[2:end]
                isempty(strip(line)) && continue
                println(out, f, ",", line)
                nrows += 1
            end
        end
    end
    return nrows
end

mkpath(ROOT)
println("[$(now())] aggregating 2050 Investment results from $ROOT")

# --- baseline: pick the most recent meta file + its cache -------------------
baseline_meta = nothing
baseline_cache = nothing
if isdir(BASEDIR)
    metas =
        sort(filter(f -> startswith(f, "meta_") && endswith(f, ".txt"), readdir(BASEDIR)))
    if !isempty(metas)
        baseline_meta = joinpath(BASEDIR, metas[end])
        run_id = replace(replace(metas[end], "meta_" => ""), ".txt" => "")
        cachef = joinpath(BASEDIR, "base_solution_$run_id.bin")
        isfile(cachef) && (baseline_cache = cachef)
    end
end

# --- continuation + arms: concat everything found ---------------------------
n_cont = concat_csvs(CONTDIR, "points_", joinpath(ROOT, "continuation_all.csv"))
n_arms_pts = concat_csvs(ARMSDIR, "points_", joinpath(ROOT, "arms_points_all.csv"))
n_arms_sum = concat_csvs(ARMSDIR, "arm_summary_", joinpath(ROOT, "arms_summary_all.csv"))

println("[$(now())] continuation points: $n_cont rows -> continuation_all.csv")
println("[$(now())] arms points:         $n_arms_pts rows -> arms_points_all.csv")
println("[$(now())] arms summaries:      $n_arms_sum rows -> arms_summary_all.csv")

# --- headline report ---------------------------------------------------------
open(joinpath(ROOT, "REPORT.txt"), "w") do io
    println(io, "2050 Investment - aggregated report")
    println(io, "generated=$(now())")
    println(io, "root=$ROOT")
    println(io)
    if baseline_meta !== nothing
        println(io, "--- baseline (", basename(baseline_meta), ") ---")
        for line in readlines(baseline_meta)
            println(io, "  ", line)
        end
    else
        println(io, "--- baseline: NOT FOUND under $BASEDIR ---")
    end
    println(io)
    println(io, "--- files ---")
    println(io, "  continuation_all.csv : $n_cont rows")
    println(io, "  arms_points_all.csv  : $n_arms_pts rows")
    println(io, "  arms_summary_all.csv : $n_arms_sum rows")
    if n_arms_sum > 0
        println(io)
        println(io, "  (see arms_summary_all.csv for median total/per-resolve time")
        println(io, "   per arm x grid size - that's where the cold_barrier vs")
        println(io, "   warm_simplex vs cold_simplex speedup comparison lives)")
    end
end
println("[$(now())] wrote $(joinpath(ROOT, "REPORT.txt"))")
println("[$(now())] DONE")
