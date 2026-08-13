# ===========================================================================
# Stage 1 of the 2050 Investment study: cold-solve the cost-minimum problem
# ONCE, record its wall-clock time, and cache the solution to disk so every
# later experiment arm (continuation, algorithm-switch/basis-reuse) can load
# x*/C* instantly instead of repeating an hours-long barrier solve.
#
# This is a single-node job - run it BEFORE the experiment arms and point
# BASE_2050_CACHE at its output. See run_dhpc_2050.slurm for the full,
# parallel-across-nodes pipeline (baseline -> arms -> aggregate).
#
# Config via environment variables:
#   BASE_2050_BACKEND       gurobi | highs           (default gurobi)
#   BASE_2050_NUMERICFOCUS  Gurobi NumericFocus 0-3   (default 3; data/5h
#                           needed 3 to avoid NUMERICAL_ERROR at this scale)
#   BASE_2050_BUILD_ONLY    true | false              (default false) - report
#                           nvars/ncons/nnz and STOP before the barrier solve.
#                           The barrier's Newton system (A*D*A' normal
#                           equations, refactored every iteration) is the real
#                           memory risk at this scale, NOT the LP itself - run
#                           a build-only pass first (cheap, minutes not hours)
#                           to see the problem size before committing a full
#                           20h job to a solve that might OOM partway through.
#   BASE_2050_OUT           output directory          (default results/2050_investment/baseline)
#   BASE_2050_ID            tag for output filenames   (default SLURM job id or "local")
#
#   JULIA_NUM_THREADS=16 julia -t 16 --project=. run_2050_baseline.jl
# ===========================================================================
using JuMP, Printf, Dates, Serialization
import MathOptInterface as MOI
import Gurobi, HiGHS
include(joinpath(@__DIR__, "dataset_2050_investment.jl"))

env(k, d) = get(ENV, k, d)
const BACKEND = Symbol(env("BASE_2050_BACKEND", "gurobi"))
const NUMERICFOCUS = parse(Int, env("BASE_2050_NUMERICFOCUS", "3"))
const BUILD_ONLY = parse(Bool, env("BASE_2050_BUILD_ONLY", "false"))
const OUTDIR =
    env("BASE_2050_OUT", joinpath(@__DIR__, "results", "2050_investment", "baseline"))
const RUN_ID = env("BASE_2050_ID", get(ENV, "SLURM_JOB_ID", "local"))
const THREADS = max(1, Threads.nthreads())

mkpath(OUTDIR)
const CACHE = joinpath(OUTDIR, "base_solution_$RUN_ID.bin")
const METAF = joinpath(OUTDIR, "meta_$RUN_ID.txt")

function optimizer()
    if BACKEND === :gurobi
        return optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag" => 0,
            "Threads" => THREADS,
            "Method" => 2,          # barrier
            "Crossover" => 0,       # interior point only - no vertex crossover
            "NumericFocus" => NUMERICFOCUS,
        )
    else
        return optimizer_with_attributes(
            HiGHS.Optimizer,
            "output_flag" => false,
            "threads" => THREADS,
            "solver" => "ipm",
            "run_crossover" => "off",
        )
    end
end

rss_gb() = Sys.maxrss() / 2^30   # peak resident set size so far, this process

println("[$(now())] building 2050 Investment model (unsolved) to report size...")
flush(stdout)
tbuild = @elapsed (model, target) = load_2050_investment(optimizer(); solve = false)
nvars = num_variables(model)
ncons = num_constraints(model; count_variable_in_set_constraints = false)
@printf(
    "[%s] build done in %.1fs: %d vars, %d cons, %d investment vars, peak RSS so far %.1f GB\n",
    now(),
    tbuild,
    nvars,
    ncons,
    length(target),
    rss_gb()
)
flush(stdout)

if BUILD_ONLY
    open(joinpath(OUTDIR, "meta_$(RUN_ID)_BUILD_ONLY.txt"), "w") do io
        println(io, "run_id=$RUN_ID  BUILD_ONLY=true")
        println(io, "datetime=$(now())")
        println(io, "host=$(gethostname())")
        println(io, "nvars=$nvars ncons=$ncons n_investment_vars=$(length(target))")
        println(io, "build_time_s=$tbuild")
        println(io, "peak_rss_gb=$(rss_gb())")
    end
    println("[$(now())] BASE_2050_BUILD_ONLY=true - stopping before the barrier solve.")
    println(
        "[$(now())] Check peak RSS above against the job's requested memory before running a full solve.",
    )
    exit(0)
end

println("[$(now())] cold-solving base cost-minimum ($BACKEND, threads=$THREADS)...")
flush(stdout)
tsolve = @elapsed JuMP.optimize!(model)
@assert is_solved_and_feasible(model) "base solve of 2050 Investment failed: $(termination_status(model))"
obj = objective_value(model)
solver_time = JuMP.solve_time(model)
gurobi_maxmem_gb =
    BACKEND === :gurobi ? (
        try
            MOI.get(model, Gurobi.ModelAttribute("MaxMemUsed"))
        catch
            NaN
        end
    ) : NaN
@printf(
    "[%s] solved in %.1fs (solver-reported %.1fs). C* = %.6g  peak RSS %.1f GB  Gurobi MaxMemUsed %.1f GB\n",
    now(),
    tsolve,
    solver_time,
    obj,
    rss_gb(),
    gurobi_maxmem_gb
)
flush(stdout)

all_v = all_variables(model)
vals = value.(all_v)
tidx = let d = Dict(v => i for (i, v) in enumerate(all_v))
    [d[v] for v in target]
end

serialize(
    CACHE,
    (
        objective = obj,
        values = vals,
        target_idx = tidx,
        nvars = nvars,
        ncons = ncons,
        backend = BACKEND,
        build_time_s = tbuild,
        solve_time_s = tsolve,
        solver_time_s = solver_time,
        peak_rss_gb = rss_gb(),
        gurobi_maxmem_gb = gurobi_maxmem_gb,
    ),
)
println("[$(now())] cached base solution -> $CACHE")

open(METAF, "w") do io
    println(io, "run_id=$RUN_ID")
    println(io, "datetime=$(now())")
    println(io, "host=$(gethostname())")
    println(io, "julia=$(VERSION)  threads=$THREADS")
    println(io, "backend=$BACKEND  numericfocus=$NUMERICFOCUS")
    println(io, "nvars=$nvars ncons=$ncons n_investment_vars=$(length(target))")
    println(io, "build_time_s=$tbuild")
    println(io, "solve_time_s=$tsolve  solver_time_s=$solver_time")
    println(io, "peak_rss_gb=$(rss_gb())  gurobi_maxmem_gb=$gurobi_maxmem_gb")
    println(io, "objective=$obj")
    println(io, "cache_file=$CACHE")
end
println("[$(now())] DONE. -> $METAF")
