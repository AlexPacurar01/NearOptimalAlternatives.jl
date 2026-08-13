# ===========================================================================
# 2050 Investment driver for the continuation-vs-SPORES suite (exact,
# Gurobi). Reuses the core functions from experiment_continuation_spores.jl.
#
# This is Stage 2a of the pipeline: it loads the model UNSOLVED and reuses
# the cached x*/C* from run_2050_baseline.jl (CONT_2050_BASE_CACHE) instead of
# repeating the cold barrier solve - at this scale (8760h, no rep-period
# compression) that solve alone can dominate the wall-clock budget of every
# other arm if repeated. Writes each point to CSV as it is found (flushed),
# since the sweep is slow and may run close to the SLURM wall-clock limit.
#
# Config via environment variables:
#   CONT_2050_BASE_CACHE  path to base_solution_*.bin from run_2050_baseline.jl
#                         (required)
#   CONT_2050_BACKEND     gurobi | highs                    (default gurobi)
#   N_DIR                 number of SPORES directions       (default 5)
#   N_BUDGET              budget-sweep points per direction (default 6)
#   EPS                   relative cost budget               (default 0.1)
#   CONT_2050_OUT         output directory (default results/2050_investment/continuation)
#   CONT_2050_ID          tag for output filenames (default SLURM job id or "local")
#
#   JULIA_NUM_THREADS=16 CONT_2050_BASE_CACHE=results/2050_investment/baseline/base_solution_123.bin \
#     julia -t 16 --project=. experiment_cont_spores_2050.jl
# ===========================================================================
using JuMP, LinearAlgebra, Printf, Serialization, Dates
import Gurobi, HiGHS
include(joinpath(@__DIR__, "experiment_continuation_spores.jl"))   # core funcs + bench_common
include(joinpath(@__DIR__, "dataset_2050_investment.jl"))

env(k, d) = get(ENV, k, d)
const THREADS = max(1, Threads.nthreads())
const BACKEND = Symbol(env("CONT_2050_BACKEND", "gurobi"))
const N_DIR = parse(Int, env("N_DIR", "5"))
const N_BUDGET = parse(Int, env("N_BUDGET", "6"))
const EPS = parse(Float64, env("EPS", "0.1"))
const OUTDIR =
    env("CONT_2050_OUT", joinpath(@__DIR__, "results", "2050_investment", "continuation"))
const RUN_ID = env("CONT_2050_ID", get(ENV, "SLURM_JOB_ID", "local"))
const CACHE_PATH = env("CONT_2050_BASE_CACHE", "")
isempty(CACHE_PATH) && error(
    "CONT_2050_BASE_CACHE is required - point it at a base_solution_*.bin from run_2050_baseline.jl",
)
mkpath(OUTDIR)

optimizer() =
    BACKEND === :gurobi ?
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Threads" => THREADS,
        "Method" => 2,
        "Crossover" => 0,
        "NumericFocus" => 3,
    ) :
    optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "threads" => THREADS)

println("[$(now())] loading cached baseline from $CACHE_PATH ...");
flush(stdout)
cache = deserialize(CACHE_PATH)
println(
    "[$(now())] cached C*=$(cache.objective)  (built in $(cache.build_time_s)s, solved in $(cache.solve_time_s)s)",
)

println("[$(now())] building 2050 Investment model (unsolved) ...");
flush(stdout)
model, target = load_2050_investment(optimizer(); solve = false)
allv = all_variables(model)
idx = Dict(v => i for (i, v) in enumerate(allv))
inv_idx = [idx[v] for v in target]
ub_inv = [
    (has_upper_bound(v) && isfinite(upper_bound(v)) && upper_bound(v) > 0) ?
    upper_bound(v) : 1.0 for v in target
]
@assert length(allv) == cache.nvars "model size ($(length(allv))) doesn't match cached baseline ($(cache.nvars)) - rebuild the cache"
xstar = cache.values
Cstar = cache.objective
@printf(
    "2050 Investment: %d vars | %d investment vars | N_DIR=%d N_BUDGET=%d EPS=%g\n",
    length(allv),
    length(inv_idx),
    N_DIR,
    N_BUDGET,
    EPS
);
flush(stdout)

# Precompute the cost coefficient vector once (cheap: one pass over allv) so
# every recorded point is just a dot product, not a per-point coefficient scan
# over a model with millions of variables.
cost_expr0 = objective_function(model)
kconst = JuMP.constant(cost_expr0)
cvec = [JuMP.coefficient(cost_expr0, v) for v in allv]

# Live CSV: written incrementally so a timeout/preemption still leaves usable
# partial results. Matches the schema of the 5h front CSVs (method label +
# direction/budget index + cost + diversity), plus a wall-clock per point.
csvpath = joinpath(OUTDIR, "points_$RUN_ID.csv")
t = @elapsed open(csvpath, "w") do io
    println(io, "method,direction,budget_idx,cost,divobj,wall_s")
    flush(io)
    t0 = time()
    run_suite(
        model,
        allv,
        inv_idx,
        ub_inv;
        n_dir = N_DIR,
        eps = EPS,
        n_budget = N_BUDGET,
        base_solution = (Cstar, xstar),
        interleaved = true,   # sweep each direction right after its anchor - see run_suite docstring
        point_sink = (method, k, bi, sol, w) -> begin
            wk = w ./ max(norm(w), 1e-12)
            cst = dot(cvec, sol) + kconst
            @printf(
                io,
                "%s,%d,%d,%.10g,%.10g,%.4f\n",
                method,
                k,
                bi,
                cst,
                dot(wk, sol),
                time() - t0
            )
            flush(io)
        end,
    )
end
@printf("\ntotal suite wall: %.1fs\n", t)
println("Wrote $csvpath")
println("[$(now())] DONE");
flush(stdout)
