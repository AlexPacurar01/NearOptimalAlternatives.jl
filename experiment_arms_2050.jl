# ===========================================================================
# 2050 Investment driver for the "faster methods that switch solver algorithm
# midway" experiment - i.e. the warm dual-simplex basis-reuse arm from
# experiment_basis_arms.jl, adapted to this dataset's scale. Same three arms,
# same budget-sweep design (the MGA sweep changes ONE right-hand side between
# solves - the textbook dual-simplex parametric-RHS case):
#
#   A  cold_barrier   N independent barrier solves (crossover off) - baseline
#   B  warm_simplex   barrier+crossover ONCE for a starting basis, then dual
#                     simplex re-solves that inherit the basis across budgets
#                     (this is the `reconfigure_solver!` callback documented
#                     on generate_alternatives_optimization! in
#                     src/generate-alternatives.jl - same idea, benchmarked
#                     directly here instead of through the public API)
#   C  cold_simplex   N dual-simplex solves with the solver state reset each
#                     time - the control separating "warm basis" from
#                     "simplex vs barrier"
#
# Stage 2b of the pipeline: loads the model UNSOLVED and reuses the cached
# x*/C* from run_2050_baseline.jl instead of repeating the cold barrier solve
# (see experiment_cont_spores_2050.jl for the same rationale).
#
# Config via environment variables:
#   ARMS_2050_BASE_CACHE  path to base_solution_*.bin (required)
#   ARMS_2050_BACKEND     gurobi | highs                    (default gurobi)
#   ARMS_2050_ARM         cold_barrier | warm_simplex | cold_simplex | "" for
#                         all three (default "") - at 2050 scale even ONE arm
#                         can consume the whole SLURM time budget (a single
#                         cold_barrier repeat at N=5 didn't finish in 20h), so
#                         run_dhpc_2050_arms.slurm gives each arm its OWN array
#                         task/node instead of running all three sequentially.
#   ARMS_2050_NPOINTS     comma list of grid sizes          (default "3")
#   ARMS_2050_EPS         relative cost budget               (default 0.1)
#   ARMS_2050_REPEATS     sweep repeats per arm (medians)    (default 1)
#   ARMS_2050_OUT         output directory (default results/2050_investment/arms)
#   ARMS_2050_ID          tag for output filenames (default SLURM job id or "local")
#
#   JULIA_NUM_THREADS=16 ARMS_2050_BASE_CACHE=results/2050_investment/baseline/base_solution_123.bin \
#     julia -t 16 --project=. experiment_arms_2050.jl
# ===========================================================================
using JuMP, LinearAlgebra, Statistics, Printf, Dates, Serialization
import MathOptInterface as MOI
import Gurobi, HiGHS
include(joinpath(@__DIR__, "dataset_2050_investment.jl"))

env(k, d) = get(ENV, k, d)
const THREADS = max(1, Threads.nthreads())
const BACKEND = Symbol(env("ARMS_2050_BACKEND", "gurobi"))
const ARM_FILTER = env("ARMS_2050_ARM", "")   # "" = run all three (small-scale use)
const NPOINTS = parse.(Int, split(env("ARMS_2050_NPOINTS", "3"), ","))
const EPS_MGA = parse(Float64, env("ARMS_2050_EPS", "0.1"))
const REPEATS = parse(Int, env("ARMS_2050_REPEATS", "1"))
const OUTDIR =
    env("ARMS_2050_OUT", joinpath(@__DIR__, "results", "2050_investment", "arms"))
const RUN_ID = env("ARMS_2050_ID", get(ENV, "SLURM_JOB_ID", "local"))
const CACHE_PATH = env("ARMS_2050_BASE_CACHE", "")
isempty(CACHE_PATH) && error(
    "ARMS_2050_BASE_CACHE is required - point it at a base_solution_*.bin from run_2050_baseline.jl",
)

mkpath(OUTDIR)
const POINTSF = joinpath(OUTDIR, "points_$RUN_ID.csv")
const SUMMARYF = joinpath(OUTDIR, "arm_summary_$RUN_ID.csv")

# NumericFocus=3: same fix data/5h needed for the Gurobi barrier at this kind
# of scale (see experiment_basis_arms.jl).
base_optimizer() =
    BACKEND === :gurobi ?
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Threads" => THREADS,
        "NumericFocus" => 3,
    ) :
    optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "threads" => THREADS)

"Point the attached solver at barrier (crossover optional) or dual simplex."
function set_algorithm!(model, alg::Symbol)
    if BACKEND === :gurobi
        if alg === :barrier
            set_optimizer_attribute(model, "Method", 2)
            set_optimizer_attribute(model, "Crossover", 0)
        elseif alg === :barrier_crossover
            set_optimizer_attribute(model, "Method", 2)
            set_optimizer_attribute(model, "Crossover", -1)   # default crossover
        else                                                   # :dual_simplex
            set_optimizer_attribute(model, "Method", 1)
        end
    else
        if alg in (:barrier, :barrier_crossover)
            set_optimizer_attribute(model, "solver", "ipm")
            set_optimizer_attribute(model, "run_crossover", "on")
        else
            set_optimizer_attribute(model, "solver", "simplex")
            set_optimizer_attribute(model, "simplex_strategy", 1)  # dual
        end
    end
end

"Discard solution/basis state so the next solve is genuinely cold."
function cold_reset!(model)
    try
        inner = JuMP.unsafe_backend(model)
        if BACKEND === :gurobi
            Gurobi.GRBreset(inner, 1)
        else
            HiGHS.Highs_clearSolver(inner.inner)
        end
    catch e
        @warn "cold_reset! failed (solve may inherit state)" exception = e
    end
end

per_solve_mem(model) =
    BACKEND === :gurobi ? (
        try
            MOI.get(model, Gurobi.ModelAttribute("MaxMemUsed"))
        catch
            NaN
        end
    ) : NaN

iters(model, getter) =
    try
        getter(model)
    catch
        -1
    end

# --- one arm: sweep the budget grid ------------------------------------------
"Same contract as experiment_basis_arms.jl's sweep_arm: `arm` is
:cold_barrier, :warm_simplex, or :cold_simplex."
function sweep_arm(model, w, all_vars, budgets, arm::Symbol)
    orig = objective_function(model)
    sense = objective_sense(model)
    off = JuMP.constant(orig)
    what = w ./ max(norm(w), eps())
    bcon = @constraint(model, orig <= budgets[1])
    @objective(model, Min, sum(w[i] * all_vars[i] for i in eachindex(w) if w[i] != 0))
    pts = NamedTuple[]
    for (k, B) in enumerate(budgets)
        set_normalized_rhs(bcon, B - off)
        if arm === :cold_barrier
            set_algorithm!(model, :barrier)
            cold_reset!(model)
        elseif arm === :cold_simplex
            set_algorithm!(model, :dual_simplex)
            cold_reset!(model)
        else # :warm_simplex
            if k == 1
                set_algorithm!(model, :barrier_crossover)
                cold_reset!(model)
            else
                set_algorithm!(model, :dual_simplex)       # keep basis, no reset
            end
        end
        optimize!(model)
        st = termination_status(model)
        t = solve_time(model)
        retried = false
        if st != MOI.OPTIMAL
            retried = true
            pre = BACKEND === :gurobi ? ("Presolve" => 0) : ("presolve" => "off")
            set_optimizer_attribute(model, pre.first, pre.second)
            cold_reset!(model)
            optimize!(model)
            st = termination_status(model)
            t += solve_time(model)
        end
        ok = st == MOI.OPTIMAL
        x = ok ? value.(all_vars) : fill(NaN, length(all_vars))
        push!(
            pts,
            (
                k = k,
                budget = B,
                status = string(st) * (retried ? "_RETRY" : ""),
                time = t,
                it_simplex = iters(model, simplex_iterations),
                it_barrier = iters(model, barrier_iterations),
                dobj = ok ? dot(what, x) : NaN,
                mem_gb = per_solve_mem(model),
            ),
        )
        if retried
            pre = BACKEND === :gurobi ? ("Presolve" => -1) : ("presolve" => "choose")
            set_optimizer_attribute(model, pre.first, pre.second)
        end
    end
    delete(model, bcon)
    set_objective_function(model, orig)
    set_objective_sense(model, sense)
    return pts
end

# --- run ----------------------------------------------------------------------
println("[$(now())] 2050 Investment basis arms: backend=$BACKEND threads=$THREADS")
println("[$(now())] loading cached baseline from $CACHE_PATH ...");
flush(stdout)
cache = deserialize(CACHE_PATH)
min_cost = cache.objective
println("  grids N=$(NPOINTS) eps=$EPS_MGA repeats=$REPEATS -> $OUTDIR")

model, target = load_2050_investment(base_optimizer(); solve = false)
all_vars = all_variables(model)
@assert length(all_vars) == cache.nvars "model size ($(length(all_vars))) doesn't match cached baseline ($(cache.nvars)) - rebuild the cache"

# HSJ-style diversity weight, computed from the CACHED base solution (no
# solve!, so we can't call `value.(...)` here) instead of a fresh solve.
w = zeros(length(all_vars))
for (jj, j) in enumerate(cache.target_idx)
    ub =
        has_upper_bound(target[jj]) && upper_bound(target[jj]) != 0 ?
        upper_bound(target[jj]) : 1.0
    w[j] = cache.values[j] / ub
end
println(
    "  model: $(length(all_vars)) vars, cached min cost $(round(min_cost, sigdigits=6))",
)

open(POINTSF, "w") do io
    println(
        io,
        "run_id,dataset,backend,npoints,arm,repeat,k,budget,status,solve_time_s," *
        "simplex_iters,barrier_iters,dobj,gurobi_maxmem_gb",
    )
end
open(SUMMARYF, "w") do io
    println(
        io,
        "run_id,dataset,backend,npoints,arm,median_total_s,median_per_resolve_s," *
        "max_dobj_reldiff_vs_A,n_failed,n_retried,peak_rss_gb",
    )
end

const ALL_ARMS = (:cold_barrier, :warm_simplex, :cold_simplex)
const ARMS = isempty(ARM_FILTER) ? ALL_ARMS : (Symbol(ARM_FILTER),)
@assert all(a -> a in ALL_ARMS, ARMS) "ARMS_2050_ARM=$ARM_FILTER must be one of $ALL_ARMS"
const DATASET_TAG = "2050_Investment"
for N in NPOINTS
    budgets = [min_cost + EPS_MGA * abs(min_cost) * i / N for i = 1:N]
    ref_dobj = nothing
    for arm in ARMS
        totals = Float64[]
        last_pts = nothing
        for rep = 1:REPEATS
            pts = sweep_arm(model, w, all_vars, budgets, arm)
            push!(totals, sum(p.time for p in pts))
            last_pts = pts
            open(POINTSF, "a") do io
                for p in pts
                    println(
                        io,
                        @sprintf(
                            "%s,%s,%s,%d,%s,%d,%d,%.8g,%s,%.4f,%d,%d,%.8g,%.3f",
                            RUN_ID,
                            DATASET_TAG,
                            BACKEND,
                            N,
                            arm,
                            rep,
                            p.k,
                            p.budget,
                            p.status,
                            p.time,
                            p.it_simplex,
                            p.it_barrier,
                            p.dobj,
                            p.mem_gb
                        )
                    )
                end
            end
            println("[$(now())] N=$N arm=$arm rep=$rep done")
            flush(stdout)
        end
        arm === :cold_barrier && (ref_dobj = [p.dobj for p in last_pts])
        nfail = count(p -> !(p.status in ("OPTIMAL", "OPTIMAL_RETRY")), last_pts)
        nretry = count(p -> endswith(p.status, "_RETRY"), last_pts)
        reldiff = NaN
        if ref_dobj !== nothing
            diffs = [
                abs(p.dobj - ref_dobj[p.k]) / max(abs(ref_dobj[p.k]), 1e-9) for
                p in last_pts if isfinite(p.dobj) && isfinite(ref_dobj[p.k])
            ]
            reldiff = isempty(diffs) ? NaN : maximum(diffs)
        end
        resolves = [p.time for p in last_pts if !(arm === :warm_simplex && p.k == 1)]
        rss = Sys.maxrss() / 2^30
        @printf(
            "  N=%-3d %-13s total=%9.2fs  per-resolve=%8.3fs  dobj reldiff vs A=%.2e  failed=%d retried=%d of %d  peakRSS=%.1fGB\n",
            N,
            arm,
            median(totals),
            median(resolves),
            reldiff,
            nfail,
            nretry,
            length(last_pts),
            rss
        )
        open(SUMMARYF, "a") do io
            println(
                io,
                @sprintf(
                    "%s,%s,%s,%d,%s,%.4f,%.4f,%.3e,%d,%d,%.2f",
                    RUN_ID,
                    DATASET_TAG,
                    BACKEND,
                    N,
                    arm,
                    median(totals),
                    median(resolves),
                    reldiff,
                    nfail,
                    nretry,
                    rss
                )
            )
        end
    end
end
println("[$(now())] DONE -> $SUMMARYF")
