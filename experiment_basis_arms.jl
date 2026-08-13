# ===========================================================================
# Exact-solver warm-start arms: does SIMPLEX BASIS REUSE make the budget sweep
# cheap at real scale, where first-order correctors and IPM warm starts do not?
#
# The MGA budget sweep changes ONE right-hand side between solves - the
# textbook dual-simplex re-optimization case (parametric LP in the RHS). Per
# dataset this script runs three arms over the same interior budget grid:
#   A  cold_barrier   N independent barrier solves (crossover off) - baseline
#   B  warm_simplex   barrier+crossover ONCE for a starting basis, then dual
#                     simplex re-solves that inherit the basis across budgets
#   C  cold_simplex   N dual-simplex solves with the solver state reset each
#                     time - the control separating "warm basis" from
#                     "simplex vs barrier"
#
# Pre-registered hypotheses (see paper_redraft.tex, Future work):
#   H1  a warm dual-simplex re-solve costs a small fraction of a cold barrier
#   H2  total(B) < total(A) beyond a modest N, growing with grid density
#   H3  peak barrier memory at hourly resolution fits one node (report maxrss)
#
# Per solve we record the solver-reported time, simplex/barrier iterations,
# the diversity objective (cross-checked across arms - all three are exact
# methods, so any disagreement flags a failed solve, not a quality trade-off),
# and Gurobi's MaxMemUsed plus the process peak RSS per arm.
#
# Config via environment variables:
#   ARMS_SOURCE   tulipa | synth          (default tulipa)
#   ARMS_DATASET  data/<x> folder / synth size      (default 5h)
#   ARMS_BACKEND  gurobi | highs                    (default gurobi)
#   ARMS_NPOINTS  comma list of grid sizes          (default "5,25")
#   ARMS_EPS      relative cost budget              (default 0.1)
#   ARMS_REPEATS  sweep repeats per arm (medians)   (default 3)
#   ARMS_OUT      output directory                  (default results/basis_arms)
#   ARMS_ID       tag for output filenames          (default SLURM task or "local")
#
# NOTE: the tulipa source needs the OLD schema (TulipaEnergyModel 0.7.0) for
# data/1h-5h. Downgrade the compat pin before running on the cluster.
#
#   JULIA_NUM_THREADS=16 julia -t 16 --project=. experiment_basis_arms.jl
# ===========================================================================
using JuMP, LinearAlgebra, Random, Statistics, Printf, Logging, Dates

env(k, d) = get(ENV, k, d)
const SOURCE = Symbol(env("ARMS_SOURCE", "tulipa"))
const DATASET = env("ARMS_DATASET", SOURCE === :synth ? "small" : "5h")
const BACKEND = Symbol(env("ARMS_BACKEND", "gurobi"))
const NPOINTS = parse.(Int, split(env("ARMS_NPOINTS", "5,25"), ","))
const EPS_MGA = parse(Float64, env("ARMS_EPS", "0.1"))
const REPEATS = parse(Int, env("ARMS_REPEATS", "3"))
const OUTDIR = env("ARMS_OUT", joinpath(pwd(), "results", "basis_arms"))
const RUN_ID = env("ARMS_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "local"))
const THREADS = max(1, Threads.nthreads())

BACKEND === :gurobi ? (using Gurobi) : (using HiGHS)
if SOURCE === :tulipa
    import TulipaEnergyModel as TEM
elseif SOURCE === :tulipa022
    # Small public instances (Tiny, Norse) through the modern Tulipa API - the
    # same loader as the small-case corrector study, so results are comparable.
    include("smallcase_tulipa.jl")
end

mkpath(OUTDIR)
const POINTSF = joinpath(OUTDIR, "points_$RUN_ID.csv")
const SUMMARYF = joinpath(OUTDIR, "arm_summary_$RUN_ID.csv")

# --- backend knobs ----------------------------------------------------------
# NumericFocus=3: the 5h Gurobi barrier hits NUMERICAL_ERROR without it.
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
            # HiGHS: crossover stays ON even for the plain barrier arm - the
            # bare IPM reports OTHER_ERROR on the loosest (most degenerate)
            # Norse budgets, and the baseline must be exact and robust.
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

# --- model construction (mirrors run_experiments.jl, old-schema Tulipa) ------
function build_model()
    if SOURCE === :synth
        return build_synth()
    elseif SOURCE === :tulipa022
        return load_tulipa_case(DATASET, base_optimizer())
    end
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(pwd(), "data", DATASET))
    TEM.create_model!(ep)
    model = ep.model
    set_optimizer(model, base_optimizer())
    set_silent(model)
    set_algorithm!(model, :barrier)
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    target = JuMP.VariableRef[]
    for sym in (:assets_investment, :flows_investment)
        if haskey(object_dictionary(model), sym)
            for v in object_dictionary(model)[sym]
                push!(target, v)
            end
        end
    end
    return model, target
end

function build_synth()
    Random.seed!(7)
    n, T, nstore = 12, 48, 4
    cap = 50 .+ 50 .* rand(n)
    ci = 10 .+ 10 .* rand(n)
    co = 1 .+ 5 .* rand(n, T)
    dem = 20 .+ 15 .* rand(T)
    eta = 0.9
    model = Model(base_optimizer())
    set_silent(model)
    @variable(model, 0 <= assets_investment[i = 1:n] <= cap[i])
    @variable(model, dispatch[i = 1:n, t = 1:T] >= 0)
    @variable(model, charge[s = 1:nstore, t = 1:T] >= 0)
    @variable(model, level[s = 1:nstore, t = 1:T] >= 0)
    @constraint(model, [i = 1:n, t = 1:T], dispatch[i, t] <= assets_investment[i])
    @constraint(model, [t = 1:T], sum(dispatch[i, t] for i = 1:n) >= dem[t])
    @constraint(
        model,
        [s = 1:nstore, t = 2:T],
        level[s, t] == level[s, t-1] + eta * charge[s, t] - dispatch[s, t]
    )
    @constraint(model, [s = 1:nstore], level[s, 1] == eta * charge[s, 1])
    @constraint(model, [s = 1:nstore, t = 1:T], level[s, t] <= 5 * assets_investment[s])
    @objective(
        model,
        Min,
        sum(ci[i] * assets_investment[i] for i = 1:n) +
        sum(co[i, t] * dispatch[i, t] for i = 1:n, t = 1:T) +
        0.1 * sum(charge)
    )
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    return model, [assets_investment[i] for i = 1:n]
end

# HSJ-style diversity weight on the structural variables (same recipe as the
# small-case study and run_experiments.jl).
function mga_weight(model, target)
    all_vars = all_variables(model)
    idx = Dict(v => i for (i, v) in enumerate(all_vars))
    xstar = value.(all_vars)
    w = zeros(length(all_vars))
    for v in target
        ub = has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0
        w[idx[v]] = xstar[idx[v]] / ub
    end
    return w, all_vars
end

# --- one arm: sweep the budget grid ------------------------------------------
"""
    sweep_arm(model, w, all_vars, budgets, arm) -> per-point NamedTuples

`arm` is :cold_barrier, :warm_simplex, or :cold_simplex. Adds the budget row,
sets the diversity objective, sweeps, and restores the model. Warm arm: first
budget solved with barrier+crossover (basis discovery), later budgets with
dual simplex inheriting the basis; RHS edits go through set_normalized_rhs so
the attached solver keeps its state.
"""
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
        # Robustness retry: on Norse the HiGHS IPM errors on a few budgets
        # because IPX fails on the PRESOLVED reduction (presolve off solves the
        # same sub-problem fine). Retry once with presolve disabled, charging
        # both attempts' time - the price a practitioner actually pays.
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
            # Restore presolve AFTER querying the solution: JuMP invalidates
            # the cached result when an optimizer attribute changes.
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
println("[$(now())] basis arms: $SOURCE/$DATASET backend=$BACKEND threads=$THREADS")
println("  grids N=$(NPOINTS) eps=$EPS_MGA repeats=$REPEATS -> $OUTDIR")

model, target = build_model()
w, all_vars = mga_weight(model, target)
min_cost = objective_value(model)
println("  model: $(num_variables(model)) vars, min cost $(round(min_cost, sigdigits=6))")

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

const ARMS = (:cold_barrier, :warm_simplex, :cold_simplex)
for N in NPOINTS
    budgets = [min_cost + EPS_MGA * abs(min_cost) * i / N for i = 1:N]
    ref_dobj = nothing                       # arm A front, correctness anchor
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
                            DATASET,
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
        end
        arm === :cold_barrier && (ref_dobj = [p.dobj for p in last_pts])
        # Correctness cross-check: exact methods must agree on the front. Failed
        # solves (non-OPTIMAL, dobj=NaN) are excluded and counted separately -
        # a failure is a robustness finding, not a front disagreement.
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
        # Warm arm: per-RE-SOLVE cost excludes the k=1 basis-discovery solve.
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
                    DATASET,
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
println("[$(now())] done -> $POINTSF, $SUMMARYF")
