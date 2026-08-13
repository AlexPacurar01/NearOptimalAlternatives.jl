# ===========================================================================
# MGA scaling & convergence benchmark.
#
# Produces the three comparison views requested for the thesis, for every
# method (SPORES, HSJ, alm_lbfgs, penalty, pdhg, oracle):
#
#   (A) time vs number of alternatives  - how the cost of the *main* call
#       (`generate_alternatives_*`) grows as more alternatives are requested,
#       on a fixed model.                              -> scaling_alts_<backend>.png
#   (B) time vs model size              - how that cost grows with the test
#       size (5h, 4h, ... or synthetic sizes).         -> scaling_size_<backend>.png
#   (C) convergence trace              - for each full-space first-order method
#       (alm_lbfgs/penalty/pdhg), the best-so-far primal infeasibility *within
#       a single solve*, vs iteration and vs wall-clock, plus the objective
#       path. Shows the early high-error phase.         -> convergence_<backend>.png
#
#   plus diversity metrics (mean pairwise / coverage / 2D-PCA hull).
#                                                       -> diversity_<backend>.png
#
# Two model sources (set BENCH_SOURCE):
#   * "synth"  - the self-contained synthetic storage model: instant, no data,
#                multiple sizes. Use this to smoke-test the script locally.
#   * "tulipa" - the real data/<Nh> TulipaEnergyModel datasets (DEFAULT).
#                Each solve is minutes; intended for a deliberate local run on
#                the small end (5h) or for the DHPC cluster.
#
# Everything is configured by environment variables (DHPC-friendly - no code
# edits needed to scale up). Defaults are conservative for a local machine.
#
#   BENCH_SOURCE   tulipa | synth                       (default tulipa)
#   BENCH_DATASETS comma list of data/<x> folders       (default "5h")
#                  or synthetic size labels for synth    (default "small,medium")
#   BENCH_BACKEND  gurobi | highs                        (default highs)
#   BENCH_NALTS    max #alternatives for view (A)        (default 3)
#   BENCH_SIZE_NALTS  #alternatives held fixed in (B)    (default 2)
#   BENCH_METHODS  comma list                            (default all six)
#   BENCH_EPS      relative cost budget (optimality gap) (default 0.1)
#
# Run (local smoke test):
#   BENCH_SOURCE=synth julia -t 4 --project=. benchmark_scaling.jl
# Run (local real data, small end):
#   BENCH_DATASETS=5h julia -t 4 --project=. benchmark_scaling.jl
# Run (DHPC, full size sweep):
#   BENCH_BACKEND=gurobi BENCH_DATASETS=5h,4h,3h,2h,1h julia -t 16 --project=. benchmark_scaling.jl
#
# Outputs: results/scaling/ (PNGs + scaling.csv).
# ===========================================================================
using JuMP, LinearAlgebra, Random, Statistics, Printf, Logging
using Plots
using NearOptimalAlternatives
include("bench_common.jl")

# TulipaEnergyModel is only needed for the real-data source; load it
# conditionally so a synthetic smoke test does not pay its (heavy) load cost.
if Symbol(get(ENV, "BENCH_SOURCE", "tulipa")) === :tulipa
    import TulipaEnergyModel as TEM
    using DuckDB
end

# --------------------------------------------------------------------------
# Configuration (all overridable by environment variables)
# --------------------------------------------------------------------------
getenv(key, default) = get(ENV, key, default)
splitlist(s) = String[strip(x) for x in split(s, ",") if !isempty(strip(x))]

const SOURCE = Symbol(getenv("BENCH_SOURCE", "tulipa"))      # :tulipa | :synth
const BACKEND = Symbol(getenv("BENCH_BACKEND", "highs"))      # :highs | :gurobi
const THREADS = max(1, Threads.nthreads())
const DATASETS =
    splitlist(getenv("BENCH_DATASETS", SOURCE === :synth ? "small,medium" : "5h"))
const MAX_NALTS = parse(Int, getenv("BENCH_NALTS", "3"))
const SIZE_NALTS = parse(Int, getenv("BENCH_SIZE_NALTS", "2"))
const EPS_MGA = parse(Float64, getenv("BENCH_EPS", "0.1"))
# Warm-start the gradient methods from the base optimum / previous alternative
# (a feasible interior point), exactly as the sequential standard MGA methods
# do. This is the fair operational comparison and is the default; set
# BENCH_WARMSTART=false for the cold-start (from-origin) robustness experiment.
const WARMSTART = parse(Bool, getenv("BENCH_WARMSTART", "true"))
# Project each gradient alternative onto the exact feasible set
# (`operational_recovery!`, a full QP solve) before returning it. Its time is
# counted in the method's wall-clock. Set BENCH_RECOVERY=false to measure the
# raw first-order points (faster, but only approximately feasible) — the honest
# way to see what the solvers deliver before the repair barrier solve.
const RECOVERY = parse(Bool, getenv("BENCH_RECOVERY", "true"))
const METHODS =
    Symbol.(splitlist(getenv("BENCH_METHODS", "Spores,HSJ,alm_lbfgs,penalty,pdhg,oracle")))
const OUTDIR = joinpath(pwd(), "results", "scaling")

const STANDARD = (:Spores, :HSJ)
isgrad(m) = !(m in STANDARD)

# Per-method iteration budgets. The largest (`rep_budget`) is the budget used
# for the timing views (A, B) and the diversity bars; the convergence trace
# (view C) instead records progress *within* one solve and uses its own
# generous trace budgets defined below.
const BUDGETS = Dict(
    :alm_lbfgs => [2, 4, 8, 16],
    :penalty => [2, 4, 8, 16],
    :oracle => [2, 4, 8],
    :pdhg => [500, 1000, 2000, 5000],
)
budget_list(m) = get(BUDGETS, m, [0])
rep_budget(m) = last(budget_list(m))

# Synthetic-model size presets (only used when SOURCE === :synth).
const SYNTH_SIZES = Dict(
    "tiny" => (n_struct = 6, T = 12, n_store = 2),
    "small" => (n_struct = 12, T = 48, n_store = 4),
    "medium" => (n_struct = 24, T = 96, n_store = 6),
    "large" => (n_struct = 40, T = 192, n_store = 8),
)

gr()
mkpath(OUTDIR)
palette = Dict(
    :Spores => :black,
    :HSJ => :purple,
    :alm_lbfgs => :seagreen,
    :penalty => :orange,
    :pdhg => :teal,
    :oracle => :indianred,
)
mcolor(m) = get(palette, m, :gray)

println("Scaling benchmark: source=$SOURCE backend=$BACKEND threads=$THREADS")
println("  datasets=$DATASETS  methods=$(METHODS)")
println("  view A: n_alts=1..$MAX_NALTS  |  view B: n_alts=$SIZE_NALTS")

# --------------------------------------------------------------------------
# Model construction: returns (model, target_vars). Each call builds a fresh,
# cold-solved model (standard methods mutate the objective, so they cannot
# share one). `size_label` selects the dataset folder (tulipa) or size preset
# (synth).
# --------------------------------------------------------------------------
function build_model(size_label::AbstractString)
    if SOURCE === :synth
        return build_synth(SYNTH_SIZES[size_label])
    else
        return build_tulipa(size_label)
    end
end

function build_synth(sz)
    Random.seed!(7)
    n, T, nstore = sz.n_struct, sz.T, sz.n_store
    cap_max = 50.0 .+ 50.0 .* rand(n)
    c_inv = 10.0 .+ 10.0 .* rand(n)
    c_op = 1.0 .+ 5.0 .* rand(n, T)
    demand = 20.0 .+ 15.0 .* rand(T)
    eta = 0.9
    model = Model(make_optimizer(BACKEND, THREADS))
    set_silent(model)
    @variable(model, 0 <= assets_investment[i = 1:n] <= cap_max[i])
    @variable(model, dispatch[i = 1:n, t = 1:T] >= 0)
    @variable(model, charge[s = 1:nstore, t = 1:T] >= 0)
    @variable(model, level[s = 1:nstore, t = 1:T] >= 0)
    @constraint(model, [i = 1:n, t = 1:T], dispatch[i, t] <= assets_investment[i])
    @constraint(model, [t = 1:T], sum(dispatch[i, t] for i = 1:n) >= demand[t])
    @constraint(
        model,
        [s = 1:nstore, t = 2:T],
        level[s, t] == level[s, t-1] + eta * charge[s, t] - dispatch[s, t]
    )
    @constraint(model, [s = 1:nstore], level[s, 1] == eta * charge[s, 1])
    @constraint(model, [s = 1:nstore, t = 1:T], charge[s, t] <= assets_investment[s])
    @constraint(model, [s = 1:nstore, t = 1:T], level[s, t] <= 5 * assets_investment[s])
    @objective(
        model,
        Min,
        sum(c_inv[i] * assets_investment[i] for i = 1:n) +
        sum(c_op[i, t] * dispatch[i, t] for i = 1:n, t = 1:T) +
        0.1 * sum(charge)
    )
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    return model, [assets_investment[i] for i = 1:n]
end

function build_tulipa(folder::AbstractString)
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(pwd(), "data", folder))
    TEM.create_model!(ep)
    model = ep.model
    set_optimizer(model, make_optimizer(BACKEND, THREADS))
    set_silent(model)
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

target_idx_of(model, target) =
    (d = Dict(v => i for (i, v) in enumerate(all_variables(model))); [d[v] for v in target])

# --------------------------------------------------------------------------
# One measured run of the *public* entry point. Returns (time_s, P) where P is
# the normalised structural alternative set. `base` optionally supplies a
# pre-solved model + cached optimum so gradient methods are not re-charged the
# base solve (the standard methods always need their own fresh model).
# --------------------------------------------------------------------------
function measure(method::Symbol, size_label, n_alts::Int, budget::Int; base = nothing)
    if method in STANDARD
        model, target = build_model(size_label)
        tidx = target_idx_of(model, target)
        all_v = all_variables(model)
        local result
        t = @elapsed result = with_logger(NullLogger()) do
            generate_alternatives_optimization!(
                model,
                EPS_MGA,
                target,
                n_alts;
                modeling_method = method,
            )
        end
        alts = [[sol[v] for v in all_v] for sol in result.solutions]
        return t, normalised_struct(alts, target, tidx)
    else
        model, target, vals, obj, tidx, all_v = base
        local result
        t = @elapsed result = with_logger(NullLogger()) do
            generate_alternatives_gradient(
                model,
                EPS_MGA,
                target,
                n_alts;
                method = method,
                operational_recovery = RECOVERY,
                x_optimal = vals,
                min_cost = obj,
                warm_start = WARMSTART,
                max_iters = (method === :pdhg ? 30 : budget),
                pdhg_iters = (method === :pdhg ? budget : 10000),
            )
        end
        alts = [[sol[v] for v in all_v] for sol in result.solutions]
        return t, normalised_struct(alts, target, tidx)
    end
end

"Pre-solve a base model for the gradient methods (built once per dataset)."
function make_base(size_label)
    model, target = build_model(size_label)
    return (
        model,
        target,
        value.(all_variables(model)),
        objective_value(model),
        target_idx_of(model, target),
        all_variables(model),
    )
end

# csv accumulation
csv =
    String["view,backend,source,size,nvars,method,budget,n_alts,time_s,dev,mean_pw,min_pw,coverage,hull"]
function logrow(view, size_label, nvars, method, budget, n_alts, t, dev, P)
    mpw, minpw = pairwise_stats(P)
    push!(
        csv,
        @sprintf(
            "%s,%s,%s,%s,%d,%s,%d,%d,%.4f,%s,%.4f,%.4f,%.4f,%.4f",
            view,
            BACKEND,
            SOURCE,
            size_label,
            nvars,
            method,
            budget,
            n_alts,
            t,
            isnan(dev) ? "NaN" : @sprintf("%.4e", dev),
            mpw,
            minpw,
            coverage(P),
            hull_area(pca2(P))
        )
    )
end

# --------------------------------------------------------------------------
# Warm-up: Julia JIT-compiles every method's code path on the first call,
# which would otherwise dump seconds of one-off compile time onto whichever
# measurement happens to run first (typically the n=1 point), making time
# appear to *fall* as more alternatives are requested. Run each method once on
# a tiny synthetic model - cheap, always available, and it compiles the shared
# solver code regardless of the real model source - and discard the result, so
# every *timed* call below is genuinely compile-free.
# --------------------------------------------------------------------------
function warmup()
    print("Warming up (JIT compile) ... ")
    flush(stdout)
    sz = SYNTH_SIZES["tiny"]
    with_logger(NullLogger()) do
        for m in METHODS
            try
                if m in STANDARD
                    mm, tt = build_synth(sz)
                    generate_alternatives_optimization!(
                        mm,
                        EPS_MGA,
                        tt,
                        2;
                        modeling_method = m,
                    )
                else
                    mm, tt = build_synth(sz)
                    generate_alternatives_gradient(
                        mm,
                        EPS_MGA,
                        tt,
                        2;
                        method = m,
                        operational_recovery = RECOVERY,
                        x_optimal = value.(all_variables(mm)),
                        min_cost = objective_value(mm),
                        warm_start = WARMSTART,
                        max_iters = (m === :pdhg ? 30 : 2),
                        pdhg_iters = (m === :pdhg ? 200 : 10000),
                    )
                end
            catch e
                @warn "warm-up failed for $m (timings for it may include compile time)" exception =
                    e
            end
        end
    end
    println("done")
end
warmup()

# ==========================================================================
# VIEW A + C + diversity: fixed dataset (first in DATASETS), full method set.
#   A: time as a function of n_alts (1..MAX_NALTS) at each method's rep budget.
#   C: best-so-far infeasibility per iteration within one solve (trace hook).
# ==========================================================================
fixed_ds = first(DATASETS)
println("\n=== fixed-dataset views (A,C,diversity) on '$fixed_ds' ===")
base_fixed = any(isgrad, METHODS) ? make_base(fixed_ds) : nothing
nvars_fixed = base_fixed === nothing ? 0 : num_variables(base_fixed[1])

# Reference exact set (SPORES) for the deviation metric, if available.
Pref = nothing
ref_t = NaN
hsj_t = NaN
if :Spores in METHODS
    ref_t, Pref = measure(:Spores, fixed_ds, SIZE_NALTS, 0)
    println("  SPORES reference: $(round(ref_t, digits=2))s")
end
if :HSJ in METHODS
    hsj_t, _ = measure(:HSJ, fixed_ds, SIZE_NALTS, 0)
end

# --- View A: time vs n_alternatives ---
alts_curve = Dict{Symbol,Vector{Tuple{Int,Float64}}}()
for m in METHODS
    alts_curve[m] = Tuple{Int,Float64}[]
    for n = 1:MAX_NALTS
        t, P = measure(m, fixed_ds, n, rep_budget(m); base = base_fixed)
        dev = (Pref === nothing || m === :Spores) ? NaN : trajectory_gap(P, Pref)
        push!(alts_curve[m], (n, t))
        logrow("A", fixed_ds, nvars_fixed, m, rep_budget(m), n, t, dev, P)
        println(@sprintf("  [A] %-10s n=%d  %.2fs", m, n, t))
    end
end

# --- Diversity: one representative run per method (at its rep budget) ---
diversity = MethodResult[]
for m in METHODS
    t, P =
        (m === :Spores && Pref !== nothing) ? (ref_t, Pref) :
        measure(m, fixed_ds, SIZE_NALTS, rep_budget(m); base = base_fixed)
    dev = (Pref === nothing || m === :Spores) ? NaN : trajectory_gap(P, Pref)
    mpw, minpw = pairwise_stats(P)
    push!(
        diversity,
        MethodResult(string(m), t, dev, P, mpw, minpw, coverage(P), hull_area(pca2(P))),
    )
    logrow("div", fixed_ds, nvars_fixed, m, rep_budget(m), SIZE_NALTS, t, dev, P)
end

# --- Convergence trace: per-iteration progress WITHIN a single solve ---
# The budget sweep only shows the *final* error of separate runs. To see the
# trajectory - how each method descends, including the early high-error phase -
# reconstruct the first-alternative MGA LP exactly as the high-level driver
# does (HSJ-style weight on the structural vars from the optimum; cost budget
# = min_cost*(1+eps)), then run each Hessian-free full-space solver *once*
# directly with its per-iteration trace hook. All three solve the identical
# Ruiz-scaled LP, so their infeasibility/objective traces are directly
# comparable. (Exact methods and the oracle are not per-iteration residual
# solvers, so they do not appear here.)
const TRACE_OUTER = 40       # ALM/penalty outer iterations
const TRACE_INNER = 1000     # inner projected-L-BFGS iterations per outer
const TRACE_PDHG = 20000    # PDHG iterations
const TRACE_SAMPLE = 50      # PDHG: record the running average every N iters
csv_trace = String["backend,source,size,start,method,iter,time_s,infeas,obj"]
# Two traces of the same first-alternative LP: warm (from the base optimum) and
# cold (from the origin), shown together so the start types can be compared
# directly. color = method, line style = start type.
trace_warm = Dict{Symbol,Vector}()
trace_cold = Dict{Symbol,Vector}()
# Anytime-overlay references for the standard methods: each is a full LP solve
# (not anytime), so it contributes a single wall-clock marker - the time to
# produce ONE alternative - plus, for the exact MGA optimum, the objective the
# first-order traces converge to (read off as the lowest objective they reach).
exact_times = Tuple{Symbol,Float64}[]
obj_ref = NaN
if base_fixed !== nothing && any(m -> m in (:alm_lbfgs, :penalty, :pdhg), METHODS)
    model, target, vals, obj, tidx, all_v = base_fixed
    w = zeros(length(all_v))
    for (j, idx) in enumerate(tidx)
        v = target[j]
        ub = (has_upper_bound(v) && upper_bound(v) != 0) ? upper_bound(v) : 1.0
        w[idx] = vals[idx] / ub
    end
    w ./= max(norm(w), eps())
    B = obj + EPS_MGA * abs(obj)
    lp = build_mga_lp(model, w, B, all_v)
    x0_warm = clamp.(vals ./ lp.d_scale, lp.lb_t, lp.ub_t)  # base optimum, scaled
    x0_cold = clamp.(zeros(lp.n), lp.lb_t, lp.ub_t)         # origin

    # Run the three full-space solvers from a given start and return their
    # per-iteration histories (also appended to the trace CSV).
    function run_trace(x0, startlabel)
        d = Dict{Symbol,Vector}()
        for m in METHODS
            m in (:alm_lbfgs, :penalty, :pdhg) || continue
            h = NamedTuple[]
            if m === :pdhg
                solve_pdhg(
                    lp.w_t,
                    lp.A_in,
                    lp.b_in,
                    lp.A_eq,
                    lp.b_eq,
                    lp.lb_t,
                    lp.ub_t,
                    copy(x0);
                    max_iter = TRACE_PDHG,
                    verbose = false,
                    history = h,
                    sample_every = TRACE_SAMPLE,
                )
            else
                solve_alm_lbfgs(
                    lp.w_t,
                    lp.A_in,
                    lp.b_in,
                    lp.A_eq,
                    lp.b_eq,
                    lp.lb_t,
                    lp.ub_t,
                    copy(x0);
                    max_outer = TRACE_OUTER,
                    max_inner = TRACE_INNER,
                    verbose = false,
                    use_multipliers = (m === :alm_lbfgs),
                    history = h,
                )
            end
            d[m] = h
            for e in h
                push!(
                    csv_trace,
                    @sprintf(
                        "%s,%s,%s,%s,%s,%d,%.4f,%.6e,%.6e",
                        BACKEND,
                        SOURCE,
                        fixed_ds,
                        startlabel,
                        m,
                        e.iter,
                        e.t,
                        e.infeas,
                        e.obj
                    )
                )
            end
            println(
                @sprintf(
                    "  [trace %-4s] %-10s %4d points  final infeas=%.2e",
                    startlabel,
                    m,
                    length(h),
                    isempty(h) ? NaN : h[end].infeas
                )
            )
        end
        return d
    end

    println("\n--- convergence trace (single solve, first alternative, warm + cold) ---")
    trace_warm = run_trace(x0_warm, "warm")
    trace_cold = run_trace(x0_cold, "cold")

    # Exact MGA optimum from one independent LP solve (min w'x s.t. model
    # constraints, cost <= B) - a rigorous reference, not the first-order
    # methods' own converged values. Reported in the traces' scaled objective
    # units via dot(w_t, x_exact ./ d_scale). Costs one extra solve of the model.
    obj_ref = let
        orig_obj = objective_function(model)
        orig_sense = objective_sense(model)
        bcon = @constraint(model, orig_obj <= B)
        @objective(model, Min, sum(w[i] * all_v[i] for i in eachindex(all_v) if w[i] != 0))
        JuMP.optimize!(model)
        ref =
            is_solved_and_feasible(model) ?
            dot(lp.w_t, JuMP.value.(all_v) ./ lp.d_scale) : NaN
        delete(model, bcon)
        set_objective_function(model, orig_obj)
        set_objective_sense(model, orig_sense)
        ref
    end
    isnan(obj_ref) && @warn "Exact MGA reference solve failed; obj_ref unavailable."

    # Time for each standard method to produce one alternative (a single full
    # LP solve) - the vertical marker the first-order anytime curve races.
    for m in (:Spores, :HSJ)
        m in METHODS || continue
        te, _ = measure(m, fixed_ds, 1, 0)
        push!(exact_times, (m, te))
        println(@sprintf("  [trace] %-10s 1-alt exact solve %.3fs", m, te))
    end
end

# ==========================================================================
# VIEW B: time vs model size, at a fixed n_alts (SIZE_NALTS), per method.
# x-axis is the measured #variables; points are labelled by dataset.
# ==========================================================================
size_curve = Dict{Symbol,Vector{Tuple{Int,Float64}}}()   # method -> (nvars, time)
size_labels = Tuple{Int,String}[]                          # (nvars, label) for ticks
for m in METHODS
    size_curve[m] = Tuple{Int,Float64}[]
end
println("\n=== size-scaling view (B) over $DATASETS ===")
for ds in DATASETS
    println("  size '$ds' ...")
    base_ds =
        (ds == fixed_ds) ? base_fixed : (any(isgrad, METHODS) ? make_base(ds) : nothing)
    nvars = base_ds === nothing ? 0 : num_variables(base_ds[1])
    push!(size_labels, (nvars, ds))
    for m in METHODS
        # reuse the already-measured fixed-dataset point to avoid re-solving
        if ds == fixed_ds && !isempty(get(alts_curve, m, []))
            ti = findfirst(p -> p[1] == SIZE_NALTS, alts_curve[m])
            if ti !== nothing
                push!(size_curve[m], (nvars, alts_curve[m][ti][2]))
                continue
            end
        end
        t, P = measure(m, ds, SIZE_NALTS, rep_budget(m); base = base_ds)
        push!(size_curve[m], (nvars, t))
        logrow(
            "B",
            ds,
            nvars,
            m,
            rep_budget(m),
            SIZE_NALTS,
            t,
            (Pref === nothing || m === :Spores) ? NaN : NaN,
            P,
        )
        println(@sprintf("    [B] %-10s nvars=%d  %.2fs", m, nvars, t))
    end
end

# ==========================================================================
# Plots
# ==========================================================================
# --- (A) time vs #alternatives ---
pA = plot(;
    xlabel = "number of alternatives",
    ylabel = "wall-clock (s)",
    title = "Time to generate alternatives ($fixed_ds, $BACKEND, $THREADS thr)",
    legend = :topleft,
    size = (820, 560),
    xticks = 1:MAX_NALTS,
)
for m in METHODS
    pts = alts_curve[m]
    plot!(
        pA,
        [p[1] for p in pts],
        [p[2] for p in pts];
        marker = :circle,
        lw = 2,
        label = string(m),
        color = mcolor(m),
    )
end
savefig(pA, joinpath(OUTDIR, "scaling_alts_$BACKEND.png"))

# --- (B) time vs model size ---
sort!(size_labels)
xticks_v = [s[1] for s in size_labels]
xticks_l = [s[2] for s in size_labels]
pB = plot(;
    xlabel = "model size (# variables)",
    ylabel = "wall-clock (s)",
    title = "Time vs test size (n_alts=$SIZE_NALTS, $BACKEND, $THREADS thr)",
    legend = :topleft,
    size = (820, 560),
    xticks = (xticks_v, xticks_l),
)
for m in METHODS
    pts = sort(size_curve[m])
    isempty(pts) && continue
    plot!(
        pB,
        [p[1] for p in pts],
        [p[2] for p in pts];
        marker = :square,
        lw = 2,
        label = string(m),
        color = mcolor(m),
    )
end
length(DATASETS) > 1 && savefig(pB, joinpath(OUTDIR, "scaling_size_$BACKEND.png"))

# --- (C) convergence trace: warm vs cold start, vs REAL wall-clock time ---
# Both panels use real elapsed time (from 0), not iteration count: an inner
# L-BFGS step and a PDHG step cost very different amounts of wall-clock, so an
# iteration axis would make the methods look comparable when they are not.
# Warm and cold starts of the same first-alternative LP are overlaid:
# color = method, line style = start type (solid = warm, dashed = cold). BOTH
# panels show the CURRENT iterate's true state at each time (not a best-so-far
# envelope), so the two panels are mutually consistent: when a method's
# objective dips below the optimum it is because that same iterate has gone
# slightly infeasible (visible as a bump in panel 1) - a weak-penalty excursion
# that rho later pulls back. Raw values stay in convergence_trace.csv.
#   Panel 1: primal infeasibility vs wall-clock (cold = clawing to feasibility;
#            warm = stays near-feasible)
#   Panel 2: objective wᵀx vs wall-clock + anytime overlay (exact optimum and
#            the standard methods' one-alternative solve times)
have_trace = any(!isempty, values(trace_warm)) || any(!isempty, values(trace_cold))
if have_trace
    floorp(v, f) = max(Float64(v), f)
    function subsample(xs, ys; n = 500)
        length(xs) <= n && return (xs, ys)
        idx = unique(round.(Int, range(1, length(xs); length = n)))
        return (xs[idx], ys[idx])
    end
    # Centred rolling median - denoises the wildly-oscillating infeasibility
    # into a readable trend while staying a *local* statistic (so a sustained
    # infeasible excursion still shows, lining up with the objective panel).
    function rollmed(v, w)
        (w <= 1 || length(v) <= w) && return v
        half = w ÷ 2
        out = similar(v)
        @inbounds for i in eachindex(v)
            out[i] = median(view(v, max(1, i - half):min(length(v), i + half)))
        end
        return out
    end
    alltraces = (trace_warm, trace_cold)
    maxt = maximum((e.t for d in alltraces for h in values(d) for e in h); init = 1e-6)
    isempty(exact_times) || (maxt = max(maxt, maximum(te for (_, te) in exact_times)))
    pC = plot(
        layout = (1, 2),
        size = (1200, 500),
        bottom_margin = 10Plots.mm,
        left_margin = 8Plots.mm,
    )
    # color = method, line style = start type; x = real elapsed time. Both
    # series are the current iterate's actual values at each recorded point.
    for (d, lsty) in ((trace_warm, :solid), (trace_cold, :dash))
        for m in METHODS
            h = get(d, m, NamedTuple[])
            isempty(h) && continue
            ts = [e.t for e in h]                          # real time, starts at 0
            infe = [floorp(e.infeas, 1e-12) for e in h]    # current infeasibility
            objs = [e.obj for e in h]                      # current objective
            infe_s = rollmed(infe, max(9, length(infe) ÷ 80))
            xr, yr = subsample(ts, infe)                   # raw (faint background)
            x1, y1 = subsample(ts, infe_s)                 # smoothed (bold trend)
            x2, y2 = subsample(ts, objs)
            plot!(
                pC[1],
                xr,
                yr;
                lw = 1,
                color = mcolor(m),
                ls = lsty,
                label = "",
                alpha = 0.15,
            )
            plot!(pC[1], x1, y1; lw = 2, color = mcolor(m), ls = lsty, label = "")
            plot!(pC[2], x2, y2; lw = 2, color = mcolor(m), ls = lsty, label = "")
        end
    end
    # Two-dimensional legend (panel 1) via empty dummy series: colour = method,
    # line style = start type.
    for m in METHODS
        m in (:alm_lbfgs, :penalty, :pdhg) || continue
        plot!(pC[1], [NaN], [NaN]; color = mcolor(m), lw = 2, label = string(m))
    end
    plot!(pC[1], [NaN], [NaN]; color = :black, lw = 2, ls = :solid, label = "warm start")
    plot!(pC[1], [NaN], [NaN]; color = :black, lw = 2, ls = :dash, label = "cold start")

    # Anytime overlay on the objective panel: the exact MGA optimum (the level
    # the first-order curves converge to) and each standard method's full
    # one-alternative solve time. A first-order curve that reaches a tolerance
    # band around `obj_ref` to the LEFT of a standard method's line is more
    # efficient at that tolerance - the anytime argument, made visible.
    isnan(obj_ref) || hline!(
        pC[2],
        [obj_ref];
        ls = :dot,
        lw = 2,
        color = :gray40,
        label = "exact optimum",
    )
    for (m, te) in exact_times
        vline!(
            pC[2],
            [te];
            ls = :dashdot,
            lw = 2,
            color = mcolor(m),
            label = "$m (1-alt solve)",
        )
    end
    plot!(
        pC[1];
        xlabel = "wall-clock (s)",
        ylabel = "primal infeasibility",
        xlims = (0.0, maxt),
        yscale = :log10,
        legend = :topright,
    )
    plot!(
        pC[2];
        xlabel = "wall-clock (s)",
        ylabel = "objective wᵀx (scaled)",
        xlims = (0.0, maxt),
        legend = :topright,
    )
    plot!(
        pC;
        plot_title = "Convergence vs wall-clock: warm vs cold start ($fixed_ds, $BACKEND)",
    )
    savefig(pC, joinpath(OUTDIR, "convergence_$BACKEND.png"))
end

# --- diversity bars ---
dnames = [r.name for r in diversity]
pD = plot(layout = (1, 3), size = (1300, 460), bottom_margin = 10Plots.mm)
bar!(
    pD[1],
    dnames,
    [r.mean_pw for r in diversity];
    legend = false,
    title = "mean pairwise dist",
    xrotation = 40,
    color = :steelblue,
)
bar!(
    pD[2],
    dnames,
    [r.cov for r in diversity];
    legend = false,
    title = "coverage (spanned range)",
    xrotation = 40,
    color = :indianred,
)
bar!(
    pD[3],
    dnames,
    [r.hull for r in diversity];
    legend = false,
    title = "2D-PCA hull volume",
    xrotation = 40,
    color = :darkgreen,
)
savefig(pD, joinpath(OUTDIR, "diversity_$BACKEND.png"))

open(joinpath(OUTDIR, "scaling.csv"), "w") do io
    println(io, join(csv, "\n"))
end
open(joinpath(OUTDIR, "convergence_trace.csv"), "w") do io
    println(io, join(csv_trace, "\n"))
end
println("\nWrote plots + scaling.csv + convergence_trace.csv to $OUTDIR")
