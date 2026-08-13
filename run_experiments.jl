# ===========================================================================
# Headless experiment runner: solves the MGA problem with every requested
# method on ONE dataset and appends every measurement to a CSV, for later
# plotting locally. No plots here - just data + a metadata file.
#
# Designed for a SLURM job array: one array task = one dataset, running in
# isolation on its own cores/memory so the wall-clock timings are clean. Run
# different datasets in parallel via separate array tasks, NOT different solvers
# inside one process (that would contend for cores and pollute the timings).
#
# Config via environment variables (all optional; sensible defaults):
#   RUN_SOURCE    tulipa | synth                  (default tulipa)
#   RUN_DATASET   one data/<x> folder / synth size (default 5h)
#   RUN_BACKEND   highs | gurobi                   (default highs)
#   RUN_METHODS   comma list                       (default Spores,HSJ,alm_lbfgs,penalty,pdhg,oracle)
#   RUN_NALTS     max #alternatives (sweeps 1..N)  (default 3)
#   RUN_REPEATS   timing repeats per config        (default 1; use 3 for medians)
#   RUN_WARMSTART true | false                      (default true)
#   RUN_RECOVERY  true | false                      (default true)
#   RUN_EPS       relative cost budget             (default 0.1)
#   RUN_TRACE     true | false  per-iteration trace (default true)
#   RUN_OUT       output directory                 (default results/experiments)
#   RUN_ID        tag for output filenames         (default $SLURM_ARRAY_TASK_ID or "local")
#
# Output (per task): RUN_OUT/summary_<ID>.csv, trace_<ID>.csv, meta_<ID>.txt
#
#   JULIA_NUM_THREADS=8 julia -t 8 --project=. run_experiments.jl
# ===========================================================================
using JuMP, LinearAlgebra, Random, Statistics, Printf, Logging, Dates
using NearOptimalAlternatives
include("bench_common.jl")

env(k, d) = get(ENV, k, d)
splitlist(s) = String[strip(x) for x in split(s, ",") if !isempty(strip(x))]

const SOURCE = Symbol(env("RUN_SOURCE", "tulipa"))
const DATASET = env("RUN_DATASET", SOURCE === :synth ? "small" : "5h")
const BACKEND = Symbol(env("RUN_BACKEND", "highs"))
const METHODS =
    Symbol.(splitlist(env("RUN_METHODS", "Spores,HSJ,alm_lbfgs,penalty,pdhg,oracle")))
const MAX_NALTS = parse(Int, env("RUN_NALTS", "3"))
const REPEATS = parse(Int, env("RUN_REPEATS", "1"))
const WARMSTART = parse(Bool, env("RUN_WARMSTART", "true"))
const RECOVERY = parse(Bool, env("RUN_RECOVERY", "true"))
const EPS_MGA = parse(Float64, env("RUN_EPS", "0.1"))
const TRACE = parse(Bool, env("RUN_TRACE", "true"))
const OUTDIR = env("RUN_OUT", joinpath(pwd(), "results", "experiments"))
const RUN_ID = env("RUN_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "local"))
const THREADS = max(1, Threads.nthreads())

const STANDARD = (:Spores, :HSJ)
isgrad(m) = !(m in STANDARD)
const BUDGETS = Dict(:alm_lbfgs => 16, :penalty => 16, :oracle => 8, :pdhg => 5000)
rep_budget(m) = get(BUDGETS, m, 0)

const SYNTH = Dict(
    "tiny" => (n_struct = 6, T = 12, n_store = 2),
    "small" => (n_struct = 12, T = 48, n_store = 4),
    "medium" => (n_struct = 24, T = 96, n_store = 6),
    "large" => (n_struct = 40, T = 192, n_store = 8),
)

if SOURCE === :tulipa
    import TulipaEnergyModel as TEM
    using DuckDB
end

mkpath(OUTDIR)
const SUMMARY = joinpath(OUTDIR, "summary_$RUN_ID.csv")
const TRACEF = joinpath(OUTDIR, "trace_$RUN_ID.csv")
const METAF = joinpath(OUTDIR, "meta_$RUN_ID.txt")

# --------------------------------------------------------------------------
# Model construction (fresh, cold-solved). Standard methods mutate the
# objective, so each measurement builds its own.
# --------------------------------------------------------------------------
function build_model()
    if SOURCE === :synth
        return build_synth(SYNTH[DATASET])
    end
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(pwd(), "data", DATASET))
    TEM.create_model!(ep)
    model = ep.model
    set_optimizer(model, make_optimizer(BACKEND, THREADS))
    set_silent(model)
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    # Flatten the structural variable containers. Tulipa stores these as
    # DenseAxisArrays/SparseAxisArrays indexed by strings, which `collect` can't
    # materialise (CartesianIndices fails on a string axis) - iterate the values
    # instead, matching the package's own `generate_alternatives_lbfgs`.
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

function build_synth(sz)
    Random.seed!(7)
    n, T, nstore = sz.n_struct, sz.T, sz.n_store
    cap = 50 .+ 50 .* rand(n)
    ci = 10 .+ 10 .* rand(n)
    co = 1 .+ 5 .* rand(n, T)
    dem = 20 .+ 15 .* rand(T)
    eta = 0.9
    model = Model(make_optimizer(BACKEND, THREADS))
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

target_idx(model, target) =
    (d = Dict(v => i for (i, v) in enumerate(all_variables(model))); [d[v] for v in target])

# --------------------------------------------------------------------------
# One measured run of the public API. Returns (time_s, P, alts_full, model).
# --------------------------------------------------------------------------
function measure(method, n_alts; base = nothing)
    if method in STANDARD
        model, target = build_model()
        tidx = target_idx(model, target)
        all_v = all_variables(model)
        local res
        t = @elapsed res = with_logger(NullLogger()) do
            generate_alternatives_optimization!(
                model,
                EPS_MGA,
                target,
                n_alts;
                modeling_method = method,
            )
        end
        alts = [[s[v] for v in all_v] for s in res.solutions]
        return t, normalised_struct(alts, target, tidx), alts, model
    else
        model, target, vals, obj, tidx, all_v = base
        local res
        t = @elapsed res = with_logger(NullLogger()) do
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
                max_iters = (method === :pdhg ? 30 : rep_budget(method)),
                pdhg_iters = (method === :pdhg ? rep_budget(method) : 10000),
            )
        end
        alts = [[s[v] for v in all_v] for s in res.solutions]
        return t, normalised_struct(alts, target, tidx), alts, model
    end
end

make_base() = (
    m = build_model();
    (
        m[1],
        m[2],
        value.(all_variables(m[1])),
        objective_value(m[1]),
        target_idx(m[1], m[2]),
        all_variables(m[1]),
    )
)

# SPORES reference set (decision-space deviation baseline). A function so the
# assignment isn't trapped in `try`'s soft scope.
function spores_reference()
    try
        _, P, _, _ = measure(:Spores, MAX_NALTS)
        println("  SPORES reference set computed")
        return P
    catch e
        @warn "SPORES reference failed" exception = e
        return nothing
    end
end

# Max original-unit physical violation (constraints + bounds) of a point.
function max_violation(model, all_v, x)
    rep = primal_feasibility_report(model, Dict(all_v[i] => x[i] for i in eachindex(all_v)))
    isempty(rep) ? 0.0 : maximum(values(rep))
end

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
open(METAF, "w") do io
    println(io, "run_id=$RUN_ID")
    println(io, "datetime=$(now())")
    println(io, "host=$(gethostname())")
    println(io, "julia=$(VERSION)  threads=$THREADS")
    println(io, "source=$SOURCE dataset=$DATASET backend=$BACKEND")
    println(io, "methods=$(METHODS) max_nalts=$MAX_NALTS repeats=$REPEATS")
    println(io, "warm_start=$WARMSTART recovery=$RECOVERY eps=$EPS_MGA")
end

const COLS =
    "run_id,host,threads,backend,source,dataset,nvars,ncons,method,n_alts," *
    "budget,warm_start,recovery,repeat,status,time_s,dev_from_exact,max_violation," *
    "mean_pairwise,min_pairwise,coverage,hull"
open(SUMMARY, "w") do io
    println(io, COLS)
end
emit(row) =
    open(SUMMARY, "a") do io
        println(io, row)
        flush(io)
    end

println("[$(now())] task $RUN_ID: $SOURCE/$DATASET, backend=$BACKEND, threads=$THREADS")
println(
    "  methods=$(METHODS) nalts=1..$MAX_NALTS repeats=$REPEATS warm=$WARMSTART recovery=$RECOVERY",
)

# No JIT warm-up: each method runs directly, exactly as a one-shot real run
# would. The first measurement of each method (n=1, repeat=1) therefore
# includes Julia's one-off compilation; later repeats are compile-free. Compare
# algorithms on the median/min over repeats (REPEATS>=2), which excludes that
# single cold measurement, while repeat=1 still records the true one-shot time.

base = any(isgrad, METHODS) ? make_base() : nothing
nvars = base === nothing ? -1 : num_variables(base[1])
ncons =
    base === nothing ? -1 :
    num_constraints(base[1]; count_variable_in_set_constraints = false)

Pref = (:Spores in METHODS) ? spores_reference() : nothing

for method in METHODS, n = 1:MAX_NALTS, rep = 1:REPEATS
    status, t, dev, viol, mpw, minpw, cov, hl = "ok", NaN, NaN, NaN, NaN, NaN, NaN, NaN
    try
        t, P, alts, model = measure(method, n; base = base)
        mpw, minpw = pairwise_stats(P)
        cov = coverage(P)
        hl = hull_area(pca2(P))
        dev = (Pref === nothing || method === :Spores) ? NaN : trajectory_gap(P, Pref)
        all_v = all_variables(model)
        viol = isempty(alts) ? NaN : maximum(max_violation(model, all_v, a) for a in alts)
    catch e
        status = "error"
        @warn "run failed: $method n=$n rep=$rep" exception = e
    end
    bud = isgrad(method) ? rep_budget(method) : 0
    emit(
        @sprintf(
            "%s,%s,%d,%s,%s,%s,%d,%d,%s,%d,%d,%s,%s,%d,%s,%.4f,%s,%.3e,%.4f,%.4f,%.4f,%.4f",
            RUN_ID,
            gethostname(),
            THREADS,
            BACKEND,
            SOURCE,
            DATASET,
            nvars,
            ncons,
            method,
            n,
            bud,
            WARMSTART,
            RECOVERY,
            rep,
            status,
            t,
            isnan(dev) ? "NaN" : @sprintf("%.4e", dev),
            viol,
            mpw,
            minpw,
            cov,
            hl
        )
    )
    println("  [$method n=$n rep=$rep] $status  t=$(round(t, digits=3))s")
end

# --------------------------------------------------------------------------
# Per-iteration convergence trace (warm + cold) for the full-space solvers,
# from one solve of the first-alternative MGA LP. Independent exact obj_ref.
# --------------------------------------------------------------------------
if TRACE && base !== nothing && any(m -> m in (:alm_lbfgs, :penalty, :pdhg), METHODS)
    model, target, vals, obj, tidx, all_v = base
    w = zeros(length(all_v))
    for (j, i) in enumerate(tidx)
        ub =
            has_upper_bound(target[j]) && upper_bound(target[j]) != 0 ?
            upper_bound(target[j]) : 1.0
        w[i] = vals[i] / ub
    end
    w ./= max(norm(w), eps())
    B = obj + EPS_MGA * abs(obj)
    lp = build_mga_lp(model, w, B, all_v)
    starts = (
        (:warm, clamp.(vals ./ lp.d_scale, lp.lb_t, lp.ub_t)),
        (:cold, clamp.(zeros(lp.n), lp.lb_t, lp.ub_t)),
    )
    open(TRACEF, "w") do io
        println(io, "run_id,dataset,start,method,iter,time_s,infeas,obj")
        for (sname, x0) in starts, m in METHODS
            m in (:alm_lbfgs, :penalty, :pdhg) || continue
            h = NamedTuple[]
            try
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
                        max_iter = 20000,
                        verbose = false,
                        history = h,
                        sample_every = 50,
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
                        max_outer = 40,
                        max_inner = 1000,
                        verbose = false,
                        use_multipliers = (m === :alm_lbfgs),
                        history = h,
                    )
                end
            catch e
                @warn "trace failed: $m $sname" exception = e
            end
            for e in h
                println(
                    io,
                    @sprintf(
                        "%s,%s,%s,%s,%d,%.5f,%.6e,%.6e",
                        RUN_ID,
                        DATASET,
                        sname,
                        m,
                        e.iter,
                        e.t,
                        e.infeas,
                        e.obj
                    )
                )
            end
        end
    end
    println("  wrote trace")
end

println("[$(now())] task $RUN_ID done. -> $SUMMARY")
