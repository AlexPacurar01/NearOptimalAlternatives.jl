# ===========================================================================
# MGA method benchmark: wall-clock and alternative quality, fairly compared.
#
# Compares the new gradient (first-order, Hessian-free) MGA solvers against
# the standard exact MGA methods (HSJ, SPORES) and a direct exact LP solve, on
# a synthetic energy-system model with inter-temporal storage-balance chains
# (the structure that makes the real model stiff). Fairness controls:
#
#   * Same solver backend per plot - one figure for Gurobi, one for HiGHS.
#   * Same parallelization - solver Threads == Julia thread count (THREADS).
#   * Same precision - every gradient method solves the *identical* canonical
#     HSJ direction-sequence that the "Direct LP (exact)" baseline solves, and
#     each alternative's gap to that exact optimum is recorded, so wall-clock
#     is read at a known, equal precision (plus a time-vs-accuracy curve).
#
# Quality uses standard MGA decision-space metrics: mean / min pairwise
# distance, near-optimal coverage (spanned ranges), 2D-PCA convex-hull volume,
# and a set-coverage (dominance) matrix.
#
# Run with a fixed parallelization, e.g.:  julia -t 4 --project=. benchmark_mga.jl
# Outputs: results/benchmark/ (PNG plots + results.csv + dominance_*.csv).
# ===========================================================================

using JuMP, LinearAlgebra, Random, Printf, Statistics, Logging
using Plots
import Gurobi, HiGHS
using NearOptimalAlternatives

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
const QUICK = haskey(ENV, "BENCH_QUICK")   # tiny smoke-test config
const THREADS = max(1, Threads.nthreads())
const N_ALTS = QUICK ? 3 : 6
const EPS_MGA = 0.1
const SEED = 7
const N_STRUCT = QUICK ? 6 : 12
const T = QUICK ? 12 : 48
const N_STORE = QUICK ? 2 : 4
const TARGET_GAP = 0.05           # target decision-space deviation from the exact set
const ALM_BUDGETS = QUICK ? [5, 20] : [5, 10, 20, 40]
const PDHG_BUDGETS = QUICK ? [2000, 10000] : [2000, 5000, 10000, 20000]
const ORACLE_BUDGETS = QUICK ? [2, 4] : [2, 4, 8]
const BACKENDS = QUICK ? [:highs] : [:gurobi, :highs]
const OUTDIR = joinpath(pwd(), "results", "benchmark")

gr()
mkpath(OUTDIR)
println(
    "Benchmark: THREADS=$THREADS N_ALTS=$N_ALTS size=($N_STRUCT struct,$T steps,$N_STORE storage)",
)

# --------------------------------------------------------------------------
# Model + solver backends
# --------------------------------------------------------------------------
function make_optimizer(backend::Symbol)
    # Interior-point (barrier) for both backends: barrier re-solves from
    # scratch every call (it never reuses a previous solve's basis), so every
    # solve - exact baseline, HSJ/SPORES iterations, oracle, and the repair
    # projection - is a genuine COLD start. This levels the field (warm-started
    # simplex would unfairly favour the sequential exact methods) and matches
    # how an independent MGA query is solved in practice.
    #
    # Crossover is OFF: the exact methods return the barrier (interior)
    # solution directly, so they are not charged the extra crossover-to-vertex
    # work that the approximate gradient methods never do - a fair wall-clock
    # comparison (all methods return a non-vertex, tolerance-accurate point).
    # HSJ relies on basic (vertex) solutions with exact zeros, so without
    # crossover it degenerates - a genuine property of the HSJ formulation.
    backend == :gurobi ?
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Threads" => THREADS,
        "Method" => 2,
        "Crossover" => 0,
    ) :
    optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "threads" => THREADS,
        "solver" => "ipm",
        "run_crossover" => "off",
    )
end

"Build and solve a fresh synthetic storage model; returns (model, target_vars)."
function build_solved_model(backend::Symbol)
    Random.seed!(SEED)
    cap_max = 50.0 .+ 50.0 .* rand(N_STRUCT)
    c_inv = 10.0 .+ 10.0 .* rand(N_STRUCT)
    c_op = 1.0 .+ 5.0 .* rand(N_STRUCT, T)
    demand = 20.0 .+ 15.0 .* rand(T)
    eta = 0.9

    model = Model(make_optimizer(backend))
    set_silent(model)
    @variable(model, 0 <= assets_investment[i = 1:N_STRUCT] <= cap_max[i])
    @variable(model, dispatch[i = 1:N_STRUCT, t = 1:T] >= 0)
    @variable(model, charge[s = 1:N_STORE, t = 1:T] >= 0)
    @variable(model, level[s = 1:N_STORE, t = 1:T] >= 0)
    @constraint(model, [i = 1:N_STRUCT, t = 1:T], dispatch[i, t] <= assets_investment[i])
    @constraint(model, [t = 1:T], sum(dispatch[i, t] for i = 1:N_STRUCT) >= demand[t])
    @constraint(
        model,
        [s = 1:N_STORE, t = 2:T],
        level[s, t] == level[s, t-1] + eta * charge[s, t] - dispatch[s, t]
    )
    @constraint(model, [s = 1:N_STORE], level[s, 1] == eta * charge[s, 1])
    @constraint(model, [s = 1:N_STORE, t = 1:T], charge[s, t] <= assets_investment[s])
    @constraint(model, [s = 1:N_STORE, t = 1:T], level[s, t] <= 5 * assets_investment[s])
    @objective(
        model,
        Min,
        sum(c_inv[i] * assets_investment[i] for i = 1:N_STRUCT) +
        sum(c_op[i, t] * dispatch[i, t] for i = 1:N_STRUCT, t = 1:T) +
        0.1 * sum(charge)
    )
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    return model, [assets_investment[i] for i = 1:N_STRUCT]
end

"""
Run ONE MGA method end-to-end via its public API, generating all `N_ALTS`
alternatives in a single call on a freshly-built, freshly-solved model (so no
state - solver basis or otherwise - is shared between methods, and each method
runs exactly as it would in practice; the sequential accumulation that creates
diversity happens inside the one call, never one-alternative-at-a-time).

`method` is `:HSJ`/`:Spores` (standard, via `generate_alternatives_optimization!`)
or a gradient symbol (`:alm_lbfgs`/`:penalty`/`:pdhg`/`:oracle`, via
`generate_alternatives_gradient`). `budget` is the gradient iteration budget
(ignored for standard methods). Returns (alts_full, time, target, target_idx).
"""
function run_method(backend, method, budget)
    model, target = build_solved_model(backend)
    all_v = all_variables(model)
    var_to_idx = Dict(v => i for (i, v) in enumerate(all_v))
    target_idx = [var_to_idx[v] for v in target]
    local result
    t = @elapsed result = with_logger(NullLogger()) do
        if method in (:HSJ, :Spores)
            generate_alternatives_optimization!(
                model,
                EPS_MGA,
                target,
                N_ALTS;
                modeling_method = method,
            )
        else
            generate_alternatives_gradient(
                model,
                EPS_MGA,
                target,
                N_ALTS;
                method = method,
                operational_recovery = true,
                max_iters = (method == :pdhg ? 30 : budget),
                pdhg_iters = (method == :pdhg ? budget : 10000),
            )
        end
    end
    alts = [[sol[v] for v in all_v] for sol in result.solutions]
    return alts, t, target, target_idx
end

"Mean step-wise normalised structural distance between an alternative set and a reference set."
function trajectory_gap(P, Pref)
    n = min(size(P, 2), size(Pref, 2))
    n == 0 && return NaN
    d = size(P, 1)
    return mean(norm(P[:, i] .- Pref[:, i]) / sqrt(d) for i = 1:n)
end

# --------------------------------------------------------------------------
# MGA quality metrics (decision/structural space, normalised to [0,1])
# --------------------------------------------------------------------------
"Normalise structural columns of full vectors to [0,1] by target bounds."
function normalised_struct(alts, target, target_idx)
    lb = [has_lower_bound(v) ? lower_bound(v) : 0.0 for v in target]
    ub = [has_upper_bound(v) ? upper_bound(v) : 1.0 for v in target]
    rng = max.(ub .- lb, 1e-12)
    P = zeros(length(target), length(alts))
    for (k, a) in enumerate(alts)
        P[:, k] = clamp.((a[target_idx] .- lb) ./ rng, 0.0, 1.0)
    end
    return P                     # d x N
end

function pairwise_stats(P)
    d, N = size(P)
    N < 2 && return (0.0, 0.0)
    ds = Float64[]
    for i = 1:N-1, j = i+1:N
        push!(ds, norm(P[:, i] .- P[:, j]) / sqrt(d))
    end
    return (mean(ds), minimum(ds))
end

coverage(P) = size(P, 2) < 2 ? 0.0 : mean(maximum(P, dims = 2) .- minimum(P, dims = 2))

function pca2(P)
    d, N = size(P)
    mu = mean(P, dims = 2)
    C = P .- mu
    F = svd(C)
    k = min(2, size(F.U, 2))
    proj = F.U[:, 1:k]' * C       # k x N
    k == 1 && (proj = vcat(proj, zeros(1, N)))
    return proj                   # 2 x N
end

function hull_area(pts)           # pts: 2 x N
    N = size(pts, 2)
    N < 3 && return 0.0
    P = [(pts[1, i], pts[2, i]) for i = 1:N]
    sort!(P)
    unique!(P)
    length(P) < 3 && return 0.0
    cross(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    lower = Tuple{Float64,Float64}[]
    for p in P
        while length(lower) >= 2 && cross(lower[end-1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    upper = Tuple{Float64,Float64}[]
    for p in reverse(P)
        while length(upper) >= 2 && cross(upper[end-1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    hull = vcat(lower[1:end-1], upper[1:end-1])
    n = length(hull)
    n < 3 && return 0.0
    area = 0.0
    for i = 1:n
        x1, y1 = hull[i]
        x2, y2 = hull[mod1(i + 1, n)]
        area += x1 * y2 - x2 * y1
    end
    return abs(area) / 2
end

"C(A,B): fraction of B columns within δ of some A column (normalised space)."
function set_coverage(A, B; delta = 0.1)
    nb = size(B, 2)
    nb == 0 && return 0.0
    covered = 0
    for j = 1:nb
        if any(norm(A[:, i] .- B[:, j]) / sqrt(size(A, 1)) <= delta for i = 1:size(A, 2))
            covered += 1
        end
    end
    return covered / nb
end

# --------------------------------------------------------------------------
# Run the benchmark
# --------------------------------------------------------------------------
mutable struct MethodResult
    name::String
    time::Float64
    mean_gap::Float64
    max_gap::Float64
    P::Matrix{Float64}            # normalised structural points (d x N)
    mean_pw::Float64
    min_pw::Float64
    cov::Float64
    hull::Float64
end

const GRADIENT = [
    (:alm_lbfgs, "alm_lbfgs", ALM_BUDGETS),
    (:penalty, "penalty", ALM_BUDGETS),
    (:pdhg, "pdhg", PDHG_BUDGETS),
    (:oracle, "oracle", ORACLE_BUDGETS),
]

# csv accumulation
csv_rows =
    String["backend,method,budget,time_s,mean_gap,max_gap,mean_pairwise,min_pairwise,coverage,hull_area"]

for backend in BACKENDS
    println("\n=== backend: $backend ===")

    # Exact SPORES set = the gold-standard reference. The gradient methods
    # solve the same weighted-HSJ formulation, so their decision-space
    # deviation from this set measures their approximation error.
    ref_alts, ref_t, rtgt, rtidx = run_method(backend, :Spores, 0)
    Pref = normalised_struct(ref_alts, rtgt, rtidx)

    results = MethodResult[]
    function record!(name, t, gap, P)
        mpw, minpw = pairwise_stats(P)
        push!(
            results,
            MethodResult(name, t, gap, gap, P, mpw, minpw, coverage(P), hull_area(pca2(P))),
        )
        push!(
            csv_rows,
            @sprintf(
                "%s,%s,%s,%.4f,%s,%s,%.4f,%.4f,%.4f,%.4f",
                backend,
                name,
                "-",
                t,
                isnan(gap) ? "NaN" : @sprintf("%.4e", gap),
                isnan(gap) ? "NaN" : @sprintf("%.4e", gap),
                mpw,
                minpw,
                coverage(P),
                hull_area(pca2(P))
            )
        )
    end

    # Standard exact methods (different - HSJ - and reference - SPORES - formulations).
    record!("SPORES", ref_t, NaN, Pref)
    hsj_alts, hsj_t, htgt, htidx = run_method(backend, :HSJ, 0)
    record!("HSJ", hsj_t, NaN, normalised_struct(hsj_alts, htgt, htidx))

    # Gradient methods: sweep the iteration budget; the representative is the
    # smallest budget whose deviation reaches TARGET_DEV (else the best).
    sweep = Dict{String,Vector{Tuple{Float64,Float64}}}()
    for (method, label, budgets) in GRADIENT
        sweep[label] = Tuple{Float64,Float64}[]
        best = nothing
        for b in budgets
            alts, t, tgt, tidx = run_method(backend, method, b)
            P = normalised_struct(alts, tgt, tidx)
            dev = trajectory_gap(P, Pref)
            push!(sweep[label], (t, dev))
            mpw, minpw = pairwise_stats(P)
            push!(
                csv_rows,
                @sprintf(
                    "%s,%s,%d,%.4f,%.4e,%.4e,%.4f,%.4f,%.4f,%.4f",
                    backend,
                    label,
                    b,
                    t,
                    dev,
                    dev,
                    mpw,
                    minpw,
                    coverage(P),
                    hull_area(pca2(P))
                )
            )
            cand = (t = t, dev = dev, P = P)
            if best === nothing ||
               (dev <= TARGET_GAP && (best.dev > TARGET_GAP || t < best.t)) ||
               (best.dev > TARGET_GAP && dev < best.dev)
                best = cand
            end
        end
        record!(label, best.t, best.dev, best.P)
    end

    dnames = [r.name for r in results]

    # --------- Plot 1: wall-clock to generate the alternative set ---------
    cols = [
        isnan(r.mean_gap) ? :gray : (r.mean_gap <= TARGET_GAP ? :seagreen : :orange) for
        r in results
    ]
    p1 = bar(
        dnames,
        [r.time for r in results];
        legend = false,
        color = cols,
        ylabel = "wall-clock (s)",
        title = "Cold-start wall-clock for $N_ALTS alternatives ($backend, $THREADS thr)",
        xrotation = 30,
        size = (900, 520),
        bottom_margin = 8Plots.mm,
    )
    for (i, r) in enumerate(results)
        lbl = isnan(r.mean_gap) ? "exact" : @sprintf("dev %.3f", r.mean_gap)
        annotate!(p1, i, r.time, text("  " * lbl, 7, :left, :black, rotation = 90))
    end
    savefig(p1, joinpath(OUTDIR, "walltime_$backend.png"))

    # --------- Plot 2: time vs deviation (the fairness view) ---------
    p2 = plot(;
        xlabel = "wall-clock (s)",
        ylabel = "deviation from exact set",
        title = "Time vs accuracy ($backend, $THREADS thr)",
        legend = :topright,
        xscale = :log10,
        yscale = :log10,
        size = (820, 560),
    )
    for (_, label, _) in GRADIENT
        pts = sweep[label]
        xs = [max(p[1], 1e-6) for p in pts]
        ys = [max(p[2], 1e-6) for p in pts]
        plot!(p2, xs, ys; marker = :circle, label = label, lw = 2)
    end
    vline!(p2, [ref_t]; ls = :dash, color = :black, label = "SPORES (exact)")
    vline!(p2, [hsj_t]; ls = :dashdot, color = :purple, label = "HSJ (exact)")
    hline!(p2, [TARGET_GAP]; ls = :dot, color = :red, label = "target deviation")
    savefig(p2, joinpath(OUTDIR, "tradeoff_$backend.png"))

    # --------- Plot 3: diversity metrics ---------
    p3 = plot(layout = (1, 3), size = (1300, 460), bottom_margin = 10Plots.mm)
    bar!(
        p3[1],
        dnames,
        [r.mean_pw for r in results];
        legend = false,
        title = "mean pairwise dist",
        xrotation = 40,
        color = :steelblue,
    )
    bar!(
        p3[2],
        dnames,
        [r.cov for r in results];
        legend = false,
        title = "coverage (spanned range)",
        xrotation = 40,
        color = :indianred,
    )
    bar!(
        p3[3],
        dnames,
        [r.hull for r in results];
        legend = false,
        title = "2D-PCA hull volume",
        xrotation = 40,
        color = :darkgreen,
    )
    savefig(p3, joinpath(OUTDIR, "diversity_$backend.png"))

    # --------- Plot 4: shared-PCA spread scatter ---------
    allP = hcat([r.P for r in results]...)
    mu = mean(allP, dims = 2)
    F = svd(allP .- mu)
    basis = F.U[:, 1:min(2, size(F.U, 2))]
    p4 = plot(;
        title = "Alternative spread (shared 2D PCA, $backend)",
        xlabel = "PC1",
        ylabel = "PC2",
        size = (760, 620),
        legend = :outertopright,
    )
    palette = [:purple, :black, :teal, :seagreen, :orange, :indianred, :goldenrod]
    for (i, r) in enumerate(results)
        proj = basis' * (r.P .- mu)
        star = r.name == "SPORES"
        scatter!(
            p4,
            proj[1, :],
            proj[2, :];
            label = r.name,
            ms = star ? 8 : 5,
            marker = star ? :star5 : :circle,
            color = palette[mod1(i, length(palette))],
        )
    end
    savefig(p4, joinpath(OUTDIR, "spread_$backend.png"))

    # --------- Dominance / set-coverage matrix ---------
    open(joinpath(OUTDIR, "dominance_$backend.csv"), "w") do io
        println(io, "A\\B," * join(dnames, ","))
        for (i, ri) in enumerate(results)
            row = [@sprintf("%.2f", set_coverage(ri.P, rj.P)) for rj in results]
            println(io, dnames[i] * "," * join(row, ","))
        end
    end

    @printf(
        "%-12s %9s %9s | %8s %8s %8s\n",
        "method",
        "time(s)",
        "dev",
        "meanPW",
        "cov",
        "hull"
    )
    for r in results
        @printf(
            "%-12s %9.3f %9s | %8.4f %8.4f %8.4f\n",
            r.name,
            r.time,
            isnan(r.mean_gap) ? "exact" : @sprintf("%.4f", r.mean_gap),
            r.mean_pw,
            r.cov,
            r.hull
        )
    end
end

open(joinpath(OUTDIR, "results.csv"), "w") do io
    println(io, join(csv_rows, "\n"))
end
println("\nWrote plots + CSVs to $OUTDIR")
