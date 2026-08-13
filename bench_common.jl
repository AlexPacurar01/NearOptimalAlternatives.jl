# ===========================================================================
# Shared benchmark utilities: cold-start optimizers, MGA quality metrics, and
# the plot/CSV outputs. Used by benchmark_mga.jl (synthetic) and
# benchmark_5h.jl (real data/5h model).
# ===========================================================================
using JuMP, LinearAlgebra, Statistics, Printf, Plots
import Gurobi, HiGHS

"""
Cold-start optimizer: interior-point (barrier / IPM) WITHOUT crossover, for
both backends. Barrier re-solves from scratch every call (no warm basis is ever
reused), so every solve is a genuine cold start; and crossover is OFF so the
exact methods return the barrier (interior) solution directly - they are not
charged the extra crossover-to-vertex work that the approximate gradient
methods never do. This makes the wall-clock comparison fair (all methods return
a non-vertex, tolerance-accurate point). Note: HSJ's diversity relies on basic
(vertex) solutions with exact zeros, so without crossover it degenerates - that
is a genuine, reportable property of the HSJ formulation, not a benchmark
artefact.
"""
function make_optimizer(backend::Symbol, threads::Int)
    backend == :gurobi ?
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Threads" => threads,
        "Method" => 2,
        "Crossover" => 0,
    ) :
    optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "threads" => threads,
        "solver" => "ipm",
        "run_crossover" => "off",
    )
end

"""
    ipm_baseline_at_budgets(model, w, vars, budgets; optimizer_factory) -> NamedTuple

Cold-IPM baseline matched to the continuation sweep: for each cost budget `B` it
solves the SAME epsilon-constraint sub-problem the continuation solves
(`min w'x  s.t. original constraints, cost(x) <= B`) from a COLD start, and
records the solver's own reported solve time (`JuMP.solve_time`, the API-call
time - NOT model build or extraction). Mutates `model`'s objective and adds one
budget constraint; intended to run AFTER the continuation has used `model`.

Returns `(times, costs, dobjs, total_time)` aligned with `budgets`.
"""
function ipm_baseline_at_budgets(
    model::Model,
    w::Vector{Float64},
    vars::Vector{VariableRef},
    budgets::Vector{Float64};
    optimizer_factory,
)
    cost_expr = objective_function(model)              # original cost c'x (+ const)
    orig_sense = objective_sense(model)
    kconst = JuMP.constant(cost_expr)
    set_optimizer(model, optimizer_factory)
    set_silent(model)
    what = w ./ max(norm(w), eps(Float64))             # report dobj in the SAME (normalised)
    @objective(model, Min, sum(w[i] * vars[i] for i in eachindex(w) if w[i] != 0))
    bud = @constraint(model, cost_expr <= budgets[1])
    times = Float64[]
    costs = Float64[]
    dobjs = Float64[]
    for B in budgets
        set_normalized_rhs(bud, B - kconst)            # cost(x) <= B  (const moved to RHS)
        JuMP.optimize!(model)
        push!(times, JuMP.solve_time(model))           # solver-reported API time
        xv = value.(vars)
        push!(costs, dot([JuMP.coefficient(cost_expr, v) for v in vars], xv) + kconst)
        push!(dobjs, dot(what, xv))                    # direction as continuation_walk's `what`
    end
    delete(model, bud)
    set_objective_function(model, cost_expr)           # RESTORE so the model can be reused
    set_objective_sense(model, orig_sense)
    return (times = times, costs = costs, dobjs = dobjs, total_time = sum(times))
end

mutable struct MethodResult
    name::String
    time::Float64
    mean_gap::Float64     # decision-space deviation from the exact set (NaN = exact method)
    P::Matrix{Float64}    # normalised structural points, d x N
    mean_pw::Float64
    min_pw::Float64
    cov::Float64
    hull::Float64
end

"Normalise the structural columns of full solution vectors to [0,1] by target bounds."
function normalised_struct(alts, target, target_idx)
    lb = [has_lower_bound(v) ? lower_bound(v) : 0.0 for v in target]
    ub = [has_upper_bound(v) ? upper_bound(v) : 1.0 for v in target]
    rng = max.(ub .- lb, 1e-12)
    P = zeros(length(target), length(alts))
    for (k, a) in enumerate(alts)
        P[:, k] = clamp.((a[target_idx] .- lb) ./ rng, 0.0, 1.0)
    end
    return P
end

function pairwise_stats(P)
    d, N = size(P)
    N < 2 && return (0.0, 0.0)
    ds = [norm(P[:, i] .- P[:, j]) / sqrt(d) for i = 1:N-1 for j = i+1:N]
    return (mean(ds), minimum(ds))
end

coverage(P) = size(P, 2) < 2 ? 0.0 : mean(maximum(P, dims = 2) .- minimum(P, dims = 2))

function pca2(P)
    d, N = size(P)
    mu = mean(P, dims = 2)
    C = P .- mu
    F = svd(C)
    k = min(2, size(F.U, 2))
    proj = F.U[:, 1:k]' * C
    k == 1 && (proj = vcat(proj, zeros(1, N)))
    return proj
end

function hull_area(pts)
    N = size(pts, 2)
    N < 3 && return 0.0
    P = unique!(sort!([(pts[1, i], pts[2, i]) for i = 1:N]))
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
    sum(
        any(norm(A[:, i] .- B[:, j]) / sqrt(size(A, 1)) <= delta for i = 1:size(A, 2)) for
        j = 1:nb
    ) / nb
end

# ===========================================================================
# Cost-dominance (global Pareto front in (original cost, diversity objective)).
#
# Each alternative is summarised by two scalars on a *fixed* search direction w:
#   cost(x) = c'x   - original implementation cost (the LP's own objective),
#   dobj(x) = w'x   - the MGA diversity objective (lower = more different along w).
# Point a dominates b iff it is no worse on both and strictly better on one:
#   c'a <= c'b  and  w'a <= w'b,  with at least one strict.
# Distances from the cost-minimum x* are deliberately NOT used: the front is a
# global trade-off between cost and along-direction difference, independent of
# how far any single point sits from x*.
# ===========================================================================

"Original implementation cost c'x for each alternative (raw full-space vectors)."
function original_costs(cvec::Vector{Float64}, alts::Vector{Vector{Float64}})
    return [dot(cvec, a) for a in alts]
end

"`true` iff (ca, da) dominates (cb, db): no worse on both, strictly better on one."
function pareto_dominates(ca, da, cb, db; atol = 1e-9)
    no_worse = ca <= cb + atol && da <= db + atol
    strictly = ca < cb - atol || da < db - atol
    return no_worse && strictly
end

"""
    nondominated_indices(costs, dobjs) -> Vector{Int}

Indices of the non-dominated (Pareto-optimal) points in the (cost, dobj) plane,
minimising both. O(N^2); N is small (a handful of alternatives per direction).
"""
function nondominated_indices(costs::Vector{Float64}, dobjs::Vector{Float64})
    n = length(costs)
    keep = Int[]
    for i = 1:n
        dominated = any(
            j != i && pareto_dominates(costs[j], dobjs[j], costs[i], dobjs[i]) for j = 1:n
        )
        dominated || push!(keep, i)
    end
    return keep
end

"""
    dominance_report(cand_cost, cand_dobj, ref_cost, ref_dobj) -> NamedTuple

How a candidate set (e.g. a boundary walk) fares against a reference set (e.g.
SPORES) on the (cost, dobj) front. Returns the number and fraction of reference
points dominated by some candidate, and the mean original-cost saving over the
dominated reference points (cost of the dominated ref minus the cheapest
candidate that dominates it).
"""
function dominance_report(
    cand_cost::Vector{Float64},
    cand_dobj::Vector{Float64},
    ref_cost::Vector{Float64},
    ref_dobj::Vector{Float64},
)
    nref = length(ref_cost)
    dominated = falses(nref)
    savings = Float64[]
    for j = 1:nref
        best = Inf
        for i in eachindex(cand_cost)
            if pareto_dominates(cand_cost[i], cand_dobj[i], ref_cost[j], ref_dobj[j])
                dominated[j] = true
                best = min(best, cand_cost[i])
            end
        end
        dominated[j] && push!(savings, ref_cost[j] - best)
    end
    ndom = count(dominated)
    return (
        n_dominated = ndom,
        frac_dominated = nref == 0 ? 0.0 : ndom / nref,
        mean_cost_saving = isempty(savings) ? 0.0 : mean(savings),
        dominated_mask = dominated,
    )
end

"Mean step-wise normalised structural distance between an alternative set and a reference set."
function trajectory_gap(P, Pref)
    n = min(size(P, 2), size(Pref, 2))
    n == 0 && return NaN
    d = size(P, 1)
    return mean(norm(P[:, i] .- Pref[:, i]) / sqrt(d) for i = 1:n)
end

"Build the four benchmark plots + dominance CSV + console table for one backend."
function make_outputs(
    results,
    sweep,
    ref_t,
    hsj_t,
    backend,
    outdir,
    n_alts,
    threads,
    target_gap,
)
    dnames = [r.name for r in results]

    cols = [
        isnan(r.mean_gap) ? :gray : (r.mean_gap <= target_gap ? :seagreen : :orange) for
        r in results
    ]
    p1 = bar(
        dnames,
        [r.time for r in results];
        legend = false,
        color = cols,
        ylabel = "wall-clock (s)",
        title = "Cold-start wall-clock for $n_alts alternatives ($backend, $threads thr)",
        xrotation = 30,
        size = (900, 520),
        bottom_margin = 8Plots.mm,
    )
    for (i, r) in enumerate(results)
        lbl = isnan(r.mean_gap) ? "exact" : @sprintf("dev %.3f", r.mean_gap)
        annotate!(p1, i, r.time, text("  " * lbl, 7, :left, :black, rotation = 90))
    end
    savefig(p1, joinpath(outdir, "walltime_$backend.png"))

    if !isempty(sweep)
        p2 = plot(;
            xlabel = "wall-clock (s)",
            ylabel = "deviation from exact set",
            title = "Time vs accuracy ($backend, $threads thr)",
            legend = :topright,
            xscale = :log10,
            yscale = :log10,
            size = (820, 560),
        )
        for (label, pts) in sweep
            isempty(pts) && continue
            plot!(
                p2,
                [max(p[1], 1e-6) for p in pts],
                [max(p[2], 1e-6) for p in pts];
                marker = :circle,
                label = label,
                lw = 2,
            )
        end
        vline!(p2, [ref_t]; ls = :dash, color = :black, label = "SPORES (exact)")
        vline!(p2, [hsj_t]; ls = :dashdot, color = :purple, label = "HSJ (exact)")
        hline!(p2, [target_gap]; ls = :dot, color = :red, label = "target deviation")
        savefig(p2, joinpath(outdir, "tradeoff_$backend.png"))
    end

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
    savefig(p3, joinpath(outdir, "diversity_$backend.png"))

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
    savefig(p4, joinpath(outdir, "spread_$backend.png"))

    open(joinpath(outdir, "dominance_$backend.csv"), "w") do io
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
            "%-12s %9.1f %9s | %8.4f %8.4f %8.4f\n",
            r.name,
            r.time,
            isnan(r.mean_gap) ? "exact" : @sprintf("%.4f", r.mean_gap),
            r.mean_pw,
            r.cov,
            r.hull
        )
    end
end
