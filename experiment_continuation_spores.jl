# ===========================================================================
# Continuation-vs-SPORES experiment suite (exact, runs entirely on Gurobi/HiGHS).
#
# Idea: SPORES picks a few search DIRECTIONS and returns one alternative per
# direction (at the full epsilon-budget). We reuse those exact directions, but
# instead of one endpoint we sweep the cost budget and collect every solution in
# between. We then ask: do any of these in-between points strictly dominate the
# SPORES endpoints (on investment usage)? If not, we measure how much denser /
# more spread the continuation set is.
#
# Pipeline (per run):
#   1. solve base cost-min -> x*, C*;  add budget cost <= (1+eps)C*.
#   2. SPORES for n_dir iterations -> directions w_1..w_n_dir and their endpoints.
#   3. for each w_k, sweep B in [C*, (1+eps)C*] (n_budget points), solve exactly,
#      remember every solution.
#   4. analyse: strict investment-dominance of continuation points over SPORES,
#      and spread/coverage of the non-dominated sets.
# ===========================================================================
using JuMP, LinearAlgebra, Printf, Statistics
import MathOptInterface as MOI
include(joinpath(@__DIR__, "bench_common.jl"))   # make_optimizer + spread metrics

"Budget grid from Cstar to Bmax, nudged off the exact Cstar endpoint: `cost_expr
<= Cstar` is a zero-slack constraint at the optimum, which a barrier solve
(Crossover=0) can fail to satisfy at floating-point tolerance on a large model
- continuation_exact's very first point on the DHPC 2050 run hit exactly this
and returned 0 solutions. A tiny relative nudge keeps the swept range visually
identical while giving the solver slack to actually find a feasible point."
budget_grid(Cstar, Bmax, n_budget) =
    collect(range(Cstar + max(abs(Cstar) * 1e-6, 1e-6), Bmax, length = n_budget))

"Run SPORES for `n_dir` iterations. Returns C*, B_max, the cost expr (+const),
the list of accumulated direction vectors (over all variables, non-zero only on
`inv_idx`), and the endpoint solution of each direction.

`base_solution = (Cstar, xstar)` skips the initial cold solve entirely (e.g. a
cached baseline from run_2050_baseline.jl / base_solution_*.bin) - essential
when the base solve alone is a multi-hour barrier run that every experiment
arm would otherwise repeat. `point_sink(k, sol, w)` is called after every
SPORES direction solves, for incremental recording on long runs. `on_direction(k, w)`
is also called after every direction (before moving on to accumulate direction
k+1) - used by `run_suite(...; interleaved=true)` to run that direction's
continuation sweep immediately, so a timeout only loses the LAST direction's
sweep instead of every direction's."
function spores_directions(
    model,
    allv,
    inv_idx,
    ub_inv;
    n_dir,
    eps,
    base_solution = nothing,
    point_sink = nothing,
    on_direction = nothing,
)
    t0 = time()
    if base_solution === nothing
        JuMP.optimize!(model)
        @assert is_solved_and_feasible(model)
        Cstar = objective_value(model)
        xstar = value.(allv)
        @printf("[%6.1fs] base solve done | C*=%.6g\n", time() - t0, Cstar)
    else
        Cstar, xstar = base_solution
        @printf("[%6.1fs] base solve SKIPPED (cached) | C*=%.6g\n", time() - t0, Cstar)
    end
    flush(stdout)
    cost_expr = objective_function(model)
    kconst = JuMP.constant(cost_expr)
    xcur = copy(xstar)                                # base solution, captured before edits
    Bmax = (1 + eps) * Cstar
    bud = @constraint(model, cost_expr <= Bmax)
    w = zeros(length(allv))
    dirs = Vector{Vector{Float64}}()
    alts = Vector{Vector{Float64}}()
    for k = 1:n_dir
        for (jj, j) in enumerate(inv_idx)
            w[j] += xcur[j] / ub_inv[jj]             # SPORES accumulation rule
        end
        @objective(model, Min, sum(w[j] * allv[j] for j in inv_idx))
        ts = time()
        JuMP.optimize!(model)
        xcur = value.(allv)
        push!(dirs, copy(w))
        push!(alts, copy(xcur))
        @printf(
            "[%6.1fs] SPORES direction %d/%d solved (%.1fs)\n",
            time() - t0,
            k,
            n_dir,
            time() - ts
        )
        flush(stdout)
        point_sink === nothing || point_sink(k, xcur, w)
        on_direction === nothing || on_direction(k, w)
    end
    delete(model, bud)
    set_objective_function(model, cost_expr)
    set_objective_sense(model, MOI.MIN_SENSE)
    return Cstar, Bmax, cost_expr, kconst, dirs, alts, xstar
end

"Continuation for a fixed direction `w`: solve min w'x s.t. constraints, cost<=B
exactly for every B in `budgets`. Returns the full solution at each budget.
`point_sink(method, k, bi, sol, w)` is called after every solve for incremental
recording; `label`/`k` identify which direction is being swept for progress logs."
function continuation_exact(
    model,
    w,
    allv,
    budgets,
    cost_expr,
    kconst;
    k = 0,
    n_dir = 0,
    point_sink = nothing,
)
    nz = findall(!iszero, w)
    @objective(model, Min, sum(w[j] * allv[j] for j in nz))
    bud = @constraint(model, cost_expr <= budgets[1])
    sols = Vector{Vector{Float64}}()
    t0 = time()
    for (bi, B) in enumerate(budgets)
        set_normalized_rhs(bud, B - kconst)
        ts = time()
        JuMP.optimize!(model)
        if !is_solved_and_feasible(model)
            # A single degenerate/numerically-hard budget point (this happens
            # right at B=Cstar, a zero-slack constraint at the barrier's
            # tolerance) used to crash the entire multi-hour task via
            # value.(allv) on a model with 0 solutions. Skip it and keep going
            # - one missing point is far cheaper than losing every direction
            # after it.
            @warn "continuation point failed, skipping" k bi B status =
                termination_status(model)
            flush(stdout)
            continue
        end
        sol = value.(allv)
        push!(sols, sol)
        @printf(
            "[%6.1fs] direction %d/%d, budget %d/%d solved (%.1fs)\n",
            time() - t0,
            k,
            n_dir,
            bi,
            length(budgets),
            time() - ts
        )
        flush(stdout)
        point_sink === nothing || point_sink("cont", k, bi, sol, w)
    end
    delete(model, bud)
    set_objective_function(model, cost_expr)
    set_objective_sense(model, MOI.MIN_SENSE)
    return sols
end

"Investment-usage dominance: a dominates b iff a_j <= b_j for all investment j
and a_k < b_k for at least one (Pareto-efficient in resource usage)."
function inv_dominates(a, b, inv_idx; atol = 1e-7)
    nob = true
    strict = false
    @inbounds for j in inv_idx
        a[j] > b[j] + atol && (nob = false; break)
        a[j] < b[j] - atol && (strict = true)
    end
    return nob && strict
end

"Indices of the non-dominated points (investment usage) within `pts`."
function nondominated(pts, inv_idx)
    n = length(pts)
    keep = Int[]
    for i = 1:n
        dom = any(j != i && inv_dominates(pts[j], pts[i], inv_idx) for j = 1:n)
        dom || push!(keep, i)
    end
    return keep
end

"""
    run_suite(model, allv, inv_idx, ub_inv; n_dir, eps, n_budget,
              base_solution = nothing, point_sink = nothing)

`base_solution = (Cstar, xstar)` forwards to `spores_directions` to skip the
initial cold solve (see its docstring). `point_sink(method, k, bi, sol, w)` is
forwarded to `continuation_exact` for incremental per-point recording (`bi`
is `0`/`n_budget` for the SPORES endpoint itself); pass one on long DHPC runs
so partial results survive a preemption or timeout.

`interleaved = true` runs each direction's continuation sweep immediately
after that direction's SPORES anchor solves, instead of solving all `n_dir`
anchors first and sweeping afterwards. On a run that may not finish in the
wall-clock budget, this means a timeout only loses the sweep for whichever
direction was in progress, rather than every direction's sweep (which is what
happened when all `n_dir` anchors alone consumed the full budget). Requires
`base_solution` to be given, since the budget grid needs Cstar up front.
"""
function run_suite(
    model,
    allv,
    inv_idx,
    ub_inv;
    n_dir = 5,
    eps = 0.1,
    n_budget = 11,
    base_solution = nothing,
    point_sink = nothing,
    interleaved = false,
)
    spores_sink =
        point_sink === nothing ? nothing :
        (k, sol, w) -> point_sink("spores", k, n_budget, sol, w)

    cont = Vector{Vector{Vector{Float64}}}()   # cont[k] = solutions along direction k
    on_direction = nothing
    budgets_box = Vector{Float64}[]
    cost_expr_box = []
    kconst_box = Float64[]
    if interleaved
        @assert base_solution !== nothing "run_suite(...; interleaved=true) requires base_solution=(Cstar, xstar) so the budget grid is known up front"
        Cstar0, _ = base_solution
        Bmax0 = (1 + eps) * Cstar0
        push!(budgets_box, budget_grid(Cstar0, Bmax0, n_budget))
        push!(cost_expr_box, objective_function(model))
        push!(kconst_box, JuMP.constant(cost_expr_box[1]))
        on_direction =
            (k, w) -> push!(
                cont,
                continuation_exact(
                    model,
                    w,
                    allv,
                    budgets_box[1],
                    cost_expr_box[1],
                    kconst_box[1];
                    k = k,
                    n_dir = n_dir,
                    point_sink = point_sink,
                ),
            )
    end

    Cstar, Bmax, cost_expr, kconst, dirs, alts, xstar = spores_directions(
        model,
        allv,
        inv_idx,
        ub_inv;
        n_dir = n_dir,
        eps = eps,
        base_solution = base_solution,
        point_sink = spores_sink,
        on_direction = on_direction,
    )
    budgets = interleaved ? budgets_box[1] : budget_grid(Cstar, Bmax, n_budget)
    if !interleaved
        for (k, w) in enumerate(dirs)
            push!(
                cont,
                continuation_exact(
                    model,
                    w,
                    allv,
                    budgets,
                    cost_expr,
                    kconst;
                    k = k,
                    n_dir = n_dir,
                    point_sink = point_sink,
                ),
            )
        end
    end

    # Continuation vs interpolation: a classic MGA user wanting intermediate points
    # along direction k would interpolate the feasible chord from x* to the endpoint
    # alts[k] (feasible by convexity). Does the exact continuation point dominate the
    # interpolated one (on investment usage) at the same budget fraction?
    dom_interp = 0
    tot_interp = 0
    divgaps = Float64[]
    for k = 1:n_dir, bi = 1:n_budget
        frac = (budgets[bi] - Cstar) / max(Bmax - Cstar, 1.0e-12)
        interp = (1 - frac) .* xstar .+ frac .* alts[k]
        tot_interp += 1
        inv_dominates(cont[k][bi], interp, inv_idx) && (dom_interp += 1)
        # diversity gap: continuation is optimal for w_k at this budget, so its w'x is
        # <= the chord's. This measures how far the true efficient front bulges below
        # the interpolation a classic MGA user would draw (higher = interpolation worse).
        wk = dirs[k]
        c_wx = dot(wk, cont[k][bi])
        i_wx = dot(wk, interp)
        push!(divgaps, (i_wx - c_wx) / max(abs(i_wx), 1.0e-12))
    end

    # --- analysis ---
    cont_flat = reduce(vcat, cont)                       # all continuation points
    # 1) strict dominance of SPORES endpoints by continuation points
    dom_count = 0
    for a in alts
        if any(inv_dominates(c, a, inv_idx) for c in cont_flat)
            dom_count += 1
        end
    end
    # 2) non-dominated structure over the union
    allpts = vcat(alts, cont_flat)
    tag = vcat(fill(:spores, length(alts)), fill(:cont, length(cont_flat)))
    nd = nondominated(allpts, inv_idx)
    nd_spores = count(i -> tag[i] == :spores, nd)
    nd_cont = count(i -> tag[i] == :cont, nd)
    # 3) spread of each set's non-dominated investment points (normalised [0,1])
    invmat(pts) = hcat([[(p[inv_idx[k]]) for k in eachindex(inv_idx)] for p in pts]...)
    rng = max.(ub_inv, 1e-9)
    norm_cols(M) = M ./ rng
    sp_nd = norm_cols(invmat(alts))
    co_nd_idx = [i - length(alts) for i in nd if tag[i] == :cont]
    co_nd = norm_cols(invmat(cont_flat[co_nd_idx]))
    sp_mean, sp_min = pairwise_stats(sp_nd)
    co_mean, co_min = pairwise_stats(co_nd)

    println("\n================  Continuation vs SPORES  ================")
    @printf("C*=%.6g  Bmax=%.6g  n_dir=%d  n_budget=%d\n", Cstar, Bmax, n_dir, n_budget)
    @printf("points: SPORES=%d  continuation=%d\n", length(alts), length(cont_flat))
    @printf(
        "SPORES endpoints strictly dominated by a continuation point: %d / %d\n",
        dom_count,
        length(alts)
    )
    @printf(
        "interpolated MGA points strictly dominated by the continuation: %d / %d\n",
        dom_interp,
        tot_interp
    )
    @printf(
        "diversity gap (continuation below interpolation chord): mean=%.2f%%  max=%.2f%%\n",
        100 * mean(divgaps),
        100 * maximum(divgaps)
    )
    @printf(
        "non-dominated union: total=%d  (SPORES=%d, continuation=%d)\n",
        length(nd),
        nd_spores,
        nd_cont
    )
    @printf(
        "spread (mean pairwise, normalised invest space): SPORES=%.4f  continuation-ND=%.4f\n",
        sp_mean,
        co_mean
    )
    return (;
        Cstar,
        Bmax,
        dirs,
        alts,
        cont,
        budgets,
        dom_count,
        nd,
        nd_spores,
        nd_cont,
        sp_mean,
        co_mean,
    )
end

# ===========================================================================
# Synthetic driver (fast validation). Swap the model block for data/5h later.
# ===========================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    using HiGHS, Random
    Random.seed!(42)
    n_struct = 8
    T = 24
    cap_max = 50.0 .+ 50.0 .* rand(n_struct)
    c_inv = 10.0 .+ 10.0 .* rand(n_struct)
    c_op = 1.0 .+ 5.0 .* rand(n_struct, T)
    demand = 20.0 .+ 10.0 .* rand(T)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= assets_investment[i = 1:n_struct] <= cap_max[i])
    @variable(model, dispatch[i = 1:n_struct, t = 1:T] >= 0)
    @constraint(model, [i = 1:n_struct, t = 1:T], dispatch[i, t] <= assets_investment[i])
    @constraint(model, [t = 1:T], sum(dispatch[i, t] for i = 1:n_struct) >= demand[t])
    @objective(
        model,
        Min,
        sum(c_inv[i] * assets_investment[i] for i = 1:n_struct) +
        sum(c_op[i, t] * dispatch[i, t] for i = 1:n_struct, t = 1:T)
    )
    allv = all_variables(model)
    idx = Dict(v => i for (i, v) in enumerate(allv))
    inv_idx = [idx[v] for v in assets_investment]
    ub_inv = [upper_bound(v) for v in assets_investment]
    run_suite(model, allv, inv_idx, ub_inv; n_dir = 5, eps = 0.1, n_budget = 11)
    println("DONE")
end
