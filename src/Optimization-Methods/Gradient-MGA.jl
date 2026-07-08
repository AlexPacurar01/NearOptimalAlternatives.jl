# ===========================================================================
# Gradient-based Modeling-to-Generate-Alternatives (MGA) optimization method
#
# Finds near-optimal alternatives by solving the MGA program with the
# Hessian-free first-order solvers in `First-Order-Solvers.jl`, either over
# the full variable space (`:fullspace`, solver selected by `method`) or via
# a structural cutting-plane decomposition with a budget-projection oracle
# (`:oracle`). This is an *optimization* method (gradient descent / L-BFGS),
# not a metaheuristic.
# ===========================================================================

export run_lbfgs_mga, operational_recovery!


"""
    recovered, max_dist, max_dev = operational_recovery!(model, lazy_solution, target_indices, max_allowed_cost)

L2-project the search result onto the exact feasible set: minimise
`sum((x[i] - max(0, lazy_solution[i]))^2)` over the target indices subject to all
model constraints and `cost(x) <= max_allowed_cost`. Temporarily mutates `model`
(budget row + objective) and restores it. Returns `nothing` if the projection
fails.
"""
function operational_recovery!(
    model::Model,
    lazy_solution::Vector{Float64},
    target_indices::Vector{Int},
    max_allowed_cost::Float64,
)
    original_obj = objective_function(model)
    original_sense = objective_sense(model)

    mga_budget_con = @constraint(model, original_obj <= max_allowed_cost)
    original_vars = all_variables(model)

    # Optimization targets only the structural indices directly
    @objective(
        model,
        Min,
        sum(
            (original_vars[idx] - max(0.0, lazy_solution[idx]))^2 for idx in target_indices
        )
    )

    set_silent(model)
    JuMP.optimize!(model)

    if termination_status(model) != MOI.OPTIMAL &&
       termination_status(model) != MOI.LOCALLY_SOLVED
        @warn "Projection failed! L-BFGS guess was completely irrecoverable."
        delete(model, mga_budget_con)
        set_objective_function(model, original_obj)
        set_objective_sense(model, original_sense)
        return nothing
    end

    max_distance = 0.0
    target_scale = 0.0

    for idx in target_indices
        target_val = max(0.0, lazy_solution[idx])
        actual_val = JuMP.value(original_vars[idx])

        max_distance = max(max_distance, abs(target_val - actual_val))
        target_scale = max(target_scale, abs(target_val), abs(actual_val))
    end

    # Relative deviation measured against the overall scale of the target, not
    # against each component: a per-component ratio explodes to meaningless
    # values (thousands of percent) whenever a target component sits near zero,
    # even though the absolute move was tiny.
    max_percentage = max_distance / max(target_scale, sqrt(eps(Float64)))

    println("Projection successful! Max distance from L-BFGS guess: $max_distance")
    println(
        "Projection successful! Max deviation relative to target scale: $(round(100 * max_percentage, digits = 4))%",
    )

    recovered_solution = JuMP.value.(all_variables(model))

    delete(model, mga_budget_con)
    set_objective_function(model, original_obj)
    set_objective_sense(model, original_sense)

    return recovered_solution, max_distance, max_percentage
end

"""
    fullspace_search(model, w, B, vars; method, max_iters, max_inner, pdhg_iters)

The full-space, barrier-free variant of the search: build the MGA LP once
(`build_mga_lp`) and solve it over *all* variables with a Hessian-free
first-order `method` (`:alm_lbfgs` default, `:penalty`, or `:pdhg`) - no LP
solver calls at all. Returns the unscaled solution. On stiff models (long
storage equality chains) first-order iterations can plateau before reaching
feasibility, which is exactly what the method comparison is meant to expose.
"""
function fullspace_search(
    model::Model,
    w::Vector{Float64},
    B::Float64,
    vars::Vector{VariableRef};
    method::Symbol = :alm_lbfgs,
    max_iters::Int = 30,
    max_inner::Int = 1000,
    pdhg_iters::Int = 10000,
    x_start::Union{Nothing,Vector{Float64}} = nothing,
)
    lp = build_mga_lp(model, w, B, vars)
    @info "Constraint extraction + equilibration: $(size(lp.A_in, 1)) inequality rows, $(size(lp.A_eq, 1)) equality rows, $(lp.n) variables in $(round(lp.t_extract, digits=1))s"

    # Initial iterate: the box-clamped origin by default, or the warm-start
    # point `x_start` (original space, scaled by `./ d_scale`) when given.
    x0 =
        x_start === nothing ? clamp.(zeros(lp.n), lp.lb_t, lp.ub_t) :
        clamp.(x_start ./ lp.d_scale, lp.lb_t, lp.ub_t)

    x_t, info = solve_firstorder(
        method,
        lp,
        x0;
        max_iters = max_iters,
        max_inner = max_inner,
        pdhg_iters = pdhg_iters,
        verbose = true,
    )
    x = lp.d_scale .* x_t
    @info "Full-space $(method) finished in $(round(info.time, digits=1))s: w'x=$(round(dot(w, x), sigdigits=6)) | infeas=$(round(info.infeas, sigdigits=4))"

    return x
end

"""
    oracle_search(model, w, B, vars, target_indices; max_iters, max_inner)

Kelley cutting-plane search in the structural space with a soft-fix oracle
(`search = :oracle`):

  - Master `min w_s'x_s s.t. cuts, lb_s <= x_s <= ub_s` (structural vars only),
    solved by `solve_alm_lbfgs`.
  - Oracle: the budget-feasibility distance

        v(x_s) = min ||x[target] - x_s||_1  s.t. model constraints, cost(x) <= B

    via slacks `s+, s- >= 0`, soft rows `x[target] - s+ + s- == x_s`, and the
    budget row, all added to `model` once (only the soft-row RHS change between
    iterations). The budget is attainable, so every oracle call is a feasible LP.
  - `v` is the LP value function (convex in the RHS `x_s`) with subgradient
    `d = dual.(soft rows)`, giving the L-shaped feasibility cut (Van Slyke &
    Wets) `d'x <= d'x_s - v(x_s)`, violated iff `v(x_s) > 0`. Kelley's method on
    `{v <= 0}` minimising `w_s'x_s` converges to the MGA optimum.
  - Every oracle solution is model-feasible and on budget, so the best one seen
    is kept as the incumbent (an early stop still returns a valid point).

Stops when `v(x_s) <= 0`, when the master saturates, or after `max_iters` calls.
"""
function oracle_search(
    model::Model,
    w::Vector{Float64},
    B::Float64,
    vars::Vector{VariableRef},
    target_indices::Vector{Int};
    max_iters::Int = 30,
    max_inner::Int = 1000,
)
    n_struct = length(target_indices)
    w_s = w[target_indices]

    lb_s = fill(-Inf, n_struct)
    ub_s = fill(Inf, n_struct)
    for (j, idx) in enumerate(target_indices)
        v = vars[idx]
        has_lower_bound(v) && (lb_s[j] = lower_bound(v))
        has_upper_bound(v) && (ub_s[j] = upper_bound(v))
    end

    original_obj = objective_function(model)
    original_sense = objective_sense(model)

    s_plus = @variable(model, [1:n_struct], lower_bound = 0.0)
    s_minus = @variable(model, [1:n_struct], lower_bound = 0.0)
    soft = @constraint(
        model,
        [j = 1:n_struct],
        vars[target_indices[j]] - s_plus[j] + s_minus[j] == 0.0
    )
    # Budget row scaled by the largest objective coefficient: identical to
    # cost(x) <= B but keeps coefficients <= 1, avoiding the wide coefficient
    # range that sends barrier into a long degenerate tail.
    budget_scale =
        original_obj isa GenericAffExpr ?
        max(maximum(abs(c) for (_, c) in original_obj.terms), 1.0) : 1.0
    budget_con = @constraint(model, original_obj / budget_scale <= B / budget_scale)
    @objective(model, Min, sum(s_plus) + sum(s_minus))

    # Barrier without crossover (Gurobi): duals come straight from the barrier
    # solution, no basis needed. BarConvTol is loosened since inexact cuts are
    # standard in Kelley/Benders schemes.
    solver = try
        solver_name(model)
    catch
        ""
    end
    if startswith(solver, "Gurobi")
        set_optimizer_attribute(model, "Method", 2)
        set_optimizer_attribute(model, "Crossover", 0)
        set_optimizer_attribute(model, "BarConvTol", 1e-6)
    end

    # First iterate: the closed-form minimiser of the cut-free master - each
    # component at the bound its weight prefers. Derived purely from the
    # problem data, *not* from any previously computed optimal solution.
    x_s = Vector{Float64}(undef, n_struct)
    for j = 1:n_struct
        preferred = w_s[j] >= 0 ? lb_s[j] : ub_s[j]
        x_s[j] = isfinite(preferred) ? preferred : clamp(0.0, lb_s[j], ub_s[j])
    end

    cuts_A = Array{Float64}(undef, 0, n_struct)
    cuts_b = Float64[]
    no_eq = spzeros(0, n_struct)

    # Incumbent = best MGA objective seen; its structural part is the proximal
    # bundle's stability center (moves only on a serious step), keeping master
    # proposals where the cut model is accurate rather than zig-zagging.
    incumbent = nothing
    incumbent_obj_s = Inf
    x_center = Float64[]
    # Proximal weight by a simplified serious/null-step rule: halved on an
    # improving (serious) step, raised toward the data-derived base weight on a
    # null step.
    prox_w = 0.0

    for it = 1:max_iters
        for j = 1:n_struct
            set_normalized_rhs(soft[j], x_s[j])
        end
        local t_solve = @elapsed JuMP.optimize!(model)

        if !(termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)) ||
           !has_duals(model)
            @warn "Budget-projection oracle failed at iteration $it ($(termination_status(model))); stopping."
            break
        end

        v_dist = objective_value(model)
        d = [JuMP.dual(soft[j]) for j = 1:n_struct]
        x_actual = [JuMP.value(vars[idx]) for idx in target_indices]
        obj_s_actual = dot(w_s, x_actual)
        # v is an L1 distance in the structural variables' own units.
        v_tol = sqrt(eps(Float64)) * max(1.0, norm(x_actual, 1))

        @info "MGA oracle iteration $it: v=$(round(v_dist, sigdigits=6)) | w_s'x=$(round(obj_s_actual, sigdigits=6)) | solve $(round(t_solve, digits=1))s"

        serious = obj_s_actual < incumbent_obj_s
        if serious
            incumbent_obj_s = obj_s_actual
            incumbent = JuMP.value.(vars)
            x_center = x_actual
        end
        base_w = norm(w_s) / max(1.0, norm(x_center))
        prox_w = serious ? prox_w / 2 : max(base_w, 2 * prox_w)

        # v == 0: the master's point is attainable within budget, and the
        # cut model already deemed it optimal for w_s - stop.
        if v_dist <= v_tol
            break
        end

        cuts_A = vcat(cuts_A, reshape(d, 1, :))
        push!(cuts_b, dot(d, x_s) - v_dist)

        # Small box problem with a few cut rows - solved to the solver's own
        # defaults (cheap), independent of the oracle budget.
        x_new, _ = solve_alm_lbfgs(
            w_s,
            sparse(cuts_A),
            copy(cuts_b),
            no_eq,
            Float64[],
            lb_s,
            ub_s,
            x_s;
            max_inner = max_inner,
            verbose = false,
            prox_weight = prox_w,
            prox_center = x_center,
        )

        # All cuts are global: a master returning the same point cannot
        # produce new information from re-solving the oracle there.
        if norm(x_new - x_s) <= sqrt(eps(Float64)) * max(1.0, norm(x_s))
            @info "Master returned the current iterate; cutting-plane model is saturated, stopping."
            break
        end
        x_s = x_new
    end

    # Teardown: remove the soft-fix structure, the budget row, and restore
    # the objective.
    delete.(model, soft)
    delete.(model, s_plus)
    delete.(model, s_minus)
    delete(model, budget_con)
    set_objective_function(model, original_obj)
    set_objective_sense(model, original_sense)

    # The incumbent is feasible and on budget by construction - it is the
    # answer whether or not a repair projection follows.
    return incumbent
end

"""
    final_point = run_lbfgs_mga(
        model, w, actual_max_cost, repair, vars, target_indices;
        max_iters, max_inner, search,
    )

Finds an MGA alternative `min w'x  s.t.  x in F, cost(x) <= actual_max_cost`
(`F` = the model's own constraint set), with the L-BFGS-driven search
selected by `search`:

  - `:oracle` (default): Kelley cutting-plane master (ALM + projected L-BFGS)
    with a soft-fix LP oracle generating one cut per call — see `oracle_search`.
    Wall-clock is dominated by a handful of barrier re-solves of the model.
  - `:fullspace`: barrier-free first-order search over all variables, no LP
    solver calls — see `fullspace_search`. The `method` keyword picks the
    Hessian-free solver (`:alm_lbfgs`, `:penalty`, `:pdhg`).

By default the search starts cold (from the box-clamped origin) and so never
uses the model's optimal solution - only the scalar budget `actual_max_cost`
that defines near-optimality. Passing `x_start` (the cost-optimal point or the
previous alternative) instead warm-starts the `:fullspace` search from that
feasible interior point, mirroring the sequential standard MGA methods. If
`repair` is set, the search result is projected onto the exact feasible set
with the budget enforced (`operational_recovery!`).
"""
function run_lbfgs_mga(
    model::Model,
    w::Vector{Float64},
    actual_max_cost::Float64,
    repair::Bool,
    vars::Vector{VariableRef},
    target_indices::Vector{Int};
    max_iters::Int = 30,
    max_inner::Int = 1000,
    search::Symbol = :oracle,
    method::Symbol = :alm_lbfgs,
    pdhg_iters::Int = 10000,
    x_start::Union{Nothing,Vector{Float64}} = nothing,
)
    w = w ./ max(norm(w), eps(Float64))
    B = actual_max_cost

    final_point =
        search === :fullspace ?
        fullspace_search(
            model,
            w,
            B,
            vars;
            method = method,
            max_iters = max_iters,
            max_inner = max_inner,
            pdhg_iters = pdhg_iters,
            x_start = x_start,
        ) : oracle_search(model, w, B, vars, target_indices; max_iters, max_inner)

    if isnothing(final_point)
        @warn "Search produced no point; no alternative generated."
        return nothing
    end

    if repair
        recovery_result =
            operational_recovery!(model, final_point, target_indices, actual_max_cost)
        if !isnothing(recovery_result)
            recovered, _, _ = recovery_result
            final_point = recovered
        end
    end

    return final_point
end
