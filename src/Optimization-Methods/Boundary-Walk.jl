# ===========================================================================
# Warm-started boundary walk for near-optimal alternatives.
#
# Idea: instead of one SPORES-style solve at the full cost budget, walk the
# cost-budget boundary outward from the cost-minimum x*, solving the MGA
# program at a sweep of budgets B_k. Every solve is warm-started from x*
# (never the previous boundary point - path-independent, so a tighter budget
# can never hide a better point that a looser one found), and every returned
# point is a converged, feasible first-order solution. The output is a set of
# alternatives spread along the (original cost, diversity objective) trade-off,
# from which the non-dominated frontier is kept.
#
# Two ways to choose the optimisation direction (Phase 4):
#   :raw        - min w'x s.t. cost <= B_k        (the literal SPORES LP; at the
#                 full budget this reproduces the SPORES point).
#   :cost_aware - min (w_hat + gamma * c_hat)'x s.t. cost <= B_k  (w and the cost
#                 gradient unit-normalised). The small cost term stops the solver
#                 from spending budget on the cost-parallel component of the
#                 SPORES move - the perpendicular (Delta_perp) construction that
#                 can dominate SPORES on original cost.
#
# The diversity axis reported for *every* point is always w'x with the original
# SPORES direction w, so :raw and :cost_aware are compared on one common front.
# ===========================================================================

export boundary_walk, BoundaryWalkPoint

"One converged boundary-walk solution and its place on the (cost, dobj) front."
struct BoundaryWalkPoint
    budget::Float64      # B_k the solve was run at
    cost::Float64        # original implementation cost c'x (+ offset)
    dobj::Float64        # diversity objective w'x (original SPORES direction)
    time::Float64        # wall-clock for this solve
    infeas::Float64      # primal infeasibility of the (scaled) solution
    x::Vector{Float64}   # full-space solution (unscaled, aligned to all_variables)
end

"`true` iff point a dominates b in (cost, dobj), minimising both (one strict)."
function _walk_dominates(a::BoundaryWalkPoint, b::BoundaryWalkPoint; atol = 1e-9)
    no_worse = a.cost <= b.cost + atol && a.dobj <= b.dobj + atol
    strictly = a.cost < b.cost - atol || a.dobj < b.dobj - atol
    return no_worse && strictly
end

"Original objective coefficients aligned to `vars`, plus the constant offset."
function _objective_costs(model::Model, vars::Vector{VariableRef})
    data = lp_matrix_data(model)
    col = Dict(v => j for (j, v) in enumerate(data.variables))
    c = [data.c[col[v]] for v in vars]
    return c, data.c_offset
end

"""
    _augment_lex(lp, w_hat, cvec, dobj1; reltol)

Second-phase LP for the lexicographic walk: same scaled problem as `lp` but
(1) with an extra inequality row pinning the diversity to phase-1's level,
`w_hat'x <= dobj1*(1+reltol)`, and (2) the objective replaced by the original
cost `c'x` (both expressed in the scaled space `x = d_scale .* x_t` and the cost
unit-normalised). Solving it finds the cheapest point that keeps SPORES-level
diversity - the parameter-free `Delta_perp` (gamma -> 0+) point.
"""
function _augment_lex(
    lp,
    w_hat::Vector{Float64},
    cvec::Vector{Float64},
    dobj1::Float64;
    reltol::Float64,
)
    a = w_hat .* lp.d_scale                       # w_hat'x = (w_hat .* d_scale)'x_t
    rhs = dobj1 + reltol * max(1.0, abs(dobj1))
    A_in2 = vcat(lp.A_in, sparse(reshape(a, 1, :)))
    b_in2 = vcat(lp.b_in, rhs)
    c_scaled = cvec .* lp.d_scale                 # c'x = (c .* d_scale)'x_t
    c_scaled ./= max(norm(c_scaled), eps(Float64))
    return merge(lp, (w_t = c_scaled, A_in = A_in2, b_in = b_in2))
end

"""
    _exact_solve(model, all_vars, obj_w, B, orig_obj, orig_sense, nz_obj; extra_w, extra_nz, extra_bound)

Solve one budget subproblem `min obj_w'x s.t. orig_obj <= B, model constraints`
(optionally with the extra diversity row `extra_w'x <= extra_bound`) using the
model's *attached* optimiser (e.g. Gurobi barrier) - the exact reference walk,
free of first-order convergence error. Temporarily mutates `model` (budget row,
optional diversity row, objective) and restores it. `nz_obj`/`extra_nz` are the
nonzero index lists of `obj_w`/`extra_w` (so the objective/constraint expressions
are built only over the variables that appear). Returns `(x, solve_time)`.
"""
function _exact_solve(
    model::Model,
    all_vars::Vector{VariableRef},
    obj_w::Vector{Float64},
    B::Float64,
    orig_obj,
    orig_sense,
    nz_obj::Vector{Int};
    extra_w::Union{Nothing,Vector{Float64}} = nothing,
    extra_nz::Vector{Int} = Int[],
    extra_bound::Float64 = 0.0,
)
    budget_con = @constraint(model, orig_obj <= B)
    extra_con =
        extra_w === nothing ? nothing :
        @constraint(model, sum(extra_w[i] * all_vars[i] for i in extra_nz) <= extra_bound)
    @objective(model, Min, sum(obj_w[i] * all_vars[i] for i in nz_obj))

    t = @elapsed JuMP.optimize!(model)
    ok = termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    x = ok ? value.(all_vars) : fill(NaN, length(all_vars))

    extra_con === nothing || delete(model, extra_con)
    delete(model, budget_con)
    set_objective_function(model, orig_obj)
    set_objective_sense(model, orig_sense)
    return x, t
end

"""
    result = boundary_walk(model, w, vars; kwargs...)

Walk the cost-budget boundary outward from the cost-minimum, returning a set of
near-optimal alternatives along the (cost, dobj) trade-off.

`model` must already be solved (its optimal point is the warm start x* and its
objective value the budget anchor `min_cost`). `w` is the full-length SPORES
search direction over `vars` (`= all_variables(model)`).

# Keywords
- `eps_slack`: relative cost slack; the widest budget is `min_cost*(1+eps_slack)`
  (the SPORES budget). Budgets sweep as `min_cost + (k/n_steps)*eps_slack*|min_cost|`.
- `n_steps`: number of budget points along the walk.
- `method`: first-order solver (`:alm_lbfgs`, `:penalty`, `:pdhg`).
- `direction`: `:raw` or `:cost_aware` (see file header).
- `gamma`: cost weight for `:cost_aware` (ignored for `:raw`).
- `early_stop_k`: stop once this many *consecutive* new points are dominated by
  the running non-dominated frontier (they add nothing). `0` disables.
- `max_iters`, `max_inner`, `pdhg_iters`: solver budgets, passed through.

# Returns
NamedTuple `(points, frontier, total_time)`:
- `points::Vector{BoundaryWalkPoint}` - every solve actually run, in walk order.
- `frontier::Vector{Int}` - indices into `points` of the non-dominated set.
- `total_time::Float64` - summed solver wall-clock.
"""
function boundary_walk(
    model::Model,
    w::Vector{Float64},
    vars::Vector{VariableRef};
    eps_slack::Float64 = 0.1,
    n_steps::Int = 10,
    method::Symbol = :alm_lbfgs,
    direction::Symbol = :raw,
    gamma::Float64 = 0.05,
    lex_tol::Float64 = 1.0e-5,
    early_stop_k::Int = 3,
    max_iters::Int = 30,
    max_inner::Int = 1000,
    pdhg_iters::Int = 10000,
)
    @assert is_solved_and_feasible(model) "boundary_walk needs a solved model (x* is the warm start)."
    direction in (:raw, :cost_aware, :lexicographic) ||
        error("Unknown direction :$direction (expected :raw, :cost_aware, :lexicographic)")

    all_vars = all_variables(model)
    @assert vars == all_vars "boundary_walk expects vars == all_variables(model)."

    x_star = value.(all_vars)
    min_cost = objective_value(model)

    w_hat = w ./ max(norm(w), eps(Float64))
    cvec, c_offset = _objective_costs(model, vars)

    # Optimisation direction. :cost_aware adds a small unit-normalised cost term
    # so the solver stops over-spending budget on the cost-parallel component.
    w_eff = w_hat
    if direction === :cost_aware
        c_hat = cvec ./ max(norm(cvec), eps(Float64))
        w_eff = w_hat .+ gamma .* c_hat
    end

    points = BoundaryWalkPoint[]
    frontier = Int[]            # indices into `points`, kept non-dominated
    total_time = 0.0
    consecutive_dominated = 0

    for k = 1:n_steps
        B_k = min_cost + (k / n_steps) * eps_slack * abs(min_cost)

        # Phase-1 objective: the diversity direction (lexicographic uses the raw
        # w; :raw/:cost_aware fold the cost tilt into w_eff up front).
        build_w = direction === :lexicographic ? w : w_eff
        lp = build_mga_lp(model, build_w, B_k, vars)
        x0 = clamp.(x_star ./ lp.d_scale, lp.lb_t, lp.ub_t)
        x_t, info = solve_firstorder(
            method,
            lp,
            x0;
            max_iters = max_iters,
            max_inner = max_inner,
            pdhg_iters = pdhg_iters,
            verbose = false,
        )
        step_time = info.time
        step_infeas = info.infeas

        # Phase 2 (lexicographic only): pin diversity to phase-1's level and
        # minimise original cost - the parameter-free Delta_perp point. Warm
        # started from phase 1 (same budget point, already feasible for the new
        # diversity row), not from a different budget - the x*-only rule governs
        # the budget sweep, not the two phases of one point.
        if direction === :lexicographic
            dobj1 = dot(w_hat, lp.d_scale .* x_t)
            lp2 = _augment_lex(lp, w_hat, cvec, dobj1; reltol = lex_tol)
            x_t, info2 = solve_firstorder(
                method,
                lp2,
                x_t;
                max_iters = max_iters,
                max_inner = max_inner,
                pdhg_iters = pdhg_iters,
                verbose = false,
            )
            step_time += info2.time
            step_infeas = max(step_infeas, info2.infeas)
        end

        x = lp.d_scale .* x_t
        total_time += step_time

        pt = BoundaryWalkPoint(
            B_k,
            dot(cvec, x) + c_offset,
            dot(w_hat, x),          # diversity axis: original direction, always
            step_time,
            step_infeas,
            x,
        )
        push!(points, pt)
        idx = length(points)

        # Update the running non-dominated frontier and the early-stop counter.
        dominated_by_existing = any(_walk_dominates(points[f], pt) for f in frontier)
        if dominated_by_existing
            consecutive_dominated += 1
        else
            consecutive_dominated = 0
            filter!(f -> !_walk_dominates(pt, points[f]), frontier)
            push!(frontier, idx)
        end

        if early_stop_k > 0 && consecutive_dominated >= early_stop_k
            break
        end
    end

    return (points = points, frontier = frontier, total_time = total_time)
end
