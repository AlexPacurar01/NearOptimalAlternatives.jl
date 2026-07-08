# ===========================================================================
# Projected-gradient boundary walk (amortised equality projection).
#
# Walk the cost-vs-diversity boundary from the cost-minimum x* by projected
# gradient descent on the diversity objective w'x, recording the whole
# trajectory. The stiff equality (balance/storage) constraints A_eq x = b_eq are
# handled *exactly* and *cheaply*: M = A_eq A_eq' is factorised ONCE per trace,
# after which projecting a gradient onto null(A_eq) or snapping a point back onto
# the balance manifold each cost one back-substitution. Because the manifold is
# affine, moving in null(A_eq) keeps every iterate exactly balance-feasible; only
# bound-clamping kicks us off, and a couple of clamp<->reproject sweeps fix that.
# The (many, mostly single-variable) inequality constraints are handled by a
# light quadratic penalty inside the gradient - the equalities, the stiff part,
# never enter the penalty.
#
# This is a cheap DENSE explorer: each recorded point is near-feasible. Exact
# feasibility / cost-efficiency is left to an end-of-trace polish on the few
# non-dominated candidates (see `projected_walk` docstring).
# ===========================================================================

export projected_walk, ProjectedWalkPoint

"One recorded point of a projected-gradient boundary walk."
struct ProjectedWalkPoint
    step::Int
    cost::Float64        # original implementation cost c'x (+ offset)
    dobj::Float64        # diversity objective w_hat'x (original direction)
    infeas::Float64      # max(inequality, equality, bound) violation, scaled space
    x::Vector{Float64}   # full-space solution (unscaled, aligned to all_variables)
end

"""
    result = projected_walk(model, w, vars; kwargs...)

Projected-gradient walk along the cost-vs-diversity boundary from x*, returning a
dense set of near-feasible alternatives trading original cost for diversity.

`model` must be solved (its optimum is x*, its objective the budget anchor). `w`
is the full-length diversity direction over `vars = all_variables(model)`.

# Keywords
- `eps_slack`: relative cost slack; the walk stops once `cost >= min_cost*(1+eps_slack)`.
- `max_steps`: hard cap on recorded points (also sets the target cost spacing).
- `inner_sweeps`: alternating equality-project<->freeze sweeps when forming the
  descent direction (slide along active bounds without a factorisation update).
- `alpha_max`: cap on the auto-sized step length.
- `reg`: if `M = A_eq A_eq'` is not positive-definite (redundant equalities),
  factorise `M + reg*I` instead.

# Returns
NamedTuple `(points, frontier, factor_time, walk_time)`:
- `points::Vector{ProjectedWalkPoint}` in walk order,
- `frontier::Vector{Int}` - non-dominated (cost, dobj) indices,
- `factor_time` - one-time Cholesky time, `walk_time` - stepping time.
"""
function projected_walk(
    model::Model,
    w::Vector{Float64},
    vars::Vector{VariableRef};
    eps_slack::Float64 = 0.1,
    max_steps::Int = 300,
    alpha0::Float64 = 0.1,
    beta0::Float64 = 1.0,
    max_corr::Int = 8,
    viol_tol::Float64 = 1.0e-4,
    infeas_cap::Float64 = 0.02,
    inner_sweeps::Int = 2,
    reg::Float64 = 0.0,
    verbose::Bool = false,
)
    @assert is_solved_and_feasible(model) "projected_walk needs a solved model (x* is the start)."
    all_vars = all_variables(model)
    @assert vars == all_vars "projected_walk expects vars == all_variables(model)."

    min_cost = objective_value(model)
    # Non-binding budget row (we stop on measured cost, not a hard constraint).
    B_huge = min_cost + 1.0e6 * abs(min_cost)
    lp = build_mga_lp(model, w, B_huge, vars)
    n = lp.n
    A_eq, A_in, b_eq, b_in = lp.A_eq, lp.A_in, lp.b_eq, lp.b_in
    m_eq = length(b_eq)
    At_eq = sparse(A_eq')
    At_in = sparse(A_in')

    # One-time factorisation of M = A_eq A_eq'.
    factor_time = @elapsed begin
        M = A_eq * At_eq
        M = sparse(Symmetric(M))
        F = cholesky(M; check = false)
        if !issuccess(F)
            r = reg > 0 ? reg : 1.0e-8 * maximum(abs, diag(M))
            F = cholesky(M + r * I; check = false)
            issuccess(F) ||
                error("Equality Gram matrix not factorisable even regularised.")
        end
    end

    # Scaled cost / original-direction diversity for metrics.
    cvec, c_offset = _objective_costs(model, vars)
    w_hat = w ./ max(norm(w), eps(Float64))
    B_budget = min_cost + eps_slack * abs(min_cost)

    # Buffers (avoid per-step allocation in the inner kernels).
    m_in = length(b_in)
    aeq = zeros(m_eq)        # A_eq * v
    z = zeros(m_eq)          # M^{-1} (A_eq * v)
    tmp_n = zeros(n)
    d = zeros(n)             # predictor diversity direction
    pd = zeros(n)            # corrector feasibility direction
    ax_in = zeros(m_in)      # A_in * x
    viol = zeros(m_in)       # max(0, A_in x - b_in)
    frozen = falses(n)

    # P(g) = g - A_eq'(M^{-1}(A_eq g)) : project a gradient onto null(A_eq).
    # Safe for aliasing (out === v): the A_eq'*z correction goes to tmp_n first.
    proj_dir!(out, v) = begin
        adjoint_mul!(aeq, At_eq, v)       # aeq = A_eq * v
        z .= F \ aeq
        adjoint_mul!(tmp_n, A_eq, z)      # tmp_n = A_eq' * z
        @. out = v - tmp_n
        return out
    end
    # Pi(x) = x - A_eq'(M^{-1}(A_eq x - b_eq)) : snap x onto the balance manifold.
    proj_point!(x) = begin
        adjoint_mul!(aeq, At_eq, x)       # aeq = A_eq * x
        @. aeq -= b_eq
        z .= F \ aeq
        adjoint_mul!(tmp_n, A_eq, z)      # tmp_n = A_eq' * z
        @. x -= tmp_n
        return x
    end

    full_infeas(x) = begin
        bound = 0.0
        @inbounds for i = 1:n
            bound = max(bound, lp.lb_t[i] - x[i], x[i] - lp.ub_t[i])
        end
        max(primal_infeasibility(A_in, b_in, A_eq, b_eq, x), bound)
    end

    # Inequality violation merit ||max(0, A_in x - b_in)||^2 (writes `viol`).
    merit!(x) = begin
        adjoint_mul!(ax_in, At_in, x)
        @. viol = max(0.0, ax_in - b_in)
        return dot(viol, viol)
    end

    # Project a gradient onto null(A_eq), freezing active-bound coordinates, by a
    # few alternating sweeps (no factorisation update). Safe in-place.
    proj_free!(out) = begin
        proj_dir!(out, out)
        @inbounds for _ = 1:inner_sweeps
            for i = 1:n
                frozen[i] && (out[i] = 0.0)
            end
            proj_dir!(out, out)
        end
        @inbounds for i = 1:n
            frozen[i] && (out[i] = 0.0)
        end
        return out
    end

    # Start on the manifold.
    x_star = value.(all_vars)
    x = clamp.(x_star ./ lp.d_scale, lp.lb_t, lp.ub_t)
    proj_point!(x)

    points = ProjectedWalkPoint[]
    frontier = Int[]
    x_trial = similar(x)
    alpha = alpha0          # adaptive predictor step length

    walk_time = @elapsed for step = 1:max_steps
        # Freeze coordinates pinned at a bound that the diversity step pushes past.
        btol = 1.0e-9
        @inbounds for i = 1:n
            at_lo = x[i] <= lp.lb_t[i] + btol
            at_hi = x[i] >= lp.ub_t[i] - btol
            frozen[i] = (at_lo && lp.w_t[i] > 0) || (at_hi && lp.w_t[i] < 0)
        end

        # Predictor: a diversity step -P(w_hat) along the manifold.
        copyto!(d, lp.w_t)
        proj_free!(d)
        @. d = -d
        norm(d) <= 1.0e-12 && break
        @. x += alpha * d
        @. x = clamp(x, lp.lb_t, lp.ub_t)
        proj_point!(x)

        # Corrector: reduce inequality violation on the manifold (this is what
        # pulls the coupled variables - e.g. dispatch - down to match the reduced
        # investment, so the move slides along the active wall instead of stalling).
        m0 = merit!(x)
        corr_used = 0
        for _ = 1:max_corr
            sqrt(m0) <= viol_tol && break
            corr_used += 1
            adjoint_mul!(pd, A_in, viol)     # pd = A_in' * viol  (ascent of merit)
            proj_free!(pd)
            beta = beta0
            improved = false
            for _ = 1:10
                @. x_trial = x - beta * pd
                @. x_trial = clamp(x_trial, lp.lb_t, lp.ub_t)
                proj_point!(x_trial)
                m1 = merit!(x_trial)
                if m1 < m0
                    copyto!(x, x_trial)
                    m0 = m1
                    improved = true
                    break
                end
                beta *= 0.5
            end
            improved || break
        end

        x_orig = lp.d_scale .* x
        cost = dot(cvec, x_orig) + c_offset
        dobj = dot(w_hat, x_orig)
        inf = full_infeas(x)

        # Adapt the predictor step to keep infeasibility near `infeas_cap`: grow it
        # while the point stays comfortably feasible (take bigger diversity steps),
        # shrink it when infeasibility exceeds the cap (step too aggressive).
        alpha =
            inf <= 0.5 * infeas_cap ? min(alpha * 1.3, alpha0 * 1.0e3) :
            inf > infeas_cap ? max(alpha * 0.5, alpha0 * 1.0e-3) : alpha
        pt = ProjectedWalkPoint(step, cost, dobj, inf, x_orig)
        push!(points, pt)
        idx = length(points)

        # Maintain the (cost, dobj) non-dominated set.
        dom = any(
            (points[f].cost <= cost + 1e-9 && points[f].dobj <= dobj + 1e-9) &&
            (points[f].cost < cost - 1e-9 || points[f].dobj < dobj - 1e-9) for
            f in frontier
        )
        if !dom
            filter!(
                f -> !(
                    (cost <= points[f].cost + 1e-9 && dobj <= points[f].dobj + 1e-9) &&
                    (cost < points[f].cost - 1e-9 || dobj < points[f].dobj - 1e-9)
                ),
                frontier,
            )
            push!(frontier, idx)
        end

        verbose &&
            step % max(1, fld(max_steps, 10)) == 0 &&
            @info "  walk step $step: cost=$(round(cost, sigdigits=8)) dobj=$(round(dobj, sigdigits=5)) infeas=$(round(inf, sigdigits=3)) alpha=$(round(alpha, sigdigits=3)) corr=$corr_used"

        cost >= B_budget && break
    end

    return (
        points = points,
        frontier = frontier,
        factor_time = factor_time,
        walk_time = walk_time,
    )
end
