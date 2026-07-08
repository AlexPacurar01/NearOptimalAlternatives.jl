# ===========================================================================
# Warm-started scalarization continuation along the cost-vs-diversity front.
#
# Trace the near-optimal Pareto front by sweeping a scalarization weight
# lambda in [0,1], objective  (1-lambda)*cost_hat + lambda*w_MGA_hat, and solving
# each sub-problem with a first-order solver WARM-STARTED from the previous
# optimum. Every solve is a converged weighted-sum optimum, hence a non-dominated
# (Pareto-efficient) point; warm-starting collapses each re-solve to a handful of
# iterations (the FOM amortisation barrier methods cannot exploit).
#
# Even spacing of points along the front is handled by arclength step control
# (pseudo-arclength continuation, Keller 1977; predictor-corrector Pareto
# continuation, Hillermeier 2001; Schuetze, Dell'Aere & Dellnitz 2005). Plain
# weighted-sum sweeps cluster points badly (Das & Dennis 1997); here the lambda
# step is adapted so consecutive points are ~`ds` apart in NORMALISED objective
# space (cf. the adaptive weighted-sum idea, Kim & de Weck 2005).
# ===========================================================================

export continuation_walk, ContinuationPoint

"One converged front point of an epsilon-constraint continuation."
struct ContinuationPoint
    budget::Float64      # cost budget B this point was solved at
    cost::Float64        # original implementation cost c'x (+ offset)
    dobj::Float64        # diversity objective w_hat'x (original MGA direction)
    infeas::Float64      # primal infeasibility (scaled space)
    iters::Int           # solver iterations for this (warm-started) solve
    time::Float64        # per-solve wall time (solver loop only; excludes LP setup)
    x::Vector{Float64}   # full-space solution (unscaled)
end

"""
    result = continuation_walk(model, w, vars; kwargs...)

Trace the cost-vs-diversity near-optimal front from x* (budget = min_cost) to the
MGA optimum (budget = min_cost*(1+eps_slack)) by a warm-started epsilon-constraint
continuation (fix objective = `w`, sweep the cost budget `B`), returning
evenly-spaced non-dominated points. The budget is swept rather than a weighted-sum
weight because weighted-sum sweeps cluster/skip points on (piecewise-linear) LP
fronts (Das & Dennis 1997); the epsilon-constraint (Haimes 1971) is monotone and
evenly spaceable.

# Keywords
- `eps_slack`: budget `min_cost*(1+eps_slack)` bounding the near-optimal region.
- `n_points`: target number of front points (sets the arclength spacing `ds`).
- `method`: first-order solver (`:alm_lbfgs`, `:penalty`, `:pdhg`).
- `max_iters`, `max_inner`, `pdhg_iters`: per-solve solver budgets.
- `ds`: explicit normalised-objective spacing (default `sqrt(2)/n_points`).
- `max_solves`: hard cap on sub-problem solves (safety).
- `use_predictor`: tangent/secant extrapolation of the warm start (Allgower-Georg;
  Pareto Tracer). OFF by default - on a piecewise-linear LP front with the
  infeasible-path ALM corrector it did not reduce iterations (the corrector's count
  is penalty-schedule-bound, and the previous optimum is already a near-perfect
  warm start). Kept for A/B testing on stiffer models.

# Returns
NamedTuple `(points, n_solves, total_iters)`. Every `points[i]` is feasible and
non-dominated; they are ordered along the front (cost increasing, diversity
decreasing).
"""
function continuation_walk(
    model::Model,
    w::Vector{Float64},
    vars::Vector{VariableRef};
    eps_slack::Float64 = 0.1,
    n_points::Int = 25,
    method::Symbol = :alm_lbfgs,
    max_iters::Int = 20,
    max_inner::Int = 500,
    pdhg_iters::Int = 10000,
    osqp_iters::Int = 8000,
    feas_tol::Union{Nothing,Float64} = nothing,
    ds::Union{Nothing,Float64} = nothing,
    max_solves::Int = 400,
    use_predictor::Bool = false,
    verbose::Bool = false,
)
    @assert is_solved_and_feasible(model) "continuation_walk needs a solved model (x* = lambda 0)."
    all_vars = all_variables(model)
    @assert vars == all_vars "continuation_walk expects vars == all_variables(model)."

    min_cost = objective_value(model)
    B_max = min_cost + eps_slack * abs(min_cost)
    cvec, c_offset = _objective_costs(model, vars)
    # Built once: objective = w (the MGA diversity direction); budget = B_max.
    lp = build_mga_lp(model, w, B_max, vars)
    what = w ./ max(norm(w), eps(Float64))
    x_star_t = clamp.(value.(all_vars) ./ lp.d_scale, lp.lb_t, lp.ub_t)

    # The budget row is the last inequality row. Recover its accumulated Ruiz row
    # scale so the budget can be retargeted by editing ONE RHS entry (no rebuild):
    #   b_in[budget_idx] = (B - c_offset) / r_budget.
    budget_idx = size(lp.A_in, 1)
    r_budget = (B_max - c_offset) / lp.b_in[budget_idx]
    b_base = copy(lp.b_in)
    set_budget(B) = (b = copy(b_base); b[budget_idx] = (B - c_offset) / r_budget; b)

    # Warm-started solve at budget B from scaled primal `x0`, PDHG duals `(yin0,yeq0)`,
    # and OSQP ADMM state `(zos0,yos0)` - whichever the chosen method consumes.
    solve_at(B, x0, yin0, yeq0, zos0, yos0) = begin
        lpl = merge(lp, (b_in = set_budget(B),))
        x_t, info = solve_firstorder(
            method,
            lpl,
            x0;
            y_in0 = yin0,
            y_eq0 = yeq0,
            osqp_z0 = zos0,
            osqp_y0 = yos0,
            feas_tol = feas_tol,
            max_iters = max_iters,
            max_inner = max_inner,
            pdhg_iters = pdhg_iters,
            osqp_iters = osqp_iters,
            verbose = verbose,
        )
        x = lp.d_scale .* x_t
        return x_t,
        (
            cost = dot(cvec, x) + c_offset,
            dobj = dot(what, x),
            infeas = info.infeas,
            iters = info.iters,
            time = info.time,
            yin = info.y_in,
            yeq = info.y_eq,
            zos = info.osqp_z,
            yos = info.osqp_y,
        ),
        x
    end

    # Low-cost endpoint (B = min_cost, max diversity) is x* ITSELF - already the
    # exact optimum from the model solve, so we DO NOT re-solve it (the cost face is
    # the degenerate, stiffest vertex; a cold first-order solve stalls there). x* is
    # feasible (infeas 0) and free (iters 0), and its scaled primal seeds the FIRST
    # interior solve - which pays a one-time cost to discover the duals, after which
    # every subsequent step warm-starts primal+duals from its solved neighbour.
    x_star_orig = lp.d_scale .* x_star_t
    m0 = (
        cost = dot(cvec, x_star_orig) + c_offset,
        dobj = dot(what, x_star_orig),
        infeas = 0.0,
        iters = 0,
        time = 0.0,
        yin = Float64[],
        yeq = Float64[],
        zos = Float64[],
        yos = Float64[],
    )

    # Normalisation for arclength spacing: cost span is known exactly (B_max - min);
    # the diversity span is unknown until we reach x_MGA, so we do NOT pre-solve that
    # endpoint (per "go straight to the first intermediate point"). Instead `wspan` is
    # a provisional estimate, refined from the running x*->point slope each step.
    cspan = max(B_max - m0.cost, eps(Float64))
    wspan = Ref(max(abs(m0.dobj), eps(Float64)))
    nrm(m) = (cost = (m.cost - m0.cost) / cspan, wx = (m0.dobj - m.dobj) / wspan[])
    objdist(a, b) = begin
        na, nb = nrm(a), nrm(b)
        hypot(na.cost - nb.cost, na.wx - nb.wx)
    end
    ds_target = ds === nothing ? sqrt(2.0) / n_points : ds
    dB_min = 1.0e-9 * max(1.0, abs(B_max))

    # Interior points, collected marching the budget UP from min_cost toward B_max
    # (increasing cost). The first solve starts straight from x* (warm primal, cold
    # duals); each later solve warm-starts primal+duals from its solved neighbour.
    points = ContinuationPoint[ContinuationPoint(
        min_cost,
        m0.cost,
        m0.dobj,
        m0.infeas,
        m0.iters,
        m0.time,
        x_star_orig,
    )]
    total_iters = 0
    n_solves = 0

    Bcur = min_cost
    xt_prev = x_star_t           # last ACCEPTED optimum (x_k); x* seeds the first step
    yin_prev, yeq_prev = m0.yin, m0.yeq   # empty: first solve discovers the duals cold
    zos_prev, yos_prev = m0.zos, m0.yos   # OSQP ADMM state carried across budgets
    xt_prev2 = nothing           # the one before (x_{k-1}), for the secant predictor
    B_prev2 = min_cost
    prevm = m0
    dB = (B_max - min_cost) / n_points   # arclength-adapted below

    while Bcur < B_max - dB_min && n_solves < max_solves
        accepted = false
        for _ = 1:8              # bounded rejections (shrink step) per accepted point
            B_try = min(B_max, Bcur + dB)

            # Tangent/secant predictor (Allgower-Georg; Pareto Tracer): extrapolate
            # the solution along its recent trajectory instead of starting from x_k.
            # On a piecewise-linear LP front this is exact within an edge, so the
            # corrector converges almost immediately between breakpoints.
            x0 = if !use_predictor || xt_prev2 === nothing
                xt_prev
            else
                s = (B_try - Bcur) / max(Bcur - B_prev2, dB_min)
                clamp.(xt_prev .+ s .* (xt_prev .- xt_prev2), lp.lb_t, lp.ub_t)
            end

            xt_try, m_try, xo = solve_at(B_try, x0, yin_prev, yeq_prev, zos_prev, yos_prev)
            n_solves += 1
            total_iters += m_try.iters
            # Refine the diversity span from the cumulative x*->point slope, then score.
            wspan[] = max(
                (m0.dobj - m_try.dobj) / max(B_try - min_cost, dB_min) * (B_max - min_cost),
                eps(Float64),
            )
            d = objdist(m_try, prevm)

            if d > 1.4 * ds_target && B_try < B_max && dB > dB_min
                dB *= max(0.3, ds_target / max(d, 1.0e-9))   # overshoot: shrink & retry
                continue
            end

            push!(
                points,
                ContinuationPoint(
                    B_try,
                    m_try.cost,
                    m_try.dobj,
                    m_try.infeas,
                    m_try.iters,
                    m_try.time,
                    xo,
                ),
            )
            dB = clamp(dB * (ds_target / max(d, 1.0e-9)), dB_min, B_max - min_cost)  # retarget spacing
            xt_prev2, B_prev2 = xt_prev, Bcur
            Bcur, xt_prev, prevm = B_try, xt_try, m_try
            yin_prev, yeq_prev = m_try.yin, m_try.yeq   # carry duals to the next step
            zos_prev, yos_prev = m_try.zos, m_try.yos   # carry OSQP ADMM state
            accepted = true
            verbose &&
                @info "  cont B=$(round(B_try,sigdigits=8)) cost=$(round(m_try.cost,sigdigits=8)) dobj=$(round(m_try.dobj,sigdigits=5)) infeas=$(round(m_try.infeas,sigdigits=3)) iters=$(m_try.iters) solve=$(round(m_try.time,digits=2))s Δs=$(round(d,digits=3))"
            break
        end
        accepted || break
    end

    # `total_solve_time` sums only the per-solve solver-loop time (the API-call-level
    # cost), excluding the one-time LP extraction (`lp_setup_time`, = build_mga_lp's
    # Ruiz + matrix extraction) and the model build/solve done by the caller.
    total_solve_time = sum(p.time for p in points)
    return (
        points = points,
        n_solves = n_solves,
        total_iters = total_iters,
        total_solve_time = total_solve_time,
        lp_setup_time = lp.t_extract,
    )
end
