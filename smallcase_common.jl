# ===========================================================================
# Shared harness for the small-case first-order MGA corrector study.
#
# Provides, in one place, the pieces every smallcase_fom_*.jl script needs:
#   * model builders   - a minimal LP with a hand-checkable optimum, and the
#                        synthetic capacity-expansion+storage model whose
#                        operational-to-structural ratio is tunable;
#   * MGA problem setup - the HSJ-style first-alternative weight and cost budget;
#   * an EXACT reference - min w'x s.t. model constraints, cost(x) <= B, solved
#                        by an interior-point method (the ground truth the
#                        correctors are graded against);
#   * a corrector runner - runs :penalty (QP), :alm_lbfgs (ALM), :pdhg, :osqp
#                        (ADMM) on the identical Ruiz-scaled LP via the package's
#                        solve_firstorder, returning per-method metrics and,
#                        optionally, the per-iteration convergence trace.
#
# The gap reported is the RELATIVE gap of the structural objective w_s'x (the MGA
# diversity direction restricted to the investment variables) to the exact value,
# in ORIGINAL units - the quantity a dominated alternative inflates.
# ===========================================================================
using JuMP, LinearAlgebra, Random, Printf, Logging
using NearOptimalAlternatives
using SCS
const NOA = NearOptimalAlternatives

# The Hessian-free correctors evaluated everywhere, grouped by TYPE rather than
# just "first-order": the three GRADIENT correctors, where the diversity
# objective competes inside the search direction (quadratic penalty, ALM, PDHG),
# versus the EXACT-LINEAR-SOLVE (operator-splitting/ADMM) corrector SCS, where the
# objective enters an exact solve as a right-hand side. All are matrix-free / no
# Hessian; the split that matters for MGA is gradient vs exact-solve, not
# first-order vs not. (The in-house matrix-free ADMM prototype `:osqp` in
# solve_firstorder is the same idea as SCS but less refined, so it is not reported;
# SCS is the mature representative of the exact-solve family.)
const CORRECTORS = [
    (:penalty, "Quadratic penalty + L-BFGS (QP)"),
    (:alm_lbfgs, "Augmented Lagrangian + L-BFGS (ALM)"),
    (:pdhg, "Restarted PDHG"),
]

# SCS (library operator-splitting) is run through a separate JuMP path (not
# solve_firstorder), so it is not in CORRECTORS; scripts append it via solve_scs_mga.
const SCS_LABEL = "Operator splitting (SCS library)"

# Consistent colours across every figure (extends the bench_common palette).
const CORRECTOR_COLOR = Dict(
    :penalty => :orange,
    :alm_lbfgs => :seagreen,
    :pdhg => :teal,
    :osqp => :indianred,
    :scs => :purple,
)

# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------

"""
    build_minimal_lp(optimizer)

A deliberately tiny MGA test with a hand-checkable optimum, used by the
correctness gate. Cost-minimal problem:

    min  x1 + 1.05 x2   s.t.  x1 + x2 >= 2,  0 <= x1, x2 <= 3.

Cost optimum is (2, 0) at cost 2.0. The HSJ weight then falls on x1 (the only
nonzero at the optimum), so within a 10% budget the MGA sub-problem
`min w'x s.t. cost <= 2.2` pushes x1 down and x2 up to (1.8, 0.2). A correct
corrector must reproduce that exactly. Returns `(model, target_vars)`.
"""
function build_minimal_lp(optimizer)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, 0 <= x1 <= 3)
    @variable(model, 0 <= x2 <= 3)
    @constraint(model, x1 + x2 >= 2)
    @objective(model, Min, x1 + 1.05 * x2)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return model, [x1, x2]
end

"""
    build_synth_model(optimizer; n_struct, T, n_store, seed)

Synthetic capacity-expansion model with inter-temporal storage - the same
structure as the real ESOM in miniature: `n_struct` investment (structural)
variables, each capping `T` dispatch (operational) variables, plus `n_store`
storage chains. The operational-to-structural variable ratio is set by `T` and
`n_store`, which is exactly the lever the ratio study sweeps. Returns
`(model, target_vars)` where `target_vars` are the investment variables that
carry the MGA direction.
"""
function build_synth_model(
    optimizer;
    n_struct::Int = 8,
    T::Int = 24,
    n_store::Int = 4,
    seed::Int = 7,
)
    Random.seed!(seed)
    cap_max = 50.0 .+ 50.0 .* rand(n_struct)
    c_inv = 10.0 .+ 10.0 .* rand(n_struct)
    c_op = 1.0 .+ 5.0 .* rand(n_struct, T)
    demand = 20.0 .+ 15.0 .* rand(T)
    eta = 0.9
    model = Model(optimizer)
    set_silent(model)
    @variable(model, 0 <= assets_investment[i = 1:n_struct] <= cap_max[i])
    @variable(model, dispatch[i = 1:n_struct, t = 1:T] >= 0)
    @constraint(model, [i = 1:n_struct, t = 1:T], dispatch[i, t] <= assets_investment[i])
    @constraint(model, [t = 1:T], sum(dispatch[i, t] for i = 1:n_struct) >= demand[t])
    if n_store > 0
        @variable(model, charge[s = 1:n_store, t = 1:T] >= 0)
        @variable(model, level[s = 1:n_store, t = 1:T] >= 0)
        @constraint(
            model,
            [s = 1:n_store, t = 2:T],
            level[s, t] == level[s, t-1] + eta * charge[s, t] - dispatch[s, t]
        )
        @constraint(model, [s = 1:n_store], level[s, 1] == eta * charge[s, 1])
        @constraint(model, [s = 1:n_store, t = 1:T], charge[s, t] <= assets_investment[s])
        @constraint(
            model,
            [s = 1:n_store, t = 1:T],
            level[s, t] <= 5 * assets_investment[s]
        )
    end
    @objective(
        model,
        Min,
        sum(c_inv[i] * assets_investment[i] for i = 1:n_struct) +
        sum(c_op[i, t] * dispatch[i, t] for i = 1:n_struct, t = 1:T) +
        (n_store > 0 ? 0.1 * sum(charge) : 0.0)
    )
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return model, [assets_investment[i] for i = 1:n_struct]
end

# --------------------------------------------------------------------------
# MGA problem setup: the first-alternative HSJ weight + cost budget, exactly as
# the high-level gradient driver (lbfgs_search_alternatives) builds them.
# --------------------------------------------------------------------------

"""
    mga_setup(model, target_vars; eps)

Returns a NamedTuple with everything needed to define and grade the first MGA
alternative: `all_vars`, `target_indices`, the full-space weight `w` and its
structural restriction `w_s`, the cost budget `B = min_cost*(1+eps)`, and the
cost optimum `base_x`. `w` is the HSJ weight (each structural variable weighted
by its optimal value over its upper bound).
"""
function mga_setup(model::Model, target_vars::Vector{VariableRef}; eps::Float64 = 0.1)
    all_vars = all_variables(model)
    var_to_idx = Dict(v => i for (i, v) in enumerate(all_vars))
    target_indices = [var_to_idx[v] for v in target_vars]
    base_x = value.(all_vars)
    base_obj = objective_value(model)

    w = zeros(length(all_vars))
    for v in target_vars
        ub = has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0
        w[var_to_idx[v]] = base_x[var_to_idx[v]] / ub
    end
    w ./= max(norm(w), eps_tol())
    w_s = w[target_indices]
    B = base_obj + eps * abs(base_obj)
    # Structural objective AT the cost optimum. The MGA sub-problem drives w_s'x
    # DOWN from this value toward the exact minimum; the difference is the
    # "achievable diversity" the correctors compete to capture.
    obj_star_s = dot(w_s, base_x[target_indices])
    # Per-structural-variable scale (bound range, or |x*| as a fallback) used to
    # normalise the decision-space distance between two alternatives to [0,1]-ish,
    # so "is the SOLUTION the same" can be judged, not only "is the objective".
    target_scale = Float64[]
    for v in target_vars
        lb = has_lower_bound(v) ? lower_bound(v) : 0.0
        ub = has_upper_bound(v) ? upper_bound(v) : lb
        rng = ub - lb
        push!(target_scale, rng > 0 ? rng : max(abs(base_x[var_to_idx[v]]), 1.0))
    end
    return (;
        all_vars,
        target_indices,
        w,
        w_s,
        B,
        base_x,
        base_obj,
        obj_star_s,
        target_scale,
    )
end

"""
    structural_distance(x, x_ref, target_indices, target_scale)

Normalised RMS distance between two alternatives on the structural (investment)
variables: `sqrt(mean(((x - x_ref)/scale)^2))` over the structural block. Zero
means the two alternatives are identical there; a positive value with an equal
objective means the MGA optimum is non-unique and the solvers picked different
(equally optimal) points on the optimal face.
"""
function structural_distance(x, x_ref, target_indices, target_scale)
    xs = x[target_indices]
    rs = x_ref[target_indices]
    d = ((xs .- rs) ./ target_scale) .^ 2
    return sqrt(sum(d) / max(length(d), 1))
end

eps_tol() = eps(Float64)

"""
    dominates(a, b; atol)

`true` iff investment vector `a` Pareto-dominates `b` on resource usage: `a`
uses no more of every variable and strictly less of at least one (thesis 2.5).
"""
function dominates(a::Vector{Float64}, b::Vector{Float64}; atol::Float64 = 1e-6)
    no_worse = all(a .<= b .+ atol)
    strictly = any(a .< b .- atol)
    return no_worse && strictly
end

"""
    exact_reference(model, w, B, all_vars, target_indices, w_s)

Ground truth: solve `min w'x s.t. model constraints, cost(x) <= B` with the
model's current (interior-point) optimizer, cold. Mutates then RESTORES the
model objective + drops the budget row, so the model can be reused. Returns
`(x, obj_full, obj_s, time)` in original units.
"""
function exact_reference(
    model::Model,
    w::Vector{Float64},
    B::Float64,
    all_vars::Vector{VariableRef},
    target_indices::Vector{Int},
    w_s::Vector{Float64},
)
    orig_obj = objective_function(model)
    orig_sense = objective_sense(model)
    bcon = @constraint(model, orig_obj <= B)
    @objective(
        model,
        Min,
        sum(w[i] * all_vars[i] for i in eachindex(all_vars) if w[i] != 0)
    )
    t = @elapsed optimize!(model)
    @assert is_solved_and_feasible(model)
    x = value.(all_vars)
    obj_full = dot(w, x)
    obj_s = dot(w_s, x[target_indices])
    delete(model, bcon)
    set_objective_function(model, orig_obj)
    set_objective_sense(model, orig_sense)
    return (; x, obj_full, obj_s, time = t)
end

"""
    solve_scs_mga(model, s, ref, lp; eps_tol) -> corrector-shaped NamedTuple

Solve the same first-alternative MGA sub-problem `min w'x s.t. constraints,
cost(x) <= B` with the SCS operator-splitting library through JuMP, and return
metrics in the SAME shape and SAME (Ruiz-scaled) feasibility units as
`run_correctors`, so SCS can be appended to the corrector comparison directly.
SCS is the exact-linear-solve corrector the thesis actually adopts; the in-house
`:osqp` is its matrix-free prototype. Mutates then restores the model objective
(and leaves SCS as the optimizer - call last on a given model).
"""
function solve_scs_mga(model::Model, s, ref, lp; eps_tol::Float64 = 1.0e-7)
    orig = objective_function(model)
    sense = objective_sense(model)
    bcon = @constraint(model, orig <= s.B)
    @objective(
        model,
        Min,
        sum(s.w[i] * s.all_vars[i] for i in eachindex(s.w) if s.w[i] != 0)
    )
    set_optimizer(
        model,
        optimizer_with_attributes(
            SCS.Optimizer,
            "verbose" => 0,
            "eps_abs" => eps_tol,
            "eps_rel" => eps_tol,
            "max_iters" => 200_000,
        ),
    )
    t = @elapsed optimize!(model)
    x = value.(s.all_vars)
    delete(model, bcon)
    set_objective_function(model, orig)
    set_objective_sense(model, sense)

    x_scaled = x ./ lp.d_scale                     # into the shared scaled space
    infeas = NOA.primal_infeasibility(lp.A_in, lp.b_in, lp.A_eq, lp.b_eq, x_scaled)
    obj_s = dot(s.w_s, x[s.target_indices])
    signal = s.obj_star_s - ref.obj_s
    gap_s = 100 * max(obj_s - ref.obj_s, 0.0) / max(abs(signal), 1e-9)
    dominated = dominates(ref.x[s.target_indices], x[s.target_indices])
    sol_dist = structural_distance(x, ref.x, s.target_indices, s.target_scale)
    # SCS does not expose an ADMM iteration count through MOI, so we report 0
    # (time is the meaningful cost axis for the library solver).
    return (;
        method = :scs,
        label = SCS_LABEL,
        time = t,
        iters = 0,
        infeas,
        obj_s,
        gap_s,
        dominated,
        sol_dist,
        x,
        history = NamedTuple[],
    )
end

# --------------------------------------------------------------------------
# Corrector runner
# --------------------------------------------------------------------------

"""
    run_correctors(model, s, ref; from, methods, trace, budgets...)

Build the shared Ruiz-scaled MGA LP once (`build_mga_lp`) and run each corrector
on it from the same start (`:origin` cold, or `:optimum` warm from x*). `s` is
the `mga_setup` NamedTuple; `ref` is the `exact_reference` result (its `x` and
`obj_s` are the ground truth). Returns a Vector of per-method NamedTuples
`(method, label, time, iters, infeas, obj_s, gap_s, dominated, x, history)`:

  - `gap_s`      PRIMARY metric: missed-diversity % (see below), robust to zero refs;
  - `dominated`  SECONDARY metric: is this corrector's alternative Pareto-DOMINATED
                 (on investment/resource usage) by the exact alternative? True means
                 the exact point uses no more of every investment variable and
                 strictly less of at least one - i.e. the corrector returned a
                 resource-dominated (useless) alternative;
  - `history`    per-iteration trace (empty unless `trace = true`).
"""
function run_correctors(
    model::Model,
    s,
    ref;
    from::Symbol = :origin,
    methods = CORRECTORS,
    trace::Bool = false,
    max_iters::Int = 60,
    max_inner::Int = 1000,
    pdhg_iters::Int = 20000,
    osqp_iters::Int = 8000,
)
    obj_ref_s = ref.obj_s
    x_ref_s = ref.x[s.target_indices]
    lp = with_logger(NullLogger()) do
        NOA.build_mga_lp(model, copy(s.w), s.B, s.all_vars)
    end
    x0 = if from === :optimum
        clamp.(s.base_x ./ lp.d_scale, lp.lb_t, lp.ub_t)
    else
        clamp.(zeros(lp.n), lp.lb_t, lp.ub_t)
    end

    results = NamedTuple[]
    for (m, label) in methods
        h = trace ? NamedTuple[] : nothing
        x_t, info = with_logger(NullLogger()) do
            run_one(
                m,
                lp,
                copy(x0);
                h = h,
                max_iters = max_iters,
                max_inner = max_inner,
                pdhg_iters = pdhg_iters,
                osqp_iters = osqp_iters,
            )
        end
        x = lp.d_scale .* x_t                          # back to original units
        obj_s = dot(s.w_s, x[s.target_indices])
        # Missed-diversity gap: fraction of the achievable diversity signal
        # (w_s'x* - exact) that the corrector FAILED to capture. 0% = reaches the
        # exact MGA value; 100% = frozen at the cost optimum (fully dominated).
        # Robust when the exact value is ~0 (unlike a plain relative gap); a small
        # negative residual (a slightly infeasible method dipping below the exact
        # value) is clamped to 0.
        signal = s.obj_star_s - obj_ref_s
        gap_s = 100 * max(obj_s - obj_ref_s, 0.0) / max(abs(signal), 1e-9)
        # Secondary: does the EXACT alternative dominate this corrector's, on
        # resource (investment) usage? (thesis 2.5 dominance test). And how far is
        # the returned alternative from the exact one in decision space?
        x_s = x[s.target_indices]
        dominated = dominates(x_ref_s, x_s)
        sol_dist = structural_distance(x, ref.x, s.target_indices, s.target_scale)
        push!(
            results,
            (;
                method = m,
                label,
                time = info.time,
                iters = info.iters,
                infeas = info.infeas,
                obj_s,
                gap_s,
                dominated,
                sol_dist,
                x,
                history = (h === nothing ? NamedTuple[] : h),
            ),
        )
    end
    return results, lp
end

# solve_firstorder wraps the non-tracing path; for tracing we must call the
# underlying solvers directly (only solve_alm_lbfgs / solve_pdhg accept history,
# so :osqp has no per-iteration trace and returns an empty history).
function run_one(m, lp, x0; h, max_iters, max_inner, pdhg_iters, osqp_iters)
    if h === nothing
        return NOA.solve_firstorder(
            m,
            lp,
            x0;
            max_iters = max_iters,
            max_inner = max_inner,
            pdhg_iters = pdhg_iters,
            osqp_iters = osqp_iters,
            verbose = false,
        )
    end
    t0 = time()
    local x_t, iters
    if m === :pdhg
        x_t, iters, _, _ = NOA.solve_pdhg(
            lp.w_t,
            lp.A_in,
            lp.b_in,
            lp.A_eq,
            lp.b_eq,
            lp.lb_t,
            lp.ub_t,
            x0;
            max_iter = pdhg_iters,
            verbose = false,
            history = h,
        )
    elseif m === :alm_lbfgs || m === :penalty
        x_t, iters = NOA.solve_alm_lbfgs(
            lp.w_t,
            lp.A_in,
            lp.b_in,
            lp.A_eq,
            lp.b_eq,
            lp.lb_t,
            lp.ub_t,
            x0;
            max_outer = max_iters,
            max_inner = max_inner,
            verbose = false,
            use_multipliers = (m === :alm_lbfgs),
            history = h,
        )
    else  # :osqp - no history hook; run normally
        return NOA.solve_firstorder(
            m,
            lp,
            x0;
            max_iters = max_iters,
            max_inner = max_inner,
            pdhg_iters = pdhg_iters,
            osqp_iters = osqp_iters,
            verbose = false,
        )
    end
    tt = time() - t0
    infeas = NOA.primal_infeasibility(lp.A_in, lp.b_in, lp.A_eq, lp.b_eq, x_t)
    return x_t,
    (method = m, iters = iters, time = tt, infeas = infeas, obj = dot(lp.w_t, x_t))
end
