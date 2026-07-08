export lbfgs_search_alternatives

"""
    alternatives = lbfgs_search_alternatives(
        model::JuMP.Model,
        target_vars::Vector{JuMP.VariableRef},
        n_alternatives::Int64,
        eps::Float64 = 0.1,
        max_allowed_cost::Float64 = Inf,
        repair_solution = false,
        optimizer_search = nothing;
        x_optimal_override = nothing,
        min_cost_override = nothing,
        max_iters::Int = 30,
        search::Symbol = :oracle,
        method::Symbol = :alm_lbfgs,
    )

Iteratively generate `n_alternatives` near-optimal alternatives with
`run_lbfgs_mga`, accumulating HSJ-style penalty weights over the structural
`target_vars` between iterations (each alternative penalises the variables
that were large in the previous ones). This is the internal driver behind
[`generate_alternatives_gradient`](@ref); it works directly with the packed
structural variables and weight vectors.

# Arguments
- `model::JuMP.Model`: a solved JuMP model for which alternatives are generated.
- `target_vars::Vector{JuMP.VariableRef}`: structural variables that carry the
  MGA penalty weights.
- `n_alternatives::Int64`: the number of alternative solutions sought.
- `eps::Float64`: relative cost slack defining the budget
  `min_cost + eps * |min_cost|` when `max_allowed_cost` is `Inf`.
- `max_allowed_cost::Float64`: absolute cost budget (overrides `eps`).
- `repair_solution`: project each alternative onto the budget-feasible space
  with `operational_recovery!` before returning it.
- `optimizer_search`: unused; kept for backward compatibility of the
  positional signature.
- `x_optimal_override`/`min_cost_override`: optimal solution values/objective
  used only to *define* the MGA problem (weights/budget); the solver itself
  never uses the optimal point.
- `max_iters::Int`: outer-iteration budget per alternative.
- `search::Symbol`: `:oracle` (structural cutting-plane decomposition) or
  `:fullspace` (whole-variable first-order search).
- `method::Symbol`: the first-order solver for the `:fullspace` search
  (`:alm_lbfgs`, `:penalty`, or `:pdhg`); ignored when `search == :oracle`.
"""
function lbfgs_search_alternatives(
    model::JuMP.Model,
    target_vars::Vector{JuMP.VariableRef},
    n_alternatives::Int64,
    eps::Float64 = 0.1,
    max_allowed_cost::Float64 = Inf,
    repair_solution = false,
    optimizer_search = nothing;
    x_optimal_override::Union{Nothing,Vector{Float64}} = nothing,
    min_cost_override::Union{Nothing,Float64} = nothing,
    max_iters::Int = 30,
    search::Symbol = :oracle,
    method::Symbol = :alm_lbfgs,
    pdhg_iters::Int = 10000,
    warm_start::Bool = false,
)
    all_vars = all_variables(model)
    # `x_optimal` defines the MGA problem (the HSJ-style penalty weights below
    # and, via `min_cost`, the cost budget). With `warm_start = false` (default)
    # the solver itself never sees it (cold start from the origin); with
    # `warm_start = true` each `:fullspace` solve is started from `x_current`
    # (the optimum for the first alternative, the previous alternative after),
    # a feasible interior point - the sequential warm start the standard MGA
    # methods use.
    x_optimal = isnothing(x_optimal_override) ? value.(all_vars) : x_optimal_override
    n_total_vars = length(x_optimal)
    var_to_idx = Dict(v => i for (i, v) in enumerate(all_vars))

    # MGA cost budget.
    min_cost = isnothing(min_cost_override) ? objective_value(model) : min_cost_override
    actual_max_cost =
        isinf(max_allowed_cost) ? min_cost + eps * abs(min_cost) : max_allowed_cost

    # Packed structural indices and their (nonzero) upper bounds for weighting.
    target_indices = Int[var_to_idx[v] for v in target_vars]
    target_ubs = [
        has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0 for
        v in target_vars
    ]

    alternatives = Vector{Float64}[]
    x_current = x_optimal
    accumulated_weights = zeros(Float64, length(target_vars))

    for _ = 1:n_alternatives
        w = zeros(Float64, n_total_vars)
        for (j, idx) in enumerate(target_indices)
            accumulated_weights[j] += x_current[idx] / target_ubs[j]
            w[idx] = accumulated_weights[j]
        end

        alternative = run_lbfgs_mga(
            model,
            w,
            actual_max_cost,
            repair_solution,
            all_vars,
            target_indices;
            max_iters = max_iters,
            search = search,
            method = method,
            pdhg_iters = pdhg_iters,
            x_start = warm_start ? copy(x_current) : nothing,
        )

        if !isnothing(alternative)
            push!(alternatives, alternative)
            x_current = alternative
        else
            @warn "Skipping an alternative due to search failure."
            continue
        end
    end

    return alternatives
end
