export generate_alternatives_optimization!,
    generate_alternatives_metaheuristics,
    generate_alternatives_gradient,
    generate_alternatives_lbfgs

"""
results = generate_alternatives_optimization!(
  model::JuMP.Model,
  optimality_gap::Float64,
  variables::AbstractArray{T,N},
  n_alternatives::Int64;
  modeling_method::Symbol = :Max_Distance,
  metric::Distances.SemiMetric = SqEuclidean(),
  fixed_variables::Vector{VariableRef} = VariableRef[],
) where {T<:Union{VariableRef,AffExpr},N}
Generate `n_alternatives` solutions to `model` which are as distant from the optimum and each other, but with a maximum `optimality_gap`, using optimization.

# Arguments
- `model::JuMP.Model`: a solved JuMP model for which alternatives are generated.
- `optimality_gap::Float64`: the maximum percentage deviation (>=0) an alternative may have compared to the optimal solution.
-  variables::AbstractArray{T,N}: the variables of `model` for which are considered when generating alternatives.
- `n_alternatives`: the number of alternative solutions sought.
- `modeling_method::Symbol = :Max_Distance`: the method used to model the problem for generating alternatives.
- `metric::Distances.Metric=SqEuclidean()`: the metric used to maximise the difference between alternatives and the optimal solution.
- `fixed_variables::Vector{VariableRef}=[]`: a subset of all variables of `model` that are not allowed to be changed when seeking for alternatives.
"""
function generate_alternatives_optimization!(
    model::JuMP.Model,
    optimality_gap::Float64,
    variables::AbstractArray{T,N},
    n_alternatives::Int64;
    weights = zeros(length(variables)),
    modeling_method::Symbol = :Max_Distance,
    metric::Distances.SemiMetric = SqEuclidean(),
    fixed_variables::Vector{VariableRef} = VariableRef[],
) where {T<:Union{VariableRef,AffExpr},N}
    if !is_solved_and_feasible(model)
        throw(ArgumentError("JuMP model has not been solved."))
    elseif optimality_gap < 0
        throw(ArgumentError("Optimality gap (= $optimality_gap) should be at least 0."))
    elseif n_alternatives < 1
        throw(
            ArgumentError(
                "Number of alternatives (= $n_alternatives) should be at least 1.",
            ),
        )
    end

    result = AlternativeSolutions([], []) # Initialize the result container for storing alternative solutions.


    @info "Creating model for generating alternatives."
    create_alternative_generating_problem!(
        model,
        optimality_gap,
        fixed_variables,
        variables;
        weights = weights,
        modeling_method = modeling_method,
        metric = metric,
    )
    @info "Solving model."
    JuMP.optimize!(model)
    @info "Solution #1/$n_alternatives found." solution_summary(model)
    update_solutions!(result, model)

    # If n_solutions > 1, we repeat the solving process to generate multiple solutions.
    for i = 2:n_alternatives
        @info "Reconfiguring model for generating alternatives."
        update_objective_function!(
            model,
            variables;
            weights = weights,
            modeling_method = modeling_method,
        )
        @info "Solving model."
        JuMP.optimize!(model)
        @info "Solution #$i/$n_alternatives found." solution_summary(model)
        update_solutions!(result, model)
    end

    return result
end

"""
    result = generate_alternatives_metaheuristics(
      model::JuMP.Model,
      optimality_gap::Float64,
      n_alternatives::Int64,
      metaheuristic_algorithm::Metaheuristics.Algorithm;
      metric::Distances.Metric = SqEuclidean(),
      selected_variables::Vector{VariableRef} = []
    )

Generate `n_alternatives` solutions to `model` which are as distant from the optimum and each other, but with a maximum `optimality_gap`, using a metaheuristic algorithm.

# Arguments
- `model::JuMP.Model`: a solved JuMP model for which alternatives are generated.
- `optimality_gap::Float64`: the maximum percentage deviation (>=0) an alternative may have compared to the optimal solution.
- `n_alternatives`: the number of alternative solutions sought.
- `metaheuristic_algorithm::Metaheuristics.Algorithm`: algorithm used to search for alternative solutions.
- `metric::Distances.Metric=SqEuclidean()`: the metric used to maximise the difference between alternatives and the optimal solution.
- `fixed_variables::Vector{VariableRef}=[]`: a subset of all variables of `model` that are not allowed to be changed when seeking for alternatives.
"""
function generate_alternatives_metaheuristics(
    model::JuMP.Model,
    optimality_gap::Float64,
    n_alternatives::Int64,
    metaheuristic_algorithm::Metaheuristics.Algorithm;
    metric::Distances.SemiMetric = SqEuclidean(),
    fixed_variables::Vector{VariableRef} = VariableRef[],
)
    if !is_solved_and_feasible(model)
        throw(ArgumentError("JuMP model has not been solved."))
    elseif optimality_gap < 0
        throw(ArgumentError("Optimality gap (= $optimality_gap) should be at least 0."))
    elseif n_alternatives < 1
        throw(
            ArgumentError(
                "Number of alternatives (= $n_alternatives) should be at least 1.",
            ),
        )
    end

    result = AlternativeSolutions([], [])

    @info "Setting up NearOptimalAlternatives problem and solver."
    # Obtain the solution values for all variables, separated in fixed and non-fixed variables.
    initial_solution = OrderedDict{VariableRef,Float64}()
    fixed_variable_solutions = Dict{MOI.VariableIndex,Float64}()
    for v in all_variables(model)
        if v in fixed_variables
            fixed_variable_solutions[v.index] = value(v)
        else
            initial_solution[v] = value(v)
        end
    end

    problem = create_alternative_generating_problem(
        model,
        metaheuristic_algorithm,
        initial_solution,
        optimality_gap,
        metric,
        fixed_variable_solutions,
    )
    @info "Solving NearOptimalAlternatives problem."
    state = run_alternative_generating_problem!(problem)
    @info "Solution #1/$n_alternatives found." state minimizer(state)


    # Update the solutions based on whether PSOGA is used or not.
    if typeof(metaheuristic_algorithm) ==
       Metaheuristics.Algorithm{NearOptimalAlternatives.PSOGA}
        update_solutions!(
            result,
            state,
            metaheuristic_algorithm.parameters.subBest,
            initial_solution,
            fixed_variable_solutions,
            model,
        )
    else
        update_solutions!(result, state, initial_solution, fixed_variable_solutions, model)
    end

    # Only run iteratively if PSOGA is not used.
    if typeof(metaheuristic_algorithm) !=
       Metaheuristics.Algorithm{NearOptimalAlternatives.PSOGA}
        for i = 2:n_alternatives
            @info "Reconfiguring NearOptimalAlternatives problem with new solution."
            add_solution!(problem, state, metric)
            @info "Solving NearOptimalAlternatives problem."
            state = run_alternative_generating_problem!(problem)
            @info "Solution #$i/$n_alternatives found." state minimizer(state)
            update_solutions!(
                result,
                state,
                initial_solution,
                fixed_variable_solutions,
                model,
            )
        end
    end

    return result
end

"""
    METHOD_DISPATCH_GRADIENT shows the gradient-based (first-order, Hessian-free)
optimization methods usable with `generate_alternatives_gradient`. Each maps a
user-facing `method` symbol to the internal `(search, solver)` pair of
`run_lbfgs_mga`: `:oracle` selects the structural cutting-plane decomposition,
while `:alm_lbfgs`, `:penalty` and `:pdhg` select the corresponding first-order
solver of the whole-variable `:fullspace` search.
"""
const METHOD_DISPATCH_GRADIENT = Dict{Symbol,Tuple{Symbol,Symbol}}(
    :oracle => (:oracle, :alm_lbfgs),
    :alm_lbfgs => (:fullspace, :alm_lbfgs),
    :penalty => (:fullspace, :penalty),
    :pdhg => (:fullspace, :pdhg),
)

"""
    result = generate_alternatives_gradient(
        model::JuMP.Model,
        optimality_gap::Float64,
        variables::AbstractArray{<:JuMP.VariableRef},
        n_alternatives::Int64;
        method::Symbol = :alm_lbfgs,
        operational_recovery::Bool = false,
        x_optimal::Union{Nothing,Vector{Float64}} = nothing,
        min_cost::Union{Nothing,Float64} = nothing,
        max_iters::Int = 30,
    )

Generate `n_alternatives` near-optimal solutions to `model` using a
gradient-based (first-order, Hessian-free) optimization method, with the same
call signature as [`generate_alternatives_optimization!`](@ref) and the
solver selected by `method`.

This is an *optimization* approach (not a metaheuristic): it minimises the
HSJ-style penalty objective `wᵀx` over the near-optimal region with the
matrix-vector-only solvers of `run_lbfgs_mga`, accumulating penalty weights on
the structural `variables` between alternatives.

# Arguments
- `model::JuMP.Model`: a solved JuMP model for which alternatives are generated
  (may be unsolved if both `x_optimal` and `min_cost` are supplied).
- `optimality_gap::Float64`: the maximum relative deviation (>= 0) an
  alternative may have in cost compared to the optimal solution.
- `variables::AbstractArray{<:JuMP.VariableRef}`: the structural variables that
  carry the MGA penalty weights.
- `n_alternatives::Int64`: the number of alternative solutions sought.
- `method::Symbol = :alm_lbfgs`: the gradient method, one of the keys of
  [`METHOD_DISPATCH_GRADIENT`](@ref): `:alm_lbfgs` (augmented Lagrangian +
  projected L-BFGS), `:penalty` (quadratic penalty + projected L-BFGS), `:pdhg`
  (primal-dual hybrid gradient), or `:oracle` (structural cutting-plane
  decomposition with a budget-projection oracle).
- `operational_recovery::Bool`: project each alternative back onto the
  budget-feasible space with `operational_recovery!` before returning it.
- `x_optimal::Union{Nothing,Vector{Float64}}`: optimal solution values used only
  to *define* the MGA problem (HSJ weights); defaults to the model's solution.
  The solver itself never uses this point.
- `min_cost::Union{Nothing,Float64}`: optimal objective value used to define the
  cost budget; defaults to the model's objective value.
- `max_iters::Int`: outer-iteration budget per alternative.
"""
function generate_alternatives_gradient(
    model::JuMP.Model,
    optimality_gap::Float64,
    variables::AbstractArray{<:JuMP.VariableRef},
    n_alternatives::Int64;
    method::Symbol = :alm_lbfgs,
    operational_recovery::Bool = false,
    x_optimal::Union{Nothing,Vector{Float64}} = nothing,
    min_cost::Union{Nothing,Float64} = nothing,
    max_iters::Int = 30,
    pdhg_iters::Int = 10000,
    warm_start::Bool = false,
)
    if isnothing(x_optimal) && !is_solved_and_feasible(model)
        throw(ArgumentError("JuMP model has not been solved."))
    elseif optimality_gap < 0
        throw(ArgumentError("Optimality gap (= $optimality_gap) should be at least 0."))
    elseif n_alternatives < 1
        throw(
            ArgumentError(
                "Number of alternatives (= $n_alternatives) should be at least 1.",
            ),
        )
    end

    search, solver = get(METHOD_DISPATCH_GRADIENT, method) do
        throw(
            ArgumentError(
                "Method $method is not supported (expected one of $(collect(keys(METHOD_DISPATCH_GRADIENT)))).",
            ),
        )
    end

    target_vars = collect(JuMP.VariableRef, variables)
    @info "Setting up gradient MGA ($method) with $(length(target_vars)) structural variables."

    alternatives = lbfgs_search_alternatives(
        model,
        target_vars,
        n_alternatives,
        optimality_gap,
        Inf,
        operational_recovery;
        x_optimal_override = x_optimal,
        min_cost_override = min_cost,
        max_iters = max_iters,
        search = search,
        method = solver,
        pdhg_iters = pdhg_iters,
        warm_start = warm_start,
    )

    result = AlternativeSolutions([], [])
    vars = all_variables(model)
    for alt in alternatives
        sol = Dict{VariableRef,Float64}()
        for i in eachindex(vars)
            sol[vars[i]] = alt[i]
        end
        push!(result.solutions, sol)
    end
    return result
end

"""
    result = generate_alternatives_lbfgs(
        model::JuMP.Model,
        n_alternatives::Int64;
        optimality_gap::Float64 = 0.1,
        method::Symbol = :alm_lbfgs,
        target_symbols::Vector{Symbol} = [:assets_investment, :flows_investment],
        target_vars::Vector{JuMP.VariableRef} = JuMP.VariableRef[],
        operational_recovery::Bool = false,
        x_optimal::Union{Nothing,Vector{Float64}} = nothing,
        min_cost::Union{Nothing,Float64} = nothing,
        max_iters::Int = 30,
    )

Convenience wrapper around [`generate_alternatives_gradient`](@ref) for the
TulipaEnergyModel workflow: it auto-resolves the structural variables from the
registered `target_symbols` (e.g. `:assets_investment`, `:flows_investment`)
when `target_vars` is not given, then forwards to the gradient method.

# Arguments
- `model::JuMP.Model`: a solved JuMP model (or unsolved if both `x_optimal` and
  `min_cost` are supplied).
- `n_alternatives`: the number of alternative solutions sought.
- `optimality_gap::Float64`: relative cost slack defining the near-optimal
  budget.
- `method::Symbol`: the gradient method (see `generate_alternatives_gradient`).
- `target_symbols::Vector{Symbol}`: registered variable containers treated as
  structural variables when `target_vars` is empty.
- `target_vars::Vector{JuMP.VariableRef}`: explicit structural variables.
- `operational_recovery`, `x_optimal`, `min_cost`, `max_iters`: forwarded to
  `generate_alternatives_gradient`.
"""
function generate_alternatives_lbfgs(
    model::JuMP.Model,
    n_alternatives::Int64;
    optimality_gap::Float64 = 0.1,
    method::Symbol = :alm_lbfgs,
    target_symbols::Vector{Symbol} = [:assets_investment, :flows_investment],
    target_vars::Vector{JuMP.VariableRef} = JuMP.VariableRef[],
    operational_recovery::Bool = false,
    x_optimal::Union{Nothing,Vector{Float64}} = nothing,
    min_cost::Union{Nothing,Float64} = nothing,
    max_iters::Int = 30,
    pdhg_iters::Int = 10000,
    warm_start::Bool = false,
)
    actual_target_vars = copy(target_vars)
    if isempty(actual_target_vars)
        dict = object_dictionary(model)
        for sym in target_symbols
            if haskey(dict, sym)
                # JuMP stores variables in n-dimensional arrays or
                # SparseAxisArrays; flatten them into a Vector{VariableRef}.
                for v in dict[sym]
                    push!(actual_target_vars, v)
                end
            end
        end
    end
    if isempty(actual_target_vars)
        @warn "No structural variables found matching $target_symbols. Falling back to all variables."
        actual_target_vars = all_variables(model)
    end

    return generate_alternatives_gradient(
        model,
        optimality_gap,
        actual_target_vars,
        n_alternatives;
        method = method,
        operational_recovery = operational_recovery,
        x_optimal = x_optimal,
        min_cost = min_cost,
        max_iters = max_iters,
        pdhg_iters = pdhg_iters,
        warm_start = warm_start,
    )
end
