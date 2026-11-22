"""
Structure holding the solutions for the near-optimal alternatives.

`tags` is optional per-solution metadata (e.g. the discovery direction and budget
level a point came from in a budget sweep). It is empty for the classic
single-budget generators and is only populated by [`generate_alternatives_sweep!`](@ref).
"""
mutable struct AlternativeSolutions
    solutions::Vector{Dict{VariableRef,Float64}}
    objective_values::Vector{Float64}
    tags::Vector{NamedTuple}
end

# Backward-compatible constructor: existing call sites build `AlternativeSolutions([], [])`
# and never touch `tags`, so default it to empty.
AlternativeSolutions(solutions, objective_values) =
    AlternativeSolutions(solutions, objective_values, NamedTuple[])

"""
    update_solutions!(results::AlternativeSolutions, model::JuMP.Model)

Update the set of results `AlternativeSolutions` with the variable values obtained when solving the JuMP model `model`.
"""
function update_solutions!(results::AlternativeSolutions, model::JuMP.Model)
    if !is_solved_and_feasible(model)
        throw(ErrorException("JuMP model $model not solved when trying to read results."))
    end

    # Retrieve all variable values from JuMP model.
    solution = Dict{VariableRef,Float64}()
    for v in all_variables(model)
        solution[v] = value(v)
    end
    push!(results.solutions, solution)

    # The `original_objective` budget constraint must exist: it marks a model prepared for
    # generating alternatives. The cost itself is not read from this constraint, because
    # its value excludes any constant term of the objective (see
    # `original_objective_value` for why).
    original_objective = constraint_by_name(model, "original_objective")
    if isnothing(original_objective)
        throw(
            ErrorException(
                "JuMP model $model has no constraint named `original_objective`",
            ),
        )
    end
    push!(results.objective_values, original_objective_value(model))
end

"""
    original_objective_value(model::JuMP.Model)

Return the value of the model's original objective at the current solution, including
any constant term. This is the cost stored in `AlternativeSolutions.objective_values`.

# Why the budget constraint's value is not used

When alternatives are generated, the original objective becomes the budget constraint
`original_objective`, and the model's objective is replaced by a diversity objective.
The cost of an alternative then has to be recovered from the original objective.

The obvious way, `value(constraint_by_name(model, "original_objective"))`, gives the
wrong number when the objective has a constant term. JuMP stores every constraint with
the constants moved to the right-hand side, and the value of a constraint is the value
of its left-hand side only. For the objective `2x + 3y + 100`, the constraint holds
`2x + 3y`, so its value is 100 lower than the true cost. The budget itself is still
enforced correctly, since the constant was subtracted from the bound as well.

The optimum's cost, `objective_value(model)` before alternatives are generated, does
include the constant. Using the constraint's value would put the optimum and its
alternatives on different scales. For example, an alternative exactly on a 20% budget
would appear to cost less than 1.2 times the optimum.

[`create_alternative_generating_problem!`](@ref) therefore stores the full original
objective in `model.ext[:noa_original_objective]`, and this function evaluates it.

# Fallback

A model whose budget constraint was built some other way has no stored objective. For
such a model this function falls back to the constraint's value, which is the previous
behaviour and is exact when the objective has no constant term.
"""
function original_objective_value(model::JuMP.Model)
    f = get(model.ext, :noa_original_objective, nothing)
    f === nothing || return value(f)
    return value(constraint_by_name(model, "original_objective"))
end

"""
    update_solutions!(results::AlternativeSolutions, model::JuMP.Model)

Update the set of results `AlternativeSolutions` with the variable values obtained when solving using Metaheuristics.

# Arguments
- `results::AlternativeSolutions`: set of solutions to add a new solution to.
- `state::Metaheuristics.State`: contains results to metaheuristic solve.
- `initial:solution::OrderedDict{VariableRef, Float64}`: used to identify the indices of the metaheuristic solution in the JuMP model.
- `fixed_variables::Dict{MOI.VariableIndex, Float64}`: set of fixed variables and their solution values.
- `model::JuMP.Model`: original model for which alternative solutions are found.
"""
function update_solutions!(
    results::AlternativeSolutions,
    state::Metaheuristics.State,
    initial_solution::OrderedDict{VariableRef,Float64},
    fixed_variables::Dict{MOI.VariableIndex,Float64},
    model::JuMP.Model,
)
    if !state.stop
        throw(
            ErrorException(
                "Metaheuristic state `state` not terminated when trying to read results.",
            ),
        )
    end

    best_solution = minimizer(state)
    solution = Dict{VariableRef,Float64}()
    index_map = Dict{Int64,Int64}()
    # Add all new results
    for (i, (k, _)) in enumerate(initial_solution)
        solution[k] = best_solution[i]
        index_map[k.index.value] = i
    end
    # Add values of fixed variables.
    for (k, v) in fixed_variables
        solution[JuMP.VariableRef(model, k)] = v
    end

    push!(results.solutions, solution)

    # Retrieve objective value to original problem.
    objective_value = extract_objective(
        JuMP.objective_function(model),
        best_solution,
        index_map,
        fixed_variables,
    )
    push!(results.objective_values, objective_value)
end


"""
    update_solutions!(results::AlternativeSolutions, model::JuMP.Model)

Update the set of results `AlternativeSolutions` with the variable values obtained when solving using PSOGA.

# Arguments
- `results::AlternativeSolutions`: set of solutions to add a new solution to.
- `state::Metaheuristics.State`: contains results to metaheuristic solve.
- `subBest::Vector{Any}`: contains the best results per subpopulation of PSOGA, which are the actual results of solving.
- `initial:solution::OrderedDict{VariableRef, Float64}`: used to identify the indices of the metaheuristic solution in the JuMP model.
- `fixed_variables::Dict{MOI.VariableIndex, Float64}`: set of fixed variables and their solution values.
- `model::JuMP.Model`: original model for which alternative solutions are found.
"""
function update_solutions!(
    results::AlternativeSolutions,
    state::Metaheuristics.State,
    subBest::Vector{Any},
    initial_solution::OrderedDict{VariableRef,Float64},
    fixed_variables::Dict{MOI.VariableIndex,Float64},
    model::JuMP.Model,
)
    if !state.stop
        throw(
            ErrorException(
                "Metaheuristic state `state` not terminated when trying to read results.",
            ),
        )
    end

    # Loop over the best solutions of each subpopulation.
    for best_solution in subBest
        solution = Dict{VariableRef,Float64}()
        index_map = Dict{Int64,Int64}()
        # Add all new results
        for (i, (k, _)) in enumerate(initial_solution)
            solution[k] = best_solution.x[i]
            index_map[k.index.value] = i
        end
        # Add values of fixed variables.
        for (k, v) in fixed_variables
            solution[JuMP.VariableRef(model, k)] = v
        end

        push!(results.solutions, solution)

        # Retrieve objective value to original problem.
        objective_value = extract_objective(
            JuMP.objective_function(model),
            best_solution.x,
            index_map,
            fixed_variables,
        )
        push!(results.objective_values, objective_value)
    end
end
