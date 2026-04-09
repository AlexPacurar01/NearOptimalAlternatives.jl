module GradientMGA

using JuMP
using LinearAlgebra
using SparseArrays
using Optim
using NLSolversBase

export run_lbfgs_mga, extract_all_constraints, operational_recovery!

"""
    extract_all_constraints(model, vars)
Extracts both Inequalities (A_in * x <= b_in) and Equalities (A_eq * x == b_eq).
"""
function extract_all_constraints(model::Model, vars::Vector{VariableRef})
    var_to_idx = Dict(v => i for (i, v) in enumerate(vars))
    n_vars = length(vars)

    # --- Inequalities ---
    I_in = Int[]
    J_in = Int[]
    V_in = Float64[]
    b_in = Float64[]
    row_in = 1

    for (i, v) in enumerate(vars)
        if has_lower_bound(v)
            push!(I_in, row_in)
            push!(J_in, i)
            push!(V_in, -1.0)
            push!(b_in, -lower_bound(v))
            row_in += 1
        end
        if has_upper_bound(v)
            push!(I_in, row_in)
            push!(J_in, i)
            push!(V_in, 1.0)
            push!(b_in, upper_bound(v))
            row_in += 1
        end
    end

    for cref in
        all_constraints(model, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
        obj = constraint_object(cref)
        push!(b_in, obj.set.upper - obj.func.constant)
        for (v, coeff) in obj.func.terms
            push!(I_in, row_in)
            push!(J_in, var_to_idx[v])
            push!(V_in, coeff)
        end
        row_in += 1
    end

    A_in = sparse(I_in, J_in, V_in, row_in - 1, n_vars)

    # --- Equalities ---
    I_eq = Int[]
    J_eq = Int[]
    V_eq = Float64[]
    b_eq = Float64[]
    row_eq = 1

    for cref in
        all_constraints(model, GenericAffExpr{Float64,VariableRef}, MOI.EqualTo{Float64})
        obj = constraint_object(cref)
        push!(b_eq, obj.set.value - obj.func.constant)
        for (v, coeff) in obj.func.terms
            push!(I_eq, row_eq)
            push!(J_eq, var_to_idx[v])
            push!(V_eq, coeff)
        end
        row_eq += 1
    end

    A_eq = sparse(I_eq, J_eq, V_eq, row_eq - 1, n_vars)

    return A_in, b_in, A_eq, b_eq
end

"""
    operational_recovery!(model, lazy_solution, vars, is_structural_mask)
Locks the capacities found by L-BFGS and runs the LP to fix the dispatch physics.
"""
function operational_recovery!(
    model::Model,
    lazy_solution::Vector{Float64},
    is_structural_mask::BitVector,
    max_allowed_cost::Float64,
    optimizer_factory,
)
    recov_model, ref_map = JuMP.copy_model(model)
    set_optimizer(recov_model, optimizer_factory)

    # 1. ENFORCE THE BUDGET!
    old_obj = objective_function(recov_model)
    @constraint(recov_model, mga_strict_budget, old_obj <= max_allowed_cost)

    original_vars = all_variables(model)

    # 2. PROJECTION: Create variables to measure the "distance" from the L-BFGS guess
    @variable(recov_model, abs_diff[1:length(original_vars)] >= 0)

    for i = 1:length(original_vars)
        if is_structural_mask[i]
            target_var = ref_map[original_vars[i]]
            # Clean up the L-BFGS guess (no negative capacities!)
            target_val = max(0.0, lazy_solution[i])

            # Formulate the absolute value difference: abs_diff = |target_var - target_val|
            @constraint(recov_model, abs_diff[i] >= target_var - target_val)
            @constraint(recov_model, abs_diff[i] >= target_val - target_var)
        end
    end

    # 3. MINIMIZE MAXIMUM VARIABLE DEVIATION (L1 norm of differences)
    @objective(
        recov_model,
        Min,
        sum(abs_diff[i] for i = 1:length(original_vars) if is_structural_mask[i])
    )

    set_silent(recov_model)
    optimize!(recov_model)

    if termination_status(recov_model) != MOI.OPTIMAL
        @warn "Projection failed! L-BFGS guess was completely irrecoverable."
        return nothing
    end

    max_distance = maximum(
        abs.(
            lazy_solution[i] - JuMP.value(ref_map[original_vars[i]]) for
            i = 1:length(original_vars) if is_structural_mask[i]
        ),
    )
    max_percentage = maximum(
        abs.(
            (lazy_solution[i] - JuMP.value(ref_map[original_vars[i]])) /
            max(1e-6, lazy_solution[i]) for
            i = 1:length(original_vars) if is_structural_mask[i]
        ),
    )
    println("Projection successful! Max distance from L-BFGS guess: $max_distance")
    println(
        "Projection successful! Max percentage deviation from L-BFGS guess: $max_percentage",
    )

    return JuMP.value.(all_variables(recov_model)), max_distance, max_percentage
end

function repair_equalities(x_guess, A_eq, b_eq)
    # 1. Calculate how much the equalities are violated
    violation = A_eq * x_guess - b_eq

    # 2. Solve for the Lagrange multipliers (lambda)
    # We solve (A * A^T) * lambda = violation to avoid explicit matrix inversion
    A_A_T = A_eq * A_eq'
    lambda = A_A_T \ violation

    # 3. Apply the gradient of the multipliers to repair the guess
    x_repaired = x_guess - A_eq' * lambda

    return x_repaired
end

"""
    run_lbfgs_mga(...)
Uses L-BFGS to maximize distance from a known point subject to soft constraint penalties.
"""
function run_lbfgs_mga(
    model::Model,
    x_start::Vector{Float64},
    eps::Float64 = 0.1,
    w::Vector{Float64} = zeros(length(x_start)),
    max_allowed_cost::Float64 = Inf,
    repair = false,
    optimizer_search = nothing,
)

    # Include MGA Cost Constraint: Only allow solutions within (1+eps) of the optimal cost (copy to ensure correct objective function reference is used)
    min_cost = objective_value(model)
    max_allowed_cost = (1 + eps) * min_cost

    new_model, ref_map = JuMP.copy_model(model)
    vars = all_variables(new_model)
    new_cost_expr = objective_function(new_model)
    @constraint(new_model, mga_cost_limit, new_cost_expr <= max_allowed_cost)

    # Feature Space Mask (Identify Structural Variables)
    is_structural = BitVector(
        occursin("investment", string(name(v))) || occursin("capacity", string(name(v))) for v in vars
    )
    # is_structural = BitVector(1 for v in vars) # For testing, treat all variables as structural
    mask_float = Float64.(is_structural) # Used for fast gradient math

    A_in, b_in, A_eq, b_eq = extract_all_constraints(new_model, vars)

    # lambda = zeros(size(A_eq, 1)) # Multipliers for Equalities
    # nu = zeros(size(A_in, 1))     # Multipliers for Inequalities

    # Penalty hyper-parameters (You may need to tune these! If the final recovered cost
    # is too high, increase rho. If L-BFGS fails to move, decrease rho.)
    rho = 1000  # Inequality penalty weight
    mu = 10000   # Equality penalty weight

    w = w ./ norm(w) # Normalize feature weights just in case

    # Optim.jl requires a combined function for performance (calculates Loss and Grad together)
    function fg!(F, G, x)
        # 1. Inequality Violations
        viol_in = (A_in * x) .- b_in
        viol_in_pos = max.(0.0, viol_in)

        # 2. Equality Violations
        viol_eq = (A_eq * x) .- b_eq

        if G !== nothing
            # Gradient: -2*Distance + rho*(A_in^T * viol_in_pos) + mu*(A_eq^T * viol_eq)
            # grad_dist = -2.0 .* sum((x .- x_known) for x_known in previous_solutions)
            grad_dist = w
            grad_in = rho .* (A_in' * viol_in_pos)
            grad_eq = mu .* (A_eq' * viol_eq)

            G .= grad_dist .+ grad_in .+ grad_eq
            # G .= w .+ grad_in .+ grad_eq # Apply feature weights
        end

        if F !== nothing
            # Loss: -Distance^2 + (rho/2)*viol_in_pos^2 + (mu/2)*viol_eq^2
            # loss_objecive = -sum(sum((x .- x_known) .^ 2) for x_known in previous_solutions)
            loss_objective = dot(w, x)
            loss_in = (rho / 2.0) * sum(viol_in_pos .^ 2)
            loss_eq = (mu / 2.0) * sum(viol_eq .^ 2)

            return loss_objective + loss_in + loss_eq
        end
    end

    # Start L-BFGS slightly perturbed from x_known so the gradient isn't perfectly zero
    x_start = x_start .+ 0.01 * randn(length(x_start))

    # println("Starting L-BFGS search...")
    # Run the optimizer
    res = optimize(
        only_fg!(fg!),
        x_start,
        LBFGS(),
        Optim.Options(iterations = 25, show_trace = false),
    )
    x_lbfgs = Optim.minimizer(res)


    # println("Starting Augmented Lagrangian L-BFGS search...")
    # x_current = x_start .+ 0.01 * randn(length(x_start)) # Perturb the start to avoid zero gradients at the known solution

    # # Outer ALM Loop
    # max_outer_iters = 10
    # viol_in = 0.0
    # viol_eq = 0.0
    # max_viol = 0.0
    # for outer_iter = 1:max_outer_iters

    #     # Inner Loop Objective & Gradient (Captures lambda and nu from outer scope)
    #     function fg!(F, G, x)
    #         viol_in = (A_in * x) .- b_in
    #         viol_eq = (A_eq * x) .- b_eq

    #         # Calculate "active" multipliers combining current multipliers and penalties
    #         active_lambda = lambda .+ mu .* viol_eq
    #         active_nu = max.(0.0, nu .+ rho .* viol_in)

    #         if G !== nothing
    #             grad_dist = w
    #             grad_eq = A_eq' * active_lambda
    #             grad_in = A_in' * active_nu
    #             G .= grad_dist .+ grad_in .+ grad_eq
    #         end

    #         if F !== nothing
    #             loss_objective = dot(w, x)
    #             loss_eq = dot(lambda, viol_eq) + (mu / 2.0) * sum(viol_eq .^ 2)
    #             # The mathematically exact ALM term for inequalities (ignoring constants w.r.t x)
    #             loss_in = (1.0 / (2.0 * rho)) * sum(active_nu .^ 2)

    #             return loss_objective + loss_eq + loss_in
    #         end
    #     end

    #     # Run a short L-BFGS optimization using current multipliers
    #     res = optimize(
    #         only_fg!(fg!),
    #         x_current,
    #         LBFGS(),
    #         Optim.Options(iterations = 10, show_trace = false),
    #     )
    #     x_current = Optim.minimizer(res)

    #     # 4. Update the Lagrange Multipliers
    #     viol_eq = (A_eq * x_current) .- b_eq
    #     viol_in = (A_in * x_current) .- b_in

    #     lambda .= lambda .+ mu .* viol_eq
    #     nu .= max.(0.0, nu .+ rho .* viol_in)

    #     # Check for feasibility to break early
    #     max_viol = max(maximum(abs.(viol_eq)), maximum(max.(0.0, viol_in)))
    #     if max_viol < 1e-5
    #         println("Converged to feasible solution at outer iteration $outer_iter")
    #         break
    #     end
    # end

    final_point = x_lbfgs

    if repair
        println("L-BFGS converged. Running Operational Recovery (Snap to Grid)...")
        final_point, max_distance, max_percentage = operational_recovery!(
            model,
            x_lbfgs,
            is_structural,
            max_allowed_cost,
            optimizer_search,
        )
        return final_point, max_distance, max_percentage
    end
    # final_point = x_current


    return final_point
end

end # module
