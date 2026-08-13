import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using NearOptimalAlternatives
using JuMP
using Gurobi
using LinearAlgebra

# ---------------------------------------------------------------------------
# Validate the ALM + L-BFGS structural/operational decomposition on the
# data/5h dataset (745 structural investment variables, ~1.27M total
# variables), mirroring the synthetic_test.jl validation: load/solve the
# baseline, run generate_alternatives_lbfgs with operational_recovery, and
# report the cost-budget feasibility of the recovered alternative.
# ---------------------------------------------------------------------------

function get_base_optimizer()
    return optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 1,
        "Method" => 2,
        "Crossover" => 1,
        "Threads" => 32,
    )
end

input_dir = joinpath(pwd(), "data/5h")

println("Reading data from: $input_dir")

energy_problem = TEM.create_energy_problem_from_csv_folder(input_dir)
TEM.create_model!(energy_problem)
base_model = energy_problem.model
set_optimizer(base_model, get_base_optimizer())

# The baseline is always solved in-session: MGA presupposes a solved model
# (it defines the HSJ weights and the cost budget). This is the standard
# workflow precondition, timed separately from the algorithm itself.
println("Solving baseline (workflow precondition, several minutes)...")
t_baseline = @elapsed optimize!(base_model)
base_obj_val = objective_value(base_model)
base_vals = value.(all_variables(base_model))
println("Baseline solved in $(round(t_baseline, digits=1))s. Obj: $base_obj_val")

println("\n=== Running ALM + L-BFGS structural/operational MGA on data/5h ===")

# Cold start: the algorithm receives only the model, the HSJ weights (derived
# from the baseline solution, which defines the MGA problem) and the scalar
# budget - no caches, no precomputed reference points.
t_alg = @elapsed begin
    # Each oracle call costs about one barrier solve of the model, so
    # max_iters is the explicit wall-clock budget for this dataset; the
    # repair projection at the end adds one more solve.
    global result = NearOptimalAlternatives.generate_alternatives_lbfgs(
        base_model,
        1;
        operational_recovery = true,
        x_optimal = base_vals,
        min_cost = base_obj_val,
        max_iters = 8,
    )
end
println(
    "Algorithm wall-clock time (cold start, incl. all sub-solves): $(round(t_alg, digits=1)) s",
)

if isempty(result.solutions)
    error("No alternative was produced - see the algorithm warnings above.")
end

alt = result.solutions[1]
all_vars = all_variables(base_model)
alt_vec = [alt[v] for v in all_vars]

eps_mga = 0.1
max_allowed_cost = (1 + eps_mga) * base_obj_val

model_obj = objective_function(base_model)
recovered_cost = sum(
    coefficient(model_obj, v) * alt[v] for
    v in all_vars if !iszero(coefficient(model_obj, v))
)

println("\n=== VALIDATION (data/5h) ===")
println("Base objective:            ", base_obj_val)
println("Budget (max allowed cost): ", max_allowed_cost)
println("Recovered solution cost:   ", recovered_cost)
println("Cost feasible:             ", recovered_cost <= max_allowed_cost + 1e-6)

# ---------------------------------------------------------------------------
# Post-hoc optimality-gap measurement (test-only): solve the true MGA LP
#   min w'x  s.t.  original constraints, cost(x) <= max_allowed_cost
# directly with Gurobi, where w is the same first-iteration HSJ weight vector
# the algorithm used internally. This reference solve is *not* part of the
# algorithm (it never sees this solution) and is timed separately as test
# setup; the gap it yields measures how close the algorithm's alternative is
# to the true optimum of the alternative-generating problem.
# ---------------------------------------------------------------------------

# Reconstruct the first-iteration HSJ weights exactly as
# lbfgs_search_alternatives does (same target symbols, same order).
target_vars = JuMP.VariableRef[]
for sym in [:assets_investment, :flows_investment]
    if haskey(object_dictionary(base_model), sym)
        for v in object_dictionary(base_model)[sym]
            push!(target_vars, v)
        end
    end
end
var_to_idx = Dict(v => i for (i, v) in enumerate(all_vars))
target_indices = [var_to_idx[v] for v in target_vars]
target_ubs =
    [has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0 for v in target_vars]

w = zeros(length(all_vars))
for (j, idx) in enumerate(target_indices)
    w[idx] = base_vals[idx] / target_ubs[j]
end
w ./= norm(w)

println("\nSolving LP-utopian reference (test setup, not part of the algorithm)...")
t_ref = @elapsed begin
    lp_model = copy(base_model)
    set_optimizer(
        lp_model,
        optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag" => 0,
            "Method" => 2,
            "Crossover" => 0,
            "Threads" => 32,
        ),
    )
    lp_vars = all_variables(lp_model)
    @constraint(lp_model, objective_function(lp_model) <= max_allowed_cost)
    @objective(
        lp_model,
        Min,
        sum(w[i] * lp_vars[i] for i in eachindex(lp_vars) if !iszero(w[i]))
    )
    optimize!(lp_model)
end
println("Reference solve time: $(round(t_ref, digits=1)) s")

if termination_status(lp_model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    lp_utopian = value.(lp_vars)
    w_s = w[target_indices]
    lp_obj_s = dot(w_s, lp_utopian[target_indices])
    alt_obj_s = dot(w_s, alt_vec[target_indices])
    lp_obj_full = dot(w, lp_utopian)
    alt_obj_full = dot(w, alt_vec)
    gap_s = abs(lp_obj_s - alt_obj_s) / max(abs(lp_obj_s), 1e-8) * 100.0
    gap_full = abs(lp_obj_full - alt_obj_full) / max(abs(lp_obj_full), 1e-8) * 100.0

    println("\n=== OPTIMALITY GAP (data/5h) ===")
    println("LP utopian w'x (structural):  ", lp_obj_s)
    println("Algorithm  w'x (structural):  ", alt_obj_s)
    println("Structural gap:               ", round(gap_s, digits = 3), "%")
    println("LP utopian w'x (full model):  ", lp_obj_full)
    println("Algorithm  w'x (full model):  ", alt_obj_full)
    println("Full-model gap:               ", round(gap_full, digits = 3), "%")
    println(
        "\nAlgorithm time:  $(round(t_alg, digits=1)) s  vs  reference LP solve: $(round(t_ref, digits=1)) s",
    )
else
    println("Reference LP did not solve: ", termination_status(lp_model))
end
