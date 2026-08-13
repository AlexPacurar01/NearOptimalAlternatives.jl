import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using Plots
using NearOptimalAlternatives
using JuMP
using Gurobi
using Statistics
using HiGHS
using Serialization

# --- SOLVER CONFIGURATION ---

function get_base_optimizer(optimizer::Symbol)
    if optimizer == :HiGHS
        return optimizer_with_attributes(
            HiGHS.Optimizer,
            "output_flag" => true,
            "solver" => "ipm",
            "run_crossover" => "on",
            "parallel" => "on",
        )
    elseif optimizer == :Gurobi
        return optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag" => 1,
            "Method" => 2,    # 2 = Barrier
            "Crossover" => 1, # 1 = Enable Crossover for the base solve
            "Threads" => 32,  # Match the SLURM cpus-per-task
        )
    end
end

function get_mga_optimizer(optimizer::Symbol)
    if optimizer == :HiGHS
        return optimizer_with_attributes(
            HiGHS.Optimizer,
            "output_flag" => true,
            "solver" => "simplex",
            "presolve" => "on",
        )
    elseif optimizer == :Gurobi
        return optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag" => 1,      # Keep on so we can see what it's doing
            "Method" => 2,          # Force Barrier (Simplex is choking on the large numbers)
            "Crossover" => 0,       # Disable Crossover (Massive time saver for MGA iterations)
            "NumericFocus" => 3,    # Maximum mathematical precision
            "ScaleFlag" => 2,       # Aggressive internal matrix scaling
            "Threads" => 16,        # Match your SLURM --cpus-per-task
        )
    end
end

# --- HELPER FUNCTIONS ---
# (is_feasible and process_alternatives remain unchanged)

function is_feasible(
    model,
    val_map::Dict{VariableRef,Float64},
    max_allowed_cost;
    tol = 1e-4,
)
    total_cost = value(v -> val_map[v], objective_function(model))
    out = true
    max_violation = 0.0

    if total_cost > max_allowed_cost + tol
        out = false
        max_violation = max(max_violation, total_cost - max_allowed_cost)
    end

    for (F, S) in list_of_constraint_types(model)
        for cref in all_constraints(model, F, S)
            c_obj = constraint_object(cref)
            val = value(v -> val_map[v], c_obj.func)
            set = c_obj.set

            if S <: MOI.LessThan && val > set.upper + tol
                out = false
                max_violation = max(max_violation, val - set.upper)
            elseif S <: MOI.GreaterThan && val < set.lower - tol
                out = false
                max_violation = max(max_violation, set.lower - val)
            elseif S <: MOI.EqualTo && abs(val - set.value) > tol
                out = false
                max_violation = max(max_violation, abs(val - set.value))
            elseif S <: MOI.Interval && (val < set.lower - tol || val > set.upper + tol)
                out = false
                viol = max(set.lower - val, val - set.upper)
                max_violation = max(max_violation, viol)
            end
        end
    end
    return out, max_violation
end

function process_alternatives(alts_raw, all_vars)
    dict_list = []
    for alt_data in alts_raw
        val_map = Dict{VariableRef,Float64}()
        if alt_data isa AbstractDict
            for (v_ref, val) in alt_data
                val_map[v_ref] = val
            end
        else
            for j = 1:length(all_vars)
                val_map[all_vars[j]] = alt_data[j]
            end
        end
        push!(dict_list, val_map)
    end
    return dict_list
end

# --- MAIN EXECUTION ---

n = 1
m = 1

lbfgs_times = zeros(5, n, m)
hsj_times = zeros(5, n, m)
spores_times = zeros(5, n, m)

infeasibility_count = 0
selected_solver = :Gurobi

for iter = 5:5
    input_dir = joinpath(pwd(), "data/$(iter)h")

    output_dir = joinpath(pwd(), "data/results_$(iter)h")

    mkpath(output_dir)

    println("\n=== Processing Dataset: $(iter)h ===")
    println("Reading data from: $input_dir")

    solution_cache_path = joinpath(output_dir, "base_solution_$(iter)h.bin")
    local base_model
    local base_obj_val
    local base_vals

    if isfile(solution_cache_path)
        println("♻️ Found binary cache! Loading...")

        # 1. Build the virgin model
        energy_problem = TEM.create_energy_problem_from_csv_folder(input_dir)
        TEM.create_model!(energy_problem)
        base_model = energy_problem.model
        set_optimizer(base_model, get_base_optimizer(selected_solver))

        # 2. Deserialize the binary array instantly
        t0 = time()
        cache = deserialize(solution_cache_path)
        base_obj_val = cache[:objective]
        base_vals = cache[:values]

        # 3. Apply the values directly to the variables via array broadcasting
        # No loops, no string names, no dictionary lookups!
        set_start_value.(all_variables(base_model), base_vals)

        println(
            "✅ Baseline loaded in $(round(time() - t0, digits=2)) seconds. Obj: $base_obj_val",
        )

        # No re-solve needed: the cached objective/values are passed directly
        # to generate_alternatives_lbfgs below. Re-solving the 1.27M-variable
        # LP just to populate JuMP's solution cache is extremely slow and adds
        # nothing (Gurobi's barrier method ignores primal start values anyway).
    else
        println("⏳ No cache found. Running full scenario...")

        energy_problem = TEM.create_energy_problem_from_csv_folder(input_dir)
        TEM.create_model!(energy_problem)
        base_model = energy_problem.model

        set_optimizer(base_model, get_base_optimizer(selected_solver))
        optimize!(base_model)

        # Grab the results
        base_obj_val = objective_value(base_model)

        # Extract a flat array of Floats instead of a String Dictionary
        base_vals = value.(all_variables(base_model))

        # Save as a NamedTuple to a binary file
        cache_data = (objective = base_obj_val, values = base_vals)
        serialize(solution_cache_path, cache_data)

        println("💾 Base solution saved instantly to $solution_cache_path")
    end

    base_vars = all_variables(base_model)

    let run_iterations = m, run_alternatives = n, slack_percentage = 0.25
        max_allowed_cost = base_obj_val * (1 + slack_percentage)

        for k = 1:run_alternatives
            println("\n--- Dataset: $(iter)h | Run $k alternatives ---")

            for i = 1:run_iterations
                println("Iteration $i")

                # # --- HSJ ---
                # copy_model = copy(base_model)
                # set_optimizer(copy_model, get_mga_optimizer(selected_solver))
                # set_start_value.(all_variables(copy_model), base_vals)
                # optimize!(copy_model)

                # t0 = time()
                # hsj_output = NearOptimalAlternatives.generate_alternatives_optimization!(
                #     copy_model,
                #     slack_percentage,
                #     all_variables(copy_model),
                #     k;
                #     modeling_method = :HSJ,
                # )
                # hsj_times[iter, k, i] = time() - t0

                # # --- SPORES ---
                # copy_model = copy(base_model)
                # set_optimizer(copy_model, get_mga_optimizer(selected_solver))
                # set_start_value.(all_variables(copy_model), base_vals)
                # optimize!(copy_model)

                # t0 = time()
                # spores_alts = NearOptimalAlternatives.generate_alternatives_optimization!(
                #     copy_model,
                #     slack_percentage,
                #     all_variables(copy_model),
                #     k;
                #     modeling_method = :Spores,
                # )
                # spores_times[iter, k, i] = time() - t0

                # --- LBFGS ---
                # copy_model = copy(base_model)
                # set_optimizer(copy_model, get_mga_optimizer(selected_solver))
                # set_start_value.(all_variables(copy_model), base_vals)
                # optimize!(copy_model)

                t0 = time()
                lbfgs_alts = NearOptimalAlternatives.generate_alternatives_lbfgs(
                    base_model,
                    k,
                    operational_recovery = true,
                    x_optimal = base_vals,
                    min_cost = base_obj_val,
                )
                lbfgs_times[iter, k, i] = time() - t0
            end
        end
    end
end


# --- DATA WRANGLING & PLOTTING ---

hsj_mean = [zeros(n) for _ = 1:5]
hsj_std = [zeros(n) for _ = 1:5]

spores_mean = [zeros(n) for _ = 1:5]
spores_std = [zeros(n) for _ = 1:5]

lbfgs_mean = [zeros(n) for _ = 1:5]
lbfgs_std = [zeros(n) for _ = 1:5]

# Slice the 3D arrays properly (linear indexing fixed)
for i = 5:5
    hsj_mean[i] = vec(mean(hsj_times[i, :, :], dims = 2))
    hsj_std[i] = vec(std(hsj_times[i, :, :], dims = 2))

    spores_mean[i] = vec(mean(spores_times[i, :, :], dims = 2))
    spores_std[i] = vec(std(spores_times[i, :, :], dims = 2))

    lbfgs_mean[i] = vec(mean(lbfgs_times[i, :, :], dims = 2))
    lbfgs_std[i] = vec(std(lbfgs_times[i, :, :], dims = 2))
end

x_values = [k for k = 1:n]
subplots = []

# Generate a subplot for each of the 5 datasets
for iter = 5:5
    p = plot(
        x_values,
        hsj_mean[iter],
        yerror = hsj_std[iter],
        label = "HSJ",
        marker = :circle,
        title = "Dataset: $(iter)h", # Cleaned up title
        xlabel = "Alternatives",
        ylabel = "Time (s)",
        legend = iter == 1 ? :topleft : false, # Only show legend on the first plot to save space
    )
    plot!(
        p,
        x_values,
        spores_mean[iter],
        yerror = spores_std[iter],
        label = "SPORES",
        marker = :square,
    )
    plot!(
        p,
        x_values,
        lbfgs_mean[iter],
        yerror = lbfgs_std[iter],
        label = "LBFGS",
        marker = :diamond,
    )
    push!(subplots, p)
end

# Combine all 5 plots into a single 3x2 grid figure
final_plot = plot(
    subplots...,
    layout = (3, 2),
    size = (1000, 1200),
    plot_title = "Algorithm Run Times Across Dataset Sizes",
    margin = 5Plots.mm,
)
# display(final_plot)

# Save all plots
savefig(final_plot, "scalability_results.png")
for plt in subplots
    raw_title = plt[1][:title]
    title = replace(raw_title, " " => "_", ":" => "")
    savefig(plt, "scalability_$(title).png")
end
