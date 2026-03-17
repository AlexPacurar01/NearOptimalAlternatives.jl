using JuMP
using NearOptimalAlternatives
using GenerationExpansionPlanning
using Gurobi
using DataFrames
using Statistics # For mean() and std()
using Plots      # For plotting the results

# --- Custom Model Builder ---
function build_raw_jump_model(
    data::GenerationExpansionPlanning.ExperimentData,
    optimizer_factory,
)
    N = data.locations
    G = data.generation_technologies
    NG = data.generators
    T = data.time_steps
    L = data.transmission_lines
    df2dict = GenerationExpansionPlanning.dataframe_to_dict

    demand_df = copy(data.demand)
    gen_avail_df = copy(data.generation_availability)
    filter!(row -> row.time_step ∈ T, demand_df)
    filter!(row -> row.time_step ∈ T, gen_avail_df)

    demand = df2dict(demand_df, [:location, :time_step], :demand)
    generation_availability =
        df2dict(gen_avail_df, [:location, :technology, :time_step], :availability)
    investment_cost = df2dict(data.generation, [:location, :technology], :investment_cost)
    variable_cost = df2dict(data.generation, [:location, :technology], :variable_cost)
    unit_capacity = df2dict(data.generation, [:location, :technology], :unit_capacity)
    ramping_rate = df2dict(data.generation, [:location, :technology], :ramping_rate)
    export_capacity = df2dict(data.transmission_capacities, [:from, :to], :export_capacity)
    import_capacity = df2dict(data.transmission_capacities, [:from, :to], :import_capacity)

    model = JuMP.Model(optimizer_factory)
    @variable(model, 0 ≤ total_investment_cost)
    @variable(model, 0 ≤ total_operational_cost)
    @variable(model, 0 ≤ investment[n ∈ N, g ∈ G; (n, g) ∈ NG], integer = !data.relaxation)
    @variable(model, 0 ≤ production[n ∈ N, g ∈ G, T; (n, g) ∈ NG])
    @variable(
        model,
        -import_capacity[n_from, n_to] ≤
        line_flow[n_from ∈ N, n_to ∈ N, t ∈ T; (n_from, n_to) ∈ L] ≤
        export_capacity[n_from, n_to]
    )
    @variable(model, 0 ≤ loss_of_load[n ∈ N, t ∈ T] ≤ demand[n, t])

    investment_MW = @expression(
        model,
        [n ∈ N, g ∈ G; (n, g) ∈ NG],
        unit_capacity[n, g] * investment[n, g]
    )
    @objective(model, Min, total_investment_cost + total_operational_cost)

    @constraint(
        model,
        total_investment_cost ==
        sum(investment_cost[n, g] * investment_MW[n, g] for (n, g) ∈ NG)
    )
    @constraint(
        model,
        total_operational_cost ==
        sum(variable_cost[n, g] * production[n, g, t] for (n, g) ∈ NG, t ∈ T) +
        data.value_of_lost_load * sum(loss_of_load[n, t] for n ∈ N, t ∈ T)
    )
    @constraint(
        model,
        [n ∈ N, t ∈ T],
        sum(production[n, g, t] for g ∈ G if (n, g) ∈ NG) +
        sum(line_flow[n_from, n_to, t] for (n_from, n_to) ∈ L if n_to == n) -
        sum(line_flow[n_from, n_to, t] for (n_from, n_to) ∈ L if n_from == n) +
        loss_of_load[n, t] == demand[n, t]
    )
    @constraint(
        model,
        [n ∈ N, g ∈ G, t ∈ T; (n, g) ∈ NG],
        production[n, g, t] ≤
        get(generation_availability, (n, g, t), 1.0) * investment_MW[n, g]
    )

    ramping = @expression(
        model,
        [n ∈ N, g ∈ G, t ∈ T; t > 1 && (n, g) ∈ NG],
        production[n, g, t] - production[n, g, t-1]
    )
    for (n, g, t) ∈ eachindex(ramping)
        @constraint(model, ramping[n, g, t] ≤ ramping_rate[n, g] * investment_MW[n, g])
        @constraint(model, ramping[n, g, t] ≥ -ramping_rate[n, g] * investment_MW[n, g])
    end
    return model
end

# --- Setup Base Model ---
println("Loading configuration and data...")
config_path = "C:/Users/pacurarav/Desktop/EU-model/spatial-model-reductions/case_studies/stylized_EU/config.toml"
config = GenerationExpansionPlanning.read_config(config_path)
experiment_data = GenerationExpansionPlanning.ExperimentData(config[:input])

println("Building base model...")
base_model = build_raw_jump_model(experiment_data, Gurobi.Optimizer)
set_silent(base_model)

for v in all_variables(base_model)
    if !has_upper_bound(v)
        set_upper_bound(v, 1e10)
    end
end

println("Solving base model for optimal cost...")
optimize!(base_model)
optimal_cost = objective_value(base_model)
println("Optimal System Cost: ", optimal_cost)

all_vars = all_variables(base_model)
investment_vars = base_model[:investment]

# --- Statistical Testing Setup ---
n_iterations = 1000
println("\nStarting $n_iterations statistical runs...")

# Data collectors
spores_times = Float64[]
lbfgs_times = Float64[]
spores_dominates_count = Int[]
lbfgs_dominates_count = Int[]

tol = 1e-5 # Floating point tolerance for dominance

for iter = 1:n_iterations
    if iter % 10 == 0
        println("  Completed $iter / $n_iterations runs...")
    end

    ## -- 1. Run SPORES --
    noa_model = copy(base_model)
    set_optimizer(noa_model, Gurobi.Optimizer)
    set_silent(noa_model)
    optimize!(noa_model)

    t0 = time()
    spores_output = NearOptimalAlternatives.generate_alternatives_optimization!(
        noa_model,
        0.1,
        all_variables(noa_model),
        5;
        modeling_method = :Spores,
    )
    push!(spores_times, time() - t0)
    spores_alts = spores_output.solutions

    ## -- 2. Run LBFGS --
    lbfgs_model = copy(base_model)
    set_optimizer(lbfgs_model, Gurobi.Optimizer)
    set_silent(lbfgs_model)
    optimize!(lbfgs_model)

    t0 = time()
    lbfgs_alts = lbfgs_search_alternatives(lbfgs_model, all_variables(lbfgs_model), 5)
    push!(lbfgs_times, time() - t0)

    ## -- 3. Extract Capacities for Dominance Checking --
    # Helper function to extract just the numeric capacities from the outputs
    function extract_caps(alt_data)
        val_map = Dict{String,Float64}()
        if alt_data isa AbstractDict
            for (v_ref, val) in alt_data
                val_map[name(v_ref)] = val
            end
        else
            for j = 1:length(all_vars)
                val_map[name(all_vars[j])] = alt_data[j]
            end
        end
        return [get(val_map, name(var), 0.0) for (index, var) in investment_vars.data]
    end

    spores_caps = [extract_caps(alt) for alt in spores_alts]
    lbfgs_caps = [extract_caps(alt) for alt in lbfgs_alts]

    ## -- 4. Check Dominance --
    # Count how many pairwise dominance events happen in this iteration
    s_dom_l = 0
    l_dom_s = 0

    for s_cap in spores_caps
        for l_cap in lbfgs_caps
            # Does SPORES dominate LBFGS?
            if all(s_cap .<= l_cap .+ tol) && any(s_cap .< l_cap .- tol)
                s_dom_l += 1
            end
            # Does LBFGS dominate SPORES?
            if all(l_cap .<= s_cap .+ tol) && any(l_cap .< s_cap .- tol)
                l_dom_s += 1
            end
        end
    end

    push!(spores_dominates_count, s_dom_l)
    push!(lbfgs_dominates_count, l_dom_s)
end

println("\n--- Testing Complete! Generating Plots... ---")

# --- Plotting the Results ---
# 1. Plot Runtimes
p1 = bar(
    ["SPORES", "LBFGS"],
    [mean(spores_times), mean(lbfgs_times)],
    yerror = [std(spores_times), std(lbfgs_times)],
    title = "Average Run Time (Seconds)",
    ylabel = "Time (s)",
    color = [:blue, :orange],
    legend = false,
)

# 2. Plot Dominance
# p2 = bar(
#     ["SPORES dom. LBFGS", "LBFGS dom. SPORES"],
#     [mean(spores_dominates_count), mean(lbfgs_dominates_count)],
#     yerror = [std(spores_dominates_count), std(lbfgs_dominates_count)],
#     title = "Average Dominance Events Per Run",
#     ylabel = "# of pairwise dominations",
#     color = [:green, :red],
#     legend = false,
# )

# Combine and display the plots
# final_plot = plot(p1, p2, layout = (1, 2), size = (800, 400), margin = 5Plots.mm)
display(p1)

# Print Summary Statistics to the console
println("\n--- Summary Statistics ($n_iterations runs) ---")
println(
    "SPORES Average Time: ",
    round(mean(spores_times), digits = 2),
    "s ± ",
    round(std(spores_times), digits = 2),
    "s",
)
println(
    "LBFGS Average Time:  ",
    round(mean(lbfgs_times), digits = 2),
    "s ± ",
    round(std(lbfgs_times), digits = 2),
    "s",
)
println("Total times SPORES strictly dominated LBFGS: ", sum(spores_dominates_count))
println("Total times LBFGS strictly dominated SPORES: ", sum(lbfgs_dominates_count))
