using JuMP
using NearOptimalAlternatives
using GenerationExpansionPlanning
using Gurobi
using DataFrames
using Statistics # For mean() and std()
using Plots      # For plotting the results
using HiGHS

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

# --- Feasibility Checker ---
# Plugs a dictionary of variable values back into the JuMP model equations
function is_feasible(model, val_map_by_name, max_allowed_cost; tol = 1e-4)
    # Helper to look up values by name
    lookup(v) = get(val_map_by_name, name(v), 0.0)

    # Compute relative cost tolerance based on the optimal cost
    # cost_tol = max(tol, max_allowed_cost * tol)
    cost_tol = tol # Using absolute tolerance for cost to avoid issues when optimal cost is near zero

    out = true

    max_violation = 0.0

    # 1. Check if the alternative respects the cost limit
    total_cost = value(lookup, objective_function(model))
    if total_cost > max_allowed_cost + cost_tol
        # println("❌ FEASIBILITY FAILED: Cost Limit Exceeded!")
        # println("   Max Allowed: $max_allowed_cost")
        # println("   Actual Cost: $total_cost")
        # println("   Difference:  $(total_cost - max_allowed_cost)")
        out = false
        if total_cost - max_allowed_cost > max_violation
            max_violation = total_cost - max_allowed_cost
        end
    end

    # 2. Check all physical grid constraints
    for (F, S) in list_of_constraint_types(model)
        for cref in all_constraints(model, F, S)
            c_obj = constraint_object(cref)
            val = value(lookup, c_obj.func)
            set = c_obj.set

            if S <: MOI.LessThan && val > set.upper + cost_tol
                # println("❌ FEASIBILITY FAILED: LessThan Constraint!")
                # println("   Constraint: $cref")
                # println(
                #     "   Value: $val | Upper Bound: $(set.upper) | Viol: $(val - set.upper)",
                # )
                out = false
                if val - set.upper > max_violation
                    max_violation = val - set.upper
                end
            end
            if S <: MOI.GreaterThan && val < set.lower - cost_tol
                # println("❌ FEASIBILITY FAILED: GreaterThan Constraint!")
                # println("   Constraint: $cref")
                # println(
                #     "   Value: $val | Lower Bound: $(set.lower) | Viol: $(set.lower - val)",
                # )
                out = false
                if set.lower - val > max_violation
                    max_violation = set.lower - val
                end
            end
            if S <: MOI.EqualTo && abs(val - set.value) > cost_tol
                # println("❌ FEASIBILITY FAILED: EqualTo Constraint (Energy Balance)!")
                # println("   Constraint: $cref")
                # println(
                #     "   Value: $val | Required: $(set.value) | Diff: $(abs(val - set.value))",
                # )
                out = false
                if abs(val - set.value) > max_violation
                    max_violation = abs(val - set.value)
                end
            end
            if S <: MOI.Interval &&
               (val < set.lower - cost_tol || val > set.upper + cost_tol)
                # println("❌ FEASIBILITY FAILED: Interval Constraint!")
                out = false
                if abs(val - set.lower) > max_violation
                    max_violation = abs(val - set.lower)
                end
            end
        end
    end

    # 3. Check variable bounds (non-negativity)
    for v in all_variables(model)
        val = lookup(v)
        if has_lower_bound(v) && val < lower_bound(v) - cost_tol
            # println("❌ FEASIBILITY FAILED: Variable Lower Bound!")
            # println("   Variable: $(name(v))")
            # println("   Value: $val | Lower Bound: $(lower_bound(v))")
            out = false
            if val - lower_bound(v) > max_violation
                max_violation = val - lower_bound(v)
            end
        end
        if has_upper_bound(v) && val > upper_bound(v) + cost_tol
            # println("❌ FEASIBILITY FAILED: Variable Upper Bound!")
            # println("   Variable: $(name(v))")
            # println("   Value: $val | Upper Bound: $(upper_bound(v))")
            out = false
            if upper_bound(v) - val > max_violation
                max_violation = upper_bound(v) - val
            end
        end
    end

    println(
        "✅ Feasibility Check Completed. Result: $(out ? "FEASIBLE" : "INFEASIBLE") | Max Violation: $max_violation",
    )
    return out, max_violation
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
t0 = time()
optimize!(base_model)
println("Base model solved in $(round(time() - t0, digits=4)) seconds.")
optimal_cost = objective_value(base_model)
slack_percentage = 0.05
max_allowed_cost = optimal_cost * (1 + slack_percentage)
println("Optimal System Cost: ", optimal_cost)
println("Max Allowed Cost (MGA Slack): ", max_allowed_cost)

all_vars = all_variables(base_model)
investment_vars = base_model[:investment]

model_size = length(all_vars)
println("Model has $model_size decision variables.")
# sleep(1) # Just to ensure the print statements are readable before we start the iterations

# --- Statistical Testing Setup ---
n_iterations = 1000 # Adjust to 1000 for final run
println("\nStarting $n_iterations statistical runs...")

# Data collectors
spores_times = Float64[];
lbfgs_times = Float64[];
hsj_times = Float64[];
spores_feasibility_rates = Int[];
lbfgs_feasibility_rates = Int[];
hsj_feasibility_rates = Int[];
spores_dominates_count = Int[];
lbfgs_dominates_count = Int[];
hsj_dominates_count = Int[];
lbfgs_repair_times = Float64[];
lbfgs_repair_feasibility_rates = Float64[];
lbfgs_repair_dominates_count = Int[];

lbfgs_max_distances = Float64[];
lbfgs_max_percentages = Float64[];

lbfgs_max_violations = Float64[];

for iter = 1:n_iterations
    if iter % 10 == 0
        println("  Completed $iter / $n_iterations runs...")
    end

    ## -- 1. Run SPORES --
    copy_model = copy(base_model)
    set_optimizer(copy_model, Gurobi.Optimizer)
    set_silent(copy_model)
    optimize!(copy_model)

    t0 = time()
    spores_output = NearOptimalAlternatives.generate_alternatives_optimization!(
        copy_model,
        slack_percentage,
        all_variables(copy_model),
        5;
        modeling_method = :Spores,
    )
    push!(spores_times, time() - t0)
    spores_alts = spores_output.solutions

    ## -- 2. Run LBFGS --
    copy_model = copy(base_model)
    set_optimizer(copy_model, Gurobi.Optimizer)
    set_silent(copy_model)
    optimize!(copy_model)

    t0 = time()
    lbfgs_alts = lbfgs_search_alternatives(
        copy_model,
        all_variables(copy_model),
        5,
        slack_percentage,
        max_allowed_cost,
        false,
        Gurobi.Optimizer,
    )
    push!(lbfgs_times, time() - t0)

    ## -- 2.5. Run LBFGS with repair --
    copy_model = copy(base_model)
    set_optimizer(copy_model, Gurobi.Optimizer)
    set_silent(copy_model)
    optimize!(copy_model)

    t0 = time()
    lbfgs_repair_alts, max_distance, max_percentage = lbfgs_search_alternatives(
        copy_model,
        all_variables(copy_model),
        5,
        slack_percentage,
        max_allowed_cost,
        true,
        Gurobi.Optimizer,
    )
    push!(lbfgs_repair_times, time() - t0)

    for dist in max_distance
        push!(lbfgs_max_distances, dist)
    end
    for perc in max_percentage
        push!(lbfgs_max_percentages, perc)
    end

    ## -- 3. Run Directionally Weighted Variable Search --
    copy_model = copy(base_model)
    set_optimizer(copy_model, Gurobi.Optimizer)
    set_silent(copy_model)
    optimize!(copy_model)

    t0 = time()
    hsj_output = NearOptimalAlternatives.generate_alternatives_optimization!(
        copy_model,
        slack_percentage,
        all_variables(copy_model),
        5;
        modeling_method = :HSJ,
    )
    push!(hsj_times, time() - t0)
    hsj_alts = hsj_output.solutions

    @info "Run $iter: SPORES Time = $(round(spores_times[end], digits=2))s, LBFGS Time = $(round(lbfgs_times[end], digits=2))s, HSJ Time = $(round(hsj_times[end], digits=2))s"
    # sleep(1) # Just to ensure the print statements are readable before we start processing the alternatives

    ## -- 4. Extract, Format, and Check Feasibility --
    function process_alternatives(alts_raw)
        caps = Float64[]
        dict_list = []
        for alt_data in alts_raw
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
            push!(dict_list, val_map)
        end
        return dict_list
    end

    spores_dicts = process_alternatives(spores_alts)
    lbfgs_dicts = process_alternatives(lbfgs_alts)
    hsj_dicts = process_alternatives(hsj_alts)
    lbfgs_repair_dicts = process_alternatives(lbfgs_repair_alts)
    # Check feasibility for all 5 generated alternatives
    s_feas_mask = [is_feasible(base_model, d, max_allowed_cost)[1] for d in spores_dicts]
    l_feas_mask = [is_feasible(base_model, d, max_allowed_cost)[1] for d in lbfgs_dicts]
    hsj_feas_mask = [is_feasible(base_model, d, max_allowed_cost)[1] for d in hsj_dicts]
    l_repair_feas_mask =
        [is_feasible(base_model, d, max_allowed_cost)[1] for d in lbfgs_repair_dicts]

    l_max_violations =
        [is_feasible(base_model, d, max_allowed_cost)[2] for d in lbfgs_dicts]

    push!(spores_feasibility_rates, sum(s_feas_mask))
    push!(lbfgs_feasibility_rates, sum(l_feas_mask))
    push!(hsj_feasibility_rates, sum(hsj_feas_mask))
    push!(lbfgs_repair_feasibility_rates, sum(l_repair_feas_mask))
    push!(lbfgs_max_violations, maximum(l_max_violations))


    ## -- 5. Check Dominance (ONLY on feasible solutions!) --
    # Extract just the capacity vectors for the feasible ones
    get_caps(d) = [get(d, name(var), 0.0) for (index, var) in investment_vars.data]

    spores_feasible_caps =
        [get_caps(spores_dicts[i]) for i = 1:length(spores_dicts) if s_feas_mask[i]]
    lbfgs_feasible_caps =
        [get_caps(lbfgs_dicts[i]) for i = 1:length(lbfgs_dicts) if l_feas_mask[i]]
    hsj_feasible_caps =
        [get_caps(hsj_dicts[i]) for i = 1:length(hsj_dicts) if hsj_feas_mask[i]]
    lbfgs_repair_feasible_caps = [
        get_caps(lbfgs_repair_dicts[i]) for
        i = 1:length(lbfgs_repair_dicts) if l_repair_feas_mask[i]
    ]

    s_dom_l = 0
    l_dom_s = 0
    s_dom_hsj = 0
    l_dom_hsj = 0
    hsj_dom_s = 0
    hsj_dom_l = 0
    tol_dom = 1e-5

    for s_cap in spores_feasible_caps
        for l_cap in lbfgs_feasible_caps
            if all(s_cap .<= l_cap .+ tol_dom) && any(s_cap .< l_cap .- tol_dom)
                s_dom_l += 1
            end
            if all(l_cap .<= s_cap .+ tol_dom) && any(l_cap .< s_cap .- tol_dom)
                l_dom_s += 1
            end
        end
    end

    for s_cap in spores_feasible_caps
        for hsj_cap in hsj_feasible_caps
            if all(s_cap .<= hsj_cap .+ tol_dom) && any(s_cap .< hsj_cap .- tol_dom)
                s_dom_hsj += 1
            end
            if all(hsj_cap .<= s_cap .+ tol_dom) && any(hsj_cap .< s_cap .- tol_dom)
                hsj_dom_s += 1
            end
        end
    end

    for l_cap in lbfgs_feasible_caps
        for hsj_cap in hsj_feasible_caps
            if all(l_cap .<= hsj_cap .+ tol_dom) && any(l_cap .< hsj_cap .- tol_dom)
                l_dom_hsj += 1
            end
            if all(hsj_cap .<= l_cap .+ tol_dom) && any(hsj_cap .< l_cap .- tol_dom)
                hsj_dom_l += 1
            end
        end
    end

    push!(spores_dominates_count, s_dom_l, s_dom_hsj)
    push!(lbfgs_dominates_count, l_dom_s, l_dom_hsj)
    push!(hsj_dominates_count, hsj_dom_s, hsj_dom_l)
end

println("\n--- Testing Complete! Generating Plots... ---")

# --- Plotting the Results ---
# 1. Plot Runtimes
p1 = bar(
    ["HSJ", "SPORES", "LBFGS", "LBFGS with Repair"],
    [mean(hsj_times), mean(spores_times), mean(lbfgs_times), mean(lbfgs_repair_times)],
    yerror = [std(hsj_times), std(spores_times), std(lbfgs_times), std(lbfgs_repair_times)],
    title = "Average Run Time (s)",
    ylabel = "Time (Seconds)",
    color = [:blue, :orange, :green, :red],
    legend = false,
)

# 2. Plot Feasibility (Out of 5 alternatives per run)
# p2 = bar(
#     ["HSJ", "SPORES", "LBFGS", "LBFGS with Repair"],
#     [
#         mean(spores_feasibility_rates),
#         mean(lbfgs_feasibility_rates),
#         mean(hsj_feasibility_rates),
#         mean(lbfgs_repair_feasibility_rates),
#     ],
#     yerror = [
#         std(spores_feasibility_rates),
#         std(lbfgs_feasibility_rates),
#         std(hsj_feasibility_rates),
#         std(lbfgs_repair_feasibility_rates),
#     ],
#     title = "Feasible Alternatives\n(Max 5 per run)",
#     ylabel = "Count",
#     color = [:green, :purple, :brown, :pink],
#     legend = false,
#     ylim = (0, 5.5),
# )

# 3. Plot Dominance
# p3 = bar(["SPORES dom. LBFGS", "LBFGS dom. SPORES", "HSJ dom. SPORES"], [mean(spores_dominates_count), mean(lbfgs_dominates_count), mean(hsj_dominates_count)],
#     yerror=[std(spores_dominates_count), std(lbfgs_dominates_count), std(hsj_dominates_count)],
#     title="Average Dominance\nEvents Per Run", ylabel="# of dominations",
#     color=[:teal, :red, :orange], legend=false)

# Plot distribution of LBFGS max distances
p3 = histogram(
    lbfgs_max_violations,
    title = "LBFGS Max Constraint Violations",
    xlabel = "Max Violation",
    ylabel = "Frequency",
    legend = false,
    color = :cyan,
)

p4 = histogram(
    lbfgs_max_distances,
    title = "LBFGS Max Distances from Solution to Feasible Solution",
    xlabel = "Max Distance",
    ylabel = "Frequency",
    legend = false,
    color = :magenta,
)

# Combine and display the plots in a 1x3 grid
times = plot(p1, size = (1000, 400), margin = 5Plots.mm)
display(times)
err_plot = plot(p3, size = (1000, 400), margin = 5Plots.mm)
display(err_plot)
dist_plot = plot(p4, size = (1000, 400), margin = 5Plots.mm)
display(dist_plot)

# Print Summary Statistics to the console
println("\n--- Summary Statistics ($n_iterations runs) ---")
println(
    "SPORES | Avg Time: $(round(mean(spores_times), digits=2))s ± $(round(std(spores_times), digits=2)) | Feasible Alts: $(mean(spores_feasibility_rates))/5 | Dominated $(sum(spores_dominates_count)) times",
)
println(
    "LBFGS  | Avg Time: $(round(mean(lbfgs_times), digits=2))s ± $(round(std(lbfgs_times), digits=2)) | Feasible Alts: $(mean(lbfgs_feasibility_rates))/5 | Dominated $(sum(lbfgs_dominates_count)) times",
)
println(
    "HSJ  | Avg Time: $(round(mean(hsj_times), digits=2))s ± $(round(std(hsj_times), digits=2)) | Feasible Alts: $(mean(hsj_feasibility_rates))/5 | Dominated $(sum(hsj_dominates_count)) times",
)
