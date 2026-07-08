using Test
using JuMP
using NearOptimalAlternatives
using Ipopt

silent_ipopt_arc() = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

# Same small capacity-expansion LP as the sweep test (bounded investment so the
# SPORES direction is defined), rebuilt here so this file is self-contained.
function build_synth_arc()
    cap = [10.0, 8.0, 6.0]
    c_inv = [3.0, 5.0, 7.0]
    c_op = [1.0 2.0 3.0 4.0; 2.5 1.5 3.5 2.0; 4.0 3.0 1.0 2.5]
    demand = [5.0, 7.0, 4.0, 6.0]
    n, T = length(cap), length(demand)
    model = JuMP.Model(silent_ipopt_arc())
    @variable(model, 0 <= inv[i = 1:n] <= cap[i])
    @variable(model, disp[i = 1:n, t = 1:T] >= 0)
    @constraint(model, [i = 1:n, t = 1:T], disp[i, t] <= inv[i])
    @constraint(model, [t = 1:T], sum(disp[i, t] for i = 1:n) >= demand[t])
    @objective(
        model,
        Min,
        sum(c_inv[i] * inv[i] for i = 1:n) +
        sum(c_op[i, t] * disp[i, t] for i = 1:n, t = 1:T)
    )
    JuMP.optimize!(model)
    return model, inv
end

@testset "Arclength: shape, budget feasibility, endpoints, distribution" begin
    model, inv = build_synth_arc()
    n_dir, n_bud, gap = 2, 5, 0.1

    result = generate_alternatives_arclength!(
        model,
        gap,
        inv,
        n_dir;
        n_budget = n_bud,
        modeling_method = :Spores,
    )

    # The marching predictor-corrector places a variable number of points per
    # direction (adaptive step, capped at n_budget), so check bounds not equality.
    @test length(result.tags) == length(result.solutions)
    @test length(result.solutions) <= n_dir * n_bud
    @test length(result.solutions) >= n_dir * 3     # 2 endpoints + >=1 interior

    for k = 1:n_dir
        tags_k = [t for t in result.tags if t.direction == k]
        @test 3 <= length(tags_k) <= n_bud
        gaps = sort!([t.gap for t in tags_k])
        # The two endpoints are always solved: [gap/n_budget, gap].
        @test isapprox(gaps[1], gap / n_bud; atol = 1e-9)
        @test isapprox(gaps[end], gap; atol = 1e-9)
        # All budgets strictly inside the near-optimal region.
        @test all(0 .< gaps .<= gap + 1e-12)

        # Sorted by budget, the diversity objective is non-increasing (a larger
        # budget can only lower min wᵀx).
        order = sortperm([t.gap for t in tags_k])
        divs = [t.diversity_objective for t in tags_k][order]
        @test all(divs[i] >= divs[i+1] - 1e-6 for i = 1:length(divs)-1)
    end

    # Every recorded cost is within its budget level.
    for i in eachindex(result.objective_values)
        @test result.objective_values[i] <= result.tags[i].budget_level + 1e-5
    end
end
