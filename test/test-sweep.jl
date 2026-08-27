using Test
using JuMP
using NearOptimalAlternatives
using Ipopt

silent_ipopt() = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

# Small structured capacity-expansion LP: investment per asset (bounded, so the
# SPORES direction is well-defined) plus per-period dispatch. Deterministic, so
# the same solver returns the same vertices across two builds.
function build_synth()
    cap = [10.0, 8.0, 6.0]
    c_inv = [3.0, 5.0, 7.0]
    c_op = [1.0 2.0 3.0 4.0; 2.5 1.5 3.5 2.0; 4.0 3.0 1.0 2.5]
    demand = [5.0, 7.0, 4.0, 6.0]
    n, T = length(cap), length(demand)
    model = JuMP.Model(silent_ipopt())
    @variable(model, 0 <= inv[i=1:n] <= cap[i])
    @variable(model, disp[i=1:n, t=1:T] >= 0)
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

@testset "Sweep: shape, budget feasibility, monotone levels" begin
    model, inv = build_synth()
    n_dir, n_bud, gap = 3, 4, 0.1

    result = generate_alternatives_sweep!(
        model,
        gap,
        inv,
        n_dir;
        n_budget = n_bud,
        modeling_method = :Spores,
    )

    # One point per (direction, budget level), with matching metadata.
    @test length(result.solutions) == n_dir * n_bud
    @test length(result.objective_values) == n_dir * n_bud
    @test length(result.tags) == n_dir * n_bud

    for i in eachindex(result.tags)
        tag = result.tags[i]
        @test 1 <= tag.direction <= n_dir
        @test 1 <= tag.budget_idx <= n_bud
        # Realised cost stays within the level it was solved at.
        @test result.objective_values[i] <= tag.budget_level + 1e-6
    end

    # Budget levels per direction sweep from the cost-optimal face up to the
    # full budget, strictly increasing.
    for k = 1:n_dir
        levels = [t.budget_level for t in result.tags if t.direction == k]
        @test length(levels) == n_bud
        @test issorted(levels)
        @test levels[end] > levels[1]
    end
end

@testset "Sweep: warm start reaches the same optima (single direction)" begin
    # With a SINGLE direction there is no SPORES weight feedback, so warm and cold
    # solve the identical LP at every budget. Warm starting changes only the
    # initial iterate, so the optimal diversity-objective value at each budget must
    # match the cold run to solver tolerance. (This is the mathematically sound
    # invariant: comparing the optimum of the same problem, not the cost ceiling
    # the diversity objective happens to spend, and not points that can differ on
    # degenerate faces.) We deliberately use n_dir = 1 because with more
    # directions a degenerate vertex would perturb the accumulated weights and the
    # two runs would legitimately follow different directions.
    n_dir, n_bud, gap = 1, 4, 0.1

    cold_model, cold_inv = build_synth()
    cold = generate_alternatives_sweep!(
        cold_model,
        gap,
        cold_inv,
        n_dir;
        n_budget = n_bud,
        modeling_method = :Spores,
        warm_start = false,
    )

    warm_model, warm_inv = build_synth()
    warm = generate_alternatives_sweep!(
        warm_model,
        gap,
        warm_inv,
        n_dir;
        n_budget = n_bud,
        modeling_method = :Spores,
        warm_start = true,
    )

    @test length(warm.solutions) == length(cold.solutions)
    for i in eachindex(cold.tags)
        # Same LP, warm vs cold start -> same optimal diversity objective.
        @test isapprox(
            warm.tags[i].diversity_objective,
            cold.tags[i].diversity_objective;
            atol = 1e-5,
            rtol = 1e-5,
        )
        # And still budget-feasible.
        @test warm.objective_values[i] <= warm.tags[i].budget_level + 1e-5
    end
end

@testset "Sweep: full-budget points reproduce plain SPORES" begin
    n_dir, n_bud, gap = 2, 4, 0.1

    # Sweep generator.
    sweep_model, sweep_inv = build_synth()
    swept = generate_alternatives_sweep!(
        sweep_model,
        gap,
        sweep_inv,
        n_dir;
        n_budget = n_bud,
        modeling_method = :Spores,
    )

    # Classic single-budget SPORES generator on an identical fresh model.
    plain_model, plain_inv = build_synth()
    plain = generate_alternatives_optimization!(
        plain_model,
        gap,
        plain_inv,
        n_dir;
        modeling_method = :Spores,
    )

    # The last (full-budget) point of each swept direction should match the
    # corresponding plain SPORES alternative: same direction sequence, same budget.
    for k = 1:n_dir
        si = findlast(t -> t.direction == k && t.budget_idx == n_bud, swept.tags)
        @test si !== nothing
        for j in eachindex(sweep_inv)
            sv = swept.solutions[si][sweep_inv[j]]
            pv = plain.solutions[k][plain_inv[j]]
            @test isapprox(sv, pv; atol = 1e-4)
        end
    end
end
