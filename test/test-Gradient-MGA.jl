using Test
using JuMP
using NearOptimalAlternatives
using Ipopt

# All four gradient methods are exercised through the public
# `generate_alternatives_gradient` entry point with operational recovery, so
# every returned alternative must be exactly feasible and within budget.
const GRADIENT_METHODS = [:alm_lbfgs, :penalty, :pdhg, :oracle]

silent_ipopt() = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

@testset "Gradient MGA: $method" for method in GRADIENT_METHODS
    # ----------------------------------------------------------------
    # Tiny LP with a unique optimum and room for an alternative.
    #   min  x1 + 1.05 x2   s.t.  x1 + x2 >= 2,  0 <= x1, x2 <= 3
    # Optimal (2, 0), cost 2.0. The HSJ penalty falls on x1 (the only
    # nonzero at the optimum), so a near-optimal alternative trades x1 for
    # x2 while staying within the 10% cost budget (<= 2.2).
    # ----------------------------------------------------------------
    model = JuMP.Model(silent_ipopt())
    @variable(model, 0 <= x_1 <= 3)
    @variable(model, 0 <= x_2 <= 3)
    @constraint(model, x_1 + x_2 >= 2)
    @objective(model, Min, x_1 + 1.05 * x_2)
    JuMP.optimize!(model)

    result = generate_alternatives_gradient(
        model,
        0.1,
        [x_1, x_2],
        1;
        method = method,
        operational_recovery = true,
    )

    @test length(result.solutions) == 1
    sol = result.solutions[1]
    a1, a2 = sol[x_1], sol[x_2]

    # Exactly feasible (recovery projects onto the model polytope) and within
    # the near-optimal cost budget.
    @test a1 + a2 >= 2 - 1e-6
    @test all((0 - 1e-6) .<= (a1, a2) .<= (3 + 1e-6))
    @test a1 + 1.05 * a2 <= 2.2 + 1e-6

    # The accurate solvers push the penalised variable x1 down (genuine
    # diversity); PDHG has a first-order accuracy tail, so only require
    # budget-feasibility for it.
    if method != :pdhg
        @test a1 <= a2
    end
end

@testset "Gradient MGA with equality: $method" for method in GRADIENT_METHODS
    # ----------------------------------------------------------------
    #   min  x1 + x2   s.t.  x1 + x2 + x3 == 4,  1 <= x1, x2, x3 <= 3
    # Optimal (1, 1, 2), cost 2.0. Exercises the equality-constraint path.
    # ----------------------------------------------------------------
    model = JuMP.Model(silent_ipopt())
    @variable(model, 1 <= x_1 <= 3)
    @variable(model, 1 <= x_2 <= 3)
    @variable(model, 1 <= x_3 <= 3)
    @constraint(model, x_1 + x_2 + x_3 == 4)
    @objective(model, Min, x_1 + x_2)
    JuMP.optimize!(model)

    result = generate_alternatives_gradient(
        model,
        0.1,
        [x_1, x_2, x_3],
        1;
        method = method,
        operational_recovery = true,
    )

    @test length(result.solutions) == 1
    sol = result.solutions[1]
    a1, a2, a3 = sol[x_1], sol[x_2], sol[x_3]

    @test isapprox(a1 + a2 + a3, 4.0; atol = 1e-5)
    @test all((1 - 1e-6) .<= (a1, a2, a3) .<= (3 + 1e-6))
    @test a1 + a2 <= 2.2 + 1e-5
end
