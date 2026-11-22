@testset "Update solutions of NearOptimalAlternatives optimization" begin
    @testset "Make sure using an unsolved NearOptimalAlternatives model fails." begin
        optimizer = Ipopt.Optimizer
        model = JuMP.Model(optimizer)
        results = NearOptimalAlternatives.AlternativeSolutions([], [])
        @test_throws ErrorException NearOptimalAlternatives.update_solutions!(
            results,
            model,
        )
    end

    @testset "Test error when no `original_objective` constraint present." begin
        optimizer = Ipopt.Optimizer
        model = JuMP.Model(optimizer)

        # initialize simple `square` JuMP model
        @variable(model, 0 ≤ x_1 ≤ 1)
        @variable(model, 0 ≤ x_2 ≤ 1)
        @objective(model, Max, (x_1 - 1)^2 + (x_2 - 1)^2)
        @constraint(model, x_1 + x_2 ≥ 1.8)

        results = NearOptimalAlternatives.AlternativeSolutions([], [])
        JuMP.optimize!(model)
        @test_throws ErrorException NearOptimalAlternatives.update_solutions!(
            results,
            model,
        )
    end

    @testset "Test correct flow for updating the set of results using a solved JuMP model." begin
        optimizer = Ipopt.Optimizer
        model = JuMP.Model(optimizer)

        # initialize simple `square` JuMP model
        @variable(model, 0 ≤ x_1 ≤ 1)
        @variable(model, 0 ≤ x_2 ≤ 1)
        @objective(model, Max, (x_1 - 1)^2 + (x_2 - 1)^2)
        @constraint(model, original_objective, x_1 + x_2 ≥ 1.8)

        results = NearOptimalAlternatives.AlternativeSolutions([], [])
        JuMP.optimize!(model)
        NearOptimalAlternatives.update_solutions!(results, model)

        # Test that only one solution is generated for 2 variables. Result should be approximately between 1.8 and 2.0 (with room for computational error).
        @test length(results.solutions) == 1 &&
              length(results.solutions[1]) == 2 &&
              length(results.objective_values) == 1 &&
              (
                  results.objective_values[1] ≥ 1.8 ||
                  isapprox(results.objective_values[1], 1.8)
              ) &&
              (
                  results.objective_values[1] ≤ 2.0 ||
                  isapprox(results.objective_values[1], 2.0)
              )
    end
    @testset "Reported cost includes a constant term of the original objective." begin
        model = JuMP.Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, 0 ≤ x_1 ≤ 10)
        @variable(model, 0 ≤ x_2 ≤ 10)
        # The constant 100 is what this test is about. When the objective becomes the
        # budget constraint, JuMP moves the constant to the right-hand side, so the
        # constraint's value is 100 below the true cost. The reported cost must be
        # computed from the full objective instead (see `original_objective_value`).
        @objective(model, Min, 2x_1 + 3x_2 + 100)
        @constraint(model, x_1 + x_2 ≥ 5)
        JuMP.optimize!(model)
        @test isapprox(objective_value(model), 110; atol = 1e-6)

        results = NearOptimalAlternatives.generate_alternatives_optimization!(
            model,
            0.1,
            all_variables(model),
            1;
            modeling_method = :HSJ,
        )
        sol = results.solutions[1]
        cost = 2 * sol[x_1] + 3 * sol[x_2] + 100
        @test isapprox(results.objective_values[1], cost; atol = 1e-5)
        @test results.objective_values[1] ≤ 1.1 * 110 + 1e-5
    end
end
