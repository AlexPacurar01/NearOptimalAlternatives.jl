```@contents
Pages = ["10-how-to-use.md"]
Depth = 5
```

# How to use

This section covers installation and gives a quick-start example for each way of generating alternatives. For the full argument list of every function, see the [IO Reference](15-io.md); for the theory behind each modeling method and generation strategy, see [Concepts](30-concepts.md).

## Install

In Julia:

- Enter package mode (press "]")

```pkg
pkg> add NearOptimalAlternatives
```

- Return to Julia mode (backspace)

```julia
julia> using NearOptimalAlternatives
```

## Generating alternatives

Given a solved JuMP model `model` and the variables you want to consider, choose one of the four functions below depending on how you want alternatives to be generated. All of them return an `AlternativeSolutions`; see [Output](@ref io-output) for its structure.

### A minimal worked example

Before looking at each function, here is a complete, tiny example you can run and check by hand: a one-constraint "energy system" that meets a fixed demand of `10` units from a cheap and an expensive generator.

```@example basic
using JuMP, Ipopt

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, 0 <= x_cheap <= 15)
@variable(model, 0 <= x_expensive <= 15)
@constraint(model, demand, x_cheap + x_expensive == 10)
@objective(model, Min, 1 * x_cheap + 3 * x_expensive)
JuMP.optimize!(model)
clean(x) = round(max(x, 0.0); digits = 3)   # clamp away solver floating-point noise near the 0 lower bound
clean(value(x_cheap)), clean(value(x_expensive)), round(objective_value(model); digits = 3)
```

Since `x_expensive` costs three times as much per unit as `x_cheap`, the cost-minimal solution uses only the cheap generator: `x_cheap = 10`, `x_expensive = 0`, at a cost of `10`. This is the single point a normal optimization run reports, and it is exactly why MGA exists: nothing here tells you whether this all-cheap solution is fragile (relying entirely on one technology) or whether a nearly-as-good alternative uses the expensive generator too.

Let's ask for one alternative within `20%` of the optimal cost:

```@example basic
using NearOptimalAlternatives

optimality_gap = 0.2
variables = [x_cheap, x_expensive]
alternatives = generate_alternatives_optimization!(model, optimality_gap, variables, 1)
round(alternatives.solutions[1][x_cheap]; digits = 3),
round(alternatives.solutions[1][x_expensive]; digits = 3),
round(alternatives.objective_values[1]; digits = 3)
```

You can check this by hand: the budget constraint is `x_cheap + 3*x_expensive <= 12` (`20%` above the optimal cost of `10`), and the default `:Max_Distance` method pushes `x_expensive` as far from its optimal value of `0` as the budget allows. Since demand must still be met exactly, `x_cheap = 10 - x_expensive`; substituting into the budget gives `x_expensive <= 1`, so the alternative is `x_cheap = 9`, `x_expensive = 1`, at a cost of exactly `12` — the corner of the near-optimal region furthest from the original solution.

The rest of this section reuses this same generator setup (rebuilt fresh each time, since a model can only be turned into an alternative-generating problem once).

### One alternative per direction: `generate_alternatives_optimization!`

The classic MGA loop: find `n_alternatives` solutions, one per direction, each at the full near-optimal budget. Demonstrated above; to only change a subset of variables, fix the rest with `fixed_variables`:

```julia
fixed_variables = [x_expensive]   # x_expensive keeps its optimal value of 0.
alternatives = generate_alternatives_optimization!(
    model, optimality_gap, variables, n_alternatives; fixed_variables = fixed_variables,
)
```

### A dense front per direction: `generate_alternatives_sweep!`

Instead of one point per direction, sweep the cost budget to return `n_budget` points per direction, tracing a near-optimal front:

```@example basic
model_sweep = Model(Ipopt.Optimizer)
set_silent(model_sweep)
@variable(model_sweep, 0 <= x_cheap <= 15)
@variable(model_sweep, 0 <= x_expensive <= 15)
@constraint(model_sweep, demand, x_cheap + x_expensive == 10)
@objective(model_sweep, Min, 1 * x_cheap + 3 * x_expensive)
JuMP.optimize!(model_sweep)

front = generate_alternatives_sweep!(model_sweep, optimality_gap, [x_cheap, x_expensive], 1; n_budget = 3)
[(round(front.solutions[i][x_cheap]; digits = 3), round(front.objective_values[i]; digits = 3)) for i in eachindex(front.solutions)]
```

Since there are only two variables tied together by one equality constraint, the whole near-optimal region is the single line segment from `(10, 0)` to `(9, 1)`, and the sweep places `n_budget` points evenly along the cost axis of that segment — you can check each entry above lies at an even fraction of the way from cost `10` to cost `12`.

### An arclength-spaced front per direction: `generate_alternatives_arclength!`

Like the sweep, but spaces the `n_budget` points evenly along the trade-off curve rather than the budget axis:

```@example basic
model_arc = Model(Ipopt.Optimizer)
set_silent(model_arc)
@variable(model_arc, 0 <= x_cheap <= 15)
@variable(model_arc, 0 <= x_expensive <= 15)
@constraint(model_arc, demand, x_cheap + x_expensive == 10)
@objective(model_arc, Min, 1 * x_cheap + 3 * x_expensive)
JuMP.optimize!(model_arc)

front_arc = generate_alternatives_arclength!(model_arc, optimality_gap, [x_cheap, x_expensive], 1; n_budget = 3)
[(round(front_arc.solutions[i][x_cheap]; digits = 3), round(front_arc.objective_values[i]; digits = 3)) for i in eachindex(front_arc.solutions)]
```

On this two-variable example the trade-off curve is a straight line, so the arclength and budget spacings coincide; the difference between the two strategies only shows up once the near-optimal front actually curves (see [Arclength Continuation](@ref arclength-continuation) for why that matters).

### Using a metaheuristic algorithm: `generate_alternatives_metaheuristics`

Generate alternatives with an algorithm from [Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl) instead of mathematical optimization:

```@example basic
using Metaheuristics

model_meta = Model(Ipopt.Optimizer)
set_silent(model_meta)
@variable(model_meta, 0 <= x_cheap <= 15)
@variable(model_meta, 0 <= x_expensive <= 15)
@constraint(model_meta, demand, x_cheap + x_expensive == 10)
@objective(model_meta, Min, 1 * x_cheap + 3 * x_expensive)
JuMP.optimize!(model_meta)

metaheuristic_algorithm = Metaheuristics.PSO()
meta_alt = generate_alternatives_metaheuristics(model_meta, optimality_gap, 1, metaheuristic_algorithm)
round(meta_alt.solutions[1][x_cheap]; digits = 3), round(meta_alt.solutions[1][x_expensive]; digits = 3)
```

Metaheuristic results are approximate and stochastic, unlike the optimization-based functions above — on a problem this small and dominated by a single equality constraint, a metaheuristic may even settle back on the exact optimum rather than a diverse alternative; they are much more useful on larger, less tightly-constrained problems.

As with the optimization-based functions, `fixed_variables` can be supplied, and the distance `metric` can be changed (weighted metrics are supported too):

```julia
using Distances

metric = Distances.Euclidean()   # Use Euclidean instead of the default SqEuclidean.
alternatives = generate_alternatives_metaheuristics(
    model, optimality_gap, n_alternatives, metaheuristic_algorithm; metric = metric,
)
```

The parameters of `metaheuristic_algorithm` are set when constructing it; see the [Metaheuristics.jl documentation](https://jmejia8.github.io/Metaheuristics.jl/stable/) for the algorithms it provides.

#### Using PSOGA, this package's own metaheuristic

`PSOGA` is used the same way as any other metaheuristic, except it needs the number of alternatives up front, so it knows how many subpopulations to keep:

```julia
metaheuristic_algorithm = PSOGA(N_solutions = n_alternatives)
alternatives = generate_alternatives_metaheuristics(model, optimality_gap, n_alternatives, metaheuristic_algorithm)
```
