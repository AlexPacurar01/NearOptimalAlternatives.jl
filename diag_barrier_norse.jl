# ===========================================================================
# Diagnose the HiGHS barrier OTHER_ERROR on specific Norse MGA budgets.
# Reproduces one failing sub-problem (N=5, k=2 grid point) with verbose output,
# then tries candidate fixes so the arms baseline can be made robust:
#   v1  ipm + crossover (the failing configuration, verbose log)
#   v2  ipm, more iterations + tighter primal feasibility handling
#   v3  ipm with presolve off
#   v4  default HiGHS (auto choose)
#   v5  dual simplex
#   Usage: julia -t 4 --project=. diag_barrier_norse.jl
# ===========================================================================
using JuMP, HiGHS, LinearAlgebra, Printf
include("smallcase_tulipa.jl")

model, target = load_tulipa_case(
    "Norse",
    optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false),
)
min_cost = objective_value(model)
all_vars = all_variables(model)
idx = Dict(v => i for (i, v) in enumerate(all_vars))
xstar = value.(all_vars)
w = zeros(length(all_vars))
for v in target
    ub = has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0
    w[idx[v]] = xstar[idx[v]] / ub
end

# The failing budget from the arms run: N=5 grid, k=2.
B = min_cost + 0.1 * abs(min_cost) * 2 / 5
orig = objective_function(model)
off = JuMP.constant(orig)
bcon = @constraint(model, orig <= B)
@objective(model, Min, sum(w[i] * all_vars[i] for i in eachindex(w) if w[i] != 0))

variants = [
    (
        "v1 ipm+crossover (failing config, verbose)",
        ["solver" => "ipm", "run_crossover" => "on", "output_flag" => true],
    ),
    (
        "v2 ipm, 300 iterations",
        [
            "solver" => "ipm",
            "run_crossover" => "on",
            "ipm_iteration_limit" => 300,
            "output_flag" => false,
        ],
    ),
    (
        "v3 ipm, presolve off",
        [
            "solver" => "ipm",
            "run_crossover" => "on",
            "presolve" => "off",
            "output_flag" => false,
        ],
    ),
    ("v4 HiGHS default (choose)", ["output_flag" => false]),
    (
        "v5 dual simplex",
        ["solver" => "simplex", "simplex_strategy" => 1, "output_flag" => false],
    ),
]

for (name, opts) in variants
    set_optimizer(model, optimizer_with_attributes(HiGHS.Optimizer, opts...))
    t = @elapsed optimize!(model)
    st = termination_status(model)
    raw = raw_status(model)
    obj = st == MOI.OPTIMAL ? objective_value(model) : NaN
    @printf("\n>>> %-42s %-18s %6.2fs  obj=%.10g\n    raw: %s\n", name, st, t, obj, raw)
end
