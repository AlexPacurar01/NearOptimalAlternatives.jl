# Decisive test: does SCS (matrix-free INDIRECT ADMM) reproduce the TRUE 5h MGA
# front (dobjs ~278/231/186) that ALM, PDHG and my hand-rolled OSQP all missed?
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, SCS, LinearAlgebra, Printf
include(joinpath(@__DIR__, "bench_common.jl"))
const THREADS = max(1, Threads.nthreads())

ep = TEM.create_energy_problem_from_csv_folder(joinpath(@__DIR__, "data", "5h"))
TEM.create_model!(ep);
model = ep.model;
set_optimizer(model, make_optimizer(:gurobi, THREADS));
set_silent(model);
t = @elapsed JuMP.optimize!(model);
@assert is_solved_and_feasible(model);
min_cost = objective_value(model)
println("built+solved 5h in $(round(t,digits=1))s | $(num_variables(model)) vars")
allv = all_variables(model);
idx = Dict(v => i for (i, v) in enumerate(allv));
target = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    haskey(object_dictionary(model), sym) && for v in object_dictionary(model)[sym]
        push!(target, v)
    end
end
w = zeros(length(allv))
for v in target
    ub = (has_upper_bound(v) && upper_bound(v) != 0) ? upper_bound(v) : 1.0
    w[idx[v]] = value(v) / ub
end

budgets = [min_cost * (1 + f) for f in (0.04, 0.07, 0.10)]
println("Running SCS indirect at $(length(budgets)) budgets ...")
scs_fac = optimizer_with_attributes(
    SCS.Optimizer,
    "verbose" => 1,
    "linear_solver" => SCS.IndirectSolver,
    "eps_abs" => 1e-4,
    "eps_rel" => 1e-4,
    "max_iters" => 20000,
)
bs = ipm_baseline_at_budgets(model, w, allv, budgets; optimizer_factory = scs_fac)
println("Running Gurobi-barrier IPM ground truth ...")
bi = ipm_baseline_at_budgets(
    model,
    w,
    allv,
    budgets;
    optimizer_factory = make_optimizer(:gurobi, THREADS),
)

println("\n================  SCS (matrix-free) vs IPM on data/5h  ================")
println("budget          SCS dobj     IPM dobj      |Δ|        SCS s    IPM s")
for (i, B) in enumerate(budgets)
    @printf(
        "%.6g   %10.3f   %10.3f   %.2e   %7.1f  %6.1f\n",
        B,
        bs.dobjs[i],
        bi.dobjs[i],
        abs(bs.dobjs[i] - bi.dobjs[i]),
        bs.times[i],
        bi.times[i]
    )
end
@printf(
    "\nmax |SCS - IPM dobj| = %.3e   (x* dobj for reference: %.3f)\n",
    maximum(abs.(bs.dobjs .- bi.dobjs)),
    dot(w ./ norm(w), value.(allv))
)
println("DONE")
