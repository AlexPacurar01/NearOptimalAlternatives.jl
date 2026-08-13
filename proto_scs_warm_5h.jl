# Warm-started SCS at 5h: budget 1 warm-started from x* (the cost optimum), budget
# 2 warm-started from budget 1. Does the warm primal collapse the solve cost vs the
# ~25 min cold solve, and does it land on the true front? Explicit flush so results
# are visible despite Julia's file buffering.
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
allv = all_variables(model);
idx = Dict(v => i for (i, v) in enumerate(allv));
xstar = value.(allv)
println("built+solved 5h in $(round(t,digits=1))s | $(length(allv)) vars");
flush(stdout);
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
what = w ./ norm(w)

cost_expr = objective_function(model);
orig_sense = objective_sense(model);
kconst = JuMP.constant(cost_expr)
budgets = [min_cost * 1.05, min_cost * 1.10]

set_optimizer(
    model,
    optimizer_with_attributes(
        SCS.Optimizer,
        "linear_solver" => SCS.IndirectSolver,
        "verbose" => 1,
        "eps_abs" => 1e-4,
        "eps_rel" => 1e-4,
        "max_iters" => 20000,
    ),
)
@objective(model, Min, sum(w[i] * allv[i] for i in eachindex(w) if w[i] != 0))
bud = @constraint(model, cost_expr <= budgets[1])

println("x* dobj (reference, max diversity end): $(round(dot(what,xstar),digits=3))");
flush(stdout);
prev = xstar;
scs_d = Float64[];
scs_t = Float64[];
for (k, B) in enumerate(budgets)
    for i in eachindex(allv)
        set_start_value(allv[i], prev[i])
    end
    set_normalized_rhs(bud, B - kconst)
    tt = @elapsed JuMP.optimize!(model)
    xv = value.(allv)
    d = dot(what, xv)
    println(
        ">>> budget $k: B=$(round(B,sigdigits=8)) warm-from=$(k==1 ? "x*" : "prev")  status=$(termination_status(model))  time=$(round(tt,digits=1))s  dobj=$(round(d,digits=3))",
    )
    flush(stdout)
    push!(scs_d, d)
    push!(scs_t, tt)
    global prev = xv
end

delete(model, bud);
set_objective_function(model, cost_expr);
set_objective_sense(model, orig_sense);
println("\nGurobi-barrier IPM ground truth ...");
flush(stdout);
bi = ipm_baseline_at_budgets(
    model,
    w,
    allv,
    budgets;
    optimizer_factory = make_optimizer(:gurobi, THREADS),
)
println("\n========  warm-SCS vs IPM on data/5h  ========")
for (k, B) in enumerate(budgets)
    @printf(
        ">>> B=%.6g  SCS dobj=%.3f (%.1fs)   IPM dobj=%.3f (%.1fs)   |Δ|=%.2e\n",
        B,
        scs_d[k],
        scs_t[k],
        bi.dobjs[k],
        bi.times[k],
        abs(scs_d[k] - bi.dobjs[k])
    )
end
println("DONE");
flush(stdout);
