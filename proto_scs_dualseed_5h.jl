# Primal-DUAL warm start: Gurobi solves the MGA subproblem (min w'x s.t. cost<=B1)
# exactly, giving VALID primal+dual. SCS is then seeded with that (x,y) and asked to
# (a) re-certify the SAME B1 (does the stall vanish with correct duals?), then
# (b) solve the adjacent B2 warm from B1. Tests whether the dual cold-start was the
# bottleneck. Explicit flush.
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
JuMP.optimize!(model);
@assert is_solved_and_feasible(model);
min_cost = objective_value(model)
allv = all_variables(model);
idx = Dict(v => i for (i, v) in enumerate(allv));
println("built+solved 5h | min_cost=$(round(min_cost,sigdigits=6))");
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
kconst = JuMP.constant(cost_expr);
B1 = min_cost * 1.05;
B2 = min_cost * 1.10;
@objective(model, Min, sum(w[i] * allv[i] for i in eachindex(w) if w[i] != 0))
bud = @constraint(model, cost_expr <= B1)

# 1) GUROBI solves the w-subproblem at B1 (valid primal+dual).
tg = @elapsed JuMP.optimize!(model)
@printf(
    "Gurobi  B1: status=%s  time=%.1fs  dobj=%.3f\n",
    termination_status(model),
    tg,
    dot(what, value.(allv))
);
flush(stdout);
xg = value.(allv)
# capture duals of ALL constraints (the valid shadow prices for the w-problem at B1)
cons = ConstraintRef[];
duals = Float64[];
for (F, S) in list_of_constraint_types(model), con in all_constraints(model, F, S)
    push!(cons, con)
    push!(duals, dual(con))
end
println("captured $(length(cons)) constraint duals");
flush(stdout);

scs = optimizer_with_attributes(
    SCS.Optimizer,
    "linear_solver" => SCS.IndirectSolver,
    "verbose" => 1,
    "eps_abs" => 1e-4,
    "eps_rel" => 1e-4,
    "max_iters" => 5000,
)
set_optimizer(model, scs)   # attach SCS BEFORE setting dual starts (Gurobi rejects ConstraintDualStart)

function scs_warm(B, xseed, conseed, dualseed; label)
    for (i, v) in enumerate(allv)
        set_start_value(v, xseed[i])
    end
    for (c, d) in zip(conseed, dualseed)
        set_dual_start_value(c, d)
    end
    set_normalized_rhs(bud, B - kconst)
    t = @elapsed JuMP.optimize!(model)
    d = dot(what, value.(allv))
    @printf(
        ">>> SCS %s: status=%s  time=%.1fs  dobj=%.3f\n",
        label,
        termination_status(model),
        t,
        d
    )
    flush(stdout)
    return value.(allv), d, t
end

# 2) SCS re-certifies the SAME B1 from Gurobi's exact (x,y).
x1, d1, t1 = scs_warm(B1, xg, cons, duals; label = "B1 (warm from Gurobi exact x,y)")
# 3) SCS solves adjacent B2 warm from B1's Gurobi seed (primal x1, same duals as a proxy).
x2, d2, t2 = scs_warm(B2, x1, cons, duals; label = "B2 (warm from B1)")

# ground truth
set_optimizer(model, make_optimizer(:gurobi, THREADS));
set_silent(model);
JuMP.optimize!(model)  # cost again, ignore
delete(model, bud);
set_objective_function(model, cost_expr);
bi = ipm_baseline_at_budgets(
    model,
    w,
    allv,
    [B1, B2];
    optimizer_factory = make_optimizer(:gurobi, THREADS),
)
@printf("\nIPM truth: B1 dobj=%.3f   B2 dobj=%.3f\n", bi.dobjs[1], bi.dobjs[2])
println("DONE");
flush(stdout);
