# ===========================================================================
# Diagnose why the FOM continuation gets stuck at x* on data/5h.
# Hypothesis: the objective w'x touches only the ~745 investment vars out of
# 1.27M, so its gradient signal is drowned by the feasibility dynamics, AND the
# true w-optimum is far from x* (a structural restructuring), so the "warm
# start" is not warm. We quantify both by solving the LP EXACTLY (Gurobi) and
# comparing x_opt to x* in the investment vs operational subspaces.
# ===========================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, LinearAlgebra, Printf
using NearOptimalAlternatives
include(joinpath(@__DIR__, "bench_common.jl"))
const THREADS = max(1, Threads.nthreads())

ep = TEM.create_energy_problem_from_csv_folder(joinpath(@__DIR__, "data", "5h"))
TEM.create_model!(ep);
model = ep.model;
set_optimizer(model, make_optimizer(:gurobi, THREADS));
set_silent(model);
@elapsed JuMP.optimize!(model);
@assert is_solved_and_feasible(model);
min_cost = objective_value(model)
allv = all_variables(model);
idx = Dict(v => i for (i, v) in enumerate(allv));
target = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    if haskey(object_dictionary(model), sym)
        for v in object_dictionary(model)[sym]
            push!(target, v)
        end
    end
end
inv = sort([idx[v] for v in target]);
n = length(allv);
op = setdiff(1:n, inv)
w = zeros(n);
for v in target
    ub = (has_upper_bound(v) && upper_bound(v) != 0) ? upper_bound(v) : 1.0
    w[idx[v]] = value(v) / ub
end;
B = min_cost + 0.1 * abs(min_cost)
println(
    "vars=$n  investment=$(length(inv)) ($(round(100*length(inv)/n,digits=3))%)  operational=$(length(op))",
)

lp = NearOptimalAlternatives.build_mga_lp(model, w, B, allv)
xstar_t = clamp.(value.(allv) ./ lp.d_scale, lp.lb_t, lp.ub_t)

# Exact min w_t'x_t s.t. LP (the true target the FOM should reach).
m = Model(make_optimizer(:gurobi, THREADS));
set_silent(m);
@variable(m, xt[j = 1:lp.n]);
for j = 1:lp.n
    isfinite(lp.lb_t[j]) && set_lower_bound(xt[j], lp.lb_t[j])
    isfinite(lp.ub_t[j]) && set_upper_bound(xt[j], lp.ub_t[j])
end;
@constraint(m, lp.A_in * xt .<= lp.b_in);
@constraint(m, lp.A_eq * xt .== lp.b_eq);
@objective(m, Min, dot(lp.w_t, xt));
JuMP.optimize!(m);
xopt_t = value.(xt)

relmove(a, b, S) = norm(a[S] .- b[S]) / max(norm(b[S]), eps())
@printf(
    "\nobjective signal:  ||w_t||=%.4f  ||w_t[inv]||=%.4f  ||w_t[op]||=%.2e\n",
    norm(lp.w_t),
    norm(lp.w_t[inv]),
    norm(lp.w_t[op])
)
@printf(
    "w_t'x*  = %.4f   w_t'x_opt = %.4f   (drop %.1f%%)\n",
    dot(lp.w_t, xstar_t),
    dot(lp.w_t, xopt_t),
    100 * (dot(lp.w_t, xstar_t) - dot(lp.w_t, xopt_t)) / abs(dot(lp.w_t, xstar_t))
)
@printf("\nDISTANCE x_opt vs x* (scaled space):\n")
@printf(
    "  total      ||Δ|| = %.3e   rel = %.3f\n",
    norm(xopt_t .- xstar_t),
    relmove(xopt_t, xstar_t, 1:n)
)
@printf(
    "  investment ||Δ|| = %.3e   rel = %.3f\n",
    norm(xopt_t[inv] .- xstar_t[inv]),
    relmove(xopt_t, xstar_t, inv)
)
@printf(
    "  operational||Δ|| = %.3e   rel = %.3f\n",
    norm(xopt_t[op] .- xstar_t[op]),
    relmove(xopt_t, xstar_t, op)
)
nz_inv = count(abs.(xopt_t[inv] .- xstar_t[inv]) .> 1e-6 .* (1 .+ abs.(xstar_t[inv])))
nz_op = count(abs.(xopt_t[op] .- xstar_t[op]) .> 1e-6 .* (1 .+ abs.(xstar_t[op])))
@printf(
    "  investment vars moved: %d/%d   operational vars moved: %d/%d\n",
    nz_inv,
    length(inv),
    nz_op,
    length(op)
)
@printf(
    "\ncost(x*)=%.4g  cost(x_opt)=%.4g  B=%.4g  (x_opt uses budget? %s)\n",
    min_cost,
    dot(
        lp.d_scale .* xopt_t,
        [JuMP.coefficient(objective_function(model), v) for v in allv],
    ),
    B,
    "see cost vs B"
)
