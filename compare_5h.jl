# ===========================================================================
# First-order method comparison on the real data/5h model.
#
# Same comparison as compare_firstorder.jl, but on the 1.27M-variable /
# 1.93M-constraint TulipaEnergy 5h model - the regime where the methods
# genuinely diverge and where Gurobi barrier is the incumbent to beat.
#
# This is a LONG run (each first-order method does many passes over a 4.9M
# nonzero matrix; the Gurobi reference is one ~685s barrier solve). Launch it
# deliberately, e.g.:  julia -t auto --project=. compare_5h.jl
#
# Budgets are intentionally modest below; raise max_iters / pdhg_iters to
# trade wall-clock for accuracy.
# ===========================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using NearOptimalAlternatives
using JuMP, Gurobi, LinearAlgebra, Printf, Logging
const GM = NearOptimalAlternatives

input_dir = joinpath(pwd(), "data/5h")
println("Reading data from: $input_dir")
energy_problem = TEM.create_energy_problem_from_csv_folder(input_dir)
TEM.create_model!(energy_problem)
model = energy_problem.model
set_optimizer(
    model,
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Method" => 2,
        "Crossover" => 1,
        "Threads" => 32,
    ),
)

println("Solving baseline (workflow precondition)...")
t_base = @elapsed optimize!(model)
base_obj = objective_value(model)
base_vals = value.(all_variables(model))
all_vars = all_variables(model)
n_total = length(all_vars)
println(
    "Baseline: obj $(round(base_obj, sigdigits=6)) in $(round(t_base, digits=1))s | $n_total vars",
)

# --- MGA weights (HSJ #1) + budget -----------------------------------------
var_to_idx = Dict(v => i for (i, v) in enumerate(all_vars))
target_vars = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    if haskey(object_dictionary(model), sym)
        for v in object_dictionary(model)[sym]
            push!(target_vars, v)
        end
    end
end
target_indices = [var_to_idx[v] for v in target_vars]
target_ubs =
    [has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0 for v in target_vars]
w = zeros(n_total)
for (j, idx) in enumerate(target_indices)
    w[idx] = base_vals[idx] / target_ubs[j]
end
w ./= norm(w)
w_s = w[target_indices]
B = 1.1 * base_obj

# --- Exact reference: solve the MGA LP directly with Gurobi barrier --------
println("Solving exact MGA reference (Gurobi barrier)...")
t_ref = @elapsed begin
    ref = copy(model)
    set_optimizer(
        ref,
        optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag" => 0,
            "Method" => 2,
            "Crossover" => 0,
            "Threads" => 32,
        ),
    )
    rv = all_variables(ref)
    @constraint(ref, objective_function(ref) <= B)
    @objective(ref, Min, sum(w[i] * rv[i] for i in eachindex(rv) if !iszero(w[i])))
    optimize!(ref)
end
obj_ref_s = dot(w_s, value.(rv)[target_indices])
println(
    "Exact reference: structural w_s'x=$(round(obj_ref_s, sigdigits=6)) in $(round(t_ref, digits=1))s\n",
)

# --- Build shared scaled LP once -------------------------------------------
lp = with_logger(NullLogger()) do
    GM.build_mga_lp(model, copy(w), B, all_vars)
end
println(
    "LP built+equilibrated in $(round(lp.t_extract, digits=1))s: $(size(lp.A_in,1)) ineq, $(size(lp.A_eq,1)) eq rows",
)
x0 = clamp.(zeros(lp.n), lp.lb_t, lp.ub_t)

# --- Run each first-order method (modest budgets; raise for accuracy) -------
configs = [
    (
        :alm_lbfgs,
        "ALM + projected L-BFGS",
        (max_iters = 15, max_inner = 1000, pdhg_iters = 0),
    ),
    (
        :penalty,
        "Quadratic penalty + L-BFGS",
        (max_iters = 15, max_inner = 1000, pdhg_iters = 0),
    ),
    (
        :pdhg,
        "PDHG (PDLP-style restarts)",
        (max_iters = 0, max_inner = 0, pdhg_iters = 2000),
    ),
]
results = []
for (m, label, cfg) in configs
    println("Running $label ...")
    x_t, info = GM.solve_firstorder(
        m,
        lp,
        copy(x0);
        max_iters = cfg.max_iters,
        max_inner = cfg.max_inner,
        pdhg_iters = cfg.pdhg_iters,
        verbose = true,
    )
    x = lp.d_scale .* x_t
    obj_s = dot(w_s, x[target_indices])
    gap_s = abs(obj_s - obj_ref_s) / max(abs(obj_ref_s), 1e-8) * 100
    push!(
        results,
        (; label, time = info.time, iters = info.iters, infeas = info.infeas, obj_s, gap_s),
    )
    println(
        "  -> $(round(info.time, digits=1))s | infeas $(round(info.infeas, sigdigits=3)) | gap $(round(gap_s, digits=2))%",
    )
end

println("\n" * "="^92)
@printf(
    "%-28s %9s %8s %12s %12s %10s\n",
    "method",
    "time(s)",
    "iters",
    "infeas",
    "w_s'x",
    "gap%"
)
println("-"^92)
for r in results
    @printf(
        "%-28s %9.1f %8d %12.2e %12.5g %10.2f\n",
        r.label,
        r.time,
        r.iters,
        r.infeas,
        r.obj_s,
        r.gap_s
    )
end
@printf(
    "%-28s %9.1f %8s %12s %12.5g %10s\n",
    "Exact LP (Gurobi barrier)",
    t_ref,
    "-",
    "0",
    obj_ref_s,
    "0"
)
println("="^92)
println(
    "Baseline solve (precondition, excluded from algorithm time): $(round(t_base, digits=1))s",
)
