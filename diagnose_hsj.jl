# Why is HSJ so slow on data/5h? Isolate the cause by solving the *same* MGA
# LP three ways and timing each:
#   (A) HSJ direction (binary 0/1 weights) + barrier + crossover   <- benchmark setting
#   (B) HSJ direction (binary 0/1 weights) + barrier, NO crossover
#   (C) SPORES direction (continuous weights) + barrier + crossover  <- reference
# If (A) >> (B), crossover on HSJ's degenerate optimal face is the culprit.
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB, JuMP, Gurobi, LinearAlgebra

const THREADS = max(1, Threads.nthreads())
opt(cross) = optimizer_with_attributes(
    Gurobi.Optimizer,
    "OutputFlag" => 1,
    "Threads" => THREADS,
    "Method" => 2,
    "Crossover" => cross,
)

ep = TEM.create_energy_problem_from_csv_folder(joinpath(pwd(), "data/5h"))
TEM.create_model!(ep)
base = ep.model
set_optimizer(base, opt(1))
set_silent(base)
@elapsed JuMP.optimize!(base)
base_obj = objective_value(base)
budget = 1.1 * base_obj

target = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    if haskey(object_dictionary(base), sym)
        for v in object_dictionary(base)[sym]
            push!(target, v)
        end
    end
end
allv = all_variables(base)
v2i = Dict(v => i for (i, v) in enumerate(allv))
tidx = [v2i[v] for v in target]
bvals = value.(allv)
ubs = [has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0 for v in target]

# HSJ binary weights (1 if structural var nonzero at the optimum) and SPORES
# continuous weights (value/ub), both over the structural variables.
w_hsj = zeros(length(allv))
w_spores = zeros(length(allv))
nactive = 0
for (j, idx) in enumerate(tidx)
    if abs(bvals[idx]) > 1e-9
        w_hsj[idx] = 1.0
        global nactive += 1
    end
    w_spores[idx] = bvals[idx] / ubs[j]
end
println("structural vars: $(length(target)), active (HSJ weight 1): $nactive")

function solve_dir(w, cross, label)
    m = copy(base)
    set_optimizer(m, opt(cross))
    set_silent(m)
    rv = all_variables(m)
    @constraint(m, objective_function(m) <= budget)
    @objective(m, Min, sum(w[i] * rv[i] for i in eachindex(rv) if !iszero(w[i])))
    t = @elapsed JuMP.optimize!(m)
    println(">>> $label : $(round(t, digits=1))s  ($(termination_status(m)))")
    return t
end

println("\n=== isolating the HSJ slowdown ===")
a = solve_dir(w_hsj, 1, "(A) HSJ binary  + crossover ON ")
b = solve_dir(w_hsj, 0, "(B) HSJ binary  + crossover OFF")
c = solve_dir(w_spores, 1, "(C) SPORES cont + crossover ON ")
println("\nA/B ratio (crossover overhead on HSJ): $(round(a / max(b, 1e-9), digits=1))x")
println("A/C ratio (HSJ vs SPORES, both crossover): $(round(a / max(c, 1e-9), digits=1))x")
