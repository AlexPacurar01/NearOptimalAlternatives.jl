# data/5h driver for the continuation-vs-SPORES suite (exact, Gurobi). Reuses the
# core functions from experiment_continuation_spores.jl. Runtime is dominated by
# (n_dir * n_budget + n_dir) cold Gurobi solves, so keep n_budget modest at first.
#   N_DIR=5 N_BUDGET=6 julia -t 8 --project=. experiment_cont_spores_5h.jl
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, LinearAlgebra, Printf
include(joinpath(@__DIR__, "experiment_continuation_spores.jl"))   # core funcs + bench_common

const THREADS = max(1, Threads.nthreads())
const N_DIR = parse(Int, get(ENV, "N_DIR", "5"))
const N_BUDGET = parse(Int, get(ENV, "N_BUDGET", "6"))
const EPS = parse(Float64, get(ENV, "EPS", "0.1"))
const OUTDIR = joinpath(@__DIR__, "results", "cont_spores_5h")
mkpath(OUTDIR)

ep = TEM.create_energy_problem_from_csv_folder(joinpath(@__DIR__, "data", "5h"))
TEM.create_model!(ep);
model = ep.model;
set_optimizer(model, make_optimizer(:gurobi, THREADS));
set_silent(model);
allv = all_variables(model);
idx = Dict(v => i for (i, v) in enumerate(allv));
target = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    haskey(object_dictionary(model), sym) && for v in object_dictionary(model)[sym]
        push!(target, v)
    end
end
inv_idx = [idx[v] for v in target]
ub_inv = [
    (has_upper_bound(v) && isfinite(upper_bound(v)) && upper_bound(v) > 0) ?
    upper_bound(v) : 1.0 for v in target
]
println(
    "5h: $(length(allv)) vars | $(length(inv_idx)) investment vars | N_DIR=$N_DIR N_BUDGET=$N_BUDGET",
);
flush(stdout);

t = @elapsed r = run_suite(
    model,
    allv,
    inv_idx,
    ub_inv;
    n_dir = N_DIR,
    eps = EPS,
    n_budget = N_BUDGET,
)
@printf("\ntotal suite wall: %.1fs (%d Gurobi solves)\n", t, N_DIR * N_BUDGET + N_DIR + 1)

# Save per-point (method, direction, budget, cost, diversity) for plotting the front.
cvec = [JuMP.coefficient(objective_function(model), v) for v in allv]
coff = JuMP.constant(objective_function(model))
open(joinpath(OUTDIR, "points.csv"), "w") do io
    println(io, "method,direction,budget_idx,cost,divobj")
    for k = 1:N_DIR
        wk = r.dirs[k]
        whatk = wk ./ max(norm(wk), 1e-12)
        # SPORES endpoint k
        a = r.alts[k]
        @printf(
            io,
            "spores,%d,%d,%.8g,%.8g\n",
            k,
            N_BUDGET,
            dot(cvec, a) + coff,
            dot(whatk, a)
        )
        for bi = 1:N_BUDGET
            c = r.cont[k][bi]
            @printf(io, "cont,%d,%d,%.8g,%.8g\n", k, bi, dot(cvec, c) + coff, dot(whatk, c))
        end
    end
end
println("Wrote $(joinpath(OUTDIR, "points.csv"))")
println("DONE");
flush(stdout);
