# =============================================================================
# Arclength-distributed front on data/5h (Gurobi), for comparison against the
# uniform-budget sweep (front_gurobi.csv). Same model, same solver settings, same
# N_DIR / N_BUDGET / GAP — only the point DISTRIBUTION differs (arclength vs
# uniform budget). Writes front_arclength.csv in the same schema.
#
# Config (env): N_DIR (5), N_BUDGET (6), GAP (0.1), TLIM (2000). Run:
#   N_DIR=5 N_BUDGET=6 TLIM=2000 julia -t 8 --project=. run_arclength_5h.jl
# =============================================================================
import TulipaEnergyModel as TEM
using DuckDB, JuMP, LinearAlgebra, Printf
import Gurobi
using NearOptimalAlternatives

const THREADS = max(1, Threads.nthreads())
const N_DIR = parse(Int, get(ENV, "N_DIR", "5"))
const N_BUDGET = parse(Int, get(ENV, "N_BUDGET", "6"))
const GAP = parse(Float64, get(ENV, "GAP", "0.1"))
const TLIM = parse(Float64, get(ENV, "TLIM", "2000"))
const DATADIR = joinpath(@__DIR__, "data", "5h")
const OUTDIR = joinpath(@__DIR__, "results", "sweep_vs_old_5h")
mkpath(OUTDIR)

# Plain barrier without crossover (same as the uniform-sweep run).
gurobi() = optimizer_with_attributes(
    Gurobi.Optimizer,
    "OutputFlag" => 0,
    "Threads" => THREADS,
    "Method" => 2,
    "Crossover" => 0,
    "TimeLimit" => TLIM,
)

println("Building + solving 5h base model (Gurobi) ...");
flush(stdout);
ep = TEM.create_energy_problem_from_csv_folder(DATADIR)
TEM.create_model!(ep)
model = ep.model
set_optimizer(model, gurobi())
set_silent(model)
tb = @elapsed JuMP.optimize!(model)
@assert is_solved_and_feasible(model) "base solve failed"
@printf("base solve: %.1fs  C*=%.6g\n", tb, objective_value(model));
flush(stdout);

target = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    haskey(object_dictionary(model), sym) && for v in object_dictionary(model)[sym]
        push!(target, v)
    end
end
@printf(
    "5h: %d vars | %d investment vars | N_DIR=%d N_BUDGET=%d\n",
    num_variables(model),
    length(target),
    N_DIR,
    N_BUDGET
);
flush(stdout)

# Write each point to the CSV as it is placed (flushed), so partial results are
# available live -- the arclength march is slow and otherwise opaque.
csvpath = joinpath(OUTDIR, "front_arclength.csv")
t = @elapsed result = open(csvpath, "w") do io
    println(io, "direction,budget_idx,cost,diversity,gap,solve_time")
    flush(io)
    sink =
        p -> begin
            @printf(
                io,
                "%d,%d,%.10g,%.10g,%.6g,%.4f\n",
                p.direction,
                p.budget_idx,
                p.cost,
                p.diversity,
                p.gap,
                p.solve_time
            )
            flush(io)
        end
    generate_alternatives_arclength!(
        model,
        GAP,
        target,
        N_DIR;
        n_budget = N_BUDGET,
        modeling_method = :Spores,
        point_sink = sink,
    )
end
@printf("\narclength wall: %.1fs  (%d front points)\n", t, length(result.solutions))
solve_sum = sum(x -> isnan(x.solve_time) ? 0.0 : x.solve_time, result.tags; init = 0.0)
@printf("solver solve-time sum: %.1fs\n", solve_sum)
println("Wrote $csvpath")
println("DONE");
flush(stdout);
