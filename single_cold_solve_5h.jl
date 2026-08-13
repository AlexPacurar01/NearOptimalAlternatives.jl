# =============================================================================
# Single cold SPORES-alternative solve per solver on data/5h.
#
# Goal: measure how long ONE cold solve of a SPORES alternative takes for each
# solver at 5h scale, to decide what is tractable.
#
# The base cost-min is solved once with Gurobi to define a common, fair problem:
#   - SPORES direction  w_j = x*_j / ub_j  on investment variables
#   - budget            B   = (1 + GAP) * C*
# Then each solver solves the SAME MGA LP  min wᵀx  s.t.  constraints, cost ≤ B
# COLD (fresh optimizer, no warm start) and we record solve time + status +
# diversity objective. Same objective for all, so the diversity values are
# directly comparable (Gurobi = exact reference).
#
# Config (env): GAP (0.1), TLIM (2000s), SOLVERS (gurobi,highs,scs), THREADS via -t.
# Run:  TLIM=2000 julia -t 8 --project=. single_cold_solve_5h.jl
# =============================================================================
import TulipaEnergyModel as TEM
using DuckDB, JuMP, LinearAlgebra, Printf
import Gurobi, HiGHS, SCS, Ipopt

const THREADS = max(1, Threads.nthreads())
const GAP = parse(Float64, get(ENV, "GAP", "0.1"))
const TLIM = parse(Float64, get(ENV, "TLIM", "2000"))
const SOLVERS = Symbol.(split(get(ENV, "SOLVERS", "gurobi,highs,scs"), ","))
const DATADIR = joinpath(@__DIR__, "data", "5h")
const OUTDIR = joinpath(@__DIR__, "results", "sweep_vs_old_5h")
mkpath(OUTDIR)

# Barrier/IPM without crossover (cannot warm-start); SCS indirect, verbose so its
# iteration progress is visible. TLIM <= 0 means "no time limit" (the limit
# attribute is omitted); SCS is still bounded by its iteration cap.
function solver_factory(s::Symbol)
    if s == :gurobi
        attrs = Pair{String,Any}[
            "OutputFlag"=>0,
            "Threads"=>THREADS,
            "Method"=>2,
            "Crossover"=>0,
        ]
        TLIM > 0 && push!(attrs, "TimeLimit" => TLIM)
        return optimizer_with_attributes(Gurobi.Optimizer, attrs...)
    elseif s == :highs
        attrs = Pair{String,Any}[
            "output_flag"=>false,
            "threads"=>THREADS,
            "solver"=>"ipm",
            "run_crossover"=>"off",
        ]
        TLIM > 0 && push!(attrs, "time_limit" => TLIM)
        return optimizer_with_attributes(HiGHS.Optimizer, attrs...)
    elseif s == :scs
        attrs = Pair{String,Any}[
            "verbose"=>1,
            "linear_solver"=>SCS.IndirectSolver,
            "eps_abs"=>1e-4,
            "eps_rel"=>1e-4,
        ]
        if TLIM > 0
            push!(attrs, "time_limit_secs" => TLIM)
            push!(attrs, "max_iters" => 20000)
        else
            push!(attrs, "max_iters" => 100_000_000)   # effectively no iteration cap
        end
        return optimizer_with_attributes(SCS.Optimizer, attrs...)
    elseif s == :ipopt
        # Interior-point NLP solver (also solves LPs). print_level 5 shows the
        # per-iteration log. No crossover concept; cannot warm-start.
        attrs = Pair{String,Any}["print_level"=>5]
        TLIM > 0 && push!(attrs, "max_cpu_time" => TLIM)
        return optimizer_with_attributes(Ipopt.Optimizer, attrs...)
    end
    error("unknown solver $s")
end

safe_solve_time(m) =
    try
        JuMP.solve_time(m)
    catch
        NaN
    end

println("Building 5h model ...");
flush(stdout);
ep = TEM.create_energy_problem_from_csv_folder(DATADIR)
TEM.create_model!(ep)
model = ep.model

# Base cost-min with Gurobi -> x*, C*, and the SPORES direction.
set_optimizer(model, solver_factory(:gurobi))
tb = @elapsed JuMP.optimize!(model)
@assert is_solved_and_feasible(model) "base solve failed"
Cstar = objective_value(model)
cost_expr = objective_function(model)
@printf("base (gurobi): C*=%.6g  in %.1fs\n", Cstar, tb);
flush(stdout);

target = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    haskey(object_dictionary(model), sym) && for v in object_dictionary(model)[sym]
        push!(target, v)
    end
end
w = Dict{VariableRef,Float64}()
for v in target
    ub =
        (has_upper_bound(v) && isfinite(upper_bound(v)) && upper_bound(v) > 0) ?
        upper_bound(v) : 1.0
    w[v] = value(v) / ub
end
Bmax = (1 + GAP) * Cstar
@printf(
    "SPORES alt: %d investment vars, budget B=%.6g (= %.1f%% over C*)\n",
    length(target),
    Bmax,
    100 * GAP
);
flush(stdout)

# Install the MGA objective + budget once; each solver re-solves it cold.
@objective(model, Min, sum(c * v for (v, c) in w))
@constraint(model, budget, cost_expr <= Bmax)

rows = NamedTuple[]
for s in SOLVERS
    println("\n>>> $s: cold SPORES-alternative solve ...")
    flush(stdout)
    set_optimizer(model, solver_factory(s))
    t = @elapsed JuMP.optimize!(model)
    st = termination_status(model)
    solved = is_solved_and_feasible(model)
    div = solved ? objective_value(model) : NaN
    stime = safe_solve_time(model)
    @printf(
        ">>> %-7s status=%s  wall=%.1fs  solver_solve_time=%.1fs  diversity(wᵀx)=%.6g\n",
        s,
        st,
        t,
        stime,
        div
    )
    flush(stdout)
    push!(rows, (; solver = s, status = st, wall = t, solve_time = stime, diversity = div))
end

println("\n" * "="^70)
@printf(
    "%-8s %-16s %10s %12s %14s\n",
    "solver",
    "status",
    "wall(s)",
    "solve(s)",
    "diversity"
)
for r in rows
    @printf(
        "%-8s %-16s %10.1f %12.1f %14.6g\n",
        r.solver,
        string(r.status),
        r.wall,
        r.solve_time,
        r.diversity
    )
end

open(joinpath(OUTDIR, "single_cold_$(join(SOLVERS, "_")).csv"), "w") do io
    println(io, "solver,status,wall,solve_time,diversity")
    for r in rows
        @printf(
            io,
            "%s,%s,%.4f,%.4f,%.10g\n",
            r.solver,
            string(r.status),
            r.wall,
            r.solve_time,
            r.diversity
        )
    end
end
println("\nDONE");
flush(stdout);
