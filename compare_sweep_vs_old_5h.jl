# =============================================================================
# Old API vs new sweep API on data/5h.
#
# Compares, on the real 5h TulipaEnergyModel, two ways of generating SPORES
# alternatives:
#   OLD  -> generate_alternatives_optimization!  (one endpoint per direction)
#   NEW  -> generate_alternatives_sweep!         (a budget-swept front per dir)
#
# Every solver runs the SAME experiment (N_DIR directions, N_BUDGET budgets). For
# each we report the wall time of each API call and the quality of the solutions
# (the cost/diversity front, and that the swept full-budget point reproduces the
# old API's endpoint). Gurobi (barrier) is the exact reference; HiGHS is IPM; SCS
# (ADMM) is warm-started along the budget continuation.
#
# Everything is exact for Gurobi/HiGHS; SCS is a first-order method and is
# expected to be slower / less accurate at this scale -- that contrast is the
# point of the comparison.
#
# Config (env):
#   N_BUDGET   budget levels per sweep            (default 6)
#   N_DIR      SPORES directions (same for all solvers)   (default 1)
#   GAP        optimality gap / top of the sweep   (default 0.1)
#   TLIM       per-solve time limit, seconds       (default 2000)
#   PLOT       1 to save front plots               (default 1)
#
# Run:  N_BUDGET=6 N_DIR=2 julia -t 8 --project=. compare_sweep_vs_old_5h.jl
# =============================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, LinearAlgebra, Printf
import Gurobi, HiGHS, SCS
using NearOptimalAlternatives

const THREADS = max(1, Threads.nthreads())
const N_BUDGET = parse(Int, get(ENV, "N_BUDGET", "6"))
const N_DIR = parse(Int, get(ENV, "N_DIR", "1"))
const GAP = parse(Float64, get(ENV, "GAP", "0.1"))
const TLIM = parse(Float64, get(ENV, "TLIM", "2000"))
const DOPLOT = get(ENV, "PLOT", "1") == "1"
const DATADIR = joinpath(@__DIR__, "data", "5h")
const OUTDIR = joinpath(@__DIR__, "results", "sweep_vs_old_5h")
mkpath(OUTDIR)

DOPLOT && @eval using Plots

# --- solvers ----------------------------------------------------------------
# Barrier/IPM WITHOUT crossover for Gurobi/HiGHS: interior-point cannot warm-start,
# so each budget is a fresh cold solve -- that is deliberately the "cannot warm
# start" baseline. SCS (ADMM, indirect) is the warm-startable method we compare it
# against. Crossover stays off (slow, and not needed to trace the front).
function solver_factory(solver::Symbol)
    if solver == :gurobi
        return optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag" => 0,
            "Threads" => THREADS,
            "Method" => 2,
            "Crossover" => 0,
            "TimeLimit" => TLIM,
        )
    elseif solver == :highs
        return optimizer_with_attributes(
            HiGHS.Optimizer,
            "output_flag" => false,
            "threads" => THREADS,
            "solver" => "ipm",
            "run_crossover" => "off",
            "time_limit" => TLIM,
        )
    elseif solver == :scs
        return optimizer_with_attributes(
            SCS.Optimizer,
            "verbose" => 0,
            "linear_solver" => SCS.IndirectSolver,
            "eps_abs" => 1e-4,
            "eps_rel" => 1e-4,
            "max_iters" => 20000,
            "time_limit_secs" => TLIM,
        )
    end
    error("unknown solver $solver")
end

# --- model ------------------------------------------------------------------
"Build the 5h model, attach `solver`, and solve the base cost-minimisation so
the alternative-generation APIs have a feasible starting point (x*, C*)."
function build_solved_model(solver::Symbol)
    ep = TEM.create_energy_problem_from_csv_folder(DATADIR)
    TEM.create_model!(ep)
    model = ep.model
    set_optimizer(model, solver_factory(solver))
    set_silent(model)
    t = @elapsed JuMP.optimize!(model)
    @printf(
        "    [%s] base solve: %.1fs  (status=%s)\n",
        solver,
        t,
        termination_status(model)
    )
    flush(stdout)
    return model
end

"Structural (investment) variables that carry the SPORES direction."
function target_vars(model)
    tv = JuMP.VariableRef[]
    for sym in (:assets_investment, :flows_investment)
        if haskey(object_dictionary(model), sym)
            for v in object_dictionary(model)[sym]
                push!(tv, v)
            end
        end
    end
    return tv
end

# Investment values of a solution, ordered by `vars` (a stable order shared
# across rebuilt models, since the model is constructed deterministically).
inv_values(sol, vars) = [sol[v] for v in vars]

# --- runs -------------------------------------------------------------------
"Old API: one SPORES endpoint per direction at the full budget."
function run_old(solver::Symbol, n_dir::Int)
    model = build_solved_model(solver)
    vars = target_vars(model)
    t = @elapsed res = generate_alternatives_optimization!(
        model,
        GAP,
        vars,
        n_dir;
        modeling_method = :Spores,
    )
    return (; time = t, res, vars)
end

"New API: a budget-swept front per direction (warm-started if `warm`)."
function run_new(solver::Symbol, n_dir::Int, warm::Bool)
    model = build_solved_model(solver)
    vars = target_vars(model)
    t = @elapsed res = generate_alternatives_sweep!(
        model,
        GAP,
        vars,
        n_dir;
        n_budget = N_BUDGET,
        modeling_method = :Spores,
        warm_start = warm,
    )
    return (; time = t, res, vars)
end

# Full-budget point of direction k in a sweep result (budget_idx == N_BUDGET).
fullbudget_index(res, k) = findlast(
    i -> res.tags[i].direction == k && res.tags[i].budget_idx == N_BUDGET,
    eachindex(res.tags),
)

# --- output -----------------------------------------------------------------
function write_front_csv(path, res)
    open(path, "w") do io
        println(io, "direction,budget_idx,cost,diversity,solve_time")
        for i in eachindex(res.solutions)
            tg = res.tags[i]
            @printf(
                io,
                "%d,%d,%.10g,%.10g,%.4f\n",
                tg.direction,
                tg.budget_idx,
                res.objective_values[i],
                tg.diversity_objective,
                tg.solve_time
            )
        end
    end
end

function plot_front(path, res, title)
    DOPLOT || return
    try
        plt = Plots.plot(;
            xlabel = "cost",
            ylabel = "diversity objective",
            title = title,
            legend = :topright,
        )
        for k in sort(unique(t.direction for t in res.tags))
            idx = [i for i in eachindex(res.tags) if res.tags[i].direction == k]
            idx = sort(idx; by = i -> res.objective_values[i])
            Plots.plot!(
                plt,
                [res.objective_values[i] for i in idx],
                [res.tags[i].diversity_objective for i in idx];
                marker = :circle,
                label = "direction $k",
            )
        end
        Plots.savefig(plt, path)
        println("    wrote $path")
    catch err
        @warn "plot skipped" err
    end
end

# =============================================================================
# Experiment
# =============================================================================
println("="^78)
@printf(
    "Old API vs sweep API on data/5h | threads=%d N_BUDGET=%d GAP=%.3f TLIM=%.0fs\n",
    THREADS,
    N_BUDGET,
    GAP,
    TLIM
)
println("="^78);
flush(stdout)

# All solvers run the SAME experiment (same number of directions and budgets);
# the only difference is that SCS (a first-order/ADMM method) warm-starts the
# budget continuation, which barrier/IPM solvers cannot exploit.
# Select which solvers to run via SOLVERS (comma-separated), e.g. SOLVERS=scs or
# SOLVERS=gurobi,highs. Handy for splitting the solvers across parallel runs.
# (solver => (directions, warm-start the sweep?))
const SPEC =
    Dict(:gurobi => (N_DIR, false), :highs => (N_DIR, false), :scs => (N_DIR, true))
const SOLVERS = Symbol.(split(get(ENV, "SOLVERS", "gurobi,highs,scs"), ","))
specs = [(s, SPEC[s][1], SPEC[s][2]) for s in SOLVERS]
summary = NamedTuple[]

for (solver, ndir, warm) in specs
    println("\n" * "-"^78)
    println("Solver: $solver  (directions=$ndir, sweep warm_start=$warm)")
    println("-"^78)
    flush(stdout)

    try

        @printf(">>> OLD API: %d alternative(s)\n", ndir)
        flush(stdout)
        old = run_old(solver, ndir)
        @printf(
            "    OLD wall: %.1fs  (%d endpoints)\n",
            old.time,
            length(old.res.solutions)
        )

        @printf(">>> NEW API: %d direction(s) x %d budgets\n", ndir, N_BUDGET)
        flush(stdout)
        new = run_new(solver, ndir, warm)
        # Solver-reported solve time summed over all sweep solves (vs the wall time,
        # which also includes JuMP/MOI overhead between solves).
        t_new_solve = sum(t -> isnan(t.solve_time) ? 0.0 : t.solve_time, new.res.tags)
        @printf(
            "    NEW wall: %.1fs  (solver solve-time sum %.1fs, %d front points)\n",
            new.time,
            t_new_solve,
            length(new.res.solutions),
        )

        # Quality: the swept full-budget point should reproduce the old endpoint.
        max_dev = 0.0
        for k = 1:ndir
            fi = fullbudget_index(new.res, k)
            if fi !== nothing && k <= length(old.res.solutions)
                ov = inv_values(old.res.solutions[k], old.vars)
                nv = inv_values(new.res.solutions[fi], new.vars)
                max_dev = max(max_dev, maximum(abs.(ov .- nv)))
            end
        end
        @printf(
            "    full-budget point vs old endpoint: max |Δinvestment| = %.3e\n",
            max_dev
        )

        # Endpoint cost/diversity of the first direction (Gurobi = exact reference).
        fi1 = fullbudget_index(new.res, 1)
        ep_cost = new.res.objective_values[fi1]
        ep_div = new.res.tags[fi1].diversity_objective
        @printf("    direction-1 full-budget: cost=%.6g  diversity=%.6g\n", ep_cost, ep_div)

        write_front_csv(joinpath(OUTDIR, "front_$(solver).csv"), new.res)
        plot_front(joinpath(OUTDIR, "front_$(solver).png"), new.res, "5h front ($solver)")

        push!(
            summary,
            (;
                solver,
                ndir,
                t_old = old.time,
                t_new = new.time,
                t_new_solve,
                n_old = length(old.res.solutions),
                n_new = length(new.res.solutions),
                max_dev,
                ep_cost,
                ep_div,
            ),
        )

    catch err
        # Isolate a solver failure (e.g. an SCS solve not converging within TLIM,
        # leaving the model not-solved) so the other solvers' results survive.
        @warn "solver $solver failed; skipping" exception = (err, catch_backtrace())
    end
    flush(stdout)
end

# --- summary table ----------------------------------------------------------
println("\n" * "="^78)
println("SUMMARY")
@printf(
    "%-8s %5s %10s %10s %12s %7s %7s %12s %12s\n",
    "solver",
    "ndir",
    "t_old(s)",
    "t_new(s)",
    "t_new_solve",
    "n_old",
    "n_new",
    "ep_cost",
    "ep_div"
)
for r in summary
    @printf(
        "%-8s %5d %10.1f %10.1f %12.1f %7d %7d %12.6g %12.6g\n",
        r.solver,
        r.ndir,
        r.t_old,
        r.t_new,
        r.t_new_solve,
        r.n_old,
        r.n_new,
        r.ep_cost,
        r.ep_div
    )
end

open(joinpath(OUTDIR, "summary_$(join(SOLVERS, "_")).csv"), "w") do io
    println(io, "solver,ndir,t_old,t_new,t_new_solve,n_old,n_new,max_dev,ep_cost,ep_div")
    for r in summary
        @printf(
            io,
            "%s,%d,%.4f,%.4f,%.4f,%d,%d,%.6e,%.10g,%.10g\n",
            r.solver,
            r.ndir,
            r.t_old,
            r.t_new,
            r.t_new_solve,
            r.n_old,
            r.n_new,
            r.max_dev,
            r.ep_cost,
            r.ep_div
        )
    end
end
println("\nWrote $(joinpath(OUTDIR, "summary_$(join(SOLVERS, "_")).csv"))")
println("DONE");
flush(stdout)
