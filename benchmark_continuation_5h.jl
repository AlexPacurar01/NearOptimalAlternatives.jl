# ===========================================================================
# Warm-started epsilon-constraint continuation on the real data/5h model.
#
# The decisive cheapness test: does `continuation_walk` (warm-started ALM,
# budget sweep) hold a FEW iterations per step on the stiff 1.27M-variable
# model, where a cold first-order solve needed thousands? If so we have a
# fast, basis-free near-optimal-space explorer - the infeasible-path FOM analog
# of Funplex.
#
# Multi-minute build + a handful of warm solves. Launch deliberately:
#     julia -t 8 --project=. benchmark_continuation_5h.jl
# Outputs: results/continuation_5h/ (front CSV + plot).
# ===========================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, LinearAlgebra, Printf
using Plots
using NearOptimalAlternatives

include(joinpath(@__DIR__, "bench_common.jl"))   # make_optimizer

const THREADS = max(1, Threads.nthreads())
const EPS_MGA = 0.1
const BACKEND = :gurobi
const N_POINTS = parse(Int, get(ENV, "N_POINTS", "8"))   # demo default; raise for a full sweep
# Continuation FOM: :alm_lbfgs (default) or :pdhg. Set via CONT_METHOD env var.
const CONT_METHOD = Symbol(get(ENV, "CONT_METHOD", "alm_lbfgs"))
const PDHG_ITERS = parse(Int, get(ENV, "PDHG_ITERS", "50000"))
const OSQP_ITERS = parse(Int, get(ENV, "OSQP_ITERS", "40000"))
const FEAS_TOL = parse(Float64, get(ENV, "FEAS_TOL", "1e-6"))  # reachable FOM tol -> warm-start early-stop
const OUTDIR = joinpath(@__DIR__, "results", "continuation_5h_$(CONT_METHOD)")
gr();
mkpath(OUTDIR);
println(
    "Continuation 5h: THREADS=$THREADS backend=$BACKEND method=$CONT_METHOD n_points=$N_POINTS",
)

function build_5h()
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(@__DIR__, "data", "5h"))
    TEM.create_model!(ep)
    model = ep.model
    set_optimizer(model, make_optimizer(BACKEND, THREADS))
    set_silent(model)
    t = @elapsed JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    target = JuMP.VariableRef[]
    for sym in (:assets_investment, :flows_investment)
        if haskey(object_dictionary(model), sym)
            for v in object_dictionary(model)[sym]
                push!(target, v)
            end
        end
    end
    println(
        "  built+solved 5h in $(round(t,digits=1))s | $(num_variables(model)) vars | $(length(target)) target vars",
    )
    return model, target
end

model, target = build_5h()
allv = all_variables(model)
idx = Dict(v => i for (i, v) in enumerate(allv))

# MGA direction: penalise the structural variables by their optimal value over
# upper bound (the SPORES-style initial weighting), as a representative direction.
w = zeros(length(allv))
for v in target
    ub = has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0
    w[idx[v]] = value(v) / ub
end

println("Running continuation_walk (warm-started budget sweep) ...")
t_walk = @elapsed r = continuation_walk(
    model,
    w,
    allv;
    eps_slack = EPS_MGA,
    n_points = N_POINTS,
    method = CONT_METHOD,
    max_iters = 25,
    max_inner = 500,
    pdhg_iters = PDHG_ITERS,
    osqp_iters = OSQP_ITERS,
    feas_tol = FEAS_TOL,
    verbose = true,
)

iters = [p.iters for p in r.points]
println("\n=== continuation_walk on data/5h ===")
@printf(
    "points=%d  solves=%d  total_outer_iters=%d  wall=%.1fs\n",
    length(r.points),
    r.n_solves,
    r.total_iters,
    t_walk
)
@printf(
    "outer iters/solve: min=%d median=%.1f max=%d\n",
    minimum(iters),
    iters[cld(length(iters), 2)],
    maximum(iters)
)
@printf(
    "TIME  lp_setup=%.2fs  total_solve(API)=%.2fs  per-solve median=%.2fs\n",
    r.lp_setup_time,
    r.total_solve_time,
    sort([p.time for p in r.points])[cld(length(r.points), 2)]
)
@printf("max infeas over front: %.2e\n", maximum(p.infeas for p in r.points))

# Matched cold-IPM baseline: same interior budgets, measuring solver API time.
ipm = nothing
if get(ENV, "RUN_IPM", "1") == "1"
    budgets = [p.budget for p in r.points[2:end]]
    println("Running cold-IPM baseline at the same $(length(budgets)) budgets ...")
    ipm = ipm_baseline_at_budgets(
        model,
        w,
        allv,
        budgets;
        optimizer_factory = make_optimizer(BACKEND, THREADS),
    )
    @printf(
        "IPM  total_solve(API)=%.2fs  per-solve median=%.2fs   (FOM was %.2fs)\n",
        ipm.total_time,
        sort(ipm.times)[cld(length(ipm.times), 2)],
        r.total_solve_time
    )
    # Sanity: FOM and IPM should agree on the diversity objective at each budget.
    dmax = maximum(abs.(ipm.dobjs .- [p.dobj for p in r.points[2:end]]))
    @printf("max |FOM dobj - IPM dobj| over front: %.2e  (optimality check)\n", dmax)
end

open(joinpath(OUTDIR, "front.csv"), "w") do io
    println(io, "budget,cost,dobj,infeas,iters,solve_time")
    for p in r.points
        @printf(
            io,
            "%.6g,%.6g,%.6g,%.3e,%d,%.4f\n",
            p.budget,
            p.cost,
            p.dobj,
            p.infeas,
            p.iters,
            p.time
        )
    end
end
if ipm !== nothing
    open(joinpath(OUTDIR, "ipm_baseline.csv"), "w") do io
        println(io, "budget,cost,dobj,solve_time")
        for (B, c, d, t) in
            zip([p.budget for p in r.points[2:end]], ipm.costs, ipm.dobjs, ipm.times)
            @printf(io, "%.6g,%.6g,%.6g,%.4f\n", B, c, d, t)
        end
    end
end

p = plot(
    [pt.cost for pt in r.points],
    [pt.dobj for pt in r.points];
    marker = :circle,
    lw = 2,
    legend = false,
    xlabel = "original cost  c'x",
    ylabel = "diversity objective  w'x",
    title = "Near-optimal front (continuation_walk, data/5h)",
    size = (760, 560),
)
savefig(p, joinpath(OUTDIR, "front_5h.png"))
println("Wrote results to $OUTDIR")
