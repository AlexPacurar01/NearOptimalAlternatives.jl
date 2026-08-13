# ===========================================================================
# Three-way comparison on data/5h, ONE process / ONE model build:
#   (1) warm-started ALM continuation,
#   (2) warm-started PDHG continuation,
#   (3) cold Gurobi-barrier IPM at the same budgets.
# Reports API-level per-solve time (solver loop / JuMP.solve_time, NOT model
# build or LP extraction), front accuracy vs the IPM ground truth, and writes a
# combined CSV + overlay plot.
#
#   N_POINTS=5 PDHG_ITERS=40000 julia -t 8 --project=. benchmark_compare_5h.jl
# Multi-hour: ~ALM 10min/pt + PDHG up to ~18min/pt + IPM ~3min/pt.
# ===========================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, LinearAlgebra, Printf
using Plots
using NearOptimalAlternatives

include(joinpath(@__DIR__, "bench_common.jl"))

const THREADS = max(1, Threads.nthreads())
const EPS_MGA = 0.1
const BACKEND = :gurobi
const N_POINTS = parse(Int, get(ENV, "N_POINTS", "5"))
const PDHG_ITERS = parse(Int, get(ENV, "PDHG_ITERS", "40000"))
const FEAS_TOL = parse(Float64, get(ENV, "FEAS_TOL", "1e-6"))
const OUTDIR = joinpath(@__DIR__, "results", "compare_5h")
gr();
mkpath(OUTDIR);
println("Compare 5h: THREADS=$THREADS N=$N_POINTS PDHG_ITERS=$PDHG_ITERS")

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
w = zeros(length(allv))
for v in target
    ub = has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0
    w[idx[v]] = value(v) / ub
end

runwalk(method) = continuation_walk(
    model,
    w,
    allv;
    eps_slack = EPS_MGA,
    n_points = N_POINTS,
    method = method,
    max_iters = 25,
    max_inner = 500,
    pdhg_iters = PDHG_ITERS,
    feas_tol = FEAS_TOL,
    verbose = true,
)

println("\n>>> [1/3] warm ALM continuation ...")
r_alm = runwalk(:alm_lbfgs)
println("\n>>> [2/3] warm PDHG continuation ...")
r_pdhg = runwalk(:pdhg)

# Cold IPM at the ALM budgets (mutates the model -> run LAST).
println("\n>>> [3/3] cold IPM baseline at ALM budgets ...")
budgets = [p.budget for p in r_alm.points[2:end]]
ipm = ipm_baseline_at_budgets(
    model,
    w,
    allv,
    budgets;
    optimizer_factory = make_optimizer(BACKEND, THREADS),
)

med(v) = isempty(v) ? 0.0 : sort(collect(v))[cld(length(v), 2)]
println("\n================  data/5h three-way comparison  ================")
@printf(
    "%-10s %7s %12s %12s %10s\n",
    "method",
    "points",
    "totSolve(s)",
    "medSolve(s)",
    "maxInfeas"
)
@printf(
    "%-10s %7d %12.1f %12.2f %10.2e\n",
    "ALM(warm)",
    length(r_alm.points),
    r_alm.total_solve_time,
    med([p.time for p in r_alm.points if p.time > 0]),
    maximum(p.infeas for p in r_alm.points)
)
@printf(
    "%-10s %7d %12.1f %12.2f %10.2e\n",
    "PDHG(warm)",
    length(r_pdhg.points),
    r_pdhg.total_solve_time,
    med([p.time for p in r_pdhg.points if p.time > 0]),
    maximum(p.infeas for p in r_pdhg.points)
)
@printf(
    "%-10s %7d %12.1f %12.2f %10s\n",
    "IPM(cold)",
    length(ipm.times),
    ipm.total_time,
    med(ipm.times),
    "exact"
)
@printf("\nLP setup (one-time, build_mga_lp): %.2fs\n", r_alm.lp_setup_time)
# Optimality check: each continuation's dobj vs cold-IPM at the same budgets.
alm_d = [p.dobj for p in r_alm.points[2:end]]
@printf(
    "max |ALM dobj  - IPM dobj| : %.2e  (front correctness)\n",
    maximum(abs.(alm_d .- ipm.dobjs))
)

open(joinpath(OUTDIR, "compare.csv"), "w") do io
    println(io, "method,budget,cost,dobj,infeas,iters,solve_time")
    for (nm, r) in (("ALM", r_alm), ("PDHG", r_pdhg))
        for p in r.points
            @printf(
                io,
                "%s,%.6g,%.6g,%.6g,%.3e,%d,%.4f\n",
                nm,
                p.budget,
                p.cost,
                p.dobj,
                p.infeas,
                p.iters,
                p.time
            )
        end
    end
    for (B, c, d, t) in zip(budgets, ipm.costs, ipm.dobjs, ipm.times)
        @printf(io, "IPM,%.6g,%.6g,%.6g,%.3e,%d,%.4f\n", B, c, d, 0.0, 0, t)
    end
end

p = plot(;
    xlabel = "original cost  c'x",
    ylabel = "diversity  ŵ'x",
    title = "Near-optimal front, data/5h (N=$N_POINTS)",
    size = (820, 600),
    legend = :topright,
)
plot!(
    p,
    [pt.cost for pt in r_alm.points],
    [pt.dobj for pt in r_alm.points];
    marker = :circle,
    lw = 2,
    label = "ALM (warm)",
)
plot!(
    p,
    [pt.cost for pt in r_pdhg.points],
    [pt.dobj for pt in r_pdhg.points];
    marker = :diamond,
    lw = 2,
    ls = :dash,
    label = "PDHG (warm)",
)
scatter!(p, ipm.costs, ipm.dobjs; marker = :xcross, ms = 7, label = "IPM (cold)")
savefig(p, joinpath(OUTDIR, "front_compare.png"))
println("Wrote results to $OUTDIR")
