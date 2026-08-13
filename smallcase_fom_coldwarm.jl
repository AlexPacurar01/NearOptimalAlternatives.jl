# ===========================================================================
# Cold vs warm start for every corrector on the same near-optimal sweep.
#
# For a fixed budget grid we solve the MGA sub-problem at each budget TWICE:
#   * cold - every budget solved from scratch (origin / no seed);
#   * warm - each budget seeded from the previous solve (primal, and duals /
#            ADMM state where the method carries them).
# and compare total time, per-step work, feasibility, and front quality against
# an exact HiGHS interior-point reference. Methods:
#   gradient correctors  :alm_lbfgs, :penalty, :pdhg   (solve_firstorder)
#   exact-solve corrector SCS (tight tolerance, so it returns FEASIBLE points)
#   barrier              HiGHS interior point (public solver, not Gurobi)
#
# Feasibility is the max ORIGINAL-unit violation of the model constraints
# (primal_feasibility_report), so every method is judged on the same footing.
#
#   Usage:  julia -t 4 --project=. smallcase_fom_coldwarm.jl [case ...]
#     env:  CW_NPOINTS (default 5), CW_SCS_EPS (default 1e-8, tight = feasible)
# ===========================================================================
using HiGHS, Printf, Statistics, LinearAlgebra
using Plots
include("smallcase_common.jl")
include("bench_common.jl")            # dominance_report

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)
const CASES = isempty(ARGS) ? ["synth"] : ARGS
if any(c -> c in ("Tiny", "Norse"), CASES)
    include("smallcase_tulipa.jl")
end

# Public interior-point baseline (HiGHS, never Gurobi). Default (auto) mode is
# robust on the degenerate MGA sub-problem; it is the exact reference and the
# "barrier" timing point.
ipm() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
scs_opt(eps_tol) = optimizer_with_attributes(
    SCS.Optimizer,
    "verbose" => 0,
    "eps_abs" => eps_tol,
    "eps_rel" => eps_tol,
    "max_iters" => 1_000_000,
)

const N_POINTS = parse(Int, get(ENV, "CW_NPOINTS", "5"))
const SCS_EPS = parse(Float64, get(ENV, "CW_SCS_EPS", "1e-8"))   # tight -> feasible
const EPS_MGA = 0.1

make_case(name) =
    name == "synth" ? build_synth_model(ipm(); n_struct = 8, T = 24, n_store = 4) :
    name in ("Tiny", "Norse") ? load_tulipa_case(name, ipm()) :
    error("unknown case '$name'")

function mga_weight(model, target)
    all_vars = all_variables(model)
    idx = Dict(v => i for (i, v) in enumerate(all_vars))
    xstar = value.(all_vars)
    w = zeros(length(all_vars))
    for v in target
        ub = has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0
        w[idx[v]] = xstar[idx[v]] / ub
    end
    return w, all_vars
end

"Max ORIGINAL-unit violation of the model constraints at point `xvec`."
function max_violation(model, all_vars, xvec)
    pt = Dict(all_vars[i] => xvec[i] for i in eachindex(all_vars))
    rep = primal_feasibility_report(model, pt)
    isempty(rep) ? 0.0 : maximum(values(rep))
end

# --- gradient-corrector sweep (in-house solve_firstorder), cold or warm -------
function firstorder_sweep(
    model,
    w,
    all_vars,
    budgets,
    method;
    warm::Bool,
    max_iters = 60,
    max_inner = 1000,
    pdhg_iters = 20000,
)
    B_max = maximum(budgets)
    lp = with_logger(NullLogger()) do
        build_mga_lp(model, copy(w), B_max, all_vars)
    end
    orig = objective_function(model)
    cvec = [coefficient(orig, v) for v in all_vars]
    off = JuMP.constant(orig)
    what = w ./ max(norm(w), eps())
    bidx = size(lp.A_in, 1)
    rbud = (B_max - off) / lp.b_in[bidx]
    bbase = copy(lp.b_in)
    setb(B) = (b = copy(bbase); b[bidx] = (B - off) / rbud; b)
    x0origin = clamp.(zeros(lp.n), lp.lb_t, lp.ub_t)
    prevx = nothing
    yin = Float64[]
    yeq = Float64[]
    pts = NamedTuple[]
    tott = 0.0
    for B in budgets
        lpl = merge(lp, (b_in = setb(B),))
        x0 = (warm && prevx !== nothing) ? copy(prevx) : copy(x0origin)
        x_t, info = with_logger(NullLogger()) do
            solve_firstorder(
                method,
                lpl,
                x0;
                y_in0 = warm ? yin : Float64[],
                y_eq0 = warm ? yeq : Float64[],
                max_iters = max_iters,
                max_inner = max_inner,
                pdhg_iters = pdhg_iters,
                verbose = false,
            )
        end
        x = lp.d_scale .* x_t
        push!(
            pts,
            (
                budget = B,
                cost = dot(cvec, x) + off,
                dobj = dot(what, x),
                viol = max_violation(model, all_vars, x),
                iters = info.iters,
                time = info.time,
            ),
        )
        tott += info.time
        prevx = x_t
        yin = info.y_in
        yeq = info.y_eq
    end
    return (points = pts, total_time = tott)
end

# --- JuMP-solver sweep (SCS or HiGHS barrier), cold or warm -------------------
function jump_sweep(model, w, all_vars, budgets, opt_factory; warm::Bool)
    orig = objective_function(model)
    sense = objective_sense(model)
    cvec = [coefficient(orig, v) for v in all_vars]
    off = JuMP.constant(orig)
    what = w ./ max(norm(w), eps())
    set_optimizer(model, opt_factory)
    bcon = @constraint(model, orig <= budgets[1])
    @objective(model, Min, sum(w[i] * all_vars[i] for i in eachindex(w) if w[i] != 0))
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    pts = NamedTuple[]
    tott = 0.0
    prevx = nothing
    prevd = nothing
    for B in budgets
        set_normalized_rhs(bcon, B - off)
        if warm && prevx !== nothing
            set_start_value.(all_vars, prevx)
            for (c, dv) in prevd
                try
                    set_dual_start_value(c, dv)
                catch
                end
            end
        end
        optimize!(model)
        x = value.(all_vars)
        prevd = Dict(c => dual(c) for c in cons)
        prevx = x
        st = solve_time(model)
        tott += st
        push!(
            pts,
            (
                budget = B,
                cost = dot(cvec, x) + off,
                dobj = dot(what, x),
                viol = max_violation(model, all_vars, x),
                iters = 0,
                time = st,
            ),
        )
    end
    delete(model, bcon)
    set_objective_function(model, orig)
    set_objective_sense(model, sense)
    return (points = pts, total_time = tott)
end

function run_case(name)
    println("\n########## Cold vs warm: $name (SCS eps=$SCS_EPS, barrier=HiGHS) ##########")
    model, target = make_case(name)
    w, all_vars = mga_weight(model, target)
    min_cost = objective_value(model)
    budgets = [min_cost + EPS_MGA * abs(min_cost) * i / N_POINTS for i = 1:N_POINTS]

    runs = Tuple{String,Any}[]           # (label, sweep result)
    for m in (:alm_lbfgs, :penalty, :pdhg)
        push!(
            runs,
            ("$m cold", firstorder_sweep(model, w, all_vars, budgets, m; warm = false)),
        )
        push!(
            runs,
            ("$m warm", firstorder_sweep(model, w, all_vars, budgets, m; warm = true)),
        )
    end
    push!(
        runs,
        (
            "scs cold",
            jump_sweep(model, w, all_vars, budgets, scs_opt(SCS_EPS); warm = false),
        ),
    )
    push!(
        runs,
        (
            "scs warm",
            jump_sweep(model, w, all_vars, budgets, scs_opt(SCS_EPS); warm = true),
        ),
    )
    push!(
        runs,
        ("barrier cold", jump_sweep(model, w, all_vars, budgets, ipm(); warm = false)),
    )
    push!(
        runs,
        ("barrier warm", jump_sweep(model, w, all_vars, budgets, ipm(); warm = true)),
    )
    set_optimizer(model, ipm())

    # Exact front = the barrier-cold sweep (ground truth).
    bc = runs[findfirst(r -> r[1] == "barrier cold", runs)][2]
    ecost = [p.cost for p in bc.points]
    edobj = [p.dobj for p in bc.points]

    @printf(
        "\n%-14s %10s %9s %11s %8s\n",
        "method/start",
        "time(s)",
        "meanIt",
        "maxViol",
        "domByEx%"
    )
    println("-"^60)
    rows = String["case,method,start,total_time_s,mean_iters,max_viol,frac_dominated"]
    for (label, res) in runs
        its = [p.iters for p in res.points if p.iters > 0]
        meanit = isempty(its) ? 0.0 : mean(its)
        maxv = maximum(p.viol for p in res.points)
        dom = dominance_report(
            ecost,
            edobj,
            [p.cost for p in res.points],
            [p.dobj for p in res.points],
        )
        @printf(
            "%-14s %10.2f %9.1f %11.2e %8.0f\n",
            label,
            res.total_time,
            meanit,
            maxv,
            100 * dom.frac_dominated
        )
        meth, start = split(label, " ")
        push!(
            rows,
            @sprintf(
                "%s,%s,%s,%.4f,%.1f,%.3e,%.3f",
                name,
                meth,
                start,
                res.total_time,
                meanit,
                maxv,
                dom.frac_dominated
            )
        )
    end
    open(joinpath(OUTDIR, "coldwarm_$name.csv"), "w") do io
        println(io, join(rows, "\n"))
    end

    # Cold-vs-warm total-time bars, dodged by method (base Plots, no StatsPlots).
    labels = ["alm_lbfgs", "penalty", "pdhg", "scs", "barrier"]
    tcold = [runs[findfirst(r -> r[1] == "$m cold", runs)][2].total_time for m in labels]
    twarm = [runs[findfirst(r -> r[1] == "$m warm", runs)][2].total_time for m in labels]
    xs = collect(1:length(labels))
    p = bar(
        xs .- 0.2,
        tcold;
        bar_width = 0.4,
        label = "cold",
        color = :gray70,
        ylabel = "total sweep time (s)",
        xticks = (xs, labels),
        xrotation = 20,
        title = "Cold vs warm start ($name, $(N_POINTS) budgets)",
        size = (900, 520),
        legend = :topright,
    )
    bar!(p, xs .+ 0.2, twarm; bar_width = 0.4, label = "warm", color = :seagreen)
    savefig(p, joinpath(OUTDIR, "coldwarm_$name.png"))
    println("wrote coldwarm_$name.csv, coldwarm_$name.png")
end

# JIT warm-up.
let (m, tv) = build_synth_model(ipm(); n_struct = 4, T = 4, n_store = 1)
    w, av = mga_weight(m, tv)
    bud = [objective_value(m) * 1.05]
    firstorder_sweep(m, w, av, bud, :alm_lbfgs; warm = false)
    jump_sweep(m, w, av, bud, scs_opt(1e-6); warm = false)
end

for c in CASES
    run_case(c)
end
