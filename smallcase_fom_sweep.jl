# ===========================================================================
# Part C - front tracing: can the correctors sweep the near-optimal front?
#
# The single-solve study (Part A) shows WHICH correctors reach the exact MGA
# point. Here we test the actual use-case: a warm-started epsilon-constraint
# CONTINUATION that sweeps the cost budget from x* up to (1+eps)C*, tracing the
# cost-vs-diversity front. Each corrector is run through `continuation_walk`
# (warm-starting primal + duals / ADMM state from the previous budget) and its
# recovered front is overlaid on an EXACT cold interior-point baseline
# (`ipm_baseline_at_budgets`). A corrector that "works" reproduces the exact
# front; a stalling one traces a front that sits ABOVE it (dominated), and the
# warm start cannot rescue it.
#
# We also probe whether the exact baseline itself can exploit a DUAL-only warm
# start (its primal is useless on the boundary, but the shadow prices might
# carry): solve two neighbouring budgets, cold vs dual-seeded, and compare.
#
# Outputs (results/fom_smallcase/):
#   sweep_front_<case>.png     recovered fronts vs the exact baseline
#   sweep_iters_<case>.png     per-step warm-started iteration count
#   sweep_<case>.csv           per (method, point) cost/dobj/iters/time
#   sweep_summary_<case>.csv   per-method totals + front dominance vs exact
#
#   Usage:  julia -t 4 --project=. smallcase_fom_sweep.jl [case ...]   (default synth)
# ===========================================================================
using HiGHS, Printf, Statistics, LinearAlgebra
using Plots
include("smallcase_common.jl")
include("smallcase_plots.jl")
include("bench_common.jl")            # ipm_baseline_at_budgets, dominance helpers

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)

const CASES = isempty(ARGS) ? ["synth"] : ARGS
if any(c -> c in ("Tiny", "Norse"), CASES)
    include("smallcase_tulipa.jl")
end

ipm() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# Configurable so the larger Norse instance stays tractable: fewer front points
# and a looser SCS tolerance (so SCS actually converges per budget, making the
# warm-vs-cold comparison meaningful rather than both hitting an iteration cap).
const N_POINTS = parse(Int, get(ENV, "SWEEP_NPOINTS", "8"))
const SCS_EPS = parse(Float64, get(ENV, "SWEEP_SCS_EPS", "1e-7"))
const EPS_MGA = 0.1

make_case(name) =
    name == "synth" ? build_synth_model(ipm(); n_struct = 8, T = 24, n_store = 4) :
    name in ("Tiny", "Norse") ? load_tulipa_case(name, ipm()) :
    error("unknown case '$name'")

# HSJ first-direction full-space weight (as the gradient driver builds it).
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

# --- dual-only warm-start probe on the exact IPM baseline -------------------
# Solve the MGA sub-problem at two neighbouring budgets; the second is solved
# cold and then again with the first solve's constraint duals seeded
# (set_dual_start_value). Reports both solver-reported times. Barrier IPMs
# typically ignore a warm basis, so this quantifies exactly how much (if any) a
# dual seed buys - part of the motivation for a warm-startable FOM corrector.
function dual_warmstart_probe(model, w, all_vars, B1, B2)
    orig = objective_function(model)
    sense = objective_sense(model)
    bcon = @constraint(model, orig <= B1)
    @objective(model, Min, sum(w[i] * all_vars[i] for i in eachindex(w) if w[i] != 0))
    optimize!(model)
    duals = Dict(
        c => dual(c) for
        c in all_constraints(model; include_variable_in_set_constraints = false)
    )
    # cold solve at B2
    set_normalized_rhs(bcon, B2 - JuMP.constant(orig))
    optimize!(model)
    t_cold = solve_time(model)
    # dual-seeded solve at B2
    for (c, dv) in duals
        try
            set_dual_start_value(c, dv)
        catch
        end
    end
    optimize!(model)
    t_warm = solve_time(model)
    delete(model, bcon)
    set_objective_function(model, orig)
    set_objective_sense(model, sense)
    return (t_cold = t_cold, t_warm = t_warm)
end

# --- warm-started SCS front (the exact-linear-solve corrector's continuation) --
# continuation_walk only drives the in-house solve_firstorder methods, so we trace
# the SCS front here directly through JuMP: sweep the budget, and when `warm`,
# seed each solve with the previous point's primal and constraint duals (the same
# primal-dual warm start the continuation exploits). Returns per-budget points and
# the total solve time; run cold too to quantify the warm-start saving.
function scs_front(model, w, all_vars, budgets; warm::Bool, eps_tol::Float64 = 1.0e-7)
    orig = objective_function(model)
    sense = objective_sense(model)
    cvec = [coefficient(orig, v) for v in all_vars]
    off = JuMP.constant(orig)
    what = w ./ max(norm(w), eps())
    set_optimizer(
        model,
        optimizer_with_attributes(
            SCS.Optimizer,
            "verbose" => 0,
            "eps_abs" => eps_tol,
            "eps_rel" => eps_tol,
            "max_iters" => 200_000,
        ),
    )
    bcon = @constraint(model, orig <= budgets[1])
    @objective(model, Min, sum(w[i] * all_vars[i] for i in eachindex(w) if w[i] != 0))
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    pts = NamedTuple[]
    total_t = 0.0
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
        total_t += st
        push!(pts, (budget = B, cost = dot(cvec, x) + off, dobj = dot(what, x), time = st))
    end
    delete(model, bcon)
    set_objective_function(model, orig)
    set_objective_sense(model, sense)
    return (points = pts, total_time = total_t)
end

function run_case(name)
    println("\n########## Part C front tracing: $name ##########")
    model, target = make_case(name)
    w, all_vars = mga_weight(model, target)
    min_cost = objective_value(model)
    budgets = [min_cost + EPS_MGA * abs(min_cost) * i / N_POINTS for i = 1:N_POINTS]

    # Each corrector's warm-started continuation front. Run these BEFORE the exact
    # baseline: continuation_walk reads objective_value(model)=C* and never mutates
    # the model, whereas ipm_baseline_at_budgets temporarily swaps the objective.
    methods = [:alm_lbfgs, :penalty, :pdhg]      # gradient correctors via continuation_walk
    fronts = Dict{Symbol,Any}()
    for m in methods
        res = continuation_walk(
            model,
            copy(w),
            all_vars;
            eps_slack = EPS_MGA,
            n_points = N_POINTS,
            method = m,
            verbose = false,
        )
        fronts[m] = res
    end

    # Exact cold-IPM baseline on an even budget grid (the ground-truth front).
    base = ipm_baseline_at_budgets(
        model,
        copy(w),
        all_vars,
        copy(budgets);
        optimizer_factory = ipm(),
    )
    println(
        "exact baseline: $(N_POINTS) cold solves in $(round(base.total_time, digits=3))s",
    )

    # SCS front, warm-started and cold, on the same budget grid (the exact-solve
    # corrector the thesis adopts). Restore the IPM optimizer afterwards.
    scs_warm = scs_front(model, copy(w), all_vars, budgets; warm = true, eps_tol = SCS_EPS)
    scs_cold = scs_front(model, copy(w), all_vars, budgets; warm = false, eps_tol = SCS_EPS)
    set_optimizer(model, ipm())

    rows = String["case,method,point,budget,cost,dobj,infeas,iters,time_s"]
    srows =
        String["case,method,n_solves,total_iters,mean_iters,total_time_s,max_infeas,frac_front_dominated_by_exact"]
    for m in methods
        res = fronts[m]
        pts = res.points
        # Dominance of this front vs the exact baseline (cost, dobj): fraction of
        # the corrector's points that some exact point dominates.
        cand_cost = [p.cost for p in pts]
        cand_dobj = [p.dobj for p in pts]
        dom = dominance_report(base.costs, base.dobjs, cand_cost, cand_dobj)
        itervec = [p.iters for p in pts if p.iters > 0]
        mean_it = isempty(itervec) ? 0.0 : mean(itervec)
        maxinf = maximum(p.infeas for p in pts)
        @printf(
            "  %-11s solves=%2d total_iters=%7d mean/step=%7.1f time=%6.2fs maxinfeas=%.1e domByExact=%.0f%%\n",
            string(m),
            res.n_solves,
            res.total_iters,
            mean_it,
            res.total_solve_time,
            maxinf,
            100 * dom.frac_dominated
        )
        push!(
            srows,
            @sprintf(
                "%s,%s,%d,%d,%.1f,%.4f,%.2e,%.3f",
                name,
                m,
                res.n_solves,
                res.total_iters,
                mean_it,
                res.total_solve_time,
                maxinf,
                dom.frac_dominated
            )
        )
        for (i, p) in enumerate(pts)
            push!(
                rows,
                @sprintf(
                    "%s,%s,%d,%.6g,%.6g,%.6g,%.2e,%d,%.4f",
                    name,
                    m,
                    i,
                    p.budget,
                    p.cost,
                    p.dobj,
                    p.infeas,
                    p.iters,
                    p.time
                )
            )
        end
    end

    # SCS front summary: dominance vs exact, and the warm-start time saving.
    scs_dom = dominance_report(
        base.costs,
        base.dobjs,
        [p.cost for p in scs_warm.points],
        [p.dobj for p in scs_warm.points],
    )
    @printf(
        "  %-11s solves=%2d %-20s time(warm)=%6.2fs time(cold)=%6.2fs (x%.2f) domByExact=%.0f%%\n",
        "scs",
        length(scs_warm.points),
        "",
        scs_warm.total_time,
        scs_cold.total_time,
        scs_cold.total_time / max(scs_warm.total_time, 1e-9),
        100 * scs_dom.frac_dominated
    )
    # Per-budget SCS solve time, warm vs cold (the warm-start effect step by step).
    println("    SCS per-step time (s):  budget |   cold |   warm")
    for i = 1:length(scs_warm.points)
        @printf(
            "      %8.4g | %6.2f | %6.2f\n",
            scs_warm.points[i].budget,
            scs_cold.points[i].time,
            scs_warm.points[i].time
        )
    end
    push!(
        srows,
        @sprintf(
            "%s,%s,%d,%d,%.1f,%.4f,%.2e,%.3f",
            name,
            "scs",
            length(scs_warm.points),
            0,
            0.0,
            scs_warm.total_time,
            NaN,
            scs_dom.frac_dominated
        )
    )
    for (i, p) in enumerate(scs_warm.points)
        push!(
            rows,
            @sprintf(
                "%s,%s,%d,%.6g,%.6g,%.6g,%.2e,%d,%.4f",
                name,
                "scs",
                i,
                p.budget,
                p.cost,
                p.dobj,
                NaN,
                0,
                0.0
            )
        )
    end

    # Dual warm-start probe on two neighbouring budgets.
    probe = dual_warmstart_probe(model, w, all_vars, budgets[end-1], budgets[end])
    println(
        @sprintf(
            "  dual-warmstart probe (exact IPM): cold=%.4fs  dual-seeded=%.4fs  (speedup x%.2f)",
            probe.t_cold,
            probe.t_warm,
            probe.t_cold / max(probe.t_warm, 1e-9)
        )
    )

    open(joinpath(OUTDIR, "sweep_$name.csv"), "w") do io
        println(io, join(rows, "\n"))
    end
    open(joinpath(OUTDIR, "sweep_summary_$name.csv"), "w") do io
        println(io, join(srows, "\n"))
    end

    # --- front plot: dobj vs cost, correctors over the exact baseline --------
    pf = plot(;
        xlabel = "implementation cost  c'x",
        ylabel = "diversity objective  wᵀx",
        title = "Near-optimal front tracing ($name): correctors vs exact baseline",
        legend = :topright,
        size = (900, 600),
        left_margin = 6Plots.mm,
        bottom_margin = 6Plots.mm,
    )
    plot!(
        pf,
        base.costs,
        base.dobjs;
        lw = 3,
        color = :black,
        ls = :dash,
        marker = :star5,
        ms = 6,
        label = "exact IPM (cold)",
    )
    for m in methods
        pts = fronts[m].points
        plot!(
            pf,
            [p.cost for p in pts],
            [p.dobj for p in pts];
            lw = 2,
            color = CORRECTOR_COLOR[m],
            marker = :circle,
            ms = 4,
            label = string(m),
        )
    end
    plot!(
        pf,
        [p.cost for p in scs_warm.points],
        [p.dobj for p in scs_warm.points];
        lw = 2,
        color = CORRECTOR_COLOR[:scs],
        marker = :diamond,
        ms = 5,
        label = "scs (warm)",
    )
    savefig(pf, joinpath(OUTDIR, "sweep_front_$name.png"))

    # --- per-step warm-started iteration count -------------------------------
    pi = plot(;
        xlabel = "front point (budget step)",
        ylabel = "solver iterations",
        title = "Warm-started iterations per step ($name)",
        legend = :topright,
        size = (900, 520),
        yscale = :log10,
    )
    for m in methods
        pts = fronts[m].points
        ys = [max(p.iters, 1) for p in pts[2:end]]   # skip x* (0 iters)
        plot!(
            pi,
            1:length(ys),
            ys;
            lw = 2,
            color = CORRECTOR_COLOR[m],
            marker = :circle,
            ms = 4,
            label = string(m),
        )
    end
    savefig(pi, joinpath(OUTDIR, "sweep_iters_$name.png"))
    println(
        "wrote sweep_front_$name.png, sweep_iters_$name.png, sweep_$name.csv, sweep_summary_$name.csv",
    )
end

# JIT warm-up (tiny synth) to keep the timings compile-free.
let (m, tv) = build_synth_model(ipm(); n_struct = 4, T = 4, n_store = 1)
    w, av = mga_weight(m, tv)
    continuation_walk(m, copy(w), av; eps_slack = 0.1, n_points = 3, method = :alm_lbfgs)
end

for c in CASES
    run_case(c)
end
