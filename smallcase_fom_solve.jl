# ===========================================================================
# Part A - single-solve comparison of the first-order MGA correctors.
#
# For each small case, build the first-alternative MGA LP once and run all four
# correctors (:penalty/QP, :alm_lbfgs/ALM, :pdhg, :osqp/ADMM) on the identical
# Ruiz-scaled problem, COLD (from the origin), graded against an exact
# interior-point reference. This answers "which correctors reach the exact MGA
# point, and at what cost (time / iterations)" on instances small enough to be
# certified correct (see smallcase_fom_correctness.jl).
#
# Outputs (results/fom_smallcase/):
#   solve_metrics_<case>.csv        per-method metrics
#   solve_bars_<case>.png           miss% / time / iterations bar panels
#   solve_trace_<case>.png          convergence: infeasibility & objective vs
#                                   wall-clock, with the exact optimum overlaid
#
#   Usage:  julia -t 4 --project=. smallcase_fom_solve.jl [case ...]
#     cases: synth (default). Tiny/Norse are added after the Tulipa upgrade via
#            smallcase_tulipa.jl (loaded on demand).
#     env:   SOLVE_REPEATS (default 3) - every solve is repeated and the
#            REPORTED time is the median, so single-run timing noise cannot
#            masquerade as a method difference.
# ===========================================================================
using HiGHS, Printf, Statistics
include("smallcase_common.jl")
include("smallcase_plots.jl")

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)

const CASES = isempty(ARGS) ? ["synth"] : ARGS
# TulipaEnergyModel is heavy; load its adapter only if a real dataset is asked
# for, and at top level (not inside a function at runtime) to respect Julia
# 1.12 world-age rules.
if any(c -> c in ("Tiny", "Norse"), CASES)
    include("smallcase_tulipa.jl")
end

# Robust exact/base solver: HiGHS default (auto solver + crossover) returns a
# clean optimum even on the degenerate MGA sub-problem, where a bare IPM without
# crossover can report a non-optimal status. The exact reference is ground truth,
# so robustness matters more than the interior-vs-vertex distinction; the primary
# metric (objective miss%) is identical on the optimal face either way.
ipm() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

const REPEATS = parse(Int, get(ENV, "SOLVE_REPEATS", "3"))

# --- case registry: name -> (model, target_vars) ---------------------------
function make_case(name::AbstractString)
    if name == "synth"
        return build_synth_model(ipm(); n_struct = 8, T = 24, n_store = 4)
    elseif name in ("Tiny", "Norse")
        return load_tulipa_case(name, ipm())
    else
        error("unknown case '$name'")
    end
end

# --- one case ---------------------------------------------------------------
function run_case(name; eps = 0.1)
    println("\n########## Part A single-solve: $name ##########")
    model, target = make_case(name)
    s = mga_setup(model, target; eps = eps)
    # Median-of-REPEATS timing: correctors and the exact reference are all
    # deterministic, so only the wall-clock varies; the median filters noise.
    ref = exact_reference(model, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    ref_times = [ref.time]
    all_runs = Vector{Any}[]
    local results, lp
    for rep = 1:REPEATS
        results, lp = run_correctors(model, s, ref; from = :origin, trace = true)
        push!(results, solve_scs_mga(model, s, ref, lp))
        push!(all_runs, results)
        if rep < REPEATS
            # solve_scs_mga leaves SCS attached; the reference must be HiGHS.
            set_optimizer(model, ipm())
            r2 = exact_reference(model, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
            push!(ref_times, r2.time)
        end
    end
    results = [
        merge(results[i], (time = median(run[i].time for run in all_runs),)) for
        i in eachindex(results)
    ]
    ref = merge(ref, (time = median(ref_times),))
    println("(timings are medians of $REPEATS runs)")
    nvars = length(s.all_vars)
    obj_ref_scaled = dot(lp.w_t, ref.x ./ lp.d_scale)

    @printf(
        "\n%-34s %8s %7s %11s %10s %9s %8s %10s\n",
        "method",
        "time(s)",
        "iters",
        "infeas",
        "miss%",
        "soldist",
        "dominated",
        "w_s'x"
    )
    println("-"^108)
    rows =
        String["case,nvars,method,time_s,iters,infeas,miss_pct,sol_dist,dominated,obj_s,exact_obj_s"]
    for r in results
        @printf(
            "%-34s %8.3f %7d %11.2e %10.4f %9.3e %8s %10.5g\n",
            r.label,
            r.time,
            r.iters,
            r.infeas,
            r.gap_s,
            r.sol_dist,
            r.dominated ? "yes" : "no",
            r.obj_s
        )
        push!(
            rows,
            @sprintf(
                "%s,%d,%s,%.4f,%d,%.3e,%.5f,%.4e,%s,%.6g,%.6g",
                name,
                nvars,
                r.method,
                r.time,
                r.iters,
                r.infeas,
                r.gap_s,
                r.sol_dist,
                r.dominated ? "yes" : "no",
                r.obj_s,
                ref.obj_s
            )
        )
    end
    @printf(
        "%-34s %8.3f %7s %11s %10.4f %9.1e %8s %10.5g\n",
        "Exact IPM (reference)",
        ref.time,
        "-",
        "0",
        0.0,
        0.0,
        "-",
        ref.obj_s
    )

    open(joinpath(OUTDIR, "solve_metrics_$name.csv"), "w") do io
        println(io, join(rows, "\n"))
    end
    plot_corrector_bars(
        joinpath(OUTDIR, "solve_bars_$name.png"),
        "First-order correctors, single solve ($name)",
        results,
    )
    plot_corrector_trace(
        joinpath(OUTDIR, "solve_trace_$name.png"),
        "Convergence of gradient correctors ($name)",
        results,
        obj_ref_scaled,
    )
    println("wrote solve_metrics_$name.csv, solve_bars_$name.png, solve_trace_$name.png")
end

# JIT warm-up on a tiny instance so timings are compile-free (including SCS).
let (m, tv) = build_synth_model(ipm(); n_struct = 4, T = 4, n_store = 1)
    s = mga_setup(m, tv)
    r = exact_reference(m, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    _, lpw = run_correctors(m, s, r; from = :origin, max_iters = 4, pdhg_iters = 300)
    solve_scs_mga(m, s, r, lpw)
end

for c in CASES
    run_case(c)
end
