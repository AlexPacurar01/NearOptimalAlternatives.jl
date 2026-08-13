# ===========================================================================
# Part B - the mechanism: missed diversity vs operational-to-structural ratio.
#
# The thesis claim is that the gradient correctors stall at DOMINATED points on
# the full ESOM because the diversity objective acts on a vanishing fraction of
# the variables (~0.06 %) and is drowned out by the dense operational feasibility
# terms. We reproduce that mechanism ON SMALL, FAST INSTANCES by holding the
# structural (investment) block fixed and growing the operational block: more
# timesteps -> more dispatch variables and more capacity/storage rows per
# structural variable -> the same "sparse objective, dense feasibility" imbalance
# at a controllable scale.
#
# All correctors get the SAME fixed compute budget and are graded by the
# missed-diversity gap (% of the achievable diversity NOT captured) against an
# exact interior-point reference. Every instance is small and dense enough that
# the correctness gate passes at the low-ratio end, so a gap that GROWS with the
# ratio isolates the drowning mechanism (not a bug).
#
# Output (results/fom_smallcase/):
#   ratio_sweep.csv   per (ratio, method) metrics
#   ratio_gap.png     miss% vs operational/structural ratio (the headline figure)
#
#   Usage:  julia -t 4 --project=. smallcase_fom_ratio.jl
# ===========================================================================
using HiGHS, Printf, Statistics
using Plots
include("smallcase_common.jl")
include("smallcase_plots.jl")

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)

ipm() = optimizer_with_attributes(
    HiGHS.Optimizer,
    "output_flag" => false,
    "solver" => "ipm",
    "run_crossover" => "off",
)

# Fixed structural block; the operational block grows with T. Storage chains add
# the stiff inter-temporal equality rows that most drown the sparse objective.
const N_STRUCT = 6
const N_STORE = 4
const T_GRID = [4, 8, 16, 32, 64, 128, 256]

# Fixed, generous compute budget shared by every instance (so a growing gap is
# about the problem, not a shrinking budget).
const MAX_ITERS = 60      # ALM/penalty outer
const MAX_INNER = 1000
const PDHG_ITERS = 20000
const OSQP_ITERS = 8000

# The gradient correctors plus SCS, in report order.
const ALL_METHODS = vcat(CORRECTORS, [(:scs, SCS_LABEL)])

# JIT warm-up (incl. SCS).
let (m, tv) = build_synth_model(ipm(); n_struct = 4, T = 4, n_store = 1)
    s = mga_setup(m, tv)
    r = exact_reference(m, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    _, lpw = run_correctors(m, s, r; from = :origin, max_iters = 4, pdhg_iters = 300)
    solve_scs_mga(m, s, r, lpw)
end

rows =
    String["T,nvars,op_struct_ratio,struct_frac_pct,method,time_s,iters,infeas,miss_pct,dominated"]
# method => Vector of (ratio, miss%, dominated)
curves = Dict(m => Tuple{Float64,Float64,Bool}[] for (m, _) in ALL_METHODS)
ratios_seen = Float64[]

hi_trace = nothing        # (results, obj_ref_scaled, ratio) at the largest ratio
for T in T_GRID
    model, target = build_synth_model(ipm(); n_struct = N_STRUCT, T = T, n_store = N_STORE)
    s = mga_setup(model, target; eps = 0.1)
    ref = exact_reference(model, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    # Trace the convergence only at the largest (most-drowned) instance, to check
    # whether the gradient methods PLATEAU above the exact objective (genuine
    # drowning) rather than merely run out of a fixed budget.
    do_trace = (T == last(T_GRID))
    results, lp = run_correctors(
        model,
        s,
        ref;
        from = :origin,
        trace = do_trace,
        max_iters = MAX_ITERS,
        max_inner = MAX_INNER,
        pdhg_iters = PDHG_ITERS,
    )
    push!(results, solve_scs_mga(model, s, ref, lp))   # add SCS at this ratio

    nvars = length(s.all_vars)
    if do_trace
        global hi_trace =
            (results, dot(lp.w_t, ref.x ./ lp.d_scale), (nvars - N_STRUCT) / N_STRUCT)
    end
    ratio = (nvars - N_STRUCT) / N_STRUCT          # operational vars per structural var
    frac = 100 * N_STRUCT / nvars                  # structural fraction (%)
    push!(ratios_seen, ratio)
    println(
        "\n--- T=$T  nvars=$nvars  op/struct=$(round(ratio, digits=1))  struct=$(round(frac, sigdigits=3))% ---",
    )
    @printf(
        "  %-12s %8s %8s %11s %9s %s\n",
        "method",
        "time",
        "iters",
        "infeas",
        "miss%",
        "dom"
    )
    for r in results
        @printf(
            "  %-12s %8.2f %8d %11.2e %9.3f  %s\n",
            string(r.method),
            r.time,
            r.iters,
            r.infeas,
            r.gap_s,
            r.dominated ? "Y" : "n"
        )
        push!(curves[r.method], (ratio, r.gap_s, r.dominated))
        push!(
            rows,
            @sprintf(
                "%d,%d,%.3f,%.4f,%s,%.3f,%d,%.3e,%.5f,%s",
                T,
                nvars,
                ratio,
                frac,
                r.method,
                r.time,
                r.iters,
                r.infeas,
                r.gap_s,
                r.dominated ? "yes" : "no"
            )
        )
    end
end

open(joinpath(OUTDIR, "ratio_sweep.csv"), "w") do io
    println(io, join(rows, "\n"))
end

# --- headline figure: miss% vs operational/structural ratio -----------------
# Solid line per method; filled markers where the returned alternative is
# resource-DOMINATED by the exact one (the alternative is useless), hollow where
# it is still non-dominated. A floor keeps zeros visible on the log axis.
YFLOOR = 1e-3
p = plot(;
    xlabel = "operational-to-structural variable ratio  (drowning →)",
    ylabel = "missed diversity  (% of achievable, log)",
    title = "Why gradient correctors stall: sparse objective drowned by operational feasibility",
    legend = :topleft,
    size = (900, 600),
    xscale = :log10,
    yscale = :log10,
    left_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm,
)
for (m, label) in ALL_METHODS
    pts = curves[m]
    isempty(pts) && continue
    xs = [p[1] for p in pts]
    ys = [max(p[2], YFLOOR) for p in pts]
    dom = [p[3] for p in pts]
    c = CORRECTOR_COLOR[m]
    plot!(
        p,
        xs,
        ys;
        lw = 2.5,
        color = c,
        label = label,
        marker = :circle,
        ms = 6,
        markercolor = [d ? c : :white for d in dom],
        markerstrokecolor = c,
    )
end
# A light band marking "dominated" territory is implicit via the filled markers;
# annotate the interpretation once.
savefig(p, joinpath(OUTDIR, "ratio_gap.png"))

# --- diagnostic: convergence trace at the most-drowned instance -------------
if hi_trace !== nothing
    res_hi, obj_ref_scaled_hi, ratio_hi = hi_trace
    plot_corrector_trace(
        joinpath(OUTDIR, "ratio_trace_hi.png"),
        "Convergence at operational/structural ≈ $(round(Int, ratio_hi)) (most drowned)",
        res_hi,
        obj_ref_scaled_hi,
    )
end

println("\nwrote ratio_sweep.csv, ratio_gap.png, ratio_trace_hi.png")
println("(filled marker = corrector's alternative is resource-dominated by the exact one)")
