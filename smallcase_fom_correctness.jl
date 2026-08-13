# ===========================================================================
# Correctness gate for the first-order MGA correctors.
#
# Before we can claim a corrector "fails at scale because the diversity
# objective is drowned out", we must first establish that it is CORRECT on
# small, dense instances where no such drowning can occur. This script runs the
# four correctors (:penalty/QP, :alm_lbfgs/ALM, :pdhg, :osqp/ADMM) on
#   (a) a minimal LP with a hand-checkable MGA optimum, and
#   (b) a small, dense synthetic capacity-expansion instance,
# and grades each against an exact interior-point reference. A method PASSES if
# its structural-objective gap is within tolerance AND it is primal-feasible.
#
# A FAIL here is an implementation/robustness issue, reported as such and kept
# strictly separate from the scale mechanism studied in smallcase_fom_ratio.jl.
#
#   Usage:  julia -t 4 --project=. smallcase_fom_correctness.jl
#   Output: results/fom_smallcase/correctness.csv + console PASS/FAIL table
# ===========================================================================
using HiGHS, Printf
include("smallcase_common.jl")

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)

# Tolerances for the gate. `gap` is the % of the achievable diversity signal the
# corrector FAILED to capture (0% = reaches the exact MGA value, 100% = frozen at
# the cost optimum). A method that is correct here misses only its first-order
# accuracy tail (a few %); a dominated-at-scale method misses tens of %. Primal
# infeasibility is in the Ruiz-scaled space and reported alongside; the gate is
# lenient on it (the mechanism is about the OBJECTIVE stalling, not feasibility).
const GAP_TOL_PCT = 2.0
const INFEAS_TOL = 1e-2

ipm() = optimizer_with_attributes(
    HiGHS.Optimizer,
    "output_flag" => false,
    "solver" => "ipm",
    "run_crossover" => "off",
)

# Warm up JIT so the first case is not charged compile time.
let (m, tv) = build_minimal_lp(ipm())
    s = mga_setup(m, tv; eps = 0.1)
    ref = exact_reference(m, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    _, lpw = run_correctors(m, s, ref; from = :origin, max_iters = 5, pdhg_iters = 200)
    solve_scs_mga(m, s, ref, lpw)
end

rows =
    String["case,nvars,method,time_s,iters,infeas,obj_s,exact_obj_s,miss_pct,sol_dist,dominated,pass"]

function run_case(name::AbstractString, model, target_vars; eps = 0.1)
    s = mga_setup(model, target_vars; eps = eps)
    ref = exact_reference(model, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    results, lp = run_correctors(model, s, ref; from = :origin)
    push!(results, solve_scs_mga(model, s, ref, lp))   # add the SCS library corrector
    nvars = length(s.all_vars)

    println(
        "\n=== $name  ($nvars vars, w_s'x*: $(round(s.obj_star_s, sigdigits=6)) -> exact $(round(ref.obj_s, sigdigits=6))) ===",
    )
    @printf(
        "%-34s %8s %7s %11s %12s %8s %9s %10s  %s\n",
        "method",
        "time(s)",
        "iters",
        "infeas",
        "w_s'x",
        "miss%",
        "soldist",
        "dominated",
        "verdict"
    )
    println("-"^120)
    for r in results
        pass = r.gap_s <= GAP_TOL_PCT && r.infeas <= INFEAS_TOL
        @printf(
            "%-34s %8.3f %7d %11.2e %12.5g %8.3f %9.3e %10s  %s\n",
            r.label,
            r.time,
            r.iters,
            r.infeas,
            r.obj_s,
            r.gap_s,
            r.sol_dist,
            r.dominated ? "yes" : "no",
            pass ? "PASS" : "FAIL"
        )
        push!(
            rows,
            @sprintf(
                "%s,%d,%s,%.4f,%d,%.3e,%.6g,%.6g,%.4f,%.4e,%s,%s",
                name,
                nvars,
                r.method,
                r.time,
                r.iters,
                r.infeas,
                r.obj_s,
                ref.obj_s,
                r.gap_s,
                r.sol_dist,
                r.dominated ? "yes" : "no",
                pass ? "PASS" : "FAIL"
            )
        )
    end
    return results
end

# (a) minimal LP: cost optimum (2,0); MGA within 10% budget -> (1.8, 0.2).
run_case("minimal_lp", build_minimal_lp(ipm())...; eps = 0.1)

# (b) small dense synthetic: few timesteps, so the structural objective is NOT a
# vanishing fraction of the variables - the regime where every corrector should
# be correct.
run_case(
    "synth_dense",
    build_synth_model(ipm(); n_struct = 8, T = 6, n_store = 2)...;
    eps = 0.1,
)

open(joinpath(OUTDIR, "correctness.csv"), "w") do io
    println(io, join(rows, "\n"))
end
println("\nWrote $(joinpath(OUTDIR, "correctness.csv"))")
