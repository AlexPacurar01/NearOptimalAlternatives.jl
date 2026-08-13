# ===========================================================================
# SCS convergence anatomy: how good is the APPROXIMATE MGA alternative, and how
# fast does it appear, versus how slow the EXACT (tight-feasible) point is?
#
# For one first-alternative MGA sub-problem we solve it exactly with a barrier
# (HiGHS) for reference, then run SCS to a sequence of hard iteration caps (with
# the tolerance set so tight that SCS never stops early, so the cap IS the
# iteration count). At each cap we record, in MGA-interpretable units:
#   * miss%   - diversity-objective gap to the exact optimum (how good the
#               alternative is on the thing MGA actually cares about);
#   * viol    - max original-unit constraint violation (how feasible it is);
#   * soldist - normalised structural distance to the exact alternative.
# If miss%/soldist collapse early while viol decays slowly, then a USABLE
# approximate alternative is cheap even though an exact-feasible point is slow -
# which matters, because MGA wants diverse near-optimal options, not exactness.
#
#   Usage:  julia -t 4 --project=. smallcase_fom_scsconv.jl [case ...]
#     env:  SCSCONV_MAXK (largest iteration cap, default 1000000)
# ===========================================================================
using HiGHS, Printf, Statistics, LinearAlgebra
using Plots
include("smallcase_common.jl")
gr()

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)
const CASES = isempty(ARGS) ? ["synth"] : ARGS
if any(c -> c in ("Tiny", "Norse"), CASES)
    include("smallcase_tulipa.jl")
end
ipm() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
const MAXK = parse(Int, get(ENV, "SCSCONV_MAXK", "1000000"))

make_case(name) =
    name == "synth" ? build_synth_model(ipm(); n_struct = 8, T = 24, n_store = 4) :
    name in ("Tiny", "Norse") ? load_tulipa_case(name, ipm()) :
    error("unknown case '$name'")

"One SCS solve of the MGA sub-problem capped at `maxit` iterations (tolerance set
tiny so the cap binds). Returns (x, time, viol) with viol the max original-unit
constraint violation at the returned iterate."
function scs_capped(model, s, maxit)
    orig = objective_function(model)
    sense = objective_sense(model)
    bcon = @constraint(model, orig <= s.B)
    @objective(
        model,
        Min,
        sum(s.w[i] * s.all_vars[i] for i in eachindex(s.w) if s.w[i] != 0)
    )
    set_optimizer(
        model,
        optimizer_with_attributes(
            SCS.Optimizer,
            "verbose" => 0,
            "eps_abs" => 1e-13,
            "eps_rel" => 1e-13,
            "max_iters" => maxit,
        ),
    )
    t = @elapsed optimize!(model)
    x = value.(s.all_vars)
    pt = Dict(s.all_vars[i] => x[i] for i in eachindex(s.all_vars))
    rep = primal_feasibility_report(model, pt)
    viol = isempty(rep) ? 0.0 : maximum(values(rep))
    delete(model, bcon)
    set_objective_function(model, orig)
    set_objective_sense(model, sense)
    return x, t, viol
end

function run_case(name)
    println("\n########## SCS convergence anatomy: $name ##########")
    model, target = make_case(name)
    s = mga_setup(model, target; eps = 0.1)
    ref = exact_reference(model, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    signal = s.obj_star_s - ref.obj_s
    println(
        "exact w_s'x = $(round(ref.obj_s, sigdigits=6)); achievable diversity = $(round(signal, sigdigits=6))",
    )

    caps = filter(<=(MAXK), [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000])
    rows = String["case,iters,time_s,miss_pct,viol,soldist,obj_s"]
    data = NamedTuple[]
    @printf("%10s %9s %10s %11s %11s\n", "iters", "time(s)", "miss%", "viol", "soldist")
    for K in caps
        x, t, viol = scs_capped(model, s, K)
        obj_s = dot(s.w_s, x[s.target_indices])
        miss = 100 * max(obj_s - ref.obj_s, 0.0) / max(abs(signal), 1e-9)
        sd = structural_distance(x, ref.x, s.target_indices, s.target_scale)
        @printf("%10d %9.2f %10.4f %11.2e %11.2e\n", K, t, miss, viol, sd)
        push!(
            rows,
            @sprintf("%s,%d,%.4f,%.5f,%.4e,%.4e,%.6g", name, K, t, miss, viol, sd, obj_s)
        )
        push!(data, (K = K, t = t, miss = miss, viol = viol, sd = sd))
    end
    open(joinpath(OUTDIR, "scsconv_$name.csv"), "w") do io
        println(io, join(rows, "\n"))
    end

    # Convergence plot: miss%, feasibility violation, and structural distance vs
    # iterations (log-log). Floors keep zeros visible.
    fl(v, f) = max(v, f)
    ks = [d.K for d in data]
    p = plot(;
        xlabel = "SCS iterations",
        ylabel = "value (log)",
        xscale = :log10,
        yscale = :log10,
        legend = :left,
        size = (900, 560),
        title = "SCS convergence on $name: approximate vs exact",
    )
    plot!(
        p,
        ks,
        [fl(d.miss, 1e-4) for d in data];
        lw = 2.5,
        marker = :circle,
        color = :purple,
        label = "missed diversity (%)",
    )
    plot!(
        p,
        ks,
        [fl(d.viol, 1e-12) for d in data];
        lw = 2.5,
        marker = :square,
        color = :indianred,
        label = "feasibility violation",
    )
    plot!(
        p,
        ks,
        [fl(d.sd, 1e-10) for d in data];
        lw = 2.5,
        marker = :diamond,
        color = :teal,
        label = "structural distance to exact",
    )
    savefig(p, joinpath(OUTDIR, "scsconv_$name.png"))
    println("wrote scsconv_$name.csv, scsconv_$name.png")
end

# warm-up
let (m, tv) = build_synth_model(ipm(); n_struct = 4, T = 4, n_store = 1)
    s = mga_setup(m, tv)
    r = exact_reference(m, s.w, s.B, s.all_vars, s.target_indices, s.w_s)
    scs_capped(m, s, 200)
end

for c in CASES
    run_case(c)
end
