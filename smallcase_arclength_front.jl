# ===========================================================================
# Grading the ARCLENGTH STEP CONTROL against a uniform budget grid, with an
# exact warm dual-simplex corrector (so corrector error cannot pollute the
# comparison). For one instance:
#   * dense reference front  - 200 warm re-solves on a uniform grid (ground
#                              truth polyline + the dobj span used to
#                              normalise objective space);
#   * uniform sweep          - N budgets uniform in B (the baseline grid);
#   * arclength sweep        - the continuation_walk control law (target
#                              spacing ds = sqrt(2)/N in the normalised
#                              (cost, dobj) plane, overshoot => shrink dB and
#                              retry, then retarget dB toward ds), run online.
# Reported per sweep: points, total solves (arclength pays for rejected
# steps), and the spacing statistics of consecutive points in the normalised
# objective plane (mean / max gap / coefficient of variation). Uniform-B gives
# even spacing only if the front has constant slope; on a curved front it
# clusters points where the front is flat and leaves gaps where it is steep -
# the arclength control is what removes that failure mode.
#
#   Usage:  julia -t 4 --project=. smallcase_arclength_front.jl [case]
#     env:  ARC_NPOINTS (default 12), ARC_EPS (default 0.1)
# ===========================================================================
using JuMP, HiGHS, LinearAlgebra, Statistics, Printf
using Plots
gr()

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)
const CASE = isempty(ARGS) ? "Norse" : ARGS[1]
const N_PTS = parse(Int, get(ENV, "ARC_NPOINTS", "12"))
const EPS_MGA = parse(Float64, get(ENV, "ARC_EPS", "0.1"))

include("smallcase_tulipa.jl")

dual_simplex() = optimizer_with_attributes(
    HiGHS.Optimizer,
    "output_flag" => false,
    "solver" => "simplex",
    "simplex_strategy" => 1,
)

model, target = load_tulipa_case(CASE, dual_simplex())
min_cost = objective_value(model)
all_vars = all_variables(model)
idx = Dict(v => i for (i, v) in enumerate(all_vars))
xstar = value.(all_vars)
w = zeros(length(all_vars))
for v in target
    ub = has_upper_bound(v) && upper_bound(v) != 0 ? upper_bound(v) : 1.0
    w[idx[v]] = xstar[idx[v]] / ub
end
what = w ./ max(norm(w), eps())

orig = objective_function(model)
off = JuMP.constant(orig)
cvec = [coefficient(orig, v) for v in all_vars]
B_max = min_cost + EPS_MGA * abs(min_cost)
B_0 = min_cost + EPS_MGA * abs(min_cost) / N_PTS    # first interior budget

bcon = @constraint(model, orig <= B_max)
@objective(model, Min, sum(w[i] * all_vars[i] for i in eachindex(w) if w[i] != 0))

nsolves = Ref(0)
"Warm dual-simplex solve at budget B (basis carried automatically)."
function solve_at(B)
    set_normalized_rhs(bcon, B - off)
    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL "solve at B=$B failed"
    nsolves[] += 1
    x = value.(all_vars)
    (cost = dot(cvec, x) + off, dobj = dot(what, x))
end

# --- dense ground-truth front (also fixes the dobj normalisation span) ------
dense = [solve_at(B) for B in range(B_0, B_max; length = 200)]
dspan = abs(dense[end].dobj - dense[1].dobj)
cspan = B_max - B_0
nrm(p) = ((p.cost - B_0) / cspan, (p.dobj - dense[1].dobj) / dspan)
dist(p, q) = hypot(nrm(p)[1] - nrm(q)[1], nrm(p)[2] - nrm(q)[2])

# --- arclength sweep (continuation_walk control law, online) ------------------
nsolves[] = 0
ds = sqrt(2.0) / N_PTS
dB_min = 1e-4 * cspan
arc = [solve_at(B_0)]
B = B_0
dB = cspan / N_PTS
while B < B_max - 1e-9 * cspan && length(arc) < 3 * N_PTS
    accepted = false
    for _ = 1:8
        B_try = min(B + dB, B_max)
        p = solve_at(B_try)
        d = dist(p, arc[end])
        if d > 1.4 * ds && B_try < B_max && dB > dB_min
            global dB *= max(0.3, ds / max(d, 1e-9))         # overshoot: retry
        else
            push!(arc, p)
            global B = B_try
            global dB = clamp(dB * ds / max(d, 1e-9), dB_min, cspan)
            accepted = true
            break
        end
    end
    accepted || break
end
arc_solves = nsolves[]

# --- uniform sweep, SAME point count as the arclength sweep produced -----------
# (a sweep with more points gets smaller gaps for free, so the grid comparison
# is only fair at equal counts; N_PTS merely sets the arclength target ds)
nsolves[] = 0
uni = [solve_at(B) for B in range(B_0, B_max; length = length(arc))]
uni_solves = nsolves[]

# --- spacing statistics --------------------------------------------------------
gaps(pts) = [dist(pts[i+1], pts[i]) for i = 1:length(pts)-1]
stats(pts) = (g = gaps(pts); (mean = mean(g), mx = maximum(g), cv = std(g) / mean(g)))
su, sa = stats(uni), stats(arc)
@printf(
    "%-10s %6s %7s %10s %9s %9s\n",
    "sweep",
    "points",
    "solves",
    "mean gap",
    "max gap",
    "CV"
)
@printf(
    "%-10s %6d %7d %10.4f %9.4f %9.3f\n",
    "uniform",
    length(uni),
    uni_solves,
    su.mean,
    su.mx,
    su.cv
)
@printf(
    "%-10s %6d %7d %10.4f %9.4f %9.3f\n",
    "arclength",
    length(arc),
    arc_solves,
    sa.mean,
    sa.mx,
    sa.cv
)

open(joinpath(OUTDIR, "arclength_front_$CASE.csv"), "w") do io
    println(io, "sweep,cost,dobj")
    for (name, pts) in (("dense", dense), ("uniform", uni), ("arclength", arc)), p in pts
        println(io, "$name,$(p.cost),$(p.dobj)")
    end
    println(
        io,
        "# uniform: points=$(length(uni)) solves=$uni_solves mean=$(su.mean) max=$(su.mx) cv=$(su.cv)",
    )
    println(
        io,
        "# arclength: points=$(length(arc)) solves=$arc_solves mean=$(sa.mean) max=$(sa.mx) cv=$(sa.cv)",
    )
end

# --- plot: two stacked panels so the sweeps can never occlude each other and
# --- the point counts are verifiable by eye ------------------------------------
pct(c) = 100 * (c / min_cost - 1)
dx, dy = pct.([q.cost for q in dense]), [q.dobj for q in dense]
function panel(pts, mk, col, name, s; xlab = "")
    q = plot(;
        ylabel = "w'x",
        xlabel = xlab,
        legend = :topright,
        grid = :y,
        gridalpha = 0.15,
    )
    plot!(q, dx, dy; lw = 1.5, color = :gray70, label = "exact front (200-pt reference)")
    scatter!(
        q,
        pct.([r.cost for r in pts]),
        [r.dobj for r in pts];
        marker = mk,
        markersize = 7,
        color = col,
        markerstrokecolor = :white,
        markerstrokewidth = 1.2,
        label = @sprintf(
            "%s (%d pts, spacing CV %.2f, max gap %.2f)",
            name,
            length(pts),
            s.cv,
            s.mx
        )
    )
    return q
end
p = plot(
    panel(uni, :circle, "#5778a4", "uniform grid", su),
    panel(arc, :diamond, "#e49444", "arclength", sa; xlab = "cost overrun above C* (%)");
    layout = (2, 1),
    size = (900, 720),
    link = :x,
    plot_title = "Uniform vs arclength budget sweep ($CASE, exact corrector)",
)
savefig(p, joinpath(OUTDIR, "arclength_front_$CASE.png"))
println("wrote arclength_front_$CASE.csv, arclength_front_$CASE.png")
