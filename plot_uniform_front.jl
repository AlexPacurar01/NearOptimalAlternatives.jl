# =============================================================================
# Visualise how the uniform-budget ("linear step") sweep distributes points along
# the near-optimal front. Reads front_gurobi.csv (the completed 5h Gurobi sweep)
# and writes PNGs into results/sweep_vs_old_5h/.
#
#   julia --project=. plot_uniform_front.jl
# =============================================================================
using DelimitedFiles, Printf
using Plots
gr()

const OUTDIR = joinpath(@__DIR__, "results", "sweep_vs_old_5h")
const CSV = joinpath(OUTDIR, "front_gurobi.csv")

raw, header = readdlm(CSV, ','; header = true)
col(name) = raw[:, findfirst(==(name), vec(header))]
dir = Int.(col("direction"))
cost = Float64.(col("cost"))
divers = Float64.(col("diversity"))
stime = Float64.(col("solve_time"))
dirs = sort(unique(dir))

# Cost in billions and gap-% above C* (C* from the sweep base solve) for readability.
const CSTAR = 1.59398e11
costB = cost ./ 1e9
gappct = 100 .* (cost ./ CSTAR .- 1)

# --- Plot 1: the front, cost vs diversity, per direction -------------------
p1 = plot(
    xlabel = "cost (billion)",
    ylabel = "diversity objective  wᵀx  (lower = more diverse)",
    title = "Uniform-budget sweep: near-optimal front (5h, SPORES)",
    legend = :topright,
)
for d in dirs
    idx = sortperm(costB[dir.==d])
    plot!(
        p1,
        costB[dir.==d][idx],
        divers[dir.==d][idx];
        marker = :circle,
        markersize = 5,
        label = "direction $d",
    )
end
savefig(p1, joinpath(OUTDIR, "uniform_front.png"))

# --- Plot 2: equal budget steps -> UNEQUAL arclength steps ------------------
# For one representative direction, normalise (cost, diversity) to [0,1] over the
# direction's range and show the arclength covered by each consecutive budget
# step. Equal budget spacing piles length into the knee and wastes the plateau.
d0 = first(dirs)
m = dir .== d0
c = costB[m];
v = divers[m];
g = gappct[m];
o = sortperm(c);
c, v, g = c[o], v[o], g[o];
cn = (c .- minimum(c)) ./ max(maximum(c) - minimum(c), eps())
vn = (v .- minimum(v)) ./ max(maximum(v) - minimum(v), eps())
steplen = [hypot(cn[i+1] - cn[i], vn[i+1] - vn[i]) for i = 1:length(c)-1]
steplab = [@sprintf("%.1f→%.1f%%", g[i], g[i+1]) for i = 1:length(c)-1]
p2 = bar(
    steplab,
    steplen;
    legend = false,
    xrotation = 30,
    ylabel = "normalised arclength of step",
    title = "Uniform budget → uneven arclength (direction $d0)",
)
savefig(p2, joinpath(OUTDIR, "uniform_arclength_steps.png"))

# --- Plot 3: diminishing returns + plateau + missing budget ----------------
p3 = plot(
    xlabel = "cost gap above C*  (%)",
    ylabel = "diversity objective  wᵀx",
    title = "Diversity vs budget (uniform steps): diminishing returns",
    legend = :topright,
)
for d in dirs
    idx = sortperm(gappct[dir.==d])
    plot!(
        p3,
        gappct[dir.==d][idx],
        divers[dir.==d][idx];
        marker = :circle,
        markersize = 5,
        label = "direction $d",
    )
end
savefig(p3, joinpath(OUTDIR, "uniform_diversity_vs_gap.png"))

# --- Plot 4: solve time per budget ----------------------------------------
p4 = plot(
    xlabel = "cost gap above C*  (%)",
    ylabel = "solve time (s)",
    title = "Per-solve time vs budget (uniform sweep)",
    legend = :topright,
)
for d in dirs
    idx = sortperm(gappct[dir.==d])
    plot!(
        p4,
        gappct[dir.==d][idx],
        stime[dir.==d][idx];
        marker = :diamond,
        markersize = 5,
        label = "direction $d",
    )
end
savefig(p4, joinpath(OUTDIR, "uniform_solvetime_vs_gap.png"))

# --- quick numeric readout of the point distribution -----------------------
println("Uniform-budget point distribution (direction $d0):")
@printf("  gaps sampled (%%): %s\n", join([@sprintf("%.1f", x) for x in g], ", "))
@printf("  diversity:        %s\n", join([@sprintf("%.3g", x) for x in v], ", "))
@printf("  arclength/step:   %s\n", join([@sprintf("%.3f", x) for x in steplen], ", "))
@printf(
    "  -> max step %.3f vs min step %.3f  (%.1fx imbalance)\n",
    maximum(steplen),
    minimum(steplen),
    maximum(steplen) / max(minimum(steplen), eps())
)
println("Wrote 4 PNGs to $OUTDIR")
