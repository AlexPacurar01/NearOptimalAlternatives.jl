# ===========================================================================
# Norse case study for the NearOptimalAlternatives.jl (NOA) SoftwareX paper,
# Section 3.3: near-optimal alternatives for the TulipaEnergyModel.jl "Norse"
# example system, generated with ONE modeling method (SPORES by default), and
# how they differ from the cost-optimal design and from each other.
#
# What it does
#   1. builds the Norse case with TulipaEnergyModel.jl (LP relaxation, i.e.
#      investment integrality dropped) and solves it to optimality;
#   2. expresses each of the 16 investment decisions as installed capacity in
#      GW (see "Units" below);
#   3. calls ONE NOA function, `generate_alternatives_optimization!`, asking
#      for N_ALT alternatives within an optimality gap EPS;
#   4. writes per-technology capacities, changes against the optimum,
#      pairwise distances, LaTeX tables and a comparison figure.
#
# Units: Tulipa's investment variables count UNITS, and unit sizes differ by
# up to ~300x in Norse (a wind unit is 3 MW, a CCGT unit 500 MW).  Methods
# that weight the decision variables (SPORES, HSJ, ...) are not invariant to
# that choice: on unit counts SPORES penalises each GW of wind ~90x harder
# than a GW of nuclear, and wind alone makes up >80% of its first objective,
# so every alternative simply removes wind.  The script therefore adds one
# bounded variable per decision holding its capacity in GW
# (cap_gw[i] == MW_per_unit[i] / 1000 * units[i]) and hands those to NOA -
# two lines of JuMP on top of the unchanged Tulipa model.  Set NOA_SPACE=units
# to reproduce the unit-count behaviour.
#
# Data: paper/data/Norse is the Norse example from TulipaEnergyModel.jl
# v0.22.2 (test/inputs/Norse) with an `investment_limit` added to every
# investable asset and transport line that had none (ten times its
# cost-optimal investment; 412.5 GW for the CCGTs and 100 GW for the
# electrolyser, which are not built at the optimum).  SPORES needs finite
# upper bounds.  See paper/README.md.
#
# Solver: HiGHS for every linear method.  Max_Distance maximises a convex
# quadratic, i.e. it is a nonconvex QP, and needs SCIP or Gurobi
# (NonConvex=2); HiGHS rejects it and Ipopt does not converge on it here.
#
# Config via env vars (all optional):
#   NOA_METHOD   Spores | HSJ | Min_Max_Variables | Random_Vector |
#                Directionally_Weighted_Variables | Max_Distance   (default Spores)
#   NOA_SOLVER   highs | scip | gurobi   (default highs; Max_Distance needs scip/gurobi)
#   NOA_SPACE    gw | units              (default gw)
#   NOA_N_ALT    number of alternatives  (default 3)
#   NOA_EPS      optimality gap          (default 0.2)
#   NOA_NORSE    Norse input folder      (default paper/data/Norse)
#   NOA_OUT      output directory        (default paper/results/norse_<method>[_units])
#   NOA_TIME     per-solve time limit, s (default 1800)
#   NOA_FIGDIR   figure directory        (default paper/figures; file named after NOA_OUT)
#
# Run from the repository root:
#   julia --project=. paper/norse_mga.jl
# ===========================================================================
using JuMP, Distances, Printf, Statistics, Dates, Logging, LinearAlgebra
using NearOptimalAlternatives
import TulipaEnergyModel as TEM
import TulipaIO as TIO
using DuckDB

env(k, d) = get(ENV, k, d)
const METHOD = Symbol(env("NOA_METHOD", "Spores"))
const SOLVER = lowercase(env("NOA_SOLVER", "highs"))
const SPACE = lowercase(env("NOA_SPACE", "gw"))
const N_ALT = parse(Int, env("NOA_N_ALT", "3"))
const EPS = parse(Float64, env("NOA_EPS", "0.2"))
const NORSE = env("NOA_NORSE", joinpath(@__DIR__, "data", "Norse"))
const OUTDIR = env(
    "NOA_OUT",
    joinpath(@__DIR__, "results", "norse_" * lowercase(string(METHOD)) * (SPACE == "units" ? "_units" : "")),
)
const TIME_LIMIT = parse(Float64, env("NOA_TIME", "1800"))
# Figures go to paper/figures/, named after the run (e.g. norse_spores.png), so the
# manuscript can include them from one place; tables and data stay in OUTDIR.
const FIGDIR = env("NOA_FIGDIR", joinpath(@__DIR__, "figures"))
const FIGNAME = basename(OUTDIR)
SPACE in ("gw", "units") || error("NOA_SPACE must be 'gw' or 'units', got '$SPACE'")
mkpath(OUTDIR)
mkpath(FIGDIR)
global_logger(ConsoleLogger(stderr, Logging.Warn))   # silence NOA's per-step @info

# --- solver ----------------------------------------------------------------
METHOD == :Max_Distance && SOLVER == "highs" &&
    error("Max_Distance is a nonconvex QP; set NOA_SOLVER=scip or NOA_SOLVER=gurobi")
if SOLVER == "highs"
    @eval using HiGHS
    optimizer() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "time_limit" => TIME_LIMIT)
elseif SOLVER == "gurobi"
    @eval using Gurobi
    optimizer() = optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "NonConvex" => 2,
        "TimeLimit" => TIME_LIMIT,
        "MIPGap" => 1e-4,
    )
elseif SOLVER == "scip"
    @eval using SCIP
    optimizer() = optimizer_with_attributes(
        SCIP.Optimizer,
        "display/verblevel" => 0,
        "limits/time" => TIME_LIMIT,
        "limits/gap" => 1e-4,
    )
else
    error("NOA_SOLVER must be 'highs', 'scip' or 'gurobi', got '$SOLVER'")
end

# --- labels ----------------------------------------------------------------
"Readable name: `Midgard_Nuclear_SMR` -> `Midgard nuclear SMR`, flows -> `Line Asgard–Midgard`."
function pretty(asset::AbstractString)
    region, tech = split(asset, "_"; limit = 2)
    words = [all(isuppercase, w) || occursin(r"\d", w) ? w : lowercase(w) for w in split(tech, "_")]
    return region * " " * join(words, " ")
end
pretty(from::AbstractString, to::AbstractString) =
    "Line " * first(split(from, "_")) * "–" * first(split(to, "_"))

# --- Norse ------------------------------------------------------------------
"""
Build the Norse case, relax integrality, solve.  Returns the solved model, the
investment variables (assets and transport lines), a readable label for each,
and the capacity per invested unit in GW.
"""
function load_norse()
    con = DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(con, NORSE)
    TEM.populate_with_defaults!(con)
    ep = TEM.EnergyProblem(con)
    TEM.create_model!(ep)
    model = ep.model
    relax_integrality(model)
    set_optimizer(model, optimizer())
    set_silent(model)
    optimize!(model)
    @assert is_solved_and_feasible(model) "base solve of Norse failed: $(termination_status(model))"

    mw = Dict{String,Float64}()
    for r in DBInterface.execute(con, "SELECT asset, capacity FROM asset")
        mw[r.asset] = r.capacity
    end
    for r in DBInterface.execute(con, "SELECT from_asset, to_asset, capacity FROM flow WHERE is_transport")
        mw[r.from_asset * "|" * r.to_asset] = r.capacity
    end

    units, labels, gw_per_unit = VariableRef[], String[], Float64[]
    for sym in (:assets_investment, :flows_investment), v in ep.variables[sym].container
        v isa VariableRef || continue
        s = string(v)
        m = match(r"assets_investment\[\d+,(.+)\]", s)
        if m !== nothing
            key, lab = String(m[1]), pretty(m[1])
        else
            m = match(r"flows_investment\[\d+,\(\"(.+)\", \"(.+)\"\)\]", s)
            key, lab = m[1] * "|" * m[2], pretty(m[1], m[2])
        end
        push!(units, v)
        push!(labels, lab)
        push!(gw_per_unit, mw[key] / 1000)
    end
    return model, units, labels, gw_per_unit
end

println("[$(now())] Norse MGA: method=$METHOD space=$SPACE solver=$SOLVER n_alt=$N_ALT eps=$EPS data=$NORSE -> $OUTDIR")
t0 = time()
model, units, labels, gw = load_norse()
base_time = time() - t0
base_cost = objective_value(model)
x0 = value.(units) .* gw                    # optimum, GW
@printf("base solve: %.1fs, cost = %.6g, %d investment decisions\n", base_time, base_cost, length(units))

# --- diversity variables: capacity in GW (default) or Tulipa's unit counts ----
if SPACE == "gw"
    n = length(units)
    @variable(model, 0 <= cap_gw[i = 1:n] <= upper_bound(units[i]) * gw[i])
    @constraint(model, [i = 1:n], cap_gw[i] == gw[i] * units[i])
    optimize!(model)                         # same optimum; gives cap_gw its values
    @assert is_solved_and_feasible(model)
    diversity_vars = collect(cap_gw)
    metric = SqEuclidean()                   # Max_Distance only: already in GW
else
    diversity_vars = units
    metric = WeightedSqEuclidean(gw .^ 2)    # Max_Distance only: distance in GW
end

# --- the one NOA call ---------------------------------------------------------
t1 = time()
res = generate_alternatives_optimization!(
    model,
    EPS,
    diversity_vars,
    N_ALT;
    modeling_method = METHOD,
    metric = metric,
)
mga_time = time() - t1
@printf("%d alternatives in %.1fs (%.1fs per alternative)\n", length(res.solutions), mga_time, mga_time / length(res.solutions))

# --- post-processing -----------------------------------------------------------
X = hcat(x0, [[s[v] for v in units] .* gw for s in res.solutions]...)   # GW; columns: optimum, alt1..altN
X[abs.(X).<5e-4] .= 0.0                                               # print solver noise (-0.00) as 0
ncol = size(X, 2)
colnames = ["optimum"; ["alt$i" for i = 1:N_ALT]]
costs = [base_cost; res.objective_values]
D = [norm(X[:, a] - X[:, b]) for a = 1:ncol, b = 1:ncol]
built = [i for i in eachindex(labels) if any(abs.(X[i, :]) .> 5e-3)]   # drop never-built technologies
never = setdiff(eachindex(labels), built)
order = sort(built; by = i -> -maximum(abs.(X[i, 2:end] .- X[i, 1])))  # largest change first

open(joinpath(OUTDIR, "alternatives.csv"), "w") do io
    println(io, "technology,gw_per_unit," * join(colnames .* "_GW", ","))
    for i in eachindex(labels)
        println(io, "\"$(labels[i])\",$(gw[i])," * join([@sprintf("%.4f", X[i, c]) for c = 1:ncol], ","))
    end
    println(io, "\"total cost\",," * join([@sprintf("%.6e", c) for c in costs], ","))
    println(io, "\"cost / optimum\",," * join([@sprintf("%.5f", c / base_cost) for c in costs], ","))
end
open(joinpath(OUTDIR, "change_vs_optimum.csv"), "w") do io
    println(io, "technology,optimum_GW," * join(["alt$(i)_minus_optimum_GW" for i = 1:N_ALT], ","))
    for i in order
        println(io, "\"$(labels[i])\",$(@sprintf("%.4f", X[i, 1]))," *
                    join([@sprintf("%.4f", X[i, c] - X[i, 1]) for c = 2:ncol], ","))
    end
end
open(joinpath(OUTDIR, "distances.csv"), "w") do io
    println(io, "," * join(colnames, ","))
    for a = 1:ncol
        println(io, colnames[a] * "," * join([@sprintf("%.3f", D[a, b]) for b = 1:ncol], ","))
    end
end
open(joinpath(OUTDIR, "summary.csv"), "w") do io
    println(io, "method,space,solver,n_alt,eps,base_solve_s,mga_total_s,base_cost,julia,noa_version")
    println(io, "$METHOD,$SPACE,$SOLVER,$N_ALT,$EPS,$(round(base_time, digits = 2)),$(round(mga_time, digits = 2))," *
                "$base_cost,$VERSION,$(pkgversion(NearOptimalAlternatives))")
end

# LaTeX table: capacities, then changes in brackets would clutter - keep capacity; changes in their own table
texlab(s) = replace(s, "H2" => "H\$_2\$")
texnum(x) = @sprintf("%.1f", x)
texdelta(x) = abs(x) < 0.05 ? "0.0" : (x > 0 ? "+" : "\$-\$") * @sprintf("%.1f", abs(x))
open(joinpath(OUTDIR, "table.tex"), "w") do io
    println(io, "\\begin{tabular}{l" * "r"^ncol * "}")
    println(io, "\\toprule")
    println(io, "\\textbf{Installed capacity (GW)} & \\textbf{Optimum} & " *
                join(["\\textbf{Alt.~$i}" for i = 1:N_ALT], " & ") * "\\\\")
    println(io, "\\midrule")
    for i in order
        println(io, texlab(labels[i]) * " & " * texnum(X[i, 1]) * " & " *
                    join([texnum(X[i, c]) * " (" * texdelta(X[i, c] - X[i, 1]) * ")" for c = 2:ncol], " & ") * "\\\\")
    end
    println(io, "\\midrule")
    println(io, "Cost relative to optimum & " * join([@sprintf("%.2f", c / base_cost) for c in costs], " & ") * "\\\\")
    println(io, "Distance to optimum (GW) & " * join([@sprintf("%.1f", D[1, c]) for c = 1:ncol], " & ") * "\\\\")
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
open(joinpath(OUTDIR, "distances.tex"), "w") do io
    hdr = ["Optimum"; ["Alt.~$i" for i = 1:N_ALT]]
    println(io, "\\begin{tabular}{l" * "r"^ncol * "}")
    println(io, "\\toprule")
    println(io, "\\textbf{Distance (GW)} & " * join(["\\textbf{$h}" for h in hdr], " & ") * "\\\\")
    println(io, "\\midrule")
    for a = 1:ncol
        println(io, hdr[a] * " & " * join([b < a ? "" : @sprintf("%.1f", D[a, b]) for b = 1:ncol], " & ") * "\\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end

# Console report
fmt(x, d) = lpad(@sprintf("%.*f", d, x), 9)
println("\n", rpad("installed capacity (GW)", 28), join(lpad.(colnames, 9)), "    change vs optimum")
for i in order
    println(rpad(labels[i], 28), join(fmt.(X[i, :], 2)), "   ", join(fmt.(X[i, 2:end] .- X[i, 1], 2)))
end
isempty(never) || println("never built: ", join(labels[never], ", "))
println(rpad("cost / optimum", 28), join(fmt.(costs ./ base_cost, 4)))
println("\npairwise distances (GW):")
for a = 1:ncol
    println(rpad(colnames[a], 10), join(fmt.(D[a, :], 1)))
end

# Figure: dot plot, one row per technology; optimum as a black diamond, the
# alternatives in the first three slots of a colour-blind-validated palette.
try
    @eval using Plots
    pal = ["#2a78d6", "#eb6834", "#1baf7a"]    # blue, orange, aqua: pass all-pairs CVD checks
    N_ALT <= length(pal) || error("figure supports up to $(length(pal)) alternatives")
    rows = [i for i in order if maximum(abs.(X[i, 2:end] .- X[i, 1])) >= 0.05]   # unchanged rows go in the caption
    nrow = length(rows)
    plt = plot(
        size = (760, 40 * nrow + 120),
        legend = :outertop,
        legend_columns = N_ALT + 1,
        xlabel = "Installed capacity (GW)",
        yticks = (1:nrow, labels[rows]),
        yflip = true,
        ylims = (0.4, nrow + 0.6),
        xlims = (-1, 1.05 * maximum(X[rows, :])),
        framestyle = :box,
        grid = :x,
        gridalpha = 0.25,
        tickfontsize = 9,
        guidefontsize = 10,
        legendfontsize = 9,
        left_margin = 4Plots.mm,
        bottom_margin = 4Plots.mm,
        fontfamily = "Helvetica",
        foreground_color_legend = nothing,
    )
    for (k, i) in enumerate(rows)        # thin connector spanning optimum and all alternatives
        plot!(plt, [minimum(X[i, :]), maximum(X[i, :])], [k, k]; color = "#c3c2b7", lw = 1.5, label = "")
    end
    offs = range(-0.18, 0.18; length = N_ALT)
    for c = 2:ncol
        scatter!(plt, X[rows, c], (1:nrow) .+ offs[c-1]; color = pal[c-1], ms = 6,
                 markerstrokecolor = :white, markerstrokewidth = 1, label = "Alternative $(c-1)")
    end
    scatter!(plt, X[rows, 1], 1:nrow; color = "#0b0b0b", marker = :diamond, ms = 7,
             markerstrokecolor = :white, markerstrokewidth = 1, label = "Cost optimum")
    savefig(plt, joinpath(FIGDIR, FIGNAME * ".png"))
    savefig(plt, joinpath(FIGDIR, FIGNAME * ".pdf"))
    println("figure -> ", joinpath(FIGDIR, FIGNAME * ".png"), "  (unchanged, not drawn: ",
            join(labels[setdiff(order, rows)], ", "), ")")
catch e
    @warn "figure skipped" exception = (e, catch_backtrace())
end
println("[$(now())] done -> $OUTDIR")
