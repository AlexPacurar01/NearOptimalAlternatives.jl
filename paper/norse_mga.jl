# ===========================================================================
# Norse case study for the NearOptimalAlternatives.jl (NOA) SoftwareX paper,
# Section 3.3: MGA alternatives on the TulipaEnergyModel.jl "Norse" example
# system, generated with ONE modeling method (SPORES by default), and the
# technology trade-offs they reveal.
#
# What it does
#   1. builds the Norse case with TulipaEnergyModel.jl (LP relaxation, i.e.
#      investment integrality dropped) and solves it to optimality;
#   2. calls ONE NOA function, `generate_alternatives_optimization!`, with
#      `modeling_method = METHOD` (default :Spores), asking for N_ALT
#      alternatives within an optimality gap EPS over the investment
#      variables.  For :Max_Distance the distance is measured in installed
#      capacity (GW) rather than in Tulipa's "number of units", through a
#      `WeightedSqEuclidean` metric (NOA accepts any Distances.jl SemiMetric);
#      the weighted-sum methods (HSJ, SPORES, ...) are unit-free;
#   3. writes, per technology, the optimal and alternative capacities, the
#      cost of every alternative, pairwise distances, a LaTeX table and a
#      bandwidth figure.
#
# Data: paper/data/Norse is the Norse example shipped with
# TulipaEnergyModel.jl v0.22.2 (test/inputs/Norse), with one change: every
# investable asset and transport flow gets an `investment_limit` of ten times
# its cost-optimal investment (asset-commission.csv, flow-commission.csv).
# Without it, Max-Distance spends the whole budget on the single technology
# whose capacity is unbounded (hydrogen storage) - see the paper.
#
# Solver: every method except Max-Distance keeps the problem an LP, solved
# here with HiGHS (default).  Max-Distance maximises a convex quadratic over
# the near-optimal polytope - a NONCONVEX QP whose optimum lies at a vertex -
# so it needs a solver with global nonconvex support (SCIP or Gurobi with
# NonConvex=2); HiGHS rejects it ("Cannot solve non-convex QP problems") and
# a local NLP solver such as Ipopt does not converge on it at this size.
#
# Config via env vars (all optional):
#   NOA_METHOD   Spores | HSJ | Min_Max_Variables | Random_Vector |
#                Directionally_Weighted_Variables | Max_Distance   (default Spores)
#   NOA_SOLVER   highs | scip | gurobi   (default highs; Max_Distance needs scip/gurobi)
#   NOA_N_ALT    number of alternatives             (default 4)
#   NOA_EPS      optimality gap                      (default 0.2)
#   NOA_NORSE    path to a Norse input folder        (default paper/data/Norse)
#   NOA_OUT      output directory                    (default paper/results/norse_<method>)
#   NOA_TIME     per-solve time limit in seconds     (default 1800)
#
# Run from the repository root (see paper/README.md):
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
const N_ALT = parse(Int, env("NOA_N_ALT", "4"))
const EPS = parse(Float64, env("NOA_EPS", "0.2"))
const NORSE = env("NOA_NORSE", joinpath(@__DIR__, "data", "Norse"))
const OUTDIR = env("NOA_OUT", joinpath(@__DIR__, "results", "norse_" * lowercase(string(METHOD))))
const TIME_LIMIT = parse(Float64, env("NOA_TIME", "1800"))
mkpath(OUTDIR)
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

# --- Norse ------------------------------------------------------------------
"""
Build the Norse case, relax integrality, solve.  Returns the solved model, the
investment variables (assets and transport flows), a short label per variable,
and the capacity per invested unit (MW) of each.
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

    cap = Dict{String,Float64}()
    for r in DBInterface.execute(con, "SELECT asset, capacity FROM asset")
        cap[r.asset] = r.capacity
    end
    for r in DBInterface.execute(con, "SELECT from_asset, to_asset, capacity FROM flow WHERE is_transport")
        cap[r.from_asset * " -> " * r.to_asset] = r.capacity
    end

    target, labels, capunit = VariableRef[], String[], Float64[]
    for sym in (:assets_investment, :flows_investment), v in ep.variables[sym].container
        v isa VariableRef || continue
        s = string(v)
        m = match(r"assets_investment\[\d+,(.+)\]", s)
        if m !== nothing
            lab = String(m[1])
        else
            m = match(r"flows_investment\[\d+,\(\"(.+)\", \"(.+)\"\)\]", s)
            lab = m[1] * " -> " * m[2]
        end
        push!(target, v)
        push!(labels, lab)
        push!(capunit, cap[lab])
    end
    return model, target, labels, capunit
end

println("[$(now())] Norse MGA: method=$METHOD solver=$SOLVER n_alt=$N_ALT eps=$EPS data=$NORSE -> $OUTDIR")
t0 = time()
model, target, labels, capunit = load_norse()
base_time = time() - t0
base_cost = objective_value(model)
x0 = value.(target)                         # units
obj_expr = objective_function(model)        # kept to evaluate the full cost (incl. constant term) per alternative
gw = capunit ./ 1000                        # GW per unit
@printf("base solve: %.1fs, cost = %.6g, %d investment variables\n", base_time, base_cost, length(target))

# --- the one NOA call ---------------------------------------------------------
t1 = time()
res = generate_alternatives_optimization!(
    model,
    EPS,
    target,
    N_ALT;
    modeling_method = METHOD,
    metric = WeightedSqEuclidean(gw .^ 2),   # Max_Distance only: distance in GW of installed capacity
)
mga_time = time() - t1
@printf(
    "%d alternatives in %.1fs (%.1fs per alternative)\n",
    length(res.solutions),
    mga_time,
    mga_time / length(res.solutions)
)

# --- post-processing -----------------------------------------------------------
# GW; columns = optimum, alt1..altN
X = hcat(x0 .* gw, [[s[v] for v in target] .* gw for s in res.solutions]...)
ncol = size(X, 2)
colnames = ["optimum"; ["alt$i" for i = 1:N_ALT]]
# NOA's `objective_values` are the value of the budget constraint's function, which JuMP
# normalises by moving the objective's constant term (250,000 in Norse) into the set;
# re-evaluate the original objective on each alternative to get the full cost.
costs = [base_cost; [value(v -> s[v], obj_expr) for s in res.solutions]]
D = [norm(X[:, a] - X[:, b]) for a = 1:ncol, b = 1:ncol]

open(joinpath(OUTDIR, "alternatives.csv"), "w") do io
    println(io, "technology,capacity_per_unit_MW," * join(colnames .* "_GW", ","))
    for i in eachindex(labels)
        println(io, "\"$(labels[i])\",$(capunit[i])," * join([@sprintf("%.4f", X[i, c]) for c = 1:ncol], ","))
    end
    println(io, "\"total cost\",," * join([@sprintf("%.6e", c) for c in costs], ","))
    println(io, "\"cost / optimum\",," * join([@sprintf("%.5f", c / base_cost) for c in costs], ","))
end
open(joinpath(OUTDIR, "distances.csv"), "w") do io
    println(io, "," * join(colnames, ","))
    for a = 1:ncol
        println(io, colnames[a] * "," * join([@sprintf("%.3f", D[a, b]) for b = 1:ncol], ","))
    end
end
open(joinpath(OUTDIR, "summary.csv"), "w") do io
    println(io, "method,solver,n_alt,eps,base_solve_s,mga_total_s,base_cost,julia,noa_version")
    println(
        io,
        "$METHOD,$SOLVER,$N_ALT,$EPS,$(round(base_time, digits = 2)),$(round(mga_time, digits = 2))," *
        "$base_cost,$VERSION,$(pkgversion(NearOptimalAlternatives))",
    )
end

# LaTeX table: one row per technology, columns optimum + alternatives (GW)
texlabel(s) = replace(s, "_" => "\\_", " -> " => " \$\\to\$ ")
open(joinpath(OUTDIR, "table.tex"), "w") do io
    println(io, "\\begin{tabular}{l" * "r"^ncol * "}")
    println(io, "\\toprule")
    println(
        io,
        "\\textbf{Technology (GW)} & \\textbf{Optimum} & " *
        join(["\\textbf{Alt.~$i}" for i = 1:N_ALT], " & ") * "\\\\",
    )
    println(io, "\\midrule")
    for i in sortperm(labels)
        println(io, texlabel(labels[i]) * " & " * join([@sprintf("%.2f", X[i, c]) for c = 1:ncol], " & ") * "\\\\")
    end
    println(io, "\\midrule")
    println(io, "Cost relative to optimum & " * join([@sprintf("%.3f", c / base_cost) for c in costs], " & ") * "\\\\")
    println(io, "Distance to optimum (GW) & " * join([@sprintf("%.1f", D[1, c]) for c = 1:ncol], " & ") * "\\\\")
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end

# Console report
fmt(x, d) = lpad(@sprintf("%.*f", d, x), 9)
println("
", rpad("technology (GW)", 48), join(lpad.(colnames, 9)))
for i in sortperm(labels)
    println(rpad(labels[i], 48), join(fmt.(X[i, :], 2)))
end
println(rpad("cost / optimum", 48), join(fmt.(costs ./ base_cost, 4)))
println(rpad("distance to optimum (GW)", 48), join(fmt.(D[1, :], 1)))
println("
pairwise distances (GW):")
for a = 1:ncol
    println(rpad(colnames[a], 8), join(fmt.(D[a, :], 1)))
end

# Bandwidth figure: per technology, the range spanned by the alternatives vs the optimum
try
    @eval using Plots
    order = sortperm(vec(maximum(X; dims = 2) .- minimum(X; dims = 2)); rev = true)
    lo = vec(minimum(X[:, 2:end]; dims = 2))
    hi = vec(maximum(X[:, 2:end]; dims = 2))
    n = length(order)
    plt = plot(
        size = (800, 520),
        legend = :topright,
        xlabel = "installed capacity (GW)",
        yticks = (1:n, replace.(labels[order], "_" => " ")),
        ytickfontsize = 7,
        left_margin = 10Plots.mm,
        yflip = true,
        framestyle = :box,
    )
    for (k, i) in enumerate(order)
        plot!(
            plt,
            [lo[i], hi[i]],
            [k, k];
            lw = 6,
            color = :steelblue,
            alpha = 0.45,
            label = k == 1 ? "range of $N_ALT alternatives" : "",
        )
    end
    for c = 2:ncol
        scatter!(
            plt,
            X[order, c],
            1:n;
            ms = 3.5,
            color = :steelblue,
            markerstrokewidth = 0,
            label = c == 2 ? "alternatives" : "",
        )
    end
    scatter!(plt, X[order, 1], 1:n; ms = 5, color = :black, marker = :diamond, label = "optimum")
    savefig(plt, joinpath(OUTDIR, "bandwidth.png"))
    savefig(plt, joinpath(OUTDIR, "bandwidth.pdf"))
    println("figure -> ", joinpath(OUTDIR, "bandwidth.png"))
catch e
    @warn "figure skipped" e
end
println("[$(now())] done -> $OUTDIR")
