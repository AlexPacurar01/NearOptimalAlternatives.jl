# Plots the per-alternative detail trace (K=100, repeat 1) from the corrected
# public-API SPORES test (experiment_spores_reconfigure_api.jl), for both
# backends: x = alternative index, y = simplex/barrier iterations (left) and
# wall time (right), barrier vs warm_primal_simplex. Plain CSV parsing (no
# CSV.jl/DataFrames.jl dependency - not in this project's Manifest).
using Plots
gr()

const OUTDIR = joinpath(pwd(), "results", "spores_native_hook")
const ARM_COLOR = Dict("barrier" => "#5778a4", "warm_primal_simplex" => "#e49444")
const ARM_LABEL = Dict("barrier" => "cold barrier", "warm_primal_simplex" => "warm simplex")

"Read detail_\$run_id.csv, keep only n_alt==100 && repeat==1, return
Dict(arm => Vector{(idx, solve_time, iters)}), iters = simplex if present
else barrier."
function load_detail(run_id)
    path = joinpath(OUTDIR, "detail_$run_id.csv")
    lines = readlines(path)
    header = split(lines[1], ",")
    col = Dict(h => i for (i, h) in enumerate(header))
    out = Dict{String,Vector{NamedTuple}}(
        "barrier" => NamedTuple[],
        "warm_primal_simplex" => NamedTuple[],
    )
    for line in lines[2:end]
        isempty(line) && continue
        f = split(line, ",")
        n_alt = parse(Int, f[col["n_alt"]])
        rep = parse(Int, f[col["repeat"]])
        (n_alt == 100 && rep == 1) || continue
        arm = f[col["arm"]]
        idx = parse(Int, f[col["idx"]])
        st = parse(Float64, f[col["solve_time_s"]])
        simplex = f[col["simplex_iterations"]]
        barrier = f[col["barrier_iterations"]]
        it =
            !isempty(simplex) && simplex != "0" ? parse(Int, simplex) :
            (!isempty(barrier) ? parse(Int, barrier) : 0)
        push!(out[arm], (idx = idx, solve_time = st, iters = it))
    end
    for arm in keys(out)
        sort!(out[arm]; by = r -> r.idx)
    end
    return out
end

function panel_pair(run_id, title)
    d = load_detail(run_id)
    p_it = plot(;
        xlabel = "alternative index",
        ylabel = "iterations (simplex or barrier)",
        legend = :topright,
        title = "$title: iterations",
    )
    p_wall = plot(;
        xlabel = "alternative index",
        ylabel = "solve time (s)",
        legend = :topright,
        title = "$title: solve time",
    )
    for arm in ("barrier", "warm_primal_simplex")
        recs = d[arm]
        plot!(
            p_it,
            [r.idx for r in recs],
            [r.iters for r in recs];
            lw = 2,
            color = ARM_COLOR[arm],
            label = ARM_LABEL[arm],
        )
        plot!(
            p_wall,
            [r.idx for r in recs],
            [r.solve_time for r in recs];
            lw = 2,
            color = ARM_COLOR[arm],
            label = ARM_LABEL[arm],
        )
    end
    return p_it, p_wall
end

g_it, g_wall = panel_pair("norse_fixed_gurobi", "Gurobi")
h_it, h_wall = panel_pair("norse_fixed_highs", "HiGHS")

fig = plot(
    g_it,
    g_wall,
    h_it,
    h_wall;
    layout = (2, 2),
    size = (1000, 760),
    plot_title = "SPORES per-alternative cost via the public API (Norse, K=100)",
)
savefig(fig, joinpath(OUTDIR, "spores_reconfigure_detail.png"))
println("wrote ", joinpath(OUTDIR, "spores_reconfigure_detail.png"))
