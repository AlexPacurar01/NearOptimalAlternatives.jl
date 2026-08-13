# ===========================================================================
# Shared plotting for the small-case corrector study (Plots/gr). Kept separate
# from smallcase_common.jl so the non-plotting correctness gate need not load
# Plots. Colours come from CORRECTOR_COLOR in smallcase_common.jl.
# ===========================================================================
using Plots
gr()

"Grouped bars: missed diversity %, solve time, iteration count per corrector."
function plot_corrector_bars(path::AbstractString, title::AbstractString, results)
    labels = [string(r.method) for r in results]
    cols = [CORRECTOR_COLOR[r.method] for r in results]
    p = plot(
        layout = (1, 3),
        size = (1250, 430),
        bottom_margin = 9Plots.mm,
        left_margin = 6Plots.mm,
    )
    bar!(
        p[1],
        labels,
        [max(r.gap_s, 1e-4) for r in results];
        legend = false,
        color = cols,
        title = "missed diversity (%)",
        xrotation = 30,
        yscale = :log10,
        ylabel = "miss% (log)",
    )
    bar!(
        p[2],
        labels,
        [r.time for r in results];
        legend = false,
        color = cols,
        title = "solve time (s)",
        xrotation = 30,
        ylabel = "s",
    )
    bar!(
        p[3],
        labels,
        [max(r.iters, 1) for r in results];
        legend = false,
        color = cols,
        title = "iterations",
        xrotation = 30,
        yscale = :log10,
        ylabel = "iters (log)",
    )
    plot!(p; plot_title = title)
    savefig(p, path)
end

"""
    plot_corrector_trace(path, title, results, obj_ref_scaled)

Two panels vs wall-clock: primal infeasibility (log) and objective wᵀx (scaled),
with the exact optimum overlaid. Only correctors with a non-empty `history`
(the gradient methods; :osqp has no per-iteration hook) are drawn. This is what
distinguishes genuine OBJECTIVE stalling (the curve plateaus ABOVE the exact
line while feasibility is met) from a method that is merely slow.
"""
function plot_corrector_trace(
    path::AbstractString,
    title::AbstractString,
    results,
    obj_ref_scaled::Float64,
)
    traced = [r for r in results if !isempty(r.history)]
    isempty(traced) && return
    p = plot(
        layout = (1, 2),
        size = (1200, 480),
        bottom_margin = 9Plots.mm,
        left_margin = 8Plots.mm,
    )
    for r in traced
        ts = [max(e.t, 1e-6) for e in r.history]
        infe = [max(e.infeas, 1e-14) for e in r.history]
        objs = [e.obj for e in r.history]
        c = CORRECTOR_COLOR[r.method]
        plot!(p[1], ts, infe; lw = 2, color = c, label = string(r.method))
        plot!(p[2], ts, objs; lw = 2, color = c, label = string(r.method))
    end
    hline!(
        p[2],
        [obj_ref_scaled];
        ls = :dot,
        lw = 2,
        color = :gray30,
        label = "exact optimum",
    )
    plot!(
        p[1];
        xlabel = "wall-clock (s)",
        ylabel = "primal infeasibility",
        yscale = :log10,
        title = "feasibility",
        legend = :topright,
    )
    plot!(
        p[2];
        xlabel = "wall-clock (s)",
        ylabel = "objective wᵀx (scaled)",
        title = "diversity objective",
        legend = :topright,
    )
    plot!(p; plot_title = title)
    savefig(p, path)
end
