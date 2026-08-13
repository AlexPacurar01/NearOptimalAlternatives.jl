# ===========================================================================
# Shared boundary-walk-vs-SPORES experiment core (dataset-agnostic).
#
# Given a `build_fn() -> (solved_model, target_vars)`, this drives the SPORES
# sequence to capture each direction w_i and its full-budget point, then runs
# `boundary_walk` along every w_i for a set of variants (:raw, a gamma sweep of
# :cost_aware, and parameter-free :lexicographic), reporting for each whether the
# walk finds a point matching SPORES' diversity at strictly lower original cost.
# Entry scripts (synthetic / data-5h) supply only `build_fn` and the config.
# ===========================================================================
using JuMP, LinearAlgebra, Printf, Statistics, Logging
using Plots
using NearOptimalAlternatives

"""
    spores_directions(build_fn, n_alts, eps_mga) -> (dirs, sp_cost, sp_dobj)

Drive the SPORES sequence on a fresh model from `build_fn`, returning for each of
`n_alts` steps the full-length accumulated direction `w_i`, and the SPORES point's
original cost `c'x` and diversity `w_hat_i'x` (unit-normalised direction). SPORES
solves at the full budget every step.
"""
function spores_directions(build_fn, n_alts::Int, eps_mga::Float64)
    A, tA = build_fn()
    all_v = all_variables(A)
    idx = Dict(v => i for (i, v) in enumerate(all_v))

    # Original cost coefficients, captured before SPORES overwrites the objective.
    data0 = lp_matrix_data(A)
    col0 = Dict(v => j for (j, v) in enumerate(data0.variables))
    cvec0 = [data0.c[col0[v]] for v in all_v]
    coff0 = data0.c_offset

    weights = zeros(length(tA))
    with_logger(NullLogger()) do
        create_alternative_generating_problem!(
            A,
            eps_mga,
            VariableRef[],
            tA;
            weights = weights,
            modeling_method = :Spores,
        )
    end

    dirs = Vector{Float64}[]
    sp_cost = Float64[]
    sp_dobj = Float64[]
    for step = 1:n_alts
        step > 1 && with_logger(NullLogger()) do
            update_objective_function!(A, tA; weights = weights, modeling_method = :Spores)
        end
        optimize!(A)
        @assert is_solved_and_feasible(A)
        x = value.(all_v)

        w_full = zeros(length(all_v))
        for (j, v) in enumerate(tA)
            w_full[idx[v]] = weights[j]
        end
        w_hat = w_full ./ max(norm(w_full), eps(Float64))
        push!(dirs, w_full)
        push!(sp_dobj, dot(w_hat, x))
        push!(sp_cost, dot(cvec0, x) + coff0)
    end
    return dirs, sp_cost, sp_dobj
end

"""
    run_walk_experiment(build_fn; kwargs...)

Run the full SPORES-vs-walk comparison and write `walk_dominance.csv` + a
representative frontier plot to `outdir`. `build_fn` is called once for the clean
walk model and once (fresh) to drive SPORES. Returns the result rows (Vector of
NamedTuple) for further aggregation.
"""
function run_walk_experiment(
    build_fn;
    n_alts::Int,
    eps_mga::Float64,
    n_steps::Int,
    method::Symbol,
    gammas::Vector{Float64},
    outdir::String,
    max_iters::Int = 30,
    max_inner::Int = 1000,
    pdhg_iters::Int = 10000,
    early_stop_k::Int = 0,
)
    mkpath(outdir)
    modelB, _ = build_fn()
    all_vB = all_variables(modelB)
    min_cost = objective_value(modelB)
    @info "walk model: $(length(all_vB)) vars | min_cost=$(round(min_cost, sigdigits=8)) | budget=$(round(min_cost*(1+eps_mga), sigdigits=8))"

    dirs, sp_cost, sp_dobj = spores_directions(build_fn, n_alts, eps_mga)

    variants = vcat(
        [("raw", :raw, 0.0)],
        [(@sprintf("cost_aware_g%.3f", g), :cost_aware, g) for g in gammas],
        [("lexicographic", :lexicographic, 0.0)],
    )

    header = "step,variant,gamma,spores_cost,spores_dobj,walk_best_cost,n_dominated,cost_saving,frac_saving,walk_time_s"
    rows = String[header]
    plotted = false

    n_variants = length(variants)
    for (i, w) in enumerate(dirs)
        for (vi, (label, dir, g)) in enumerate(variants)
            @info "walk step $i/$(length(dirs)), variant $vi/$n_variants ($label) ..."
            r = boundary_walk(
                modelB,
                w,
                all_vB;
                eps_slack = eps_mga,
                n_steps = n_steps,
                method = method,
                direction = dir,
                gamma = g,
                max_iters = max_iters,
                max_inner = max_inner,
                pdhg_iters = pdhg_iters,
                early_stop_k = early_stop_k,
            )
            cand_cost = [p.cost for p in r.points]
            cand_dobj = [p.dobj for p in r.points]

            # Cheapest walk point matching SPORES' diversity within a small
            # relative band (FOM objective accuracy >> 1e-9). Dominates iff that
            # point is materially cheaper than SPORES.
            dtol = 1e-4 * max(1.0, abs(sp_dobj[i]))
            ctol = 1e-6 * max(1.0, abs(sp_cost[i]))
            matching = [p.cost for p in r.points if p.dobj <= sp_dobj[i] + dtol]
            best = isempty(matching) ? NaN : minimum(matching)
            saving = isnan(best) ? 0.0 : sp_cost[i] - best
            n_dom = (!isnan(best) && saving > ctol) ? 1 : 0

            push!(
                rows,
                @sprintf(
                    "%d,%s,%.4f,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.3f",
                    i,
                    label,
                    g,
                    sp_cost[i],
                    sp_dobj[i],
                    best,
                    n_dom,
                    saving,
                    saving / max(abs(sp_cost[i]), eps()),
                    r.total_time,
                )
            )
            @info "  step $i $label: dominated=$(n_dom==1) saving=$(round(saving, sigdigits=4)) ($(round(100*saving/max(abs(sp_cost[i]),eps()), sigdigits=3))%) in $(round(r.total_time, digits=1))s"

            if !plotted && dir == :lexicographic
                plotted = true
                p = plot(;
                    xlabel = "original cost  c'x",
                    ylabel = "diversity objective  w'x",
                    title = "Boundary walk vs SPORES (step $i, lexicographic)",
                    legend = :topright,
                    size = (760, 560),
                )
                scatter!(p, cand_cost, cand_dobj; label = "walk", marker = :circle, ms = 5)
                scatter!(
                    p,
                    [sp_cost[i]],
                    [sp_dobj[i]];
                    label = "SPORES",
                    marker = :star5,
                    ms = 10,
                    color = :red,
                )
                savefig(p, joinpath(outdir, "frontier_step$(i)_lex.png"))
            end
        end
    end

    open(joinpath(outdir, "walk_dominance.csv"), "w") do io
        println(io, join(rows, "\n"))
    end
    println("\n", join(rows, "\n"))
    println("\nWrote results to $outdir")
    return rows
end
