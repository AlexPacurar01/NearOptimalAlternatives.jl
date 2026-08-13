# ===========================================================================
# SPORES warm-start test via the PUBLIC API ONLY - this is the corrected
# replacement for the custom re-implemented loop in
# experiment_spores_native_hook.jl, which had a real bug: SPORES accumulates
# its weights vector across alternatives (Spores_update!/Spores_initial! do
# `weights[i] += value(v)/upper_bound(v)`, never resetting), and the correct
# behaviour depends on ONE `weights` array being threaded through every call.
# `generate_alternatives_optimization!` does this correctly (one `weights`
# keyword, reused for the whole run); the custom loop never passed `weights`
# to `create_alternative_generating_problem!`/`update_objective_function!` at
# all, so every call silently fell back to a fresh `zeros(...)` default - no
# accumulation, a materially different (and wrong) trajectory. This script
# goes through NOTHING but `generate_alternatives_optimization!` itself, so
# that bug class cannot recur: there is no second, hand-rolled copy of the
# loop to get out of sync with the real one.
#
# Two arms, both expressible with the existing `reconfigure_solver!` hook
# (fires once, between alternative #1 and #2) - no model/solver reset of any
# kind, no per-solve intervention:
#   barrier              reconfigure_solver! = nothing - the model keeps
#                         using the base solve's barrier+crossover config for
#                         every alternative; the package never touches a
#                         solver attribute for this arm.
#   warm_primal_simplex   reconfigure_solver! switches to primal simplex once,
#                         after alternative #1; every alternative after that
#                         inherits the basis, warm-started by the solver's own
#                         defaults - nothing in this script or the package
#                         resets anything.
#
# Per-alternative diagnostics (solve time, simplex/barrier iterations) come
# from the `"alternative_solved"` structured @info record now emitted by
# `generate_alternatives_optimization!` itself (pure logging addition to the
# package, no behaviour change) - captured here with a small custom logger,
# not by re-deriving them from a separate solve loop.
#
# Config via env vars (same meanings as experiment_spores_native_hook.jl):
#   SPN_DATASET, SPN_BACKEND, SPN_KLIST, SPN_EPS, SPN_REPEATS, SPN_OUT, SPN_ID
#
#   julia --project=. experiment_spores_reconfigure_api.jl
# ===========================================================================
using JuMP, Statistics, Printf, Logging, Dates
using NearOptimalAlternatives

env(k, d) = get(ENV, k, d)
const DATASET = env("SPN_DATASET", "Norse")
const BACKEND = Symbol(env("SPN_BACKEND", "highs"))
const KLIST = sort(parse.(Int, split(env("SPN_KLIST", "2,5,10,25,50,100"), ",")))
const EPS_MGA = parse(Float64, env("SPN_EPS", "0.2"))
const REPEATS = parse(Int, env("SPN_REPEATS", "3"))
const OUTDIR = env("SPN_OUT", joinpath(pwd(), "results", "spores_native_hook"))
const RUN_ID = env("SPN_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "local"))
const THREADS = max(1, Threads.nthreads())

BACKEND === :gurobi ? (using Gurobi) : (using HiGHS)
include("smallcase_tulipa.jl")

mkpath(OUTDIR)
const POINTSF = joinpath(OUTDIR, "points_$RUN_ID.csv")
const DETAILF = joinpath(OUTDIR, "detail_$RUN_ID.csv")

base_optimizer() =
    BACKEND === :gurobi ?
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Threads" => THREADS,
        "NumericFocus" => 3,
        "Method" => 2,
        "Crossover" => -1,
    ) :
    optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "threads" => THREADS,
        "solver" => "ipm",
        "run_crossover" => "on",
        "presolve" => "off",
    )

"reconfigure_solver! for the warm arm: switch the already-attached solver to
primal simplex, inheriting the basis from the crossover base solve. The
barrier arm passes `nothing` - it never touches a solver attribute."
function warm_reconfigure!(model)
    if BACKEND === :gurobi
        set_optimizer_attribute(model, "Method", 0)
    else
        set_optimizer_attribute(model, "solver", "simplex")
        set_optimizer_attribute(model, "simplex_strategy", 4)
        set_optimizer_attribute(model, "presolve", "off")
    end
end
reconfigure_for(arm::Symbol) = arm === :warm_primal_simplex ? warm_reconfigure! : nothing

# --- capture the package's "alternative_solved" log records, nothing else ---
struct SolveLogCollector <: Logging.AbstractLogger
    records::Vector{NamedTuple}
end
Logging.min_enabled_level(::SolveLogCollector) = Logging.Info
Logging.shouldlog(::SolveLogCollector, level, _module, group, id) = true
Logging.catch_exceptions(::SolveLogCollector) = false
function Logging.handle_message(
    logger::SolveLogCollector,
    level,
    message,
    _module,
    group,
    id,
    file,
    line;
    kwargs...,
)
    if message == "alternative_solved"
        push!(logger.records, NamedTuple(kwargs))
    end
    return nothing
end

println(
    "[$(now())] spores reconfigure-API test: $DATASET backend=$BACKEND threads=$THREADS",
)
println("  K=$(KLIST) eps=$EPS_MGA repeats=$REPEATS -> $OUTDIR")

open(POINTSF, "w") do io
    println(
        io,
        "run_id,dataset,backend,n_alt,arm,repeat,base_solve_time_s," *
        "spores_total_s,spores_per_alt_avg_s,total_simplex_iters,total_barrier_iters," *
        "min_cost,final_diversity_l2",
    )
end
open(DETAILF, "w") do io
    println(
        io,
        "run_id,dataset,backend,n_alt,arm,repeat,idx,solve_time_s,simplex_iterations,barrier_iterations,status",
    )
end

const ARMS = (:barrier, :warm_primal_simplex)
for K in KLIST
    for arm in ARMS
        base_times = Float64[]
        spores_times = Float64[]
        for rep = 1:REPEATS
            t0 = time()
            model, target = load_tulipa_case(DATASET, base_optimizer())
            base_time = time() - t0
            min_cost = objective_value(model)
            x0 = Dict(v => value(v) for v in target)

            collector = SolveLogCollector(NamedTuple[])
            local result
            spores_time = @elapsed Logging.with_logger(collector) do
                result = generate_alternatives_optimization!(
                    model,
                    EPS_MGA,
                    target,
                    K;
                    modeling_method = :Spores,
                    reconfigure_solver! = reconfigure_for(arm),
                )
            end

            final_sol = result.solutions[end]
            final_div = sqrt(sum((final_sol[v] - x0[v])^2 for v in target))

            recs = sort(collector.records; by = r -> r.index)
            total_simplex = sum(
                r -> r.simplex_iterations === missing ? 0 : r.simplex_iterations,
                recs;
                init = 0,
            )
            total_barrier = sum(
                r -> r.barrier_iterations === missing ? 0 : r.barrier_iterations,
                recs;
                init = 0,
            )

            push!(base_times, base_time)
            push!(spores_times, spores_time)

            open(POINTSF, "a") do io
                println(
                    io,
                    @sprintf(
                        "%s,%s,%s,%d,%s,%d,%.4f,%.4f,%.4f,%d,%d,%.6g,%.6g",
                        RUN_ID,
                        DATASET,
                        BACKEND,
                        K,
                        arm,
                        rep,
                        base_time,
                        spores_time,
                        spores_time / K,
                        total_simplex,
                        total_barrier,
                        min_cost,
                        final_div
                    )
                )
            end
            open(DETAILF, "a") do io
                for r in recs
                    println(
                        io,
                        @sprintf(
                            "%s,%s,%s,%d,%s,%d,%d,%.4f,%s,%s,%s",
                            RUN_ID,
                            DATASET,
                            BACKEND,
                            K,
                            arm,
                            rep,
                            r.index,
                            r.solve_time,
                            r.simplex_iterations === missing ? "" :
                            string(r.simplex_iterations),
                            r.barrier_iterations === missing ? "" :
                            string(r.barrier_iterations),
                            r.status
                        )
                    )
                end
            end
        end
        @printf(
            "  K=%-4d %-19s base(excluded)=%7.3fs  spores_total=%9.4fs  per_alt=%8.4fs\n",
            K,
            arm,
            median(base_times),
            median(spores_times),
            median(spores_times) / K
        )
    end
end
println("[$(now())] done -> $POINTSF, $DETAILF")
