# ===========================================================================
# 2050 Investment: one corrector arm per run, for a real scaling/value
# comparison of the ideas this project adds on top of the package's plain
# default (repeated independent barrier solves). Each invocation builds its
# OWN model and does its OWN base cost-min solve - no state or cache is
# shared between arms, so all five runs are fully independent (run them in
# parallel as separate SLURM array tasks; see run_dhpc_2050_corrector_arms.slurm).
# Every arm calls only the package's public API
# (generate_alternatives_optimization!/generate_alternatives_arclength! plus
# reconfigure_solver!) - nothing here re-implements internal package logic, so
# the five arms are a fair, apples-to-apples comparison of the same real loop
# under different solver configurations.
#
# Arm regular            all_barrier_no_crossover
#     Every alternative solved independently with barrier, no crossover - the
#     package's plain default behaviour, no reconfigure_solver! at all. The
#     baseline the "warm"/"exact" arms are measured against.
# Arm exact               all_barrier_with_crossover
#     Base solve is barrier, no crossover (same reference config as "regular"),
#     then one re-solve switches to barrier + crossover before generating
#     alternatives; every alternative after that still solves via barrier +
#     crossover (the attribute persists - no further reconfigure_solver! is
#     needed). This is the "clean vertex every time" reference the warm arms'
#     returned points are checked against, and it directly measures the
#     crossover overhead itself (base solve time with vs without crossover).
# Arm warm                crossover_then_warm_primal_simplex
#     Same barrier-no-crossover -> barrier-crossover pre-step as "exact", to
#     obtain a starting basis; generate_alternatives_optimization!'s own
#     reconfigure_solver! hook then switches to PRIMAL simplex after
#     alternative #1. Primal, not dual: between successive alternatives here
#     only the *objective* changes (the SPORES weight update), not the budget
#     constraint, which is the textbook case for primal-simplex re-optimization
#     (Section 3.5), the mirror image of the dual-simplex case the two
#     continuation arms below use.
# Arm continuation         arclength_all_barrier_no_crossover
#     generate_alternatives_arclength! (single direction, n_budget points along
#     its near-optimal front), barrier, no crossover throughout, no
#     reconfigure_solver! at all. The baseline "warm_continuation" is measured
#     against - the arclength-side counterpart of "regular".
# Arm warm_continuation    arclength_warm_dual_simplex
#     Same barrier-no-crossover -> barrier-crossover pre-step, then
#     generate_alternatives_arclength! with the same reconfigure_solver! switch
#     to DUAL simplex after its first point. Dual, not primal: within one
#     arclength direction only the *budget* (right-hand side) moves, not the
#     objective, the textbook case for dual-simplex re-optimization (Section
#     3.4) already validated at small scale in Table basisarms. The budget
#     constraint itself is created once and its RHS updated in place by that
#     function's own internal mechanism (constraint_by_name + set_normalized_rhs)
#     - nothing about that needed changing here.
#
# Per-alternative diagnostics (solve time, simplex/barrier iterations, status)
# come from the "alternative_solved" structured @info record both
# generate_alternatives_optimization! and generate_alternatives_arclength!
# emit after every solve - captured here with a small custom logger, same as
# experiment_spores_reconfigure_api.jl.
#
# Config via environment variables:
#   ARMS2050_ARM      regular | exact | warm | continuation | warm_continuation
#                     (required)
#   ARMS2050_K        alternatives for regular/exact/warm (default 5; the first
#                     real run measured ~136 min/alternative for "regular"
#                     without crossover on this model - a K large enough to
#                     exceed the 24h Education-account SLURM time limit is
#                     fine, since results are written to disk after every
#                     alternative, so a time-limit kill still keeps whatever
#                     completed)
#   ARMS2050_NBUDGET  arclength points for continuation/warm_continuation
#                     (default 5)
#   ARMS2050_EPS      relative cost budget (default 0.1)
#   ARMS2050_NUMERICFOCUS  Gurobi NumericFocus (default 3)
#   ARMS2050_OUT      output directory (default results/2050_investment/corrector_arms)
#   ARMS2050_ID       tag for output filenames (default SLURM job id or "local")
#
#   ARMS2050_ARM=regular JULIA_NUM_THREADS=96 julia -t 96 --project=. experiment_2050_corrector_arms.jl
# ===========================================================================
using JuMP, Statistics, Printf, Logging, Dates
import Gurobi
using NearOptimalAlternatives
include(joinpath(@__DIR__, "dataset_2050_investment.jl"))

env(k, d) = get(ENV, k, d)
const VALID_ARMS = (:regular, :exact, :warm, :continuation, :warm_continuation)
const ARM = Symbol(env("ARMS2050_ARM", ""))
ARM in VALID_ARMS || error("ARMS2050_ARM=$ARM must be one of $VALID_ARMS")
const K = parse(Int, env("ARMS2050_K", "5"))
const NBUDGET = parse(Int, env("ARMS2050_NBUDGET", "5"))
const EPS_MGA = parse(Float64, env("ARMS2050_EPS", "0.1"))
const NUMERICFOCUS = parse(Int, env("ARMS2050_NUMERICFOCUS", "3"))
const OUTDIR =
    env("ARMS2050_OUT", joinpath(@__DIR__, "results", "2050_investment", "corrector_arms"))
const RUN_ID = env("ARMS2050_ID", get(ENV, "SLURM_JOB_ID", "local")) * "_" * string(ARM)
const THREADS = max(1, Threads.nthreads())

mkpath(OUTDIR)
const POINTSF = joinpath(OUTDIR, "arms_$RUN_ID.csv")
const DETAILF = joinpath(OUTDIR, "detail_$RUN_ID.csv")

# --- solver algorithm switches (Gurobi) ---------------------------------------
to_barrier_no_crossover!(model) = begin
    set_optimizer_attribute(model, "Method", 2)
    set_optimizer_attribute(model, "Crossover", 0)
end
to_barrier_with_crossover!(model) = begin
    set_optimizer_attribute(model, "Method", 2)
    set_optimizer_attribute(model, "Crossover", -1)
end
to_primal_simplex!(model) = set_optimizer_attribute(model, "Method", 0)
to_dual_simplex!(model) = set_optimizer_attribute(model, "Method", 1)

base_optimizer() = optimizer_with_attributes(
    Gurobi.Optimizer,
    "OutputFlag" => 1,
    "LogFile" => joinpath(OUTDIR, "gurobi_$RUN_ID.log"),
    "Threads" => THREADS,
    "NumericFocus" => NUMERICFOCUS,
    "Method" => 2,
    "Crossover" => 0,
)

# --- capture "alternative_solved" log records, nothing else -------------------
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
    message == "alternative_solved" && push!(logger.records, NamedTuple(kwargs))
    return nothing
end

println(
    "[$(now())] 2050 corrector arm run: arm=$ARM K=$K n_budget=$NBUDGET eps=$EPS_MGA threads=$THREADS -> $OUTDIR",
)

open(POINTSF, "w") do io
    println(
        io,
        "run_id,arm,n_points,total_wall_s,total_solver_s,total_simplex_iters,total_barrier_iters,build_time_s,base_solve_time_s,precrossover_time_s,min_cost",
    )
end
open(DETAILF, "w") do io
    println(
        io,
        "run_id,arm,idx,direction,budget_idx,solve_time_s,simplex_iterations,barrier_iterations,status",
    )
end

# --- build + base cost-min solve (barrier, no crossover) - fresh every run ----
println("[$(now())] building 2050 Investment model...")
flush(stdout)
tbuild = @elapsed (model, target) = load_2050_investment(base_optimizer(); solve = false)
nvars = num_variables(model)
ncons = num_constraints(model; count_variable_in_set_constraints = false)
@printf(
    "[%s] build done in %.1fs: %d vars, %d cons, %d investment vars\n",
    now(),
    tbuild,
    nvars,
    ncons,
    length(target),
)
flush(stdout)

println("[$(now())] base cost-min solve (barrier, no crossover)...")
flush(stdout)
tbase = @elapsed JuMP.optimize!(model)
@assert is_solved_and_feasible(model) "base 2050 solve failed: $(termination_status(model))"
min_cost = objective_value(model)
@printf("[%s] base solved (no crossover) in %.1fs. C* = %.6g\n", now(), tbase, min_cost)
flush(stdout)

# --- arm-specific pre-step (all but "regular"/"continuation") + generation ----
# Every arm that needs a simplex basis to warm-start from (exact, warm,
# warm_continuation) re-solves the SAME base problem once more with crossover
# enabled, so both the without-crossover (tbase, above) and with-crossover
# (tprecross, below) initial-solve times are always recorded and directly
# comparable across arms.
tprecross = 0.0
if ARM !== :regular && ARM !== :continuation
    println(
        "[$(now())] pre-step: switching to barrier + crossover before generating alternatives...",
    )
    flush(stdout)
    to_barrier_with_crossover!(model)
    tprecross = @elapsed JuMP.optimize!(model)
    @assert is_solved_and_feasible(model) "crossover re-confirm of x* failed"
    @printf("[%s] base re-solved (with crossover) in %.1fs\n", now(), tprecross)
    flush(stdout)
end

println("[$(now())] generating alternatives (arm=$ARM)...")
flush(stdout)
collector = SolveLogCollector(NamedTuple[])
wall = @elapsed Logging.with_logger(collector) do
    if ARM === :regular
        generate_alternatives_optimization!(
            model,
            EPS_MGA,
            target,
            K;
            modeling_method = :Spores,
            reconfigure_solver! = nothing,
        )
    elseif ARM === :exact
        # Model is already on barrier+crossover after the pre-step above, and
        # that setting persists across solves - no reconfigure_solver! needed
        # to keep every alternative on the same "clean vertex every time" config.
        generate_alternatives_optimization!(
            model,
            EPS_MGA,
            target,
            K;
            modeling_method = :Spores,
            reconfigure_solver! = nothing,
        )
    elseif ARM === :warm
        generate_alternatives_optimization!(
            model,
            EPS_MGA,
            target,
            K;
            modeling_method = :Spores,
            reconfigure_solver! = to_primal_simplex!,
        )
    elseif ARM === :continuation
        generate_alternatives_arclength!(
            model,
            EPS_MGA,
            target,
            1;   # single direction
            n_budget = NBUDGET,
            modeling_method = :Spores,
            reconfigure_solver! = nothing,
        )
    else   # :warm_continuation
        # n_directions=1, so its returned restore closure never actually fires
        # (there is no next direction) - included anyway to match the real
        # contract (reconfigure_solver! must return either nothing or a
        # restore closure).
        generate_alternatives_arclength!(
            model,
            EPS_MGA,
            target,
            1;   # single direction
            n_budget = NBUDGET,
            modeling_method = :Spores,
            reconfigure_solver! = m -> begin
                to_dual_simplex!(m)
                to_barrier_with_crossover!
            end,
        )
    end
end

# --- record results -------------------------------------------------------------
points = collector.records
n = length(points)
total_solver = sum(p.solve_time for p in points; init = 0.0)
total_simplex =
    sum(p -> p.simplex_iterations === missing ? 0 : p.simplex_iterations, points; init = 0)
total_barrier =
    sum(p -> p.barrier_iterations === missing ? 0 : p.barrier_iterations, points; init = 0)

open(POINTSF, "a") do io
    println(
        io,
        @sprintf(
            "%s,%s,%d,%.4f,%.4f,%d,%d,%.4f,%.4f,%.4f,%.6g",
            RUN_ID,
            ARM,
            n,
            wall,
            total_solver,
            total_simplex,
            total_barrier,
            tbuild,
            tbase,
            tprecross,
            min_cost
        )
    )
end
open(DETAILF, "a") do io
    for (i, p) in enumerate(points)
        println(
            io,
            @sprintf(
                "%s,%s,%d,%s,%s,%.4f,%s,%s,%s",
                RUN_ID,
                ARM,
                i,
                haskey(p, :direction) ? string(p.direction) : "",
                haskey(p, :budget_idx) ? string(p.budget_idx) : "",
                p.solve_time,
                p.simplex_iterations === missing ? "" : string(p.simplex_iterations),
                p.barrier_iterations === missing ? "" : string(p.barrier_iterations),
                p.status,
            )
        )
    end
end
avg_per_alt = n > 0 ? total_solver / n : 0.0
@printf(
    "\n[%s] DONE arm=%s n=%d wall=%.2fs solver=%.2fs avg_per_alt=%.2fs simplex_it=%d barrier_it=%d build=%.1fs base(no-cross)=%.1fs base(cross)=%.1fs -> %s\n",
    now(),
    ARM,
    n,
    wall,
    total_solver,
    avg_per_alt,
    total_simplex,
    total_barrier,
    tbuild,
    tbase,
    tprecross,
    POINTSF,
)
