# ===========================================================================
# SPORES objective-warm-start experiment: does keeping the SIMPLEX BASIS across
# SPORES iterations make generating n alternatives cheap, the way a warm
# dual-simplex budget sweep makes the RHS-parametric continuation cheap
# (experiment_basis_arms.jl)?
#
# SPORES (Spores_initial!/Spores_update! in src/MGA-Methods/Spores.jl) fixes
# the near-optimal budget ONCE, then iterates
#   min w_k' x   s.t.  cost(x) <= B,  (all other constraints unchanged)
# where w_k is built ADDITIVELY from the value of iterate k-1. Only the
# OBJECTIVE changes between iterations -> the optimal basis of iteration k-1
# stays PRIMAL feasible for iteration k (parametric-cost LP / sensitivity
# "ranging" - the mirror image of the RHS case experiment_basis_arms.jl
# already validates with dual simplex; see memory
# spores-objective-warm-start-idea.md).
#
# Two arms per alternative count N:
#   A  cold_barrier          every SPORES solve is an independent barrier
#                             solve (solver state reset each time) - the
#                             package's current default behaviour
#   B  warm_primal_simplex   iteration 1 solved with barrier+crossover (basis
#                             discovery in the budget-constrained feasible
#                             region), iterations 2..N re-solved with PRIMAL
#                             simplex INHERITING the basis - only the
#                             objective coefficients change, no solver reset
#
# Reproducibility: the SPORES objective sequence w_1..w_Nmax is computed ONCE
# from a genuine reference trajectory (using the package's own
# create_alternative_generating_problem!/update_objective_function! with
# modeling_method=:Spores) and then REPLAYED identically for every arm and
# every N in the sweep. Both arms therefore solve the exact same sequence of
# LPs, so a per-iteration objective-value mismatch is a genuine solve
# discrepancy, not the arms taking different alternatives at a degenerate
# vertex.
#
# Config via env vars (mirrors experiment_basis_arms.jl):
#   SPW_SOURCE    tulipa022 | synth              (default tulipa022)
#   SPW_DATASET   Norse | Tiny (tulipa022) / synth size (default Norse)
#   SPW_BACKEND   gurobi | highs                 (default highs)
#   SPW_NLIST     comma list of alternative counts (default "2,5,10")
#   SPW_EPS       relative cost budget (optimality_gap) (default 0.1)
#   SPW_REPEATS   sweep repeats per arm (medians)  (default 3)
#   SPW_OUT       output directory                (default results/spores_warmstart)
#   SPW_ID        tag for output filenames         (default "local")
#
#   julia --project=. experiment_spores_warmstart.jl
# ===========================================================================
using JuMP, LinearAlgebra, Random, Statistics, Printf, Logging, Dates
using NearOptimalAlternatives

env(k, d) = get(ENV, k, d)
const SOURCE = Symbol(env("SPW_SOURCE", "tulipa022"))
const DATASET = env("SPW_DATASET", SOURCE === :synth ? "small" : "Norse")
const BACKEND = Symbol(env("SPW_BACKEND", "highs"))
const NLIST = sort(parse.(Int, split(env("SPW_NLIST", "2,5,10"), ",")))
const EPS_MGA = parse(Float64, env("SPW_EPS", "0.1"))
const REPEATS = parse(Int, env("SPW_REPEATS", "3"))
const OUTDIR = env("SPW_OUT", joinpath(pwd(), "results", "spores_warmstart"))
const RUN_ID = env("SPW_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "local"))
const THREADS = max(1, Threads.nthreads())

BACKEND === :gurobi ? (using Gurobi) : (using HiGHS)
SOURCE === :tulipa022 && include("smallcase_tulipa.jl")

mkpath(OUTDIR)
const POINTSF = joinpath(OUTDIR, "points_$RUN_ID.csv")
const SUMMARYF = joinpath(OUTDIR, "arm_summary_$RUN_ID.csv")

# --- backend knobs ----------------------------------------------------------
base_optimizer() =
    BACKEND === :gurobi ?
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Threads" => THREADS,
        "NumericFocus" => 3,
    ) :
    optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "threads" => THREADS)

"Point the attached solver at barrier (crossover optional), primal simplex, or dual simplex."
function set_algorithm!(model, alg::Symbol)
    if BACKEND === :gurobi
        if alg === :barrier
            set_optimizer_attribute(model, "Method", 2)
            set_optimizer_attribute(model, "Crossover", 0)
        elseif alg === :barrier_crossover
            set_optimizer_attribute(model, "Method", 2)
            set_optimizer_attribute(model, "Crossover", -1)   # default crossover
        elseif alg === :primal_simplex
            set_optimizer_attribute(model, "Method", 0)
        else                                                   # :dual_simplex
            set_optimizer_attribute(model, "Method", 1)
        end
    else
        if alg in (:barrier, :barrier_crossover)
            # bare IPM can error on degenerate small instances; crossover keeps
            # the baseline robust (mirrors experiment_basis_arms.jl).
            set_optimizer_attribute(model, "solver", "ipm")
            set_optimizer_attribute(model, "run_crossover", "on")
        elseif alg === :primal_simplex
            set_optimizer_attribute(model, "solver", "simplex")
            set_optimizer_attribute(model, "simplex_strategy", 4)  # HiGHS primal
        else
            set_optimizer_attribute(model, "solver", "simplex")
            set_optimizer_attribute(model, "simplex_strategy", 1)  # HiGHS dual
        end
    end
end

"Discard solution/basis state so the next solve is genuinely cold."
function cold_reset!(model)
    try
        inner = JuMP.unsafe_backend(model)
        if BACKEND === :gurobi
            Gurobi.GRBreset(inner, 1)
        else
            HiGHS.Highs_clearSolver(inner.inner)
        end
    catch e
        @warn "cold_reset! failed (solve may inherit state)" exception = e
    end
end

per_solve_mem(model) =
    BACKEND === :gurobi ? (
        try
            MOI.get(model, Gurobi.ModelAttribute("MaxMemUsed"))
        catch
            NaN
        end
    ) : NaN

iters(model, getter) =
    try
        getter(model)
    catch
        -1
    end

# --- model construction -------------------------------------------------------
function build_model()
    SOURCE === :tulipa022 && return load_tulipa_case(DATASET, base_optimizer())
    return build_synth()
end

function build_synth()
    Random.seed!(7)
    n, T, nstore = 12, 48, 4
    cap = 50 .+ 50 .* rand(n)
    ci = 10 .+ 10 .* rand(n)
    co = 1 .+ 5 .* rand(n, T)
    dem = 20 .+ 15 .* rand(T)
    eta = 0.9
    model = Model(base_optimizer())
    set_silent(model)
    @variable(model, 0 <= assets_investment[i = 1:n] <= cap[i])
    @variable(model, dispatch[i = 1:n, t = 1:T] >= 0)
    @variable(model, charge[s = 1:nstore, t = 1:T] >= 0)
    @variable(model, level[s = 1:nstore, t = 1:T] >= 0)
    @constraint(model, [i = 1:n, t = 1:T], dispatch[i, t] <= assets_investment[i])
    @constraint(model, [t = 1:T], sum(dispatch[i, t] for i = 1:n) >= dem[t])
    @constraint(
        model,
        [s = 1:nstore, t = 2:T],
        level[s, t] == level[s, t-1] + eta * charge[s, t] - dispatch[s, t]
    )
    @constraint(model, [s = 1:nstore], level[s, 1] == eta * charge[s, 1])
    @constraint(model, [s = 1:nstore, t = 1:T], level[s, t] <= 5 * assets_investment[s])
    @objective(
        model,
        Min,
        sum(ci[i] * assets_investment[i] for i = 1:n) +
        sum(co[i, t] * dispatch[i, t] for i = 1:n, t = 1:T) +
        0.1 * sum(charge)
    )
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    return model, [assets_investment[i] for i = 1:n]
end

# --- reference SPORES trajectory: real Spores_initial!/Spores_update! ---------
"""
    precompute_weight_sequence(model, variables, optimality_gap, nmax) -> w_snapshots

Run one genuine SPORES trajectory (cold barrier, the package's real
Spores_initial!/Spores_update!) to `nmax` alternatives, snapshotting the exact
weight vector used for each iteration's objective. Restores `model` to its
original (cost-minimising) objective and removes the budget constraint before
returning, so the caller gets a clean model plus a reproducible objective
sequence to replay for every arm/N.
"""
function precompute_weight_sequence(model, variables, optimality_gap, nmax)
    base_objective = objective_function(model)
    base_sense = objective_sense(model)

    weights = zeros(length(variables))
    w_snapshots = Vector{Vector{Float64}}(undef, nmax)

    create_alternative_generating_problem!(
        model,
        optimality_gap,
        VariableRef[],
        variables;
        weights = weights,
        modeling_method = :Spores,
    )
    w_snapshots[1] = copy(weights)
    set_algorithm!(model, :barrier)
    cold_reset!(model)
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model) "reference SPORES solve #1 failed"

    for k = 2:nmax
        update_objective_function!(
            model,
            variables;
            weights = weights,
            modeling_method = :Spores,
        )
        w_snapshots[k] = copy(weights)
        set_algorithm!(model, :barrier)
        cold_reset!(model)
        JuMP.optimize!(model)
        @assert is_solved_and_feasible(model) "reference SPORES solve #$k failed"
    end

    delete(model, model[:original_objective])
    unregister(model, :original_objective)
    set_objective_function(model, base_objective)
    set_objective_sense(model, base_sense)
    return w_snapshots
end

# --- one arm: replay the precomputed objective sequence -----------------------
"""
    run_spores_arm(model, variables, base_objective, min_cost, optimality_gap,
                   w_snapshots, n_alt, arm) -> per-point NamedTuples

`arm` is :cold_barrier or :warm_primal_simplex. Adds the budget constraint,
replays w_snapshots[1:n_alt] as the objective at each step, sweeps, and
restores the model.
"""
function run_spores_arm(
    model,
    variables,
    base_objective,
    base_sense,
    min_cost,
    optimality_gap,
    w_snapshots,
    n_alt,
    arm::Symbol,
)
    bcon = @constraint(model, base_objective <= min_cost * (1 + optimality_gap))
    pts = NamedTuple[]
    for k = 1:n_alt
        w = w_snapshots[k]
        @objective(model, Min, sum(w[i] * variables[i] for i in eachindex(w) if w[i] != 0))
        if arm === :cold_barrier
            set_algorithm!(model, :barrier)
            cold_reset!(model)
        elseif k == 1
            set_algorithm!(model, :barrier_crossover)  # basis discovery
            cold_reset!(model)
        else
            set_algorithm!(model, :primal_simplex)      # keep basis, no reset
        end
        optimize!(model)
        st = termination_status(model)
        t = solve_time(model)
        ok = st == MOI.OPTIMAL
        dobj = ok ? objective_value(model) : NaN
        push!(
            pts,
            (
                k = k,
                status = string(st),
                time = t,
                it_simplex = iters(model, simplex_iterations),
                it_barrier = iters(model, barrier_iterations),
                dobj = dobj,
                mem_gb = per_solve_mem(model),
            ),
        )
    end
    delete(model, bcon)
    set_objective_function(model, base_objective)
    set_objective_sense(model, base_sense)
    return pts
end

# --- run ------------------------------------------------------------------------
println("[$(now())] spores warmstart: $SOURCE/$DATASET backend=$BACKEND threads=$THREADS")
println("  N=$(NLIST) eps=$EPS_MGA repeats=$REPEATS -> $OUTDIR")

model, target = build_model()
base_objective = objective_function(model)
base_sense = objective_sense(model)
min_cost = objective_value(model)
println(
    "  model: $(num_variables(model)) vars, $(length(target)) structural, min cost $(round(min_cost, sigdigits=6))",
)

nmax = maximum(NLIST)
w_snapshots = precompute_weight_sequence(model, target, EPS_MGA, nmax)
println("  precomputed $nmax-step reference SPORES objective sequence")

open(POINTSF, "w") do io
    println(
        io,
        "run_id,dataset,backend,n_alt,arm,repeat,k,status,solve_time_s," *
        "simplex_iters,barrier_iters,dobj,solver_maxmem_gb",
    )
end
open(SUMMARYF, "w") do io
    println(
        io,
        "run_id,dataset,backend,n_alt,arm,median_total_s,median_per_resolve_s," *
        "max_dobj_reldiff_vs_cold,n_failed,peak_rss_gb",
    )
end

const ARMS = (:cold_barrier, :warm_primal_simplex)
for N in NLIST
    ref_dobj = nothing                       # cold_barrier front, correctness anchor
    for arm in ARMS
        totals = Float64[]
        last_pts = nothing
        for rep = 1:REPEATS
            pts = run_spores_arm(
                model,
                target,
                base_objective,
                base_sense,
                min_cost,
                EPS_MGA,
                w_snapshots,
                N,
                arm,
            )
            push!(totals, sum(p.time for p in pts))
            last_pts = pts
            open(POINTSF, "a") do io
                for p in pts
                    println(
                        io,
                        @sprintf(
                            "%s,%s,%s,%d,%s,%d,%d,%s,%.4f,%d,%d,%.8g,%.3f",
                            RUN_ID,
                            DATASET,
                            BACKEND,
                            N,
                            arm,
                            rep,
                            p.k,
                            p.status,
                            p.time,
                            p.it_simplex,
                            p.it_barrier,
                            p.dobj,
                            p.mem_gb
                        )
                    )
                end
            end
        end
        arm === :cold_barrier && (ref_dobj = [p.dobj for p in last_pts])
        nfail = count(p -> p.status != "OPTIMAL", last_pts)
        reldiff = NaN
        if ref_dobj !== nothing
            diffs = [
                abs(p.dobj - ref_dobj[p.k]) / max(abs(ref_dobj[p.k]), 1e-9) for
                p in last_pts if isfinite(p.dobj) && isfinite(ref_dobj[p.k])
            ]
            reldiff = isempty(diffs) ? NaN : maximum(diffs)
        end
        # Warm arm: per-RE-SOLVE cost excludes the k=1 basis-discovery solve.
        resolves = [p.time for p in last_pts if !(arm === :warm_primal_simplex && p.k == 1)]
        rss = Sys.maxrss() / 2^30
        @printf(
            "  N=%-3d %-19s total=%9.4fs  per-resolve=%8.4fs  dobj reldiff vs cold=%.2e  failed=%d of %d  peakRSS=%.2fGB\n",
            N,
            arm,
            median(totals),
            median(resolves),
            reldiff,
            nfail,
            length(last_pts),
            rss
        )
        open(SUMMARYF, "a") do io
            println(
                io,
                @sprintf(
                    "%s,%s,%s,%d,%s,%.4f,%.4f,%.3e,%d,%.2f",
                    RUN_ID,
                    DATASET,
                    BACKEND,
                    N,
                    arm,
                    median(totals),
                    median(resolves),
                    reldiff,
                    nfail,
                    rss
                )
            )
        end
    end
end
println("[$(now())] done -> $POINTSF, $SUMMARYF")
