# ===========================================================================
# Isolates the four costs that get conflated when worrying "switching to warm
# primal simplex needs a re-solve, and re-solves are as expensive as the
# original barrier solve":
#
#   1. barrier_only        cold barrier, NO crossover  - fastest cost-min solve
#   2. barrier_crossover    cold barrier, WITH crossover - the one-time cost of
#                            getting a valid simplex basis (crossover_overhead
#                            = #2 - #1)
#   3. switch_verify         from the #2 basis, flip Method to primal simplex
#                            and re-solve the SAME UNCHANGED problem - this is
#                            the mandatory re-solve JuMP forces after any
#                            set_optimizer_attribute call (invalidates the
#                            cached "solved" flag even though nothing about
#                            the problem or the solver's in-memory basis
#                            changed). Hypothesis: near-instant (few/zero
#                            pivots), NOT comparable to #1 or #2, because nothing
#                            about the LP changed - pivot cost tracks how much
#                            the problem moved, not how long the ORIGINAL
#                            solve took.
#   4. first_spores_warm     the real first SPORES step (budget constraint +
#                            reweighted objective) solved warm from the #3
#                            basis, vs...
#   4c. first_spores_cold     ...the same step solved cold barrier - the actual
#                            payoff comparison.
#
# Run this on your real (e.g. 40-minute) dataset to confirm the decomposition
# holds at scale before trusting the SPORES warm-start numbers measured on
# Norse: if #3 stays cheap while #1/#2 are large, the "expensive re-solve"
# fear is unfounded - the switch cost does not scale with solve time.
#
# Config via env vars (mirrors experiment_basis_arms.jl):
#   CO_SOURCE    tulipa | tulipa022 | synth      (default tulipa022)
#   CO_DATASET   Norse | Tiny (tulipa022) / data/<x> folder (tulipa) (default Norse)
#   CO_BACKEND   gurobi | highs                  (default highs)
#   CO_EPS       relative cost budget for the first SPORES step (default 0.1)
#   CO_REPEATS   repeats (medians)                (default 3)
#   CO_OUT       output directory                 (default results/crossover_overhead)
#   CO_ID        tag for output filenames          (default "local")
#
#   julia --project=. experiment_crossover_overhead.jl
# ===========================================================================
using JuMP, Random, Statistics, Printf, Logging, Dates
using NearOptimalAlternatives

env(k, d) = get(ENV, k, d)
const SOURCE = Symbol(env("CO_SOURCE", "tulipa022"))
const DATASET = env("CO_DATASET", SOURCE === :synth ? "small" : "Norse")
const BACKEND = Symbol(env("CO_BACKEND", "highs"))
const EPS_MGA = parse(Float64, env("CO_EPS", "0.1"))
const REPEATS = parse(Int, env("CO_REPEATS", "3"))
const OUTDIR = env("CO_OUT", joinpath(pwd(), "results", "crossover_overhead"))
const RUN_ID = env("CO_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "local"))
const THREADS = max(1, Threads.nthreads())

BACKEND === :gurobi ? (using Gurobi) : (using HiGHS)
if SOURCE === :tulipa
    import TulipaEnergyModel as TEM
elseif SOURCE === :tulipa022
    include("smallcase_tulipa.jl")
end

Logging.disable_logging(Logging.Info)
mkpath(OUTDIR)
const POINTSF = joinpath(OUTDIR, "points_$RUN_ID.csv")

# --- backend knobs ----------------------------------------------------------
"Cold barrier, no crossover."
opt_barrier_only() =
    BACKEND === :gurobi ?
    optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 0,
        "Threads" => THREADS,
        "NumericFocus" => 3,
        "Method" => 2,
        "Crossover" => 0,
    ) :
    optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "threads" => THREADS,
        "solver" => "ipm",
        "run_crossover" => "off",
        "presolve" => "off",
    )

"Cold barrier, with crossover - produces a valid simplex basis."
opt_barrier_crossover() =
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

"Switch the ALREADY-ATTACHED solver to primal simplex - no reset, keep the basis."
function switch_to_primal_simplex!(model)
    if BACKEND === :gurobi
        set_optimizer_attribute(model, "Method", 0)
    else
        set_optimizer_attribute(model, "solver", "simplex")
        set_optimizer_attribute(model, "simplex_strategy", 4)  # HiGHS primal
    end
end

# --- model construction -------------------------------------------------------
function build_model(optimizer)
    if SOURCE === :tulipa022
        return load_tulipa_case(DATASET, optimizer)
    elseif SOURCE === :synth
        return build_synth(optimizer)
    end
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(pwd(), "data", DATASET))
    TEM.create_model!(ep)
    model = ep.model
    set_optimizer(model, optimizer)
    set_silent(model)
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model)
    target = JuMP.VariableRef[]
    for sym in (:assets_investment, :flows_investment)
        if haskey(object_dictionary(model), sym)
            for v in object_dictionary(model)[sym]
                push!(target, v)
            end
        end
    end
    return model, target
end

function build_synth(optimizer)
    Random.seed!(7)
    n, T, nstore = 12, 48, 4
    cap = 50 .+ 50 .* rand(n)
    ci = 10 .+ 10 .* rand(n)
    co = 1 .+ 5 .* rand(n, T)
    dem = 20 .+ 15 .* rand(T)
    eta = 0.9
    model = Model(optimizer)
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

# --- run ------------------------------------------------------------------------
println("[$(now())] crossover overhead: $SOURCE/$DATASET backend=$BACKEND threads=$THREADS")
println("  eps=$EPS_MGA repeats=$REPEATS -> $OUTDIR")

open(POINTSF, "w") do io
    println(
        io,
        "run_id,dataset,backend,repeat,t_barrier_only_s,t_barrier_crossover_s," *
        "crossover_overhead_s,t_switch_verify_s,t_first_spores_warm_s,t_first_spores_cold_s",
    )
end

t1s, t2s, t3s, t4w, t4c = Float64[], Float64[], Float64[], Float64[], Float64[]
for rep = 1:REPEATS
    # 1. cold barrier, no crossover
    t0 = time()
    m1, _ = build_model(opt_barrier_only())
    t_barrier_only = time() - t0

    # 2. cold barrier, with crossover -> valid basis
    t0 = time()
    model, target = build_model(opt_barrier_crossover())
    t_barrier_crossover = time() - t0
    min_cost = objective_value(model)
    base_objective = objective_function(model)
    base_sense = objective_sense(model)

    # 3. switch to primal simplex, re-solve the SAME UNCHANGED problem
    switch_to_primal_simplex!(model)
    t0 = time()
    JuMP.optimize!(model)
    t_switch_verify = time() - t0
    @assert is_solved_and_feasible(model) "switch-verify solve failed"

    # 4. the real first SPORES step: add budget constraint + reweighted
    # objective (Spores_initial!), solved warm (continuing from the basis
    # just verified in step 3, no reset).
    weights = zeros(length(target))
    t0 = time()
    create_alternative_generating_problem!(
        model,
        EPS_MGA,
        VariableRef[],
        target;
        weights = weights,
        modeling_method = :Spores,
    )
    JuMP.optimize!(model)
    t_first_spores_warm = time() - t0
    @assert is_solved_and_feasible(model) "first warm SPORES solve failed"

    # For the cold-arm comparison, build a genuinely FRESH model (a real
    # "always cold barrier" pipeline never runs crossover at all) rather than
    # reusing the warm-arm model - reusing it would need another re-solve just
    # to re-establish true cost-optimality after the warm SPORES objective
    # was set, which would conflate "restoring the model" cost with the
    # thing we're actually measuring.
    model_cold, target_cold = build_model(opt_barrier_only())
    weights2 = zeros(length(target_cold))
    t0 = time()
    create_alternative_generating_problem!(
        model_cold,
        EPS_MGA,
        VariableRef[],
        target_cold;
        weights = weights2,
        modeling_method = :Spores,
    )
    JuMP.optimize!(model_cold)
    t_first_spores_cold = time() - t0
    @assert is_solved_and_feasible(model_cold) "first cold SPORES solve failed"

    push!(t1s, t_barrier_only)
    push!(t2s, t_barrier_crossover)
    push!(t3s, t_switch_verify)
    push!(t4w, t_first_spores_warm)
    push!(t4c, t_first_spores_cold)

    open(POINTSF, "a") do io
        println(
            io,
            @sprintf(
                "%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                RUN_ID,
                DATASET,
                BACKEND,
                rep,
                t_barrier_only,
                t_barrier_crossover,
                t_barrier_crossover - t_barrier_only,
                t_switch_verify,
                t_first_spores_warm,
                t_first_spores_cold
            )
        )
    end
end

@printf("\n  --- medians over %d repeats ---\n", REPEATS)
@printf("  1. barrier_only          %8.4fs\n", median(t1s))
@printf(
    "  2. barrier_crossover     %8.4fs  (overhead = %.4fs)\n",
    median(t2s),
    median(t2s) - median(t1s)
)
@printf(
    "  3. switch_verify         %8.4fs  <- the re-solve you're worried about\n",
    median(t3s)
)
@printf("  4. first_spores_warm     %8.4fs\n", median(t4w))
@printf("  4c. first_spores_cold    %8.4fs\n", median(t4c))
println("[$(now())] done -> $POINTSF")
