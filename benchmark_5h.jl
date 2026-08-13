# ===========================================================================
# MGA method benchmark on the real data/5h TulipaEnergy model.
#
# Same comparison as benchmark_mga.jl (cold-start wall-clock + decision-space
# deviation + diversity metrics) but on the 1.27M-variable model. Because each
# solve is ~minutes, the config is intentionally minimal: Gurobi only,
# N_ALTS=2, ONE capped iteration budget per gradient method (no sweep). Even so
# this is a multi-HOUR run - launch it deliberately and keep the machine awake:
#
#     julia -t 8 --project=. benchmark_5h.jl
#
# Outputs: results/benchmark_5h/ (PNG plots + dominance CSV).
# ===========================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, LinearAlgebra, Logging
using NearOptimalAlternatives
include("bench_common.jl")

const THREADS = max(1, Threads.nthreads())
const N_ALTS = 2
const EPS_MGA = 0.1
const TARGET_GAP = 0.05
const BACKEND = :gurobi
const OUTDIR = joinpath(pwd(), "results", "benchmark_5h")
# capped gradient budgets (single value each; raise to trade time for accuracy)
const GRAD_BUDGET = Dict(:alm_lbfgs => 4, :penalty => 4, :pdhg => 1000, :oracle => 2)
gr()
mkpath(OUTDIR)

"Build + cold-solve a fresh data/5h model; returns (model, target_vars)."
function build_5h()
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(pwd(), "data/5h"))
    TEM.create_model!(ep)
    model = ep.model
    set_optimizer(model, make_optimizer(BACKEND, THREADS))
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

target_idx_of(model, target) =
    (d = Dict(v => i for (i, v) in enumerate(all_variables(model))); [d[v] for v in target])

println("Benchmark 5h: THREADS=$THREADS N_ALTS=$N_ALTS backend=$BACKEND")

# --- Standard methods: each needs its own freshly solved model (they mutate it) ---
function run_standard_5h(modeling_method)
    println("  building + solving fresh 5h model for $modeling_method ...")
    model, target = build_5h()
    tidx = target_idx_of(model, target)
    all_v = all_variables(model)
    local result
    t = @elapsed result = with_logger(NullLogger()) do
        generate_alternatives_optimization!(
            model,
            EPS_MGA,
            target,
            N_ALTS;
            modeling_method = modeling_method,
        )
    end
    alts = [[sol[v] for v in all_v] for sol in result.solutions]
    return alts, t, target, tidx
end

println("== SPORES (exact reference) ==")
ref_alts, ref_t, rtgt, rtidx = run_standard_5h(:Spores)
Pref = normalised_struct(ref_alts, rtgt, rtidx)
println("   SPORES: $(round(ref_t, digits=1))s")

println("== HSJ (exact) ==")
hsj_alts, hsj_t, htgt, htidx = run_standard_5h(:HSJ)
Ph = normalised_struct(hsj_alts, htgt, htidx)
println("   HSJ: $(round(hsj_t, digits=1))s")

# --- Gradient methods: reuse ONE base model (pass the cached optimum so no
# re-solve is needed; each method restores the model after itself). ---
println("== building base 5h model for gradient methods ==")
base_model, base_target = build_5h()
base_vals = value.(all_variables(base_model))
base_obj = objective_value(base_model)
base_tidx = target_idx_of(base_model, base_target)
base_all = all_variables(base_model)

function run_gradient_5h(method)
    println("  running $method (budget $(GRAD_BUDGET[method])) ...")
    local result
    t = @elapsed result = with_logger(NullLogger()) do
        generate_alternatives_gradient(
            base_model,
            EPS_MGA,
            base_target,
            N_ALTS;
            method = method,
            operational_recovery = true,
            x_optimal = base_vals,
            min_cost = base_obj,
            max_iters = (method == :pdhg ? 30 : GRAD_BUDGET[method]),
            pdhg_iters = (method == :pdhg ? GRAD_BUDGET[method] : 10000),
        )
    end
    alts = [[sol[v] for v in base_all] for sol in result.solutions]
    return alts, t
end

results = MethodResult[]
function record!(name, t, gap, P)
    mpw, minpw = pairwise_stats(P)
    push!(
        results,
        MethodResult(name, t, gap, P, mpw, minpw, coverage(P), hull_area(pca2(P))),
    )
end

record!("SPORES", ref_t, NaN, Pref)
record!("HSJ", hsj_t, NaN, Ph)
for method in (:alm_lbfgs, :penalty, :pdhg, :oracle)
    alts, t = run_gradient_5h(method)
    P = normalised_struct(alts, base_target, base_tidx)
    record!(string(method), t, trajectory_gap(P, Pref), P)
    println(
        "   $method: $(round(t, digits=1))s | dev $(round(trajectory_gap(P, Pref), digits=4))",
    )
end

make_outputs(
    results,
    Dict{String,Vector{Tuple{Float64,Float64}}}(),
    ref_t,
    hsj_t,
    BACKEND,
    OUTDIR,
    N_ALTS,
    THREADS,
    TARGET_GAP,
)
println("\nWrote plots + CSV to $OUTDIR")
