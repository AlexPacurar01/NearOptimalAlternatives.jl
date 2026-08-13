# ===========================================================================
# Boundary-walk vs SPORES on the synthetic stiff-storage model (HiGHS).
#
# Hypothesis: SPORES solves min w_i'x s.t. cost <= B(full budget) and spends the
# whole budget, including on the cost-parallel component of its move - cost that
# buys no extra diversity. Walking the cost boundary from x* along w_i should
# find a point matching SPORES' diversity (w_i'x) at strictly lower cost.
#
# This is the cheap/fast harness; the decisive test is benchmark_walk_5h.jl.
# Run:  julia -t 4 --project=. benchmark_walk.jl   (BENCH_QUICK=1 for a smoke test)
# Outputs: results/walk/ (dominance CSV + frontier plot).
# ===========================================================================
using JuMP, Random
import HiGHS

include(joinpath(@__DIR__, "bench_common.jl"))      # dominance core + helpers
include(joinpath(@__DIR__, "walk_experiment.jl"))   # shared experiment core

const QUICK = haskey(ENV, "BENCH_QUICK")
const THREADS = max(1, Threads.nthreads())
const N_ALTS = QUICK ? 3 : 6
const EPS_MGA = 0.1
const SEED = 7
const N_STRUCT = QUICK ? 6 : 12
const T = QUICK ? 12 : 48
const N_STORE = QUICK ? 2 : 4
const N_STEPS = QUICK ? 6 : 12
const GAMMAS = QUICK ? [0.02, 0.1] : [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
const OUTDIR = joinpath(@__DIR__, "results", "walk")

gr()
println(
    "Walk benchmark (synthetic): THREADS=$THREADS N_ALTS=$N_ALTS size=($N_STRUCT struct,$T steps,$N_STORE storage)",
)

opt() =
    optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "threads" => THREADS)

"Build and solve the stiff synthetic storage model; returns (model, target investment vars)."
function build_solved_model()
    Random.seed!(SEED)
    cap_max = 50.0 .+ 50.0 .* rand(N_STRUCT)
    c_inv = 10.0 .+ 10.0 .* rand(N_STRUCT)
    c_op = 1.0 .+ 5.0 .* rand(N_STRUCT, T)
    demand = 20.0 .+ 15.0 .* rand(T)
    eta = 0.9

    model = Model(opt())
    set_silent(model)
    @variable(model, 0 <= assets_investment[i = 1:N_STRUCT] <= cap_max[i])
    @variable(model, dispatch[i = 1:N_STRUCT, t = 1:T] >= 0)
    @variable(model, charge[s = 1:N_STORE, t = 1:T] >= 0)
    @variable(model, level[s = 1:N_STORE, t = 1:T] >= 0)
    @constraint(model, [i = 1:N_STRUCT, t = 1:T], dispatch[i, t] <= assets_investment[i])
    @constraint(model, [t = 1:T], sum(dispatch[i, t] for i = 1:N_STRUCT) >= demand[t])
    @constraint(
        model,
        [s = 1:N_STORE, t = 2:T],
        level[s, t] == level[s, t-1] + eta * charge[s, t] - dispatch[s, t]
    )
    @constraint(model, [s = 1:N_STORE], level[s, 1] == eta * charge[s, 1])
    @constraint(model, [s = 1:N_STORE, t = 1:T], charge[s, t] <= assets_investment[s])
    @constraint(model, [s = 1:N_STORE, t = 1:T], level[s, t] <= 5 * assets_investment[s])
    @objective(
        model,
        Min,
        sum(c_inv[i] * assets_investment[i] for i = 1:N_STRUCT) +
        sum(c_op[i, t] * dispatch[i, t] for i = 1:N_STRUCT, t = 1:T) +
        0.1 * sum(charge)
    )
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return model, [assets_investment[i] for i = 1:N_STRUCT]
end

run_walk_experiment(
    build_solved_model;
    n_alts = N_ALTS,
    eps_mga = EPS_MGA,
    n_steps = N_STEPS,
    method = :alm_lbfgs,
    gammas = GAMMAS,
    outdir = OUTDIR,
)
