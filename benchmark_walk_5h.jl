# ===========================================================================
# Boundary-walk vs SPORES on the real data/5h TulipaEnergy model (Gurobi).
#
# The decisive test: data/5h is stiff and degenerate, so SPORES is far more
# likely to overspend budget on the cost-parallel component of its move than on
# the well-conditioned synthetic model. If the walk dominates SPORES anywhere,
# the margin should be materially larger here.
#
# This is a multi-HOUR run (each solve is minutes; SPORES + walk do many solves).
# Config is intentionally minimal. Launch deliberately and keep the machine awake:
#     julia -t 8 --project=. benchmark_walk_5h.jl
# Outputs: results/walk_5h/ (dominance CSV + frontier plot).
# ===========================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP

include(joinpath(@__DIR__, "bench_common.jl"))      # make_optimizer + dominance core
include(joinpath(@__DIR__, "walk_experiment.jl"))   # shared experiment core

const THREADS = max(1, Threads.nthreads())
const N_ALTS = 2
const EPS_MGA = 0.1
const BACKEND = :gurobi
const N_STEPS = 3                       # budget points per walk (kept small: each solve is minutes)
const GAMMAS = Float64[]                # skip cost_aware on 5h; :lexicographic supersedes it (see synthetic)
const WALK_MAX_ITERS = 8                # ALM outer iters per solve (5h needs a tight cap)
const WALK_MAX_INNER = 300              # inner L-BFGS iters per outer (bounds per-solve wall-clock)
const OUTDIR = joinpath(@__DIR__, "results", "walk_5h")

gr()
println(
    "Walk benchmark 5h: THREADS=$THREADS N_ALTS=$N_ALTS backend=$BACKEND N_STEPS=$N_STEPS",
)

"Build + cold-solve a fresh data/5h model; returns (model, target investment/flow vars)."
function build_5h()
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(@__DIR__, "data", "5h"))
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

run_walk_experiment(
    build_5h;
    n_alts = N_ALTS,
    eps_mga = EPS_MGA,
    n_steps = N_STEPS,
    method = :alm_lbfgs,
    gammas = GAMMAS,
    outdir = OUTDIR,
    max_iters = WALK_MAX_ITERS,
    max_inner = WALK_MAX_INNER,
)
