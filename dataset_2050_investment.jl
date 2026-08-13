# ===========================================================================
# Loader for the "2050 Investment" dataset (data/2050_Investment, produced by
# convert_2050_investment.jl from the raw data/2050 Investment/2050
# Investment export). A full calendar year of hourly data (8760 timesteps,
# ONE representative period spanning the whole year, weight 1) - the largest,
# most memory-heavy case in the study, meant for the DHPC memory partition.
#
# Follows the same modern-Tulipa (v0.22) loading pattern as smallcase_tulipa.jl
# (TIO.read_csv_folder -> populate_with_defaults! -> EnergyProblem ->
# create_model!), pointed at the converted folder instead of a bundled
# package example. Convex study: integrality is relaxed, same as Tiny/Norse.
#
# Run convert_2050_investment.jl once before using this loader if
# data/2050_Investment does not exist yet.
# ===========================================================================
import TulipaEnergyModel as TEM
import TulipaIO as TIO
using DuckDB
using JuMP

const DATASET_2050_DIR = joinpath(@__DIR__, "data", "2050_Investment")

"""
    load_2050_investment(optimizer; solve = true) -> (model, target_vars)

Build (and, by default, cold-solve) the LP relaxation of the 2050 Investment
dataset, returning the JuMP model and its structural (investment) variables -
the same `(model, target_vars)` contract `load_tulipa_case` and the synthetic
builders use elsewhere in this repo.

Set `solve = false` to get the built-but-unsolved model back immediately
(e.g. to inspect size, or to attach a `x_optimal`/`min_cost` pair recovered
from a serialized baseline instead of re-solving).
"""
function load_2050_investment(optimizer; solve::Bool = true)
    isdir(DATASET_2050_DIR) || error(
        "data/2050_Investment not found. Run `julia --project=. convert_2050_investment.jl` first.",
    )
    con = DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(con, DATASET_2050_DIR)
    TEM.populate_with_defaults!(con)
    ep = TEM.EnergyProblem(con)
    TEM.create_model!(ep)
    model = ep.model
    JuMP.relax_integrality(model)
    set_optimizer(model, optimizer)
    set_silent(model)

    if solve
        JuMP.optimize!(model)
        @assert is_solved_and_feasible(model) "base solve of 2050 Investment failed"
    end

    target = VariableRef[]
    for sym in (:assets_investment, :flows_investment)
        haskey(ep.variables, sym) || continue
        for v in ep.variables[sym].container
            v isa VariableRef && push!(target, v)
        end
    end
    isempty(target) && error("no investment variables found in 2050 Investment")
    return model, target
end
