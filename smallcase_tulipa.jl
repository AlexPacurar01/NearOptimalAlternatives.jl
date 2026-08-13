# ===========================================================================
# Loader for the public TulipaEnergyModel example datasets (Tiny, Norse) using
# the modern (v0.22) DuckDB-connection API, following the package's own
# "Basics" tutorial:
#
#     connection = DBInterface.connect(DuckDB.DB)
#     TIO.read_csv_folder(connection, input_dir)
#     TEM.populate_with_defaults!(connection)
#     ep = TEM.EnergyProblem(connection); TEM.create_model!(ep)
#
# The datasets are read from the INSTALLED package's test inputs, so they always
# match the resolved TulipaEnergyModel version (the repo's own data/Tiny is an
# older-schema copy that a modern Tulipa rejects). The MGA study is convex, so we
# relax any integrality (investment_integer) to the LP relaxation, then solve the
# base problem with the supplied interior-point optimizer.
# ===========================================================================
import TulipaEnergyModel as TEM
import TulipaIO as TIO
using DuckDB
using JuMP

"Absolute path to a bundled TulipaEnergyModel example dataset."
function tulipa_input_dir(name::AbstractString)
    pkg = dirname(dirname(pathof(TEM)))
    dir = joinpath(pkg, "test", "inputs", name)
    isdir(dir) || error("bundled Tulipa dataset '$name' not found at $dir")
    return dir
end

"""
    load_tulipa_case(name, optimizer) -> (model, target_vars)

Build and cold-solve the LP relaxation of the TulipaEnergyModel example `name`
(e.g. "Tiny", "Norse"), returning the solved JuMP model and its structural
(investment) variables - the same `(model, target_vars)` contract the synthetic
builders use, so the corrector harness is schema-agnostic.
"""
function load_tulipa_case(name::AbstractString, optimizer)
    con = DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(con, tulipa_input_dir(name))
    TEM.populate_with_defaults!(con)
    ep = TEM.EnergyProblem(con)
    TEM.create_model!(ep)
    model = ep.model
    # Convex study: drop integrality (investment_integer / unit_commitment_integer).
    JuMP.relax_integrality(model)
    set_optimizer(model, optimizer)
    set_silent(model)
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model) "base solve of $name failed"

    target = VariableRef[]
    for sym in (:assets_investment, :flows_investment)
        haskey(ep.variables, sym) || continue
        for v in ep.variables[sym].container
            v isa VariableRef && push!(target, v)
        end
    end
    isempty(target) && error("no investment variables found in $name")
    return model, target
end
