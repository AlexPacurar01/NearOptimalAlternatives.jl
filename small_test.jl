import TulipaIO as TIO
import TulipaEnergyModel as TEM
using JuMP
using Gurobi
using NearOptimalAlternatives

function try_dataset(name)
    input_dir = joinpath(pwd(), "data", name)
    println("\n=== Trying dataset: $name ($input_dir) ===")
    try
        energy_problem = TEM.create_energy_problem_from_csv_folder(input_dir)
        TEM.create_model!(energy_problem)
        model = energy_problem.model
        println(
            "  Loaded OK. n_vars=$(length(all_variables(model))), n_constraints=$(sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))",
        )
        return model
    catch e
        println("  FAILED: ", sprint(showerror, e))
        return nothing
    end
end

for name in ["Tiny", "Norse"]
    try_dataset(name)
end
