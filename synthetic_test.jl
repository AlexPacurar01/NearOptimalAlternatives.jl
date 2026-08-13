using JuMP
using HiGHS
using LinearAlgebra
using Random
using NearOptimalAlternatives

# ---------------------------------------------------------------------------
# Small synthetic capacity-expansion model that mirrors the structure of the
# real ESOM: a handful of "structural" (investment) variables, each of which
# bounds many "operational" (dispatch) variables via per-timestep capacity
# constraints. This gives the same K-constraints-per-structural-variable
# gradient-scale-mismatch that stalls a full-space L-BFGS, but at a size
# (a few hundred variables) that solves and iterates in milliseconds - so the
# ALM + L-BFGS structural solver and the operational recovery QP can be
# debugged quickly without the 1.27M-variable dataset.
# ---------------------------------------------------------------------------

Random.seed!(42)

n_struct = 8     # number of investment/capacity variables
T = 24           # timesteps -> T capacity constraints per structural variable

cap_max = 50.0 .+ 50.0 .* rand(n_struct)        # max investable capacity
c_inv = 10.0 .+ 10.0 .* rand(n_struct)        # investment cost per unit capacity
c_op = 1.0 .+ 5.0 .* rand(n_struct, T)       # operating cost per unit dispatch
demand = 20.0 .+ 10.0 .* rand(T)               # demand per timestep

model = Model(HiGHS.Optimizer)
set_silent(model)

@variable(model, 0 <= assets_investment[i = 1:n_struct] <= cap_max[i])
@variable(model, dispatch[i = 1:n_struct, t = 1:T] >= 0)

@constraint(
    model,
    capacity[i = 1:n_struct, t = 1:T],
    dispatch[i, t] <= assets_investment[i]
)
@constraint(model, balance[t = 1:T], sum(dispatch[i, t] for i = 1:n_struct) >= demand[t])

@objective(
    model,
    Min,
    sum(c_inv[i] * assets_investment[i] for i = 1:n_struct) +
    sum(c_op[i, t] * dispatch[i, t] for i = 1:n_struct, t = 1:T)
)

optimize!(model)
@assert is_solved_and_feasible(model)

base_obj_val = objective_value(model)
base_x = value.(all_variables(model))

println("Base objective: ", base_obj_val)
println("Base investments: ", value.(assets_investment))

# ---------------------------------------------------------------------------
# Run the structural ALM + L-BFGS / operational-recovery alternative search.
# ---------------------------------------------------------------------------

t_alg = @elapsed begin
    global result = NearOptimalAlternatives.generate_alternatives_lbfgs(
        model,
        1;
        operational_recovery = true,
    )
end
println(
    "Algorithm wall-clock time (cold start, incl. all sub-solves): $(round(t_alg, digits=2)) s",
)

alt = result.solutions[1]
all_vars = all_variables(model)
alt_vec = [alt[v] for v in all_vars]

println("\nAlternative investments: ", [alt[assets_investment[i]] for i = 1:n_struct])
println(
    "Alternative objective (full model): ",
    sum(
        coefficient(objective_function(model), v) * alt[v] for
        v in all_vars if !iszero(coefficient(objective_function(model), v))
    ),
)

# ---------------------------------------------------------------------------
# Independent LP-utopian check for this small model: solve
#   min  w' x   s.t.  objective(x) <= max_allowed_cost,  original bounds/constraints
# directly with HiGHS, where w is the same accumulated-weight vector used
# internally for iteration 1 (recomputed here for verification).
# ---------------------------------------------------------------------------

eps_mga = 0.1
min_cost = base_obj_val
max_allowed_cost = (1 + eps_mga) * min_cost

x_optimal = base_x
target_vars = [assets_investment[i] for i = 1:n_struct]
target_ubs = [upper_bound(v) for v in target_vars]
var_to_idx = Dict(v => i for (i, v) in enumerate(all_vars))

w_full = zeros(length(all_vars))
for (j, v) in enumerate(target_vars)
    idx = var_to_idx[v]
    w_full[idx] = x_optimal[idx] / target_ubs[j]
end
w_full ./= norm(w_full)

lp_model = copy(model)
set_optimizer(lp_model, HiGHS.Optimizer)
set_silent(lp_model)
lp_vars = all_variables(lp_model)
orig_obj = objective_function(lp_model)
@constraint(lp_model, orig_obj <= max_allowed_cost)
@objective(lp_model, Min, sum(w_full[i] * lp_vars[i] for i = 1:length(lp_vars)))
optimize!(lp_model)
@assert is_solved_and_feasible(lp_model)
lp_utopian = value.(lp_vars)

lp_obj_full = dot(w_full, lp_utopian)
alt_obj_full = dot(w_full, alt_vec)
gap_full = abs(lp_obj_full - alt_obj_full) / max(abs(lp_obj_full), 1e-8) * 100.0

println("\n=== VALIDATION ===")
println("LP utopian wᵀx (full model):     ", lp_obj_full)
println("Recovered  wᵀx (full model):     ", alt_obj_full)
println("Gap to LP utopian:                ", round(gap_full, digits = 4), "%")

model_obj = objective_function(model)
orig_obj_val = sum(
    coefficient(model_obj, v) * alt[v] for
    v in all_vars if !iszero(coefficient(model_obj, v))
)
println(
    "Recovered solution cost:          ",
    orig_obj_val,
    " (budget = ",
    max_allowed_cost,
    ")",
)
println("Cost feasible:                    ", orig_obj_val <= max_allowed_cost + 1e-6)
