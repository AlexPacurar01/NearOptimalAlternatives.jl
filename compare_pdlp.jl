# ===========================================================================
# Compare the custom first-order MGA solvers against a library PDLP (HiGHS's
# built-in `solver = "pdlp"`) and barrier (exact reference) on the *same*
# first-alternative MGA LP. All cold-started; violations reported in original
# units via `primal_feasibility_report`.
#
#   julia -t 4 --project=. compare_pdlp.jl
# ===========================================================================
using JuMP, HiGHS, LinearAlgebra, Random, Printf
using NearOptimalAlternatives

const N = 12
const T = 48
const NS = 4

function build_model()
    Random.seed!(7)
    cap = 50 .+ 50 .* rand(N)
    ci = 10 .+ 10 .* rand(N)
    co = 1 .+ 5 .* rand(N, T)
    dem = 20 .+ 15 .* rand(T)
    eta = 0.9
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "threads", max(1, Threads.nthreads()))
    @variable(m, 0 <= ai[i = 1:N] <= cap[i])
    @variable(m, dp[i = 1:N, t = 1:T] >= 0)
    @variable(m, ch[s = 1:NS, t = 1:T] >= 0)
    @variable(m, lv[s = 1:NS, t = 1:T] >= 0)
    @constraint(m, [i = 1:N, t = 1:T], dp[i, t] <= ai[i])
    @constraint(m, [t = 1:T], sum(dp[i, t] for i = 1:N) >= dem[t])
    @constraint(m, [s = 1:NS, t = 2:T], lv[s, t] == lv[s, t-1] + eta * ch[s, t] - dp[s, t])
    @constraint(m, [s = 1:NS], lv[s, 1] == eta * ch[s, 1])
    @constraint(m, [s = 1:NS, t = 1:T], lv[s, t] <= 5 * ai[s])
    @objective(
        m,
        Min,
        sum(ci[i] * ai[i] for i = 1:N) +
        sum(co[i, t] * dp[i, t] for i = 1:N, t = 1:T) +
        0.1 * sum(ch)
    )
    optimize!(m)
    return m, [ai[i] for i = 1:N]
end

m, target = build_model()
allv = all_variables(m)
idx = Dict(v => i for (i, v) in enumerate(allv))
tidx = [idx[v] for v in target]
xstar = value.(allv)
B = objective_value(m) * 1.1

# SPORES-style weight on the structural variables.
w = zeros(length(allv))
for (j, i) in enumerate(tidx)
    ub =
        has_upper_bound(target[j]) && upper_bound(target[j]) != 0 ? upper_bound(target[j]) :
        1.0
    w[i] = xstar[i] / ub
end

# Max original-unit constraint violation of a point (incl. the budget row).
function max_violation(point_vec)
    pt = Dict(allv[i] => point_vec[i] for i in eachindex(allv))
    rep = primal_feasibility_report(m, pt)
    isempty(rep) ? 0.0 : maximum(values(rep))
end
mga_obj(x) = dot(w, x)

results = Tuple{String,Float64,Float64,Float64}[]   # name, time, obj, viol

# --- Custom first-order solvers on the shared scaled LP (cold start) ---
lp = build_mga_lp(m, w, B, allv)
x0 = clamp.(zeros(lp.n), lp.lb_t, lp.ub_t)
# Warm up JIT so the timings below exclude Julia compilation (HiGHS paths are
# already compiled from the base solve, so this levels the field).
solve_alm_lbfgs(
    lp.w_t,
    lp.A_in,
    lp.b_in,
    lp.A_eq,
    lp.b_eq,
    lp.lb_t,
    lp.ub_t,
    copy(x0);
    max_outer = 2,
    max_inner = 20,
    verbose = false,
)
solve_pdhg(
    lp.w_t,
    lp.A_in,
    lp.b_in,
    lp.A_eq,
    lp.b_eq,
    lp.lb_t,
    lp.ub_t,
    copy(x0);
    max_iter = 50,
    verbose = false,
)
let
    t = @elapsed (xs, _) = solve_alm_lbfgs(
        lp.w_t,
        lp.A_in,
        lp.b_in,
        lp.A_eq,
        lp.b_eq,
        lp.lb_t,
        lp.ub_t,
        copy(x0);
        max_outer = 50,
        max_inner = 1000,
        verbose = false,
    )
    x = lp.d_scale .* xs
    push!(results, ("custom alm_lbfgs", t, mga_obj(x), max_violation(x)))
end
let
    t = @elapsed (xs, _) = solve_pdhg(
        lp.w_t,
        lp.A_in,
        lp.b_in,
        lp.A_eq,
        lp.b_eq,
        lp.lb_t,
        lp.ub_t,
        copy(x0);
        max_iter = 50000,
        verbose = false,
    )
    x = lp.d_scale .* xs
    push!(results, ("custom pdhg", t, mga_obj(x), max_violation(x)))
end

# --- Library solves of the same MGA LP via JuMP (budget row + MGA objective) ---
bcon = @constraint(m, objective_function(m) <= B)
@objective(m, Min, sum(w[i] * allv[i] for i in eachindex(allv) if w[i] != 0))
for (name, solver) in (("HiGHS barrier (exact)", "ipm"), ("HiGHS PDLP (library)", "pdlp"))
    set_optimizer_attribute(m, "solver", solver)
    solver == "ipm" && set_optimizer_attribute(m, "run_crossover", "off")
    t = @elapsed optimize!(m)
    x = value.(allv)
    push!(results, (name, t, mga_obj(x), max_violation(x)))
end

@printf("\n%-22s %10s %14s %14s\n", "method", "time(s)", "MGA obj", "max viol")
for (name, t, o, v) in results
    @printf("%-22s %10.3f %14.4f %14.2e\n", name, t, o, v)
end
