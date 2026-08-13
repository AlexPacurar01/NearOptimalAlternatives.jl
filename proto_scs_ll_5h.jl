# SCS low-level (x,y,s) warm start on data/5h. B1 solved from x* (warm primal,
# cold dual), B2 warm-started from B1's FULL ADMM state. Does the warm dual+slack
# collapse the iterations, and does it land on the true front? Explicit flush.
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, SCS, LinearAlgebra, SparseArrays, Printf
using NearOptimalAlternatives
const NOA = NearOptimalAlternatives
include(joinpath(@__DIR__, "bench_common.jl"))
const THREADS = max(1, Threads.nthreads())

struct ScsLP
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    c::Vector{Float64}
    z::Int
    l::Int
    n::Int
    m::Int
    budrow::Int
end
function build_scslp(lp)
    n = lp.n
    m_in = size(lp.A_in, 1)
    m_eq = size(lp.A_eq, 1)
    flb = [j for j = 1:n if isfinite(lp.lb_t[j])]
    fub = [j for j = 1:n if isfinite(lp.ub_t[j])]
    Alb = sparse(1:length(flb), flb, fill(-1.0, length(flb)), length(flb), n)
    Aub = sparse(1:length(fub), fub, fill(1.0, length(fub)), length(fub), n)
    A = vcat(lp.A_eq, lp.A_in, Alb, Aub)
    b = vcat(lp.b_eq, lp.b_in, -lp.lb_t[flb], lp.ub_t[fub])
    return ScsLP(
        A,
        b,
        lp.w_t,
        m_eq,
        m_in + length(flb) + length(fub),
        n,
        m_eq + m_in + length(flb) + length(fub),
        m_eq + m_in,
    )
end
function scs_solve_lp(
    p;
    x0 = nothing,
    y0 = nothing,
    s0 = nothing,
    eps = 1e-4,
    maxit = 20000,
    verbose = 1,
)
    warm = x0 !== nothing
    xs = warm ? copy(x0) : zeros(p.n)
    ys = warm ? copy(y0) : zeros(p.m)
    ss = warm ? copy(s0) : zeros(p.m)
    SCS.scs_solve(
        SCS.IndirectSolver,
        p.m,
        p.n,
        p.A,
        spzeros(p.n, p.n),
        p.b,
        p.c,
        p.z,
        p.l,
        Float64[],
        Float64[],
        Int[],
        Int[],
        Int[],
        0,
        0,
        Float64[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        xs,
        ys,
        ss;
        warm_start = warm,
        eps_abs = eps,
        eps_rel = eps,
        max_iters = maxit,
        verbose = verbose,
    )
end

ep = TEM.create_energy_problem_from_csv_folder(joinpath(@__DIR__, "data", "5h"));
TEM.create_model!(ep);
model = ep.model;
set_optimizer(model, make_optimizer(:gurobi, THREADS));
set_silent(model);
JuMP.optimize!(model);
@assert is_solved_and_feasible(model);
min_cost = objective_value(model);
allv = all_variables(model);
idx = Dict(v => i for (i, v) in enumerate(allv));
xstar = value.(allv);
println("built+solved 5h | min_cost=$(round(min_cost,sigdigits=6))");
flush(stdout);
target = JuMP.VariableRef[]
for sym in (:assets_investment, :flows_investment)
    haskey(object_dictionary(model), sym) && for v in object_dictionary(model)[sym]
        push!(target, v)
    end
end
w = zeros(length(allv));
for v in target
    ub = (has_upper_bound(v) && upper_bound(v) != 0) ? upper_bound(v) : 1.0
    w[idx[v]] = value(v) / ub
end;
what = w ./ norm(w)

cvec, coff = NOA._objective_costs(model, allv)
B1 = min_cost * 1.05;
B2 = min_cost * 1.10;
Bmax = min_cost * 1.10;
lp = NOA.build_mga_lp(model, w, Bmax, allv);
p = build_scslp(lp);
bidx = size(lp.A_in, 1);
r_budget = (Bmax - coff) / lp.b_in[bidx];
scaled(B) = (B - coff) / r_budget
x0t = clamp.(xstar ./ lp.d_scale, lp.lb_t, lp.ub_t)
println("SCS LP: m=$(p.m) rows, n=$(p.n) vars");
flush(stdout);

p.b[p.budrow] = scaled(B1)
s0 = p.b .- p.A * x0t
println(">>> SCS B1 warm-from-x* (cold dual) ...");
flush(stdout);
t1 = @elapsed sol1 = scs_solve_lp(p; x0 = x0t, y0 = zeros(p.m), s0 = s0)
d1 = dot(what, lp.d_scale .* sol1.x)
@printf(
    ">>> B1: status=%s iters=%d time=%.1fs dobj=%.3f\n",
    SCS.raw_status(sol1.info),
    sol1.info.iter,
    t1,
    d1
);
flush(stdout);

p.b[p.budrow] = scaled(B2)
println(">>> SCS B2 warm-from-B1 (full x,y,s) ...");
flush(stdout);
t2 = @elapsed sol2 = scs_solve_lp(p; x0 = sol1.x, y0 = sol1.y, s0 = sol1.s)
d2 = dot(what, lp.d_scale .* sol2.x)
@printf(
    ">>> B2: status=%s iters=%d time=%.1fs dobj=%.3f\n",
    SCS.raw_status(sol2.info),
    sol2.info.iter,
    t2,
    d2
);
flush(stdout);

bi = ipm_baseline_at_budgets(
    model,
    w,
    allv,
    [B1, B2];
    optimizer_factory = make_optimizer(:gurobi, THREADS),
)
@printf(
    "\nIPM truth: B1=%.3f  B2=%.3f   (x* dobj=%.3f)\n",
    bi.dobjs[1],
    bi.dobjs[2],
    dot(what, xstar)
)
@printf("SCS vs IPM: |B1|=%.2e  |B2|=%.2e\n", abs(d1 - bi.dobjs[1]), abs(d2 - bi.dobjs[2]))
println("DONE");
flush(stdout);
