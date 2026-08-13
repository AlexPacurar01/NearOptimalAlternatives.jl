# ===========================================================================
# Probe: is a cheap, amortised projection onto the equality manifold feasible
# on the real data/5h model? Decides whether the projected-gradient boundary
# walk (factorise A_eq A_eq' ONCE, then back-substitute per step) is viable.
#
# Reports, for the 5h model's constraint matrix:
#   - variable / constraint-row counts and the equality / inequality split,
#   - how many inequality rows are single-variable (clamp-able as bounds) vs
#     genuine multi-variable coupling rows,
#   - sparse Cholesky of M = A_eq A_eq': success, factorise time, fill-in
#     (nnz(factor)/nnz(M)), and one back-substitution (= per-step projection) time.
#
# No solve needed - we only read the constraint matrix. Run:
#     julia -t 8 --project=. probe_projection_5h.jl
# ===========================================================================
import TulipaIO as TIO
import TulipaEnergyModel as TEM
using DuckDB
using JuMP, LinearAlgebra, SparseArrays, Printf

println("Building data/5h model (no solve - matrices only) ...")
t_build = @elapsed begin
    ep = TEM.create_energy_problem_from_csv_folder(joinpath(@__DIR__, "data", "5h"))
    TEM.create_model!(ep)
    model = ep.model
end
n = num_variables(model)
println("  built in $(round(t_build, digits=1))s | $n variables")

println("Extracting constraint matrix (lp_matrix_data) ...")
t_extract = @elapsed data = lp_matrix_data(model)
A = data.A                       # m x n sparse constraint matrix (no var bounds)
bl, bu = data.b_lower, data.b_upper
m = size(A, 1)
println(
    "  extracted in $(round(t_extract, digits=1))s | $m constraint rows | nnz(A)=$(nnz(A))",
)

# Row classification.
eq_rows = Int[]
in_rows = Int[]
for i = 1:m
    if isfinite(bl[i]) && isfinite(bu[i]) && bl[i] == bu[i]
        push!(eq_rows, i)
    else
        push!(in_rows, i)
    end
end

# Count nonzeros per row to find single-variable (bound-like) inequality rows.
rows_nnz = zeros(Int, m)
rv = rowvals(A)
for j = 1:n, k in nzrange(A, j)
    rows_nnz[rv[k]] += 1
end
in_single = count(i -> rows_nnz[i] == 1, in_rows)
in_multi = length(in_rows) - in_single
eq_single = count(i -> rows_nnz[i] == 1, eq_rows)

@printf("\nRow breakdown:\n")
@printf("  equality rows      : %8d  (single-var: %d)\n", length(eq_rows), eq_single)
@printf(
    "  inequality rows    : %8d  (single-var/bound-like: %d, multi-var coupling: %d)\n",
    length(in_rows),
    in_single,
    in_multi
)

# Equality matrix and its normal-equations Gram matrix M = A_eq A_eq'.
A_eq = A[eq_rows, :]
m_eq = size(A_eq, 1)
println("\nForming M = A_eq * A_eq'  ($(m_eq) x $(m_eq)) ...")
t_gram = @elapsed M = A_eq * transpose(A_eq)
M = sparse(Symmetric(M))
@printf(
    "  formed in %.1fs | nnz(A_eq)=%d | nnz(M)=%d | avg nnz/row(M)=%.1f\n",
    t_gram,
    nnz(A_eq),
    nnz(M),
    nnz(M) / max(m_eq, 1)
)

println("\nSparse Cholesky of M (CHOLMOD, AMD ordering) ...")
F = nothing
t_chol = @elapsed F = cholesky(M; check = false)
if issuccess(F)
    nnzF = nnz(sparse(F.L))
    @printf(
        "  SUCCESS in %.1fs | nnz(factor)=%d | fill ratio nnz(L)/nnz(M)=%.2f\n",
        t_chol,
        nnzF,
        nnzF / max(nnz(M), 1)
    )
else
    println("  M not positive-definite (redundant equality rows). Retrying M + reg*I ...")
    reg = 1e-8 * maximum(abs, diag(M))
    t_chol = @elapsed F = cholesky(M + reg * I; check = false)
    if issuccess(F)
        nnzF = nnz(sparse(F.L))
        @printf(
            "  SUCCESS (regularised, reg=%.2e) in %.1fs | nnz(factor)=%d | fill ratio=%.2f\n",
            reg,
            t_chol,
            nnzF,
            nnzF / max(nnz(M), 1)
        )
    else
        println(
            "  FAILED even regularised - normal-equations Cholesky not viable; use iterative corrector.",
        )
    end
end

# One back-substitution = the per-step projection cost.
if F !== nothing && issuccess(F)
    rhs = randn(m_eq)
    F \ rhs                       # warm up
    t_solve = @elapsed (F \ rhs)
    @printf("\nPer-step projection (one back-substitution): %.3fs\n", t_solve)
    @printf(
        "=> a 200-step trace would spend ~%.0fs in projections (+ gradient mat-vecs).\n",
        200 * t_solve
    )
end
println("\nProbe done.")
