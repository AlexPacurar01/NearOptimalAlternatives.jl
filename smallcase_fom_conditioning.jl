# ===========================================================================
# Conditioning diagnostic: measure, per instance, the quantities that govern
# first-order-corrector convergence, so the report's "conditioning, not size,
# is binding" claim rests on numbers instead of inference.
#
# For each case we build the same MGA LP the correctors see (build_mga_lp) and
# record, RAW (as extracted from the model) and SCALED (post Ruiz + budget-row
# rescale, i.e. exactly what SCS/PDHG/ALM iterate on):
#   * size          - n vars, inequality/equality rows, nonzeros
#   * struct frac   - fraction of variables the diversity objective touches
#   * cost range    - max|c| / min nonzero |c|   (classic ESOM ill-conditioning)
#   * entry range   - max|a| / min nonzero |a| of the constraint matrix
#   * rhs range     - max|b| / min nonzero |b| over finite rows
#   * row-norm spread, col-norm spread of the stacked [A_in; A_eq]
#   * ||A||_2 estimate (power iteration) of the scaled operator
# A well-conditioned instance has all spreads near 1 after Ruiz; residual
# spread orders of magnitude above 1 is the ill-conditioning the equilibration
# cannot remove and the first-order iteration pays for.
#
#   Usage:  julia -t 4 --project=. smallcase_fom_conditioning.jl [case ...]
#           (default: synth Tiny Norse)
# ===========================================================================
using HiGHS, Printf, Statistics, LinearAlgebra, SparseArrays
include("smallcase_common.jl")

const OUTDIR = joinpath(pwd(), "results", "fom_smallcase")
mkpath(OUTDIR)
const CASES = isempty(ARGS) ? ["synth", "Tiny", "Norse"] : ARGS
if any(c -> c in ("Tiny", "Norse"), CASES)
    include("smallcase_tulipa.jl")
end
ipm() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

make_case(name) =
    name == "synth" ? build_synth_model(ipm(); n_struct = 8, T = 24, n_store = 4) :
    name in ("Tiny", "Norse") ? load_tulipa_case(name, ipm()) :
    error("unknown case '$name'")

"max/min ratio over the nonzero absolute values of `v` (1.0 if empty)."
function dynrange(v)
    nz = filter(x -> x > 0, abs.(v))
    isempty(nz) ? 1.0 : maximum(nz) / minimum(nz)
end

"Row and column 2-norm spreads (max/min over nonzero norms) of sparse `A`."
function norm_spreads(A::SparseMatrixCSC)
    m, n = size(A)
    rown = zeros(m)
    coln = zeros(n)
    rv = rowvals(A)
    nz = nonzeros(A)
    for j = 1:n, k in nzrange(A, j)
        rown[rv[k]] += nz[k]^2
        coln[j] += nz[k]^2
    end
    r = filter(>(0), sqrt.(rown))
    c = filter(>(0), sqrt.(coln))
    (
        row = isempty(r) ? 1.0 : maximum(r) / minimum(r),
        col = isempty(c) ? 1.0 : maximum(c) / minimum(c),
    )
end

"Power-iteration estimate of ||A||_2 (50 iterations)."
function opnorm_est(A::SparseMatrixCSC)
    n = size(A, 2)
    v = normalize!(randn(n))
    s = 0.0
    for _ = 1:50
        u = A * v
        v = A' * u
        s = norm(v)
        s > 0 && (v ./= s)
    end
    sqrt(s)
end

function run_case(name)
    println("\n########## conditioning: $name ##########")
    model, target = make_case(name)
    s = mga_setup(model, target; eps = 0.1)

    data = lp_matrix_data(model)
    finite_b = filter(isfinite, vcat(data.b_lower, data.b_upper))
    raw = (
        c_range = dynrange(data.c),
        a_range = dynrange(nonzeros(data.A)),
        b_range = dynrange(finite_b),
        spreads = norm_spreads(data.A),
    )

    lp = with_logger(NullLogger()) do
        build_mga_lp(model, copy(s.w), s.B, s.all_vars)
    end
    S = vcat(lp.A_in, lp.A_eq)
    scal =
        (a_range = dynrange(nonzeros(S)), spreads = norm_spreads(S), opnorm = opnorm_est(S))

    nvars = lp.n
    m_in, m_eq = size(lp.A_in, 1), size(lp.A_eq, 1)
    frac = 100 * length(s.target_indices) / nvars
    @printf(
        "size: n=%d  m_in=%d  m_eq=%d  nnz=%d  structural=%.2f%%\n",
        nvars,
        m_in,
        m_eq,
        nnz(S),
        frac
    )
    @printf(
        "raw    : cost range %.1e | entry range %.1e | rhs range %.1e | row spread %.1e | col spread %.1e\n",
        raw.c_range,
        raw.a_range,
        raw.b_range,
        raw.spreads.row,
        raw.spreads.col
    )
    @printf(
        "scaled : entry range %.1e | row spread %.1e | col spread %.1e | ||A||_2 ~ %.2f\n",
        scal.a_range,
        scal.spreads.row,
        scal.spreads.col,
        scal.opnorm
    )

    return @sprintf(
        "%s,%d,%d,%d,%d,%.4f,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3f",
        name,
        nvars,
        m_in,
        m_eq,
        nnz(S),
        frac,
        raw.c_range,
        raw.a_range,
        raw.b_range,
        raw.spreads.row,
        raw.spreads.col,
        scal.a_range,
        scal.spreads.row,
        scal.spreads.col,
        scal.opnorm
    )
end

rows =
    String["case,n,m_in,m_eq,nnz,struct_pct,raw_c_range,raw_a_range,raw_b_range,"*"raw_row_spread,raw_col_spread,scaled_a_range,scaled_row_spread,"*"scaled_col_spread,scaled_opnorm"]
for c in CASES
    push!(rows, run_case(c))
end
open(joinpath(OUTDIR, "conditioning.csv"), "w") do io
    println(io, join(rows, "\n"))
end
println("\nwrote conditioning.csv")
