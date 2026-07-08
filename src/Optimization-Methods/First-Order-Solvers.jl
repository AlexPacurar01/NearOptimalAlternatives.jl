# ===========================================================================
# Hessian-free first-order solvers for the full-space MGA linear program
#
#     min  w'x   s.t.  A_in x <= b_in,  A_eq x = b_eq,  lb <= x <= ub
#
# A shared, factorization-free numerical engine (only sparse matrix-vector
# products, threaded via `adjoint_mul!`) used by the gradient MGA method.
# All solvers operate on the same Ruiz-equilibrated problem produced by
# `build_mga_lp` and are dispatched through `solve_firstorder`.
# ===========================================================================

export build_mga_lp, solve_firstorder, solve_alm_lbfgs, solve_pdhg, primal_infeasibility

"""
    d = ruiz_equilibrate!(A_in, b_in, A_eq, b_eq, n; passes = 10)

In-place iterated Ruiz scaling (Ruiz, 2001) of `[A_in; A_eq]`: each pass divides
every row and column by the sqrt of its largest absolute entry (columns shared
across both matrices), driving all row/column ∞-norms to ~1. `b_in`/`b_eq` are
rescaled with the rows. Returns the accumulated column scaling `d > 0`, defining
`x = d .* x_scaled`; the caller transforms bounds (`./d`), objective (`.*d`) and
back-transforms the solution.
"""
function ruiz_equilibrate!(
    A_in::SparseMatrixCSC{Float64,Int},
    b_in::Vector{Float64},
    A_eq::SparseMatrixCSC{Float64,Int},
    b_eq::Vector{Float64},
    n::Int;
    passes::Int = 10,
)
    d = ones(n)
    m_in = size(A_in, 1)
    m_eq = size(A_eq, 1)
    c = zeros(n)
    r_in = zeros(m_in)
    r_eq = zeros(m_eq)

    for _ = 1:passes
        fill!(c, 0.0)
        fill!(r_in, 0.0)
        fill!(r_eq, 0.0)
        for (A, r) in ((A_in, r_in), (A_eq, r_eq))
            rv = rowvals(A)
            nz = nonzeros(A)
            for j = 1:n, k in nzrange(A, j)
                a = abs(nz[k])
                c[j] = max(c[j], a)
                r[rv[k]] = max(r[rv[k]], a)
            end
        end
        @. c = ifelse(c > 0, sqrt(c), 1.0)
        @. r_in = ifelse(r_in > 0, sqrt(r_in), 1.0)
        @. r_eq = ifelse(r_eq > 0, sqrt(r_eq), 1.0)

        for (A, r) in ((A_in, r_in), (A_eq, r_eq))
            rv = rowvals(A)
            nz = nonzeros(A)
            for j = 1:n, k in nzrange(A, j)
                nz[k] /= r[rv[k]] * c[j]
            end
        end
        b_in ./= r_in
        b_eq ./= r_eq
        d ./= c
    end

    return d
end

"""
    adjoint_mul!(y, M, x)

`y = M' * x` for a `SparseMatrixCSC`, threaded over columns (each `y[j]` reads
only column `j`, so no write conflicts). With `A` and its materialised transpose
`At` in CSC form, `A*x = adjoint_mul!(y, At, x)` and `A'*z = adjoint_mul!(y, A, z)`
— the inner-loop kernel of every solver here.
"""
function adjoint_mul!(
    y::Vector{Float64},
    M::SparseMatrixCSC{Float64,Int},
    x::Vector{Float64},
)
    rv = rowvals(M)
    nz = nonzeros(M)
    Threads.@threads :static for j = 1:size(M, 2)
        s = 0.0
        @inbounds for k in nzrange(M, j)
            s += nz[k] * x[rv[k]]
        end
        @inbounds y[j] = s
    end
    return y
end

"""
    f, iters, reason = projected_lbfgs!(fg!, x, lb, ub; memory, max_iter, g_tol, callback)

Box-constrained L-BFGS with two-metric gradient projection (Bertsekas, 1982;
cf. L-BFGS-B, Byrd et al. 1995): two-loop direction with `s'y/y'y` scaling on the
free subspace, scaled steepest descent on coordinates pinned at an active bound,
projected-arc Armijo line search (`c1 = 1e-4`, backtracking by halving). On
line-search failure the curvature memory is reset and a projected-gradient step
is tried; failing that, `x` is a projected stationary point and the loop stops.
Converges when `||P(x - g) - x||_inf <= g_tol`; curvature pairs with
`s'y <= sqrt(eps)||s|| ||y||` are skipped. `fg!(true, g, x)` returns `f(x)` and
fills `g`. `x` is updated in place; `callback(iter, x)`, if given, runs after each
accepted step (convergence tracing).
"""
function projected_lbfgs!(
    fg!,
    x::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64};
    memory::Int = 10,
    max_iter::Int = 200,
    g_tol::Float64 = sqrt(eps(Float64)),
    callback = nothing,
    precond! = nothing,
)
    n = length(x)
    @. x = clamp(x, lb, ub)

    g = zeros(n)
    gnew = zeros(n)
    xnew = zeros(n)
    p = zeros(n)
    My = precond! === nothing ? Float64[] : zeros(n)   # buffer for M*y (preconditioned scale)

    f = fg!(true, g, x)

    S = Vector{Float64}[]
    Y = Vector{Float64}[]
    rho_pairs = Float64[]
    a = Float64[]
    active = falses(n)

    iters = 0
    reason = :max_iter

    for _ = 1:max_iter
        iters += 1
        # KKT measure for box-constrained problems: the projected gradient
        # step. Zero iff x is stationary for min f s.t. lb <= x <= ub.
        pg = 0.0
        @inbounds for i = 1:n
            pg = max(pg, abs(clamp(x[i] - g[i], lb[i], ub[i]) - x[i]))
        end
        if pg <= g_tol
            reason = :converged
            break
        end

        # Two-metric projection (Bertsekas, 1982): the quasi-Newton model is
        # applied only on the estimated-free subspace. Components pinned at
        # an active bound - at the bound with the gradient pushing further
        # outward - take a scaled steepest-descent move instead; including
        # them in the two-loop recursion would let bound-clipped coordinates
        # distort the direction for the free ones and stall the line search.
        @inbounds for i = 1:n
            tol_a = sqrt(eps(Float64)) * max(1.0, abs(x[i]))
            active[i] =
                (x[i] - lb[i] <= tol_a && g[i] > 0) || (ub[i] - x[i] <= tol_a && g[i] < 0)
        end

        # Two-loop recursion on the free subspace: p = -H*g_free.
        copyto!(p, g)
        @inbounds for i = 1:n
            active[i] && (p[i] = 0.0)
        end
        k = length(S)
        resize!(a, k)
        for i = k:-1:1
            a[i] = rho_pairs[i] * dot(S[i], p)
            axpy!(-a[i], Y[i], p)
        end
        # H0 step of the two-loop recursion. Unpreconditioned: H0 = gamma*I with the
        # Barzilai-Borwein scale gamma = s'y/y'y. Preconditioned (M = matrix-free
        # whitening of the stiff temporal/chain modes, e.g. inverse storage Laplacian
        # via an O(T) tridiagonal solve): H0 = gamma_M * M with the M-METRIC scale
        # gamma_M = s'y / (y'M y). Using the plain y'y here would double-scale and
        # produce non-descent directions - the bug that made naive preconditioning
        # diverge. The (s,y) pairs stay in original coordinates.
        if precond! === nothing
            gamma =
                k > 0 ? dot(S[k], Y[k]) / dot(Y[k], Y[k]) : 1.0 / max(norm(g), eps(Float64))
            rmul!(p, gamma)
        else
            if k > 0
                copyto!(My, Y[k])
                precond!(My)
                gamma = dot(S[k], Y[k]) / max(dot(Y[k], My), eps(Float64))
            else
                gamma = 1.0
            end
            precond!(p)
            rmul!(p, gamma)
        end
        for i = 1:k
            b = rho_pairs[i] * dot(Y[i], p)
            axpy!(a[i] - b, S[i], p)
        end
        @inbounds for i = 1:n
            active[i] && (p[i] = gamma * g[i])
        end
        rmul!(p, -1.0)

        accepted = false
        f_new = f
        for attempt = 1:2
            alpha = 1.0
            for _ = 1:60
                @. xnew = clamp(x + alpha * p, lb, ub)
                delta = 0.0
                @inbounds for i = 1:n
                    delta += g[i] * (xnew[i] - x[i])
                end
                if delta < 0
                    f_new = fg!(true, gnew, xnew)
                    if isfinite(f_new) && f_new <= f + 1.0e-4 * delta
                        accepted = true
                        break
                    end
                end
                alpha /= 2
            end
            accepted && break
            attempt == 2 && break
            # The quasi-Newton direction gave no acceptable projected step:
            # discard the (no longer trustworthy) curvature memory and retry
            # with a scaled steepest-descent direction.
            empty!(S)
            empty!(Y)
            empty!(rho_pairs)
            copyto!(p, g)
            rmul!(p, -gamma)
        end
        if !accepted
            reason = :linesearch
            break
        end

        s = xnew .- x
        y = gnew .- g
        sy = dot(s, y)
        if sy > sqrt(eps(Float64)) * norm(s) * norm(y)
            push!(S, s)
            push!(Y, y)
            push!(rho_pairs, 1 / sy)
            if length(S) > memory
                popfirst!(S)
                popfirst!(Y)
                popfirst!(rho_pairs)
            end
        end

        copyto!(x, xnew)
        copyto!(g, gnew)
        f = f_new

        # Optional per-iteration trace hook (used by the convergence
        # benchmark): records the residual/objective trajectory *within* a
        # single solve, including the early high-error phase. Off by default,
        # so the search itself pays nothing.
        callback === nothing || callback(iters, x)
    end

    return f, iters, reason
end

"""
    solve_alm_lbfgs(w, A_in, b_in, A_eq, b_eq, lb, ub, x0; max_outer, max_inner)

Solves the box-constrained LP

    minimize    w' * x
    subject to  A_in * x <= b_in
                A_eq * x == b_eq
                lb <= x <= ub

with an augmented Lagrangian method, inner subproblems by projected L-BFGS
(`projected_lbfgs!`) — gradient-only, no factorisation. Hestenes-Powell form
(Nocedal & Wright, Ch. 17):

    L(x) = w'x + lambda'r_eq + (rho/2)||r_eq||^2
              + (1/2rho) * (||max(0, mu + rho*r_in)||^2 - ||mu||^2)

with `r_in = A_in x - b_in`, `r_eq = A_eq x - b_eq`; each L/grad-L costs four
sparse matrix-vector products. Inputs should be Ruiz-scaled (see `build_mga_lp`)
so a single `rho` is commensurate across rows. `rho` starts by matching the
penalty-gradient magnitude to `||w||` and rises x10, per the Alg. 17.4 safeguard
(ALM) or every iteration (quadratic penalty, when `use_multipliers = false`).
`max_outer`/`max_inner` are the budget (inexact inner solves are fine for ALM).

`prox_weight > 0` adds `(prox_weight/2)||x - prox_center||^2` (bundle
stabilisation for `oracle_search`). `history`, if given, collects the
per-iteration trace.
"""
function solve_alm_lbfgs(
    w::Vector{Float64},
    A_in::SparseMatrixCSC{Float64,Int},
    b_in::Vector{Float64},
    A_eq::SparseMatrixCSC{Float64,Int},
    b_eq::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64},
    x0::Vector{Float64};
    max_outer::Int = 30,
    max_inner::Int = 1000,
    verbose::Bool = true,
    prox_weight::Float64 = 0.0,
    prox_center::Vector{Float64} = Float64[],
    use_multipliers::Bool = true,
    feas_tol_override::Union{Nothing,Float64} = nothing,
    history::Union{Nothing,Vector} = nothing,
    precond! = nothing,
)
    n = length(w)
    m_in = length(b_in)
    m_eq = length(b_eq)
    has_prox = prox_weight > 0.0 && length(prox_center) == n
    t0 = time()
    cum_inner = Ref(0)   # inner-iteration counter shared across outer loops

    # Explicit transposes: SparseMatrixCSC adjoint-vector products are slow
    # row-wise traversals; materialising the transpose once makes A'*y a
    # plain CSC product.
    At_in = sparse(A_in')
    At_eq = sparse(A_eq')

    # Preallocated buffers - the inner L-BFGS evaluates the augmented
    # Lagrangian thousands of times and must not allocate O(n)/O(m) vectors
    # per call.
    r_in = zeros(m_in)
    r_eq = zeros(m_eq)
    bracket = zeros(m_in)
    y_eq = zeros(m_eq)
    g_buf = zeros(n)

    mu = zeros(m_in)
    lambda = zeros(m_eq)

    x = clamp.(x0, lb, ub)

    function residuals!(xx)
        if m_in > 0
            adjoint_mul!(r_in, At_in, xx)   # r_in = A_in * xx
            r_in .-= b_in
        end
        if m_eq > 0
            adjoint_mul!(r_eq, At_eq, xx)   # r_eq = A_eq * xx
            r_eq .-= b_eq
        end
        return nothing
    end

    # max(0, r_in) and |r_eq| infinity norms without allocating.
    infeasibility() = max(
        m_in == 0 ? 0.0 : mapreduce(r -> max(r, 0.0), max, r_in; init = 0.0),
        m_eq == 0 ? 0.0 : mapreduce(abs, max, r_eq; init = 0.0),
    )

    if m_in == 0 && m_eq == 0
        # Pure box LP: closed form, each component at the bound w prefers.
        boxsol = [
            w[j] >= 0 ? (isfinite(lb[j]) ? lb[j] : clamp(0.0, lb[j], ub[j])) :
            (isfinite(ub[j]) ? ub[j] : clamp(0.0, lb[j], ub[j])) for j = 1:n
        ]
        return boxsol, 0
    end

    # Initial penalty: make the constraint-gradient term comparable to w.
    residuals!(x)
    bracket .= max.(0.0, r_in)
    g_init = zeros(n)
    if m_in > 0
        adjoint_mul!(g_buf, A_in, bracket)   # g_buf = A_in' * bracket
        g_init .+= g_buf
    end
    if m_eq > 0
        adjoint_mul!(g_buf, A_eq, r_eq)      # g_buf = A_eq' * r_eq
        g_init .+= g_buf
    end
    # Match the penalty-gradient magnitude to ||w||. If the start point is
    # feasible (to the solver tolerance that produced it) the penalty gradient
    # is negligible and this scaling is meaningless - fall back to a neutral
    # rho = 1 (the problem is Ruiz-scaled, so O(1) is the right scale; ALM then
    # adapts rho upward as needed). The threshold is relative to ||w|| so it
    # triggers for any near-feasible warm start, not just an exactly feasible
    # one: without it rho would explode to ||w|| / (tiny residual) ~ 1e7-1e8.
    g_init_norm = norm(g_init)
    rho = g_init_norm > 1.0e-3 * max(norm(w), eps(Float64)) ? norm(w) / g_init_norm : 1.0

    b_scale =
        max(1.0, m_in == 0 ? 0.0 : maximum(abs, b_in), m_eq == 0 ? 0.0 : maximum(abs, b_eq))
    feas_tol =
        feas_tol_override === nothing ? sqrt(eps(Float64)) * b_scale : feas_tol_override

    prev_viol = Inf
    outers_done = 0
    # Inner-solve accuracy (Conn-Gould-Toint / LANCELOT): solving each augmented-
    # Lagrangian subproblem to full precision is wasted while the multipliers are
    # still far off. We instead require the inner projected gradient only as tight as
    # `inner_omega / rho_k` (loosened early when rho is small, tightening as rho grows),
    # floored at the final tolerance `g_final`. This lets early/warm subproblems stop
    # in a handful of inner iters instead of always hitting `max_inner`.
    g_final = sqrt(eps(Float64)) * max(1.0, norm(w))
    inner_omega = max(1.0, norm(w))

    # Record the starting point so the convergence trace begins at the initial
    # (high) error rather than after the first inner solve.
    if history !== nothing
        residuals!(x)
        push!(history, (iter = 0, t = 0.0, infeas = infeasibility(), obj = dot(w, x)))
    end

    for outer = 1:max_outer
        outers_done = outer
        rho_k = rho
        mu_k = copy(mu)
        lambda_k = copy(lambda)

        function fg!(F, G, xx)
            residuals!(xx)
            if m_in > 0
                @. bracket = max(0.0, mu_k + rho_k * r_in)
            end
            if m_eq > 0
                @. y_eq = lambda_k + rho_k * r_eq
            end
            if G !== nothing
                copyto!(G, w)
                if has_prox
                    @. G += prox_weight * (xx - prox_center)
                end
                if m_in > 0
                    adjoint_mul!(g_buf, A_in, bracket)   # A_in' * bracket
                    G .+= g_buf
                end
                if m_eq > 0
                    adjoint_mul!(g_buf, A_eq, y_eq)      # A_eq' * y_eq
                    G .+= g_buf
                end
            end
            if F !== nothing
                f = dot(w, xx)
                if has_prox
                    f += (prox_weight / 2) * sum(abs2, xx .- prox_center)
                end
                if m_in > 0
                    f += (sum(abs2, bracket) - sum(abs2, mu_k)) / (2 * rho_k)
                end
                if m_eq > 0
                    f += dot(lambda_k, r_eq) + (rho_k / 2) * sum(abs2, r_eq)
                end
                return f
            end
            return nothing
        end

        # Per-inner-iteration trace hook: records the *true* infeasibility and
        # objective at every accepted L-BFGS step (recomputing the residuals
        # costs two extra mat-vecs per step, paid only when tracing).
        trace_cb =
            history === nothing ? nothing :
            (_, xx) -> begin
                residuals!(xx)
                cum_inner[] += 1
                push!(
                    history,
                    (
                        iter = cum_inner[],
                        t = time() - t0,
                        infeas = infeasibility(),
                        obj = dot(w, xx),
                    ),
                )
            end

        # Loosen with small rho, tighten as rho grows, but never looser than 1e-6
        # (relative): a feasible warm start still demands a near-stationary inner
        # solve, else the outer breaks on `viol<=feas_tol` WITHOUT minimising w'x.
        inner_g_tol = clamp(inner_omega / rho_k, g_final, 1.0e-6 * inner_omega)
        local inner_iters, inner_reason
        t_inner = @elapsed _, inner_iters, inner_reason = projected_lbfgs!(
            fg!,
            x,
            lb,
            ub;
            max_iter = max_inner,
            g_tol = inner_g_tol,
            callback = trace_cb,
            precond! = precond!,
        )

        residuals!(x)
        # Hestenes-Powell multiplier update. With `use_multipliers = false`
        # the multipliers stay at zero and the method degrades exactly to the
        # classical *quadratic-penalty* method (the augmented Lagrangian with
        # mu = lambda = 0 is the pure penalty function) - this is how the
        # comparison harness obtains the penalty baseline from the same code.
        if use_multipliers
            if m_in > 0
                @. mu = max(0.0, mu + rho * r_in)
            end
            if m_eq > 0
                @. lambda = lambda + rho * r_eq
            end
        end

        viol = infeasibility()
        verbose &&
            @info "ALM outer $outer: w'x=$(round(dot(w, x), sigdigits=6)) | viol=$(round(viol, sigdigits=4)) | rho=$(round(rho, sigdigits=3)) | inner $(inner_iters) iters ($inner_reason) $(round(t_inner, digits=1))s"

        if viol <= feas_tol
            break
        end
        # Penalty-parameter update, by method:
        #   - ALM (use_multipliers): raise rho only when the outer step failed
        #     to cut infeasibility to a quarter of its previous value - the
        #     Nocedal & Wright Alg. 17.4 safeguard. At fixed rho the multiplier
        #     update still drives convergence, so re-solving is productive.
        #   - Quadratic penalty (use_multipliers = false): raise rho every
        #     iteration - the classical Framework 17.1 schedule. With the
        #     multipliers pinned at zero, re-minimising the *same* penalty
        #     function from its own minimiser is a no-op, so an adaptive rule
        #     would burn one wasted inner solve per real step.
        if !use_multipliers || viol > 0.25 * prev_viol
            rho *= 10
        end
        prev_viol = viol
    end

    return x, outers_done
end

"""
    lp = build_mga_lp(model, w, B, vars)

Extracts the full-space MGA LP `min w'x s.t. model constraints, cost(x) <= B,
lb <= x <= ub` once into Ruiz-equilibrated sparse matrices shared by every
first-order solver in the comparison. Returns a NamedTuple with the *scaled*
problem (`w_t, A_in, b_in, A_eq, b_eq, lb_t, ub_t`), the column scaling
`d_scale` (so `x_original = d_scale .* x_scaled`), and the extraction time.
The objective is rescaled and unit-normalised so `||w_t|| = 1`.
"""
function build_mga_lp(
    model::Model,
    w::Vector{Float64},
    B::Float64,
    vars::Vector{VariableRef},
)
    n = length(vars)

    t_extract = @elapsed begin
        # Full linear problem in matrix form via JuMP's own extractor:
        # `b_lower <= A x <= b_upper`, variable box `x_lower`/`x_upper`,
        # objective `c'x + c_offset`. Captures every linear constraint type
        # (including single-variable ones) so nothing is silently dropped.
        data = lp_matrix_data(model)
        col = Dict(v => j for (j, v) in enumerate(data.variables))
        perm = [col[v] for v in vars]                  # output column j is vars[j]

        At = sparse(data.A[:, perm]')                  # transpose for cheap row access
        rv = rowvals(At)
        nz = nonzeros(At)
        bl, bu = data.b_lower, data.b_upper

        I_in, J_in, V_in, b_in = Int[], Int[], Float64[], Float64[]
        I_eq, J_eq, V_eq, b_eq = Int[], Int[], Float64[], Float64[]
        row_in = row_eq = 0
        for i = 1:size(At, 2)                           # column i of At = constraint row i
            li, ui = bl[i], bu[i]
            if isfinite(li) && li == ui                 # equality
                row_eq += 1
                for k in nzrange(At, i)
                    push!(I_eq, row_eq)
                    push!(J_eq, rv[k])
                    push!(V_eq, nz[k])
                end
                push!(b_eq, li)
            else
                if isfinite(ui)                         # A x <= ui
                    row_in += 1
                    for k in nzrange(At, i)
                        push!(I_in, row_in)
                        push!(J_in, rv[k])
                        push!(V_in, nz[k])
                    end
                    push!(b_in, ui)
                end
                if isfinite(li)                         # -A x <= -li
                    row_in += 1
                    for k in nzrange(At, i)
                        push!(I_in, row_in)
                        push!(J_in, rv[k])
                        push!(V_in, -nz[k])
                    end
                    push!(b_in, -li)
                end
            end
        end

        # Budget row cost(x) <= B (objective assumed minimised).
        cvec = data.c[perm]
        row_in += 1
        for j = 1:n
            if cvec[j] != 0
                push!(I_in, row_in)
                push!(J_in, j)
                push!(V_in, cvec[j])
            end
        end
        push!(b_in, B - data.c_offset)

        A_in = sparse(I_in, J_in, V_in, row_in, n)
        A_eq = sparse(I_eq, J_eq, V_eq, row_eq, n)
        lb = collect(Float64, data.x_lower[perm])
        ub = collect(Float64, data.x_upper[perm])

        # Ruiz equilibration: the variable transform x = d .* x_scaled makes all
        # constraint row/column infinity norms ~1 - the shared preconditioning
        # every first-order solver depends on.
        d_scale = ruiz_equilibrate!(A_in, b_in, A_eq, b_eq, n)

        # Rescale the budget row (the last inequality row) to unit 2-norm. Its
        # coefficients are the whole cost vector, so `cost(x) <= B` is a dense sum
        # over all variables; Ruiz's infinity-norm scaling under-scales such rows
        # and leaves a huge RHS (~1e11 on data/5h) that poisons `feas_tol` and the
        # PDHG primal weight. A 2-norm rescale gives an O(1) RHS with unit-norm,
        # well-conditioned coefficients (equivalent constraint).
        budget_idx = row_in
        rv_in = rowvals(A_in)
        nz_in = nonzeros(A_in)
        rownorm2 = 0.0
        @inbounds for j = 1:n, k in nzrange(A_in, j)
            rv_in[k] == budget_idx && (rownorm2 += nz_in[k]^2)
        end
        rownorm2 = sqrt(rownorm2)
        if rownorm2 > eps(Float64)
            @inbounds for j = 1:n, k in nzrange(A_in, j)
                rv_in[k] == budget_idx && (nz_in[k] /= rownorm2)
            end
            b_in[budget_idx] /= rownorm2
        end
    end

    lb_t = lb ./ d_scale
    ub_t = ub ./ d_scale
    w_t = w .* d_scale
    w_t ./= max(norm(w_t), eps(Float64))

    return (
        w_t = w_t,
        A_in = A_in,
        b_in = b_in,
        A_eq = A_eq,
        b_eq = b_eq,
        lb_t = lb_t,
        ub_t = ub_t,
        d_scale = d_scale,
        n = n,
        t_extract = t_extract,
    )
end

"""
    primal_infeasibility(A_in, b_in, A_eq, b_eq, x)

Infinity-norm primal infeasibility `max(||max(0, A_in x - b_in)||_inf,
||A_eq x - b_eq||_inf)` - the common feasibility metric for comparing the
first-order solvers on a shared (Ruiz-scaled) problem.
"""
function primal_infeasibility(
    A_in::SparseMatrixCSC{Float64,Int},
    b_in::Vector{Float64},
    A_eq::SparseMatrixCSC{Float64,Int},
    b_eq::Vector{Float64},
    x::Vector{Float64},
)
    vin = isempty(b_in) ? 0.0 : maximum(max.(A_in * x .- b_in, 0.0))
    veq = isempty(b_eq) ? 0.0 : maximum(abs.(A_eq * x .- b_eq))
    return max(vin, veq)
end

"""
    solve_pdhg(w, A_in, b_in, A_eq, b_eq, lb, ub, x0; max_iter, restart_every)

Primal-dual hybrid gradient (Chambolle-Pock, 2011; PDLP-style, Applegate et al.
2021) for the box LP `min w'x s.t. A_in x <= b_in, A_eq x == b_eq, lb <= x <= ub`:

    y_in <- max(0, y_in + sigma (A_in xbar - b_in))     (dual ascent, ineq cone)
    y_eq <- y_eq + sigma (A_eq xbar - b_eq)              (dual ascent, eq)
    x    <- proj_box(x - tau (w + A_in' y_in + A_eq' y_eq))   (primal descent)
    xbar <- 2 x - x_prev                                 (extrapolation)

`tau = sigma = 0.9 / ||[A_in; A_eq]||_2` (operator norm by power iteration).

Accelerated with **restarted Halpern anchoring** (Lu & Yang 2024): each step is
`z <- lam*z0 + (1-lam)*F(z)` with `lam = 1/(k+2)`, and the anchor `z0` is reset to
the current iterate whenever the fixed-point residual `||F(z)-z||` drops by `1/e`.
This gives LAST-ITERATE (not merely ergodic) linear convergence on LP, which is
what lets a warm start `(x0, y_in0, y_eq0)` from a nearby budget actually pay off.
Returns the last iterate, its iteration count, and its final duals `(y_in, y_eq)`
for warm-starting the next solve. `history`, if given, collects the per-sample trace.
"""
function solve_pdhg(
    w::Vector{Float64},
    A_in::SparseMatrixCSC{Float64,Int},
    b_in::Vector{Float64},
    A_eq::SparseMatrixCSC{Float64,Int},
    b_eq::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64},
    x0::Vector{Float64};
    y_in0::Vector{Float64} = Float64[],
    y_eq0::Vector{Float64} = Float64[],
    feas_tol_override::Union{Nothing,Float64} = nothing,
    max_iter::Int = 10000,
    restart_every::Int = 1000,   # base artificial-restart period; it GROWS (doubles)
    # per artificial restart and resets on a decay restart
    # (see the restart block), so it only bites when
    # decay-restarts are absent (stiff/ill-cond. LP).
    verbose::Bool = true,
    progress_every::Int = 2000,
    kkt_every::Int = 256,
    opt_tol::Float64 = 1.0e-4,   # relative duality-gap / dual-feas tolerance (PDLP
    # default). Primal feasibility uses the tight
    # `feas_tol`; the gap only CERTIFIES optimality (the
    # primal objective is optimal long before it closes),
    # so a looser gap tol early-stops without changing the
    # returned alternative.
    history::Union{Nothing,Vector} = nothing,
    sample_every::Int = 50,
)
    n = length(w)
    m_in = length(b_in)
    m_eq = length(b_eq)
    t0 = time()
    At_in = sparse(A_in')
    At_eq = sparse(A_eq')

    ax_in = zeros(m_in)
    ax_eq = zeros(m_eq)
    aty = zeros(n)
    tmp = zeros(n)

    # A * x into (ax_in, ax_eq).
    Aop!(xx) = begin
        m_in > 0 && adjoint_mul!(ax_in, At_in, xx)
        m_eq > 0 && adjoint_mul!(ax_eq, At_eq, xx)
        nothing
    end
    # A' * [y_in; y_eq] into aty.
    Atop!(yin, yeq) = begin
        fill!(aty, 0.0)
        if m_in > 0
            adjoint_mul!(tmp, A_in, yin)
            aty .+= tmp
        end
        if m_eq > 0
            adjoint_mul!(tmp, A_eq, yeq)
            aty .+= tmp
        end
        nothing
    end

    # Operator norm L = ||[A_in; A_eq]||_2 by power iteration on A'A.
    z = randn(n)
    z ./= max(norm(z), eps(Float64))
    L = 1.0
    for _ = 1:20
        Aop!(z)
        Atop!(ax_in, ax_eq)         # aty = A'(A z)
        L = sqrt(max(dot(z, aty), eps(Float64)))
        nz = norm(aty)
        nz <= eps(Float64) && break
        z .= aty ./ nz
    end

    # Step size eta with tau*sigma = eta^2 <= 1/L^2 keeping F nonexpansive. We scale
    # the nominal 0.9/L by an adaptive factor `alpha in (0,1]` that BACKS OFF when an
    # epoch diverges (residual grows) - the PDLP-style safeguard that prevents the
    # primal-weight runaway observed on stiff LPs, where shrinking omega shrinks the
    # dual step until the dual can no longer enforce feasibility.
    eta0 = 0.9 / L
    alpha = 1.0
    eta = alpha * eta0
    # PDLP primal weight: primal step tau = eta/omega, dual step sigma = eta*omega.
    # omega rebalances primal-vs-dual progress (the LP conditioning lever) without
    # affecting stability (tau*sigma = eta^2 regardless). Start neutral (omega = 1)
    # and adapt at each restart from the primal/dual movement (bounded + smoothed).
    omega = 1.0
    tau = eta / omega
    sigma = eta * omega

    # Feasibility tolerance at the scale of the (Ruiz-scaled) right-hand sides,
    # matching the ALM solver so the methods stop at a commensurate accuracy.
    b_scale =
        max(1.0, m_in == 0 ? 0.0 : maximum(abs, b_in), m_eq == 0 ? 0.0 : maximum(abs, b_eq))
    feas_tol =
        feas_tol_override === nothing ? sqrt(eps(Float64)) * b_scale : feas_tol_override
    wnorm = max(1.0, maximum(abs, w))   # scale for relative dual-feasibility residual

    # State z = (x, y_in, y_eq). Primal from x0; duals warm-started (or zero).
    x = clamp.(x0, lb, ub)
    y_in = (m_in > 0 && length(y_in0) == m_in) ? copy(y_in0) : zeros(m_in)
    y_eq = (m_eq > 0 && length(y_eq0) == m_eq) ? copy(y_eq0) : zeros(m_eq)

    # One PDHG operator application F(z): a nonexpansive Chambolle-Pock step.
    fx = copy(x)
    fyin = zeros(m_in)
    fyeq = zeros(m_eq)
    xbar = copy(x)
    apply_F!() = begin
        Atop!(y_in, y_eq)                       # aty = A' y
        @. fx = clamp(x - tau * (w + aty), lb, ub)
        @. xbar = 2.0 * fx - x
        Aop!(xbar)                              # ax_in = A_in xbar, ax_eq = A_eq xbar
        m_in > 0 && (@. fyin = max(0.0, y_in + sigma * (ax_in - b_in)))
        m_eq > 0 && (@. fyeq = y_eq + sigma * (ax_eq - b_eq))
        r2 = 0.0
        @inbounds for i = 1:n
            r2 += (fx[i] - x[i])^2
        end
        @inbounds for i = 1:m_in
            r2 += (fyin[i] - y_in[i])^2
        end
        @inbounds for i = 1:m_eq
            r2 += (fyeq[i] - y_eq[i])^2
        end
        return sqrt(r2)
    end

    # Halpern anchor z0 (Lu & Yang 2024): z_{k+1} = lam*z0 + (1-lam)*F(z), lam =
    # 1/(k_inner+2), giving LAST-ITERATE O(1/k) on the fixed-point residual.
    # Restarting the anchor when the residual drops by 1/e makes it linear for LP -
    # and, unlike the ergodic average, the converged last iterate is exactly what
    # the next warm start exploits.
    anc_x = copy(x)
    anc_yin = copy(y_in)
    anc_yeq = copy(y_eq)
    k_inner = 0
    R_anchor = Inf
    iters_done = max_iter
    beta = 1.0 / MathConstants.e
    rel_p = rel_gap = rel_d = NaN          # last computed relative KKT residuals
    # Artificial-restart period (PDLP, Applegate et al. 2021, sec. 5): re-anchor +
    # re-balance omega if `period` iters pass without a sufficient-decay restart -
    # needed because otherwise lam = 1/(k_inner+2) -> 0 degrades Halpern into plain
    # (oscillatory, non-converging) Chambolle-Pock and the residual freezes. The
    # period GROWS after each artificial restart and resets on a decay restart, so a
    # well-behaved stiff solve gets ever-longer epochs to converge its last iterate,
    # while a stalled one keeps getting omega adaptations.
    period = restart_every

    if history !== nothing
        push!(
            history,
            (
                iter = 0,
                t = 0.0,
                infeas = primal_infeasibility(A_in, b_in, A_eq, b_eq, x),
                obj = dot(w, x),
            ),
        )
    end

    for k = 1:max_iter
        R = apply_F!()                          # fills (fx,fyin,fyeq); residual ||F(z)-z||
        R_anchor == Inf && (R_anchor = R)

        lam = 1.0 / (k_inner + 2)
        oml = 1.0 - lam
        @. x = lam * anc_x + oml * fx
        m_in > 0 && (@. y_in = lam * anc_yin + oml * fyin)
        m_eq > 0 && (@. y_eq = lam * anc_yeq + oml * fyeq)
        k_inner += 1

        if history !== nothing && k % sample_every == 0
            push!(
                history,
                (
                    iter = k,
                    t = time() - t0,
                    infeas = primal_infeasibility(A_in, b_in, A_eq, b_eq, x),
                    obj = dot(w, x),
                ),
            )
        end

        # Termination on the relative KKT residuals (PDLP, Applegate et al. 2021),
        # NOT the fixed-point residual R: under dual degeneracy R has a floor (the
        # dual drifts along the optimal face) so R never reaches tol even at the
        # optimum. We instead check, every `kkt_every` iters, relative primal
        # infeasibility, relative duality gap, and dual feasibility of the reduced
        # cost r = w + A'y against the box.
        if k % kkt_every == 0 || k == max_iter
            # Primal feasibility kept ABSOLUTE-tight (same criterion as the ALM solver,
            # so the two methods stop at commensurate feasibility - no precision loss).
            rel_p = primal_infeasibility(A_in, b_in, A_eq, b_eq, x)
            Atop!(y_in, y_eq)                       # aty = A' y  (current, post-update y)
            P = dot(w, x)
            D = -dot(b_in, y_in) - dot(b_eq, y_eq)  # dual objective (box folded in below)
            dinf = 0.0
            @inbounds for j = 1:n
                rj = w[j] + aty[j]                  # reduced cost
                if rj >= 0.0
                    isfinite(lb[j]) ? (D += rj * lb[j]) : (dinf = max(dinf, rj))
                else
                    isfinite(ub[j]) ? (D += rj * ub[j]) : (dinf = max(dinf, -rj))
                end
            end
            rel_gap = abs(P - D) / (1.0 + abs(P) + abs(D))
            rel_d = dinf / (1.0 + wnorm)
            if rel_p <= feas_tol && rel_gap <= opt_tol && rel_d <= opt_tol
                iters_done = k
                break
            end
        end

        # Restart logic (PDLP, Applegate et al. 2021, sec. 5; Lu & Yang 2024). Two
        # triggers re-anchor the Halpern iterate (resetting lam = 1/2, which keeps the
        # last-iterate contraction alive instead of degrading into oscillatory CP):
        #   (a) SUFFICIENT DECAY - residual dropped by 1/e vs the anchor;
        #   (b) ARTIFICIAL    - `period` iters passed without (a) (stiff LP, residual
        #                       plateaus and (a) never fires).
        # The primal weight `omega` is updated ONLY on a decay restart, where the
        # primal/dual movement is a trustworthy balance signal. On an artificial restart
        # we re-anchor but leave omega ALONE: on a cold-dual solve the movement ratio
        # points the wrong way (the dual is being established and lags, so the heuristic
        # would shrink the dual step it needs to grow), which previously ran omega away.
        decay_restart = R <= beta * R_anchor
        if decay_restart || k_inner >= period
            if R > 4.0 * R_anchor
                # Real instability (residual blew up >4x, not the normal non-monotone
                # wiggle): back off the step size rather than touch omega.
                alpha = max(alpha * 0.5, 1.0e-3)
                eta = alpha * eta0
                tau = eta / omega
                sigma = eta * omega
            elseif decay_restart
                # Trustworthy progress: smoothed PDLP primal-weight update (theta=1/2,
                # omega <- sqrt(omega*||dy||/||dx||)), per-restart change BOUNDED to
                # [1/4,4] and omega clamped to [1e-3,1e3]; let the step recover gently.
                dxn = 0.0
                @inbounds for i = 1:n
                    dxn += (x[i] - anc_x[i])^2
                end
                dyn = 0.0
                @inbounds for i = 1:m_in
                    dyn += (y_in[i] - anc_yin[i])^2
                end
                @inbounds for i = 1:m_eq
                    dyn += (y_eq[i] - anc_yeq[i])^2
                end
                if dxn > eps(Float64) && dyn > eps(Float64)
                    ratio = clamp(sqrt(omega * sqrt(dyn / dxn)) / omega, 0.25, 4.0)
                    omega = clamp(omega * ratio, 1.0e-3, 1.0e3)
                end
                alpha = min(alpha * 1.2, 1.0)
                eta = alpha * eta0
                tau = eta / omega
                sigma = eta * omega
            end

            copyto!(anc_x, x)
            m_in > 0 && copyto!(anc_yin, y_in)
            m_eq > 0 && copyto!(anc_yeq, y_eq)
            R_anchor = R
            k_inner = 0
            # A genuine decay restart resets the epoch length to the base; a purely
            # artificial one doubles it, so repeated stalls get geometrically longer
            # epochs (preserving the last-iterate contraction) rather than thrashing.
            period = decay_restart ? restart_every : min(2 * period, max_iter)
        end

        verbose &&
            k % progress_every == 0 &&
            @info "    HPDHG iter $k/$max_iter ($(round(time()-t0,digits=1))s): w'x=$(round(dot(w, x), sigdigits=6)) | relP=$(round(rel_p, sigdigits=3)) relGap=$(round(rel_gap, sigdigits=3)) relD=$(round(rel_d, sigdigits=3)) | R=$(round(R, sigdigits=3)) omega=$(round(omega,sigdigits=3)) alpha=$(round(alpha,sigdigits=3))"
    end

    # Halpern converges in the LAST iterate, so return it (with its duals) directly.
    return copy(x), iters_done, copy(y_in), copy(y_eq)
end

"""
    solve_osqp(w, A_in, b_in, A_eq, b_eq, lb, ub, x0; z0, y0, rho0, ...)

Canonical operator-splitting LP solver (OSQP; Stellato et al. 2020 / Boyd et al.
2011) for `min w'x s.t. A_in x <= b_in, A_eq x == b_eq, lb <= x <= ub`. Stacks the
constraints as `A = [A_in; A_eq; I]` with `l <= A x <= u`, and runs ADMM:

  - x-update: solve the FIXED SPD system `(sigma I + rho A'A) x = sigma x - w +
    A'(rho z - y)` by Jacobi-preconditioned CG (matrix-free: matvecs only, no
    factorisation - the memory-bound-friendly "indirect" mode);
  - z-update: `z = clamp(alpha A x + (1-alpha) z + y/rho, l, u)` (over-relaxed box
    projection);
  - y-update: `y += rho(alpha A x + (1-alpha) z - z)` (dual ascent),

with OSQP adaptive `rho` (balance scaled primal/dual residuals) and warm-startable
`(x, z, y)`. Unlike a gradient method, the x-update is an EXACT solve, so a sparse
objective `w` (few investment vars among many operational) is handled natively -
no drowning. Returns `(x, iters, z, y)` (the last two for warm starting the sweep).
"""
function solve_osqp(
    w::Vector{Float64},
    A_in::SparseMatrixCSC{Float64,Int},
    b_in::Vector{Float64},
    A_eq::SparseMatrixCSC{Float64,Int},
    b_eq::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64},
    x0::Vector{Float64};
    z0::Vector{Float64} = Float64[],
    y0::Vector{Float64} = Float64[],
    rho0::Float64 = 1.0,
    sigma::Float64 = 1.0e-6,
    relax::Float64 = 1.6,
    max_iter::Int = 8000,
    cg_maxit::Int = 500,
    adapt_rho::Bool = false,   # OSQP rho-adaptation MISFIRES from a feasible warm
    # start (primal already satisfied -> rho ratchets to
    # the floor -> stuck). The continuation starts feasible
    # (x*), so we default to a FIXED per-constraint rho
    # (SCS-style); enable only for cold starts.
    feas_tol_override::Union{Nothing,Float64} = nothing,
    verbose::Bool = true,
    progress_every::Int = 500,
)
    n = length(w)
    m_in = length(b_in)
    m_eq = length(b_eq)
    t0 = time()
    # Stacked constraint operator A = [A_in; A_eq; I_n], bounds l <= A x <= u.
    A = vcat(A_in, A_eq, sparse(I, n, n))
    At = sparse(A')
    m = size(A, 1)
    l = vcat(fill(-Inf, m_in), b_eq, lb)
    u = vcat(b_in, b_eq, ub)
    eps_t = feas_tol_override === nothing ? 1.0e-5 : feas_tol_override

    # Per-constraint rho (OSQP, Stellato et al. 2020 sec. 5.2): EQUALITY rows get
    # `rho_eq_fac` x the inequality/box rows. A single scalar rho lets the (loosely
    # scaled, huge) inequality residuals dominate the adaptive update and drive rho to
    # its floor -> constraints unenforced -> infeasible. Per-constraint weighting fixes
    # this. rho_vec = rho * bfac; bfac = rho_eq_fac on equality rows, 1 elsewhere.
    rv = rowvals(A)
    nz = nonzeros(A)
    rho_eq_fac = 1.0e3
    bfac = ones(m)
    @inbounds for i = (m_in+1):(m_in+m_eq)
        bfac[i] = rho_eq_fac
    end
    rho = rho0
    rho_vec = rho .* bfac
    # Jacobi diag(M)_j = sigma + sum_i rho_i A[i,j]^2 = sigma + rho * cn2w_j.
    cn2w = zeros(n)
    @inbounds for j = 1:n, k in nzrange(A, j)
        cn2w[j] += bfac[rv[k]] * nz[k]^2
    end
    dinv = 1.0 ./ (sigma .+ rho .* cn2w)

    x = clamp.(x0, lb, ub)
    z = (length(z0) == m) ? copy(z0) : (mul!(zeros(m), A, x))
    y = (length(y0) == m) ? copy(y0) : zeros(m)

    # Buffers + threaded matvecs (adjoint_mul!: y = M' x). A*v via At, A'*u via A.
    Av = zeros(m)
    Mx = zeros(n)
    rcg = zeros(n)
    zcg = zeros(n)
    pcg = zeros(n)
    Mp = zeros(n)
    rhs = zeros(n)
    Aty = zeros(n)
    zt = zeros(m)
    zr = zeros(m)
    iters_done = max_iter
    Aop!(out, v) = adjoint_mul!(out, At, v)    # out = A v
    Atop!(out, u) = adjoint_mul!(out, A, u)    # out = A' u

    applyM!(out, v) = begin                    # out = (sigma I + A' diag(rho) A) v
        Aop!(Av, v)
        @. Av *= rho_vec
        Atop!(out, Av)
        @. out = sigma * v + out
        nothing
    end
    pcg_solve!() = begin                       # solve M x = rhs (warm: current x), Jacobi-PCG
        applyM!(Mx, x)
        @. rcg = rhs - Mx
        @. zcg = dinv * rcg
        copyto!(pcg, zcg)
        rzv = dot(rcg, zcg)
        its = 0
        bnrm = norm(rhs)
        for c = 1:cg_maxit
            its = c
            applyM!(Mp, pcg)
            a = rzv / max(dot(pcg, Mp), eps(Float64))
            @. x += a * pcg
            @. rcg -= a * Mp
            norm(rcg) <= 1.0e-7 * (1.0 + bnrm) && break
            @. zcg = dinv * rcg
            rz2 = dot(rcg, zcg)
            beta = rz2 / max(rzv, eps(Float64))
            @. pcg = zcg + beta * pcg
            rzv = rz2
        end
        return its
    end

    cg_total = 0
    for k = 1:max_iter
        # x-update: rhs = sigma x - w + A'(diag(rho) z - y).
        @. zr = rho_vec * z - y
        Atop!(rhs, zr)
        @. rhs += sigma * x - w
        cg_total += pcg_solve!()
        # z-update (over-relaxed) + box projection; y-update (per-constraint rho).
        Aop!(zt, x)
        @. zr = relax * zt + (1.0 - relax) * z
        @inbounds for i = 1:m
            znew = clamp(zr[i] + y[i] / rho_vec[i], l[i], u[i])
            y[i] += rho_vec[i] * (zr[i] - znew)
            z[i] = znew
        end

        if k % 25 == 0 || k == max_iter
            Aop!(zt, x)
            Atop!(Aty, y)
            rp = 0.0
            sp_a = 0.0
            sp_z = 0.0
            @inbounds for i = 1:m
                rp = max(rp, abs(zt[i] - z[i]))
                sp_a = max(sp_a, abs(zt[i]))
                sp_z = max(sp_z, abs(z[i]))
            end
            rd = 0.0
            sd = abs_maxv(w)
            @inbounds for j = 1:n
                rd = max(rd, abs(w[j] + Aty[j]))
                sd = max(sd, abs(Aty[j]))
            end
            sp = max(sp_a, sp_z)
            if rp <= eps_t * (1.0 + sp) && rd <= eps_t * (1.0 + sd)
                iters_done = k
                break
            end
            # OSQP adaptive rho (scales the whole per-constraint vector): rho <-
            # rho * sqrt((rp/sp)/(rd/sd)). OFF by default - misfires from a feasible
            # warm start (see kwarg note).
            if adapt_rho
                ratio = sqrt((rp / max(sp, 1.0e-12)) / max(rd / max(sd, 1.0e-12), 1.0e-12))
                if ratio > 5.0 || ratio < 0.2
                    rho = clamp(rho * ratio, 1.0e-3, 1.0e6)
                    @. rho_vec = rho * bfac
                    @. dinv = 1.0 / (sigma + rho * cn2w)
                end
            end
            verbose &&
                k % progress_every == 0 &&
                @info "    OSQP iter $k/$max_iter ($(round(time()-t0,digits=1))s): w'x=$(round(dot(w,x),sigdigits=6)) | rPrim=$(round(rp,sigdigits=3)) rDual=$(round(rd,sigdigits=3)) rho=$(round(rho,sigdigits=3)) cg/it=$(round(cg_total/k,digits=1))"
        end
    end
    return copy(x), iters_done, copy(z), copy(y)
end

abs_maxv(v) = isempty(v) ? 0.0 : maximum(abs, v)

"""
    x_t, info = solve_firstorder(method, lp, x0; max_iters, max_inner, pdhg_iters)

Runs one of the Hessian-free first-order solvers on the shared scaled LP `lp`
(from `build_mga_lp`) and returns the scaled solution plus a metrics
NamedTuple `(method, iters, time, infeas, obj)`. Methods:

  - `:alm_lbfgs` - augmented Lagrangian + projected L-BFGS (the anchor),
  - `:penalty`   - quadratic-penalty + projected L-BFGS (ALM with multipliers
    disabled),
  - `:pdhg`      - primal-dual hybrid gradient (PDLP-style).
"""
function solve_firstorder(
    method::Symbol,
    lp,
    x0::Vector{Float64};
    y_in0::Vector{Float64} = Float64[],
    y_eq0::Vector{Float64} = Float64[],
    osqp_z0::Vector{Float64} = Float64[],
    osqp_y0::Vector{Float64} = Float64[],
    feas_tol::Union{Nothing,Float64} = nothing,
    max_iters::Int = 30,
    max_inner::Int = 1000,
    pdhg_iters::Int = 10000,
    osqp_iters::Int = 8000,
    verbose::Bool = false,
)
    local x_t, iters
    y_in = Float64[]
    y_eq = Float64[]
    osqp_z = Float64[]
    osqp_y = Float64[]
    t = @elapsed begin
        if method === :alm_lbfgs
            x_t, iters = solve_alm_lbfgs(
                lp.w_t,
                lp.A_in,
                lp.b_in,
                lp.A_eq,
                lp.b_eq,
                lp.lb_t,
                lp.ub_t,
                x0;
                max_outer = max_iters,
                max_inner = max_inner,
                verbose = verbose,
                use_multipliers = true,
                feas_tol_override = feas_tol,
            )
        elseif method === :penalty
            x_t, iters = solve_alm_lbfgs(
                lp.w_t,
                lp.A_in,
                lp.b_in,
                lp.A_eq,
                lp.b_eq,
                lp.lb_t,
                lp.ub_t,
                x0;
                max_outer = max_iters,
                max_inner = max_inner,
                verbose = verbose,
                use_multipliers = false,
                feas_tol_override = feas_tol,
            )
        elseif method === :pdhg
            x_t, iters, y_in, y_eq = solve_pdhg(
                lp.w_t,
                lp.A_in,
                lp.b_in,
                lp.A_eq,
                lp.b_eq,
                lp.lb_t,
                lp.ub_t,
                x0;
                y_in0 = y_in0,
                y_eq0 = y_eq0,
                feas_tol_override = feas_tol,
                max_iter = pdhg_iters,
                verbose = verbose,
            )
        elseif method === :osqp
            x_t, iters, osqp_z, osqp_y = solve_osqp(
                lp.w_t,
                lp.A_in,
                lp.b_in,
                lp.A_eq,
                lp.b_eq,
                lp.lb_t,
                lp.ub_t,
                x0;
                z0 = osqp_z0,
                y0 = osqp_y0,
                feas_tol_override = feas_tol,
                max_iter = osqp_iters,
                verbose = verbose,
            )
        else
            error(
                "Unknown first-order method :$method (expected :alm_lbfgs, :penalty, :pdhg, :osqp)",
            )
        end
    end
    infeas = primal_infeasibility(lp.A_in, lp.b_in, lp.A_eq, lp.b_eq, x_t)
    return x_t,
    (
        method = method,
        iters = iters,
        time = t,
        infeas = infeas,
        obj = dot(lp.w_t, x_t),
        y_in = y_in,
        y_eq = y_eq,
        osqp_z = osqp_z,
        osqp_y = osqp_y,
    )
end
