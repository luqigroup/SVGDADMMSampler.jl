# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Residual Whiteness Principle: automatic selection of the augmented-Lagrangian
# penalty `mu`, which trades data fit against enforcement of the wave equation
# in the adjoint-wavefield subproblem `lambda = W (Q + mu I)^-1 delta_d`.
#
# With `Q = V diag(lambda) V^H`, the residual left by the regularized solve is
#
#     r(mu) = V diag( mu / (lambda + mu) ) V^H delta_d,
#
# scored by a normalized fourth moment of its Fourier transform,
#
#     phi(mu) = ||F r(mu)||_4^4 / ||F r(mu)||_2^4.
#
# Under-regularization leaves coherent signal in the residual and over-
# regularization leaves unexplained data; `phi` rises either way, and its
# minimum marks the residual closest to white. `phi` is homogeneous of degree
# zero, so it cannot be gamed by `mu` merely shrinking everything.
#
# Aghazade et al. (2025), arXiv:2512.16757.
#
# Two changes to the SEARCH, none to the criterion: `V^H delta_d` is projected
# once rather than per evaluation, and a logarithmic scan brackets the minimum
# before golden-section refinement, since `phi` is not unimodal over a range
# spanning four decades.

export rwp_penalty, whiteness

using FFTW
using LinearAlgebra

"""
    whiteness(r)

Normalized fourth moment `||F r||_4^4 / ||F r||_2^4` of a residual block,
computed on the two-dimensional Fourier transform of `r`.

Smaller is whiter: a residual with no spatial structure spreads its energy
evenly across wavenumbers and minimizes this ratio, while coherent structure
concentrates energy into few components and raises it. The value is invariant
under `r -> c r`.
"""
function whiteness(r::AbstractMatrix)
    R = fft(r)
    m2 = zero(real(eltype(R)))
    m4 = zero(real(eltype(R)))
    @inbounds for z in R
        a2 = abs2(z)
        m2 += a2
        m4 += a2 * a2
    end
    return m4 / (m2^2 + eps(typeof(m2)))
end

"""
    rwp_penalty(λ, V, delta_d; mu_min, mu_max, n_scan, tol)

Penalty parameter minimizing the residual whiteness criterion, given the
eigendecomposition `Q = V diag(λ) V^H` of the data-space Gram matrix and the
data residual `delta_d` (`nr x ns`).

The default bracket spans `1e-6 * λ_min_positive` to `1e-2 * λ_max`, matching
the reference implementation. `n_scan` sets the number of logarithmically
spaced probe points, and `tol` the relative width at which golden-section
refinement stops.

Returns the selected `mu`.
"""
function rwp_penalty(λ::AbstractVector, V::AbstractMatrix, delta_d::AbstractMatrix;
                     mu_min::Union{Nothing,Real} = nothing,
                     mu_max::Union{Nothing,Real} = nothing,
                     n_scan::Int = 48, tol::Real = 1e-4)
    size(V, 1) == size(delta_d, 1) ||
        throw(DimensionMismatch("V has $(size(V, 1)) rows, residual has " *
                                "$(size(delta_d, 1))"))
    n_scan ≥ 3 || throw(ArgumentError("n_scan must be at least 3, got $n_scan"))

    # Discard eigenvalues indistinguishable from zero: they are null directions
    # of Q and carry no information about the fit.
    λc = [x < 1e-12 ? 0.0 : float(x) for x in λ]
    λ_max = maximum(λc)
    λ_max > 0 || throw(ArgumentError("Gram matrix has no positive eigenvalues"))
    positive = filter(>(0), λc)
    λ_min = isempty(positive) ? λ_max : minimum(positive)

    lo = something(mu_min, 1e-6 * λ_min)
    hi = something(mu_max, 1e-2 * λ_max)
    # Keep the bracket strictly positive and ordered: at mu = 0 the residual
    # vanishes identically and the criterion degenerates to 0/eps.
    lo = max(lo, 1e-12 * hi)
    lo < hi || return hi

    # mu-independent projection, computed once.
    proj = V' * delta_d
    objective(μ) = whiteness(V * ((μ ./ (λc .+ μ)) .* proj))

    # Track the best point ever evaluated. Golden-section reports its final
    # bracket's midpoint, which when the optimum sits on an endpoint is worse
    # than a point already probed; keeping the incumbent removes that.
    best_mu, best_f = lo, Inf
    function probe(logmu)
        μ = exp(logmu)
        f = objective(μ)
        if f < best_f
            best_f, best_mu = f, μ
        end
        return f
    end

    # --- Logarithmic scan to bracket the global minimum.
    logs = range(log(lo), log(hi); length = n_scan)
    scan = [probe(lg) for lg in logs]
    best_i = argmin(scan)

    # --- Golden-section refinement in log space, where the criterion varies on
    # a natural scale.
    a = logs[max(best_i - 1, 1)]
    b = logs[min(best_i + 1, n_scan)]
    a < b && _golden_section(probe, a, b, tol)

    return best_mu
end

"""
    rwp_penalty(Q, delta_d; kwargs...)

Convenience form that takes the Gram matrix directly and eigendecomposes it.
Prefer the `(λ, V, delta_d)` method when the eigendecomposition is already
available — the ADMM iteration needs it for the shifted solve anyway, so
recomputing it here would double the cost.
"""
function rwp_penalty(Q::AbstractMatrix, delta_d::AbstractMatrix; kwargs...)
    λ, V = gram_eigen(Q)
    return rwp_penalty(λ, V, delta_d; kwargs...)
end

# Golden-section search for a minimum of `f` on `[a, b]`, stopping when the
# bracket is narrower than `tol` times its initial width.
function _golden_section(f, a::Real, b::Real, tol::Real)
    invphi = (sqrt(5) - 1) / 2
    width0 = b - a
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)

    while (b - a) > tol * width0
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        end
    end
    return (a + b) / 2
end
