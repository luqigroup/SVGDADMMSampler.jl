# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Stationary Gaussian random field prior on a periodic grid, with sampling and
# score in O(N log N) by FFT. The spectrum is Matern-like,
#
#     lambda(k) = ( sigma (4 pi^2 |k|^2 + tau^2)^(-alpha/2) )^2,
#
# so `alpha` controls smoothness and `tau` the correlation length. On a
# periodic grid the covariance is diagonal in the Fourier basis, making both
# operations one transform each -- the only practical route at 57600 points.
#
# The zero-frequency coefficient is zeroed, so every realization has exactly
# zero spatial mean and is a perturbation about its background model.
#
# Adapted from the reference `grf_init`/`grf_sample`/`grf_score`. The spectrum,
# scaling and regularization carry over unchanged; the two deliberate
# departures are documented on `grf_sample!` and `_score_grid`.

export GaussianRandomField, grf_sample, grf_sample!, grf_score, grf_logpdf,
       grf_covariance_spectrum, grf_pointwise_std

using FFTW
using Random
using Statistics

"""
    GaussianRandomField(size; alpha = 2.0, tau = 2.0, sigma = nothing,
                        pointwise_std = nothing, eig_rtol = 1e-6)

Stationary Gaussian random field on a periodic `(nz, nx)` grid.

  * `alpha` — spectral decay exponent; larger means smoother realizations.
  * `tau` — inverse correlation length.
  * `sigma` — spectral amplitude parameter, matching the reference
    implementation's meaning. When omitted it defaults to
    `tau^(alpha - dim/2)`.
  * `pointwise_std` — alternative to `sigma`: rescale the field so that each
    realization has exactly this marginal standard deviation. This is usually
    the knob you want, because it is stated in the units of the quantity being
    perturbed (squared slowness, here), whereas the realized spread implied by
    a given `sigma` depends on `alpha`, `tau`, and the grid size all at once.
    Give one of `sigma` or `pointwise_std`, not both.
  * `eig_rtol` — spectral components below `eig_rtol` times the largest are
    treated as null directions: they are not sampled and contribute nothing to
    the score. This regularizes the inverse covariance, which is otherwise
    unbounded on the near-null tail of the spectrum.

!!! note "Periodic boundary"
    The diagonalization assumes periodicity. On the PML-extended grid, where
    this prior is applied, the absorbing layers already isolate the physical
    region from the boundary, so wrap-around correlation across the domain
    edge is confined to the layers.
"""
struct GaussianRandomField
    n::Tuple{Int,Int}
    alpha::Float64
    tau::Float64
    sigma::Float64
    sqrt_eig::Matrix{Float64}     # sqrt of the covariance spectrum
    inv_eig::Matrix{Float64}      # regularized inverse spectrum (0 on null directions)
end

function GaussianRandomField(n::Tuple{Int,Int}; alpha::Real = 2.0, tau::Real = 2.0,
                             sigma::Union{Nothing,Real} = nothing,
                             pointwise_std::Union{Nothing,Real} = nothing,
                             eig_rtol::Real = 1e-6)
    nz, nx = n
    (nz > 0 && nx > 0) || throw(ArgumentError("grid size must be positive, got $n"))
    tau > 0 || throw(ArgumentError("tau must be positive, got $tau"))
    eig_rtol ≥ 0 || throw(ArgumentError("eig_rtol must be non-negative"))
    (isnothing(sigma) || isnothing(pointwise_std)) ||
        throw(ArgumentError("give either sigma or pointwise_std, not both"))

    dim = 2
    σ = something(sigma, tau^(0.5 * (2 * alpha - dim)))
    σ > 0 || throw(ArgumentError("sigma must be positive, got $σ"))

    kz = _fft_wavenumbers(nz)
    kx = _fft_wavenumbers(nx)

    spectrum = Matrix{Float64}(undef, nz, nx)
    @inbounds for j in 1:nx, i in 1:nz
        spectrum[i, j] = (4π^2 * (kz[i]^2 + kx[j]^2) + tau^2)^(-alpha / 2)
    end

    # The (nz * nx) factor cancels the 1/N in the inverse transform; the
    # sqrt(2) is absorbed by drawing CIRCULARLY-SYMMETRIC complex coefficients
    # in `grf_sample!` (see the note there), which is what makes the sampled
    # covariance exactly the spectrum `sqrt_eig .^ 2` that `inv_eig` inverts.
    sqrt_eig = (nz * nx) * sqrt(2) * σ .* spectrum
    sqrt_eig[1, 1] = 0.0                       # zero mean

    # Marginal variance follows from Parseval: Var(x_n) = sum_k lambda_k / N^2.
    if !isnothing(pointwise_std)
        pointwise_std > 0 ||
            throw(ArgumentError("pointwise_std must be positive, got $pointwise_std"))
        current = sqrt(sum(abs2, sqrt_eig)) / (nz * nx)
        scale = pointwise_std / current
        sqrt_eig .*= scale
        σ *= scale
    end

    eig = sqrt_eig .^ 2
    inv_eig = zeros(nz, nx)
    thresh = eig_rtol * maximum(eig)
    @inbounds for idx in eachindex(eig)
        eig[idx] > thresh && (inv_eig[idx] = 1 / eig[idx])
    end

    return GaussianRandomField((nz, nx), Float64(alpha), Float64(tau), Float64(σ),
                               sqrt_eig, inv_eig)
end

GaussianRandomField(nz::Int, nx::Int; kwargs...) =
    GaussianRandomField((nz, nx); kwargs...)

GaussianRandomField(g::PMLGrid; extended::Bool = true, kwargs...) =
    GaussianRandomField(extended ? g.ne : g.n; kwargs...)

Base.size(f::GaussianRandomField) = f.n
Base.length(f::GaussianRandomField) = prod(f.n)

Base.show(io::IO, f::GaussianRandomField) = print(
    io, "GaussianRandomField(", f.n[1], "x", f.n[2], ", alpha=", f.alpha,
    ", tau=", f.tau, ", sigma=", round(f.sigma; sigdigits = 4), ")")

"""
    grf_covariance_spectrum(field)

The covariance eigenvalues `lambda(k)` in FFT wavenumber order. The empirical
periodogram of many realizations converges to this, which is what
`test_grf.jl` checks.
"""
grf_covariance_spectrum(f::GaussianRandomField) = f.sqrt_eig .^ 2

"""
    grf_pointwise_std(field)

Marginal standard deviation of a realization at any single grid point. Follows
from Parseval as `sqrt(sum(lambda)) / N`, and is the quantity `pointwise_std`
sets directly at construction.
"""
grf_pointwise_std(f::GaussianRandomField) =
    sqrt(sum(abs2, f.sqrt_eig)) / prod(f.n)

"""
    grf_sample(field, n_samples; rng = Random.default_rng())

Draw `n_samples` realizations, returned as a `prod(size(field)) x n_samples`
matrix whose columns are vectorized fields — the layout the particle ensemble
uses. Pass `n_samples = nothing` for a single field returned in grid shape.

`rng` is threaded explicitly so ensembles are reproducible without touching
the global stream.
"""
function grf_sample(f::GaussianRandomField, n_samples::Int;
                    rng::AbstractRNG = Random.default_rng())
    n_samples ≥ 0 || throw(ArgumentError("n_samples must be non-negative"))
    out = Matrix{Float64}(undef, length(f), n_samples)
    buf = Matrix{ComplexF64}(undef, f.n)
    for j in 1:n_samples
        grf_sample!(view(out, :, j), f, buf; rng = rng)
    end
    return out
end

function grf_sample(f::GaussianRandomField; rng::AbstractRNG = Random.default_rng())
    return reshape(grf_sample(f, 1; rng = rng)[:, 1], f.n)
end

"""
    grf_sample!(out, field, buf; rng)

In-place single realization written into the length-`N` vector `out`, using
`buf` (a `size(field)` complex matrix) as scratch.

!!! note "Why the coefficients are complex"
    The reference implementation draws REAL Gaussian coefficients and takes
    the real part of the inverse transform. Writing `theta_k = 2 pi k.n / N`,
    that gives

        Cov(x_n, x_m) prop. sum_k s_k^2 [cos(theta_k^{n-m}) + cos(theta_k^{n+m})],

    whose second term depends on `n + m` rather than on `n - m`: the resulting
    field is not stationary, and its covariance is therefore not the spectrum
    that `inv_eig` inverts. Since `grf_score` is used as the prior term of the
    posterior gradient, a sampler and a score built on different covariances
    would mean the ensemble is initialized from one prior and pulled toward
    another.

    Drawing circularly-symmetric complex coefficients instead cancels the
    `n + m` term exactly, leaving `Cov(x_n, x_m)` a function of `n - m` alone
    and equal to the spectrum in `sqrt_eig .^ 2`. The variance is unchanged, so
    realizations look the same and have the same amplitude; only the spurious
    non-stationarity goes away. `test_grf.jl` verifies the result against
    Stein's identity, `E[score(x) x'] = -I`, which the real-coefficient
    construction fails.
"""
function grf_sample!(out::AbstractVector, f::GaussianRandomField,
                     buf::Matrix{ComplexF64};
                     rng::AbstractRNG = Random.default_rng())
    length(out) == length(f) ||
        throw(DimensionMismatch("output has $(length(out)) entries, expected $(length(f))"))
    # Circularly-symmetric complex Gaussian: E[zeta] = 0, E[|zeta|^2] = 2,
    # E[zeta^2] = 0. The last identity is what kills the non-stationary term,
    # and the unit coefficient is what makes E[|F x|^2] equal `sqrt_eig^2`
    # exactly -- see `grf_covariance_spectrum`.
    @inbounds for idx in eachindex(buf)
        buf[idx] = f.sqrt_eig[idx] * complex(randn(rng), randn(rng))
    end
    field = real.(ifft(buf))
    copyto!(out, vec(field))
    return out
end

"""
    grf_score(field, x)

Score of the prior, `grad_x log p(x) = -K^-1 x`, evaluated by FFT.

`x` may be a grid-shaped matrix, a vectorized field, or an `N x k` matrix of
`k` particles (columns treated independently); the return matches the input
shape.
"""
function grf_score(f::GaussianRandomField, x::AbstractMatrix)
    if size(x) == f.n
        return _score_grid(f, x)
    end
    size(x, 1) == length(f) ||
        throw(DimensionMismatch("expected $(length(f)) rows or a $(f.n) grid, " *
                                "got $(size(x))"))
    out = similar(x, Float64)
    for j in axes(x, 2)
        out[:, j] .= vec(_score_grid(f, reshape(view(x, :, j), f.n)))
    end
    return out
end

function grf_score(f::GaussianRandomField, x::AbstractVector)
    length(x) == length(f) ||
        throw(DimensionMismatch("expected $(length(f)) entries, got $(length(x))"))
    return vec(_score_grid(f, reshape(x, f.n)))
end

# Score of a grid-shaped field.
#
# THE FACTOR OF N. With `F` the unnormalized DFT (`fft`) and `Lambda` the
# covariance spectrum, the sampler realizes `E[F x (F x)^H] = Lambda`, hence
#
#     K = F^-1 Lambda F^-H = F^H Lambda F / N^2   (using F^-1 = F^H / N),
#     K^-1 = F^H Lambda^-1 F = N * ifft( Lambda^-1 .* fft( . ) ).
#
# The `N` is easy to lose because `ifft` already carries a `1/N`, and dropping
# it leaves a score that is internally consistent -- it still integrates to
# `grf_logpdf`, and a finite-difference check still passes -- while being the
# score of a distribution `N` times wider than the one being sampled from.
# Stein's identity in `test_grf.jl` is what pins it down. The reference
# implementation's `grf_score` omits this factor; the demo never calls it, so
# the discrepancy does not surface there.
_score_grid(f::GaussianRandomField, X::AbstractMatrix) =
    -prod(f.n) .* real.(ifft(f.inv_eig .* fft(complex.(Float64.(X)))))

"""
    grf_logpdf(field, x)

Log density up to an additive constant, `-x' K^-1 x / 2`, computed from the
same spectral representation as the score. Components on the regularized null
directions are excluded, consistent with `grf_score`.
"""
function grf_logpdf(f::GaussianRandomField, x::AbstractVecOrMat)
    X = size(x) == f.n ? x : reshape(x, f.n)
    return -0.5 * dot(vec(X), vec(-_score_grid(f, X)))
end

# FFT wavenumber ordering: 0, 1, ..., n/2 - 1, -n/2, ..., -1.
function _fft_wavenumbers(n::Int)
    kmax = n ÷ 2
    k = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k[i] = i - 1 < kmax ? i - 1 : i - 1 - n
    end
    return k
end
