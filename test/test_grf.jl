# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Verification of the Gaussian random field prior.
#
# The load-bearing test is STEIN'S IDENTITY. For x ~ N(0, K) the score is
# -K^-1 x, so
#
#     E[ score(x) x' ] = -K^-1 K = -I.
#
# This ties the sampler and the score together in a single check: it fails if
# the realized covariance is not the one the score inverts, whatever the reason
# -- a missing factor, a wrong normalization, or a non-stationary construction.
# It is also what distinguishes the complex-coefficient draw used here from the
# reference implementation's real-coefficient draw, and that comparison is made
# explicit at the end of this file.

using Test
using LinearAlgebra
using Statistics
using Random
using FFTW
using SVGDADMMSampler

@testset "GRF construction" begin
    f = GaussianRandomField((16, 16); alpha = 2.0, tau = 2.0, sigma = 1.0)
    @test size(f) == (16, 16)
    @test length(f) == 256

    # Zero-mean constraint: the DC component carries no power and is a null
    # direction of the score.
    λ = grf_covariance_spectrum(f)
    @test λ[1, 1] == 0
    @test f.inv_eig[1, 1] == 0

    # The spectrum is isotropic in wavenumber, so it must be symmetric under
    # k -> -k along both axes.
    @test λ[2, 1] ≈ λ[16, 1]
    @test λ[1, 2] ≈ λ[1, 16]
    @test λ[3, 5] ≈ λ[15, 13]

    # Smoother fields (larger alpha) put less power at high wavenumber.
    f_rough = GaussianRandomField((16, 16); alpha = 1.0, tau = 2.0, sigma = 1.0)
    f_smooth = GaussianRandomField((16, 16); alpha = 4.0, tau = 2.0, sigma = 1.0)
    hi = CartesianIndex(9, 9)                       # Nyquist corner
    ratio_rough = grf_covariance_spectrum(f_rough)[hi] / grf_covariance_spectrum(f_rough)[1, 2]
    ratio_smooth = grf_covariance_spectrum(f_smooth)[hi] / grf_covariance_spectrum(f_smooth)[1, 2]
    @test ratio_smooth < ratio_rough

    @test_throws ArgumentError GaussianRandomField((16, 16); tau = -1.0)
    @test_throws ArgumentError GaussianRandomField((0, 4))
    @test_throws ArgumentError GaussianRandomField((16, 16); sigma = 1.0,
                                                   pointwise_std = 1.0)

    # Construction from a PMLGrid targets the extended grid by default, since
    # that is where the model particles live.
    g = PMLGrid((10, 12), (10.0, 10.0); pml = 3)
    @test size(GaussianRandomField(g)) == extended_size(g)
    @test size(GaussianRandomField(g; extended = false)) == physical_size(g)
end

@testset "GRF sampling statistics" begin
    rng = Xoshiro(20260727)
    f = GaussianRandomField((24, 24); alpha = 2.0, tau = 3.0, sigma = 1.0)
    X = grf_sample(f, 4000; rng = rng)

    @test size(X) == (length(f), 4000)

    # Every realization has exactly zero spatial mean, by construction.
    @test maximum(abs, mean(X; dims = 1)) < 1e-10

    # Marginal standard deviation matches the closed form from Parseval.
    @test std(vec(X)) ≈ grf_pointwise_std(f) rtol = 0.05

    # The empirical periodogram converges to the covariance spectrum. This is
    # the direct check that the sampler realizes the intended covariance.
    P = zeros(f.n)
    for j in axes(X, 2)
        P .+= abs2.(fft(reshape(view(X, :, j), f.n)))
    end
    P ./= size(X, 2)
    λ = grf_covariance_spectrum(f)
    keep = λ .> 1e-3 * maximum(λ)
    @test median(abs.(P[keep] .- λ[keep]) ./ λ[keep]) < 0.10

    # pointwise_std sets the realized spread directly.
    f_std = GaussianRandomField((24, 24); alpha = 2.0, tau = 3.0, pointwise_std = 0.37)
    @test grf_pointwise_std(f_std) ≈ 0.37
    @test std(vec(grf_sample(f_std, 2000; rng = rng))) ≈ 0.37 rtol = 0.05
end

@testset "GRF score" begin
    f = GaussianRandomField((12, 12); alpha = 2.0, tau = 2.0, sigma = 1.0)

    # Shape handling: grid, vector, and ensemble forms agree.
    X = reshape(collect(1.0:144.0), 12, 12)
    s_grid = grf_score(f, X)
    s_vec = grf_score(f, vec(X))
    @test size(s_grid) == (12, 12)
    @test s_vec ≈ vec(s_grid)

    ens = hcat(vec(X), vec(2 .* X))
    s_ens = grf_score(f, ens)
    @test size(s_ens) == (144, 2)
    @test s_ens[:, 1] ≈ s_vec
    @test s_ens[:, 2] ≈ 2 .* s_vec               # linearity

    @test eltype(s_grid) <: Real                 # the score of a real field is real

    # The score is the gradient of grf_logpdf: check against a finite difference.
    x = vec(grf_sample(f; rng = Xoshiro(7)))
    g = grf_score(f, x)
    δ = 1e-6
    for j in (5, 40, 111)
        xp = copy(x); xp[j] += δ
        xm = copy(x); xm[j] -= δ
        fd = (grf_logpdf(f, xp) - grf_logpdf(f, xm)) / (2δ)
        @test isapprox(fd, g[j]; rtol = 1e-4, atol = 1e-8 * maximum(abs, g))
    end

    @test_throws DimensionMismatch grf_score(f, zeros(5))
end

@testset "Stein's identity ties the sampler to the score" begin
    # E[score(x) x'] = -I on the retained spectral directions. A small grid
    # keeps the outer product cheap enough to average directly.
    rng = Xoshiro(31337)
    f = GaussianRandomField((8, 8); alpha = 2.0, tau = 2.0, sigma = 1.0)
    N = length(f)
    n_draws = 60_000

    M = zeros(N, N)
    for _ in 1:n_draws
        x = vec(grf_sample(f; rng = rng))
        s = grf_score(f, x)
        M .+= s * x'
    end
    M ./= n_draws

    # The DC mode is a null direction (zero-mean constraint), so the identity
    # holds on the orthogonal complement of the constant vector: the projector
    # onto the retained subspace is I - 11'/N.
    n_retained = count(!iszero, f.inv_eig)
    @test n_retained == N - 1
    target = -(I(N) - fill(1 / N, N, N))

    # The TRACE is the discriminating statistic: it averages N terms each with
    # mean -1, so it concentrates tightly. The off-diagonal entries do not --
    # Var(s_i x_j) involves (K^-1)_ii K_jj, and this spectrum spans several
    # orders of magnitude -- so the Frobenius threshold is deliberately loose.
    # It still separates the constructions by a wide margin: a missing factor
    # of N or of 2 moves the diagonal itself, giving a relative error near 1
    # or 0.5 respectively, against the ~0.13 of Monte-Carlo noise here.
    @test isapprox(tr(M), -(N - 1); rtol = 0.02)
    @test isapprox(mean(diag(M)), -(N - 1) / N; rtol = 0.02)
    @test norm(M - target) / norm(target) < 0.20

    # The reference construction -- REAL Fourier coefficients, real part of the
    # inverse transform -- fails the same test, which is why it is not used.
    # Its realized covariance is half the spectrum the score inverts, and it
    # carries a non-stationary component, so the trace lands near -(N-1)/2.
    M_ref = zeros(N, N)
    for _ in 1:n_draws
        buf = f.sqrt_eig .* randn(rng, 8, 8)
        x = vec(real.(ifft(complex.(buf))))
        s = grf_score(f, x)
        M_ref .+= s * x'
    end
    M_ref ./= n_draws

    @test norm(M_ref - target) / norm(target) > 0.35      # clearly not -I
    @test isapprox(tr(M_ref), -(N - 1) / 2; rtol = 0.05)  # off by exactly a factor of two
end
