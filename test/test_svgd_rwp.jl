# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Verification of the generic Stein machinery and the RWP penalty selection.
#
# For SVGD the decisive test is that the algorithm actually samples: run it on a
# target whose mean and covariance are known in closed form and check the
# converged ensemble against them. Structural checks (symmetry, sign of the
# repulsion, kernel limits) localize a failure when that one breaks.
#
# For the RWP the decisive test is that the search is sound: the penalty it
# returns must be no worse, on the criterion being minimized, than any point of
# a dense independent scan of the same bracket -- endpoints included, since the
# optimum is not always interior. The line search itself is tested separately
# against functions whose minimizers are known in closed form.

using Test
using LinearAlgebra
using Statistics
using Random
using FFTW
using SVGDADMMSampler

@testset "Pairwise distances and bandwidth" begin
    X = [0.0 0.0; 3.0 4.0; -1.0 0.0]
    D2 = SVGDADMMSampler.pairwise_sqdist(X)

    @test D2 ≈ D2'                        # symmetric
    @test all(diag(D2) .== 0)             # exact zeros on the diagonal
    @test D2[1, 2] ≈ 25.0
    @test D2[1, 3] ≈ 1.0
    @test all(D2 .≥ 0)

    # Cancellation must not produce negative distances for far-apart particles.
    Y = 1e6 .+ randn(20, 5)
    @test all(SVGDADMMSampler.pairwise_sqdist(Y) .≥ 0)

    # The two median conventions differ in the documented direction: including
    # the zero self-distances pulls the median down.
    D2r = SVGDADMMSampler.pairwise_sqdist(randn(30, 4))
    h_incl = median_bandwidth(D2r; include_diagonal = true)
    h_excl = median_bandwidth(D2r; include_diagonal = false)
    @test h_incl < h_excl

    @test median_bandwidth(zeros(1, 1)) == 1.0            # single particle
    @test median_bandwidth(zeros(5, 5)) > 0               # collapsed ensemble
end

@testset "Stein direction structure" begin
    Np, d = 6, 3
    rng = Xoshiro(11)
    X = randn(rng, Np, d)
    scores = randn(rng, Np, d)

    phi, h = stein_direction(X, scores)
    @test size(phi) == (Np, d)
    @test h > 0

    # With zero score the direction is pure repulsion: two particles must move
    # apart, never together.
    X2 = [-1.0 0.0; 1.0 0.0]
    phi2, _ = stein_direction(X2, zeros(2, 2))
    @test phi2[1, 1] < 0                 # left particle pushed further left
    @test phi2[2, 1] > 0                 # right particle pushed further right
    @test phi2[:, 2] ≈ [0.0, 0.0]        # no motion in the orthogonal direction
    @test phi2[1, 1] ≈ -phi2[2, 1]       # antisymmetric

    # Halving the repulsion weight halves the repulsive part exactly, which is
    # the knob that reproduces the reference implementation's 1/(2h^2).
    phi_half, _ = stein_direction(X2, zeros(2, 2); repulsion = 0.5)
    @test phi_half ≈ 0.5 .* phi2

    # A large bandwidth makes the kernel nearly constant, so the attraction
    # term approaches the ensemble-mean score.
    phi_wide, _ = stein_direction(X, scores; h = 1e4)
    @test all(isapprox.(phi_wide, mean(scores; dims = 1); atol = 1e-3))

    @test_throws DimensionMismatch stein_direction(X, randn(Np, d + 1))
    @test_throws ArgumentError stein_direction(X, scores; h = -1.0)
end

@testset "SVGD samples a correlated Gaussian" begin
    # The end-to-end check: a target with known mean and covariance.
    rng = Xoshiro(4242)
    Σ = [1.0 0.6; 0.6 0.8]
    μ = [1.5, -0.75]
    Σinv = inv(Σ)

    Np = 400
    X = randn(rng, Np, 2) .* 0.3 .+ 3.0          # deliberately offset and narrow
    for _ in 1:4000
        scores = -(X .- μ') * Σinv
        phi, _ = stein_direction(X, scores)
        X .+= 0.05 .* phi
    end

    m̂ = vec(mean(X; dims = 1))
    Ĉ = cov(X)

    @test m̂ ≈ μ atol = 0.08
    @test Ĉ ≈ Σ rtol = 0.20
    # The correlation structure is the part a collapsed ensemble would lose.
    @test Ĉ[1, 2] / sqrt(Ĉ[1, 1] * Ĉ[2, 2]) ≈ 0.6 / sqrt(1.0 * 0.8) atol = 0.08

    # Halved repulsion under-disperses, which is exactly why the weight is
    # exposed rather than hard-coded.
    Xh = randn(Xoshiro(4242), Np, 2) .* 0.3 .+ 3.0
    for _ in 1:4000
        scores = -(Xh .- μ') * Σinv
        phi, _ = stein_direction(Xh, scores; repulsion = 0.5)
        Xh .+= 0.05 .* phi
    end
    @test tr(cov(Xh)) < tr(Ĉ)
end

@testset "AdaGrad scaling" begin
    s = AdaGradState(; decay = 0.9, fudge = 1e-6)
    phi = fill(2.0, 3, 2)

    a1 = adagrad_step!(s, phi)
    @test size(a1) == size(phi)
    @test all(a1 .≈ 2.0 ./ (1e-6 .+ 2.0))       # first call initializes from phi^2

    # A persistently large gradient direction gets damped toward unit scale.
    for _ in 1:50
        a1 = adagrad_step!(s, phi)
    end
    @test all(abs.(a1) .< 1.01)
    @test !isnothing(s.history)
end

@testset "Whiteness criterion" begin
    rng = Xoshiro(99)
    n = 64

    # White noise scores lower than a coherent plane wave of equal energy.
    noise = randn(rng, n, n)
    coherent = [cos(2π * 3 * i / n) for i in 1:n, j in 1:n]
    coherent .*= norm(noise) / norm(coherent)
    @test whiteness(noise) < whiteness(coherent)

    # Scale invariance: the criterion is homogeneous of degree zero.
    @test whiteness(noise) ≈ whiteness(7.3 .* noise) rtol = 1e-10

    # Adding structure to noise raises the score.
    @test whiteness(noise .+ 3 .* coherent) > whiteness(noise)
end

@testset "Golden-section line search" begin
    # The refinement step, tested where the answer is known in closed form.
    for x0 in (0.7, 0.05, 0.99)
        f = x -> (x - x0)^2
        @test SVGDADMMSampler._golden_section(f, 0.0, 1.0, 1e-8) ≈ x0 atol = 1e-5
    end
    # A monotone objective drives the bracket to the correct endpoint.
    @test SVGDADMMSampler._golden_section(x -> x, 0.0, 1.0, 1e-8) ≈ 0.0 atol = 1e-5
    @test SVGDADMMSampler._golden_section(x -> -x, 0.0, 1.0, 1e-8) ≈ 1.0 atol = 1e-5
end

@testset "RWP penalty selection" begin
    rng = Xoshiro(2718)
    nr, ns = 48, 6

    # A Gram matrix with a spectrum decaying over three decades, and a residual
    # made of white noise plus a strong coherent component confined to the
    # well-determined subspace -- the structure the criterion is built to see.
    V = Matrix(qr(randn(rng, nr, nr)).Q)
    λ = [10.0^(-3 * (i - 1) / (nr - 1)) for i in 1:nr]
    λ = reverse(sort(λ))

    coeff = randn(rng, nr, ns)
    coeff[1:8, :] .*= 25.0                       # coherent, high-eigenvalue part
    delta_d = V * coeff

    μ = rwp_penalty(λ, V, delta_d)
    @test μ > 0
    @test 1e-6 * minimum(filter(>(0), λ)) ≤ μ ≤ 1e-2 * maximum(λ)

    # The returned penalty must be no worse than ANY point of a dense
    # independent scan of the same bracket, including both endpoints. Because
    # the search reports the best point it ever evaluated rather than the
    # midpoint of its final bracket, this holds whether the optimum is interior
    # or sits on an endpoint.
    obj(m) = whiteness(V * ((m ./ (λ .+ m)) .* (V' * delta_d)))
    lo_b = 1e-6 * minimum(filter(>(0), λ))
    hi_b = 1e-2 * maximum(λ)
    grid = exp.(range(log(lo_b), log(hi_b); length = 400))
    @test obj(μ) ≤ minimum(obj, grid) * (1 + 1e-9)
    @test obj(μ) ≤ obj(lo_b) * (1 + 1e-9)
    @test obj(μ) ≤ obj(hi_b) * (1 + 1e-9)

    # Determinism, and agreement of the two entry points.
    @test rwp_penalty(λ, V, delta_d) == μ
    Q = Hermitian(V * Diagonal(λ) * V')
    @test rwp_penalty(Q, delta_d) ≈ μ rtol = 1e-3

    @test_throws DimensionMismatch rwp_penalty(λ, V, randn(nr + 1, ns))
    @test_throws ArgumentError rwp_penalty(λ, V, delta_d; n_scan = 2)
    @test_throws ArgumentError rwp_penalty(zeros(nr), V, delta_d)
end
