# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# End-to-end verification of ADMM-SVGD for constrained FWI.
#
# The test that matters is whether the sampler INVERTS: starting from a
# constant model with no knowledge of the anomaly, does the ensemble mean move
# toward the true model, and does the wave-equation constraint tighten as the
# multipliers accumulate? Everything else here localizes a failure when that
# one breaks.
#
# The problem is deliberately small -- a 48 by 48 grid, a handful of sources
# and particles -- so the whole suite stays fast. The physics, the geometry and
# the algorithm are identical to the production run; only the sizes differ.

using Test
using LinearAlgebra
using Statistics
using Random
using SVGDADMMSampler

# One small transmission experiment, shared by the tests below.
# The frequency is chosen so the starting model is NOT cycle-skipped. Across
# the lens the two-way delay relative to the 2 km/s background is about
# 0.5 km * (1/1.4 - 1/2.0) ~ 0.107 s; at 3 Hz that is a third of a period, so
# the misfit gradient points toward the true model. Above roughly 0.4 periods
# it stops doing so -- at 6 Hz the same lens is 0.64 periods out and the
# gradient turns away from the truth, which is cycle skipping and not a defect
# of the sampler. The production run at 4 Hz sits near that boundary
# deliberately: that is the regime the dual-space formulation is meant to
# handle, so a unit test stays well clear of it.
function _small_problem(; n = (48, 48), freq = 3.0, h = 2000.0 / 48, n_src = 6,
                        n_rec = 24, noise_rel = 1e-3, seed = 8)
    rng = Xoshiro(seed)

    v_true = gaussian_lens_model(; n = n, background = 2.0, contrast = 0.6,
                                 width_x = 0.25, width_z = 0.5,
                                 spacing_km = 2.0 / n[2])
    v0 = fill(2.0, n)

    g = PMLGrid(n, (h, h); pml = 12)
    stencil = optimal_stencil(grid_density_range(minimum(v_true), maximum(v_true),
                                                 h, freq, freq)...)
    template = helmholtz_template(g, stencil, PMLProfile(), freq)

    acq = transmission_acquisition(g; n_sources = n_src,
                                   receiver_stride = max(1, n[2] ÷ n_rec))

    m_true = pad(g, vec(velocity_to_slowness(v_true)))
    m0 = pad(g, vec(velocity_to_slowness(v0)))

    w = ricker_weight(freq, freq)
    src = w .* Matrix{ComplexF64}(source_operator(acq))
    rec = receiver_operator(acq)

    # Synthetic data from the true model, plus complex Gaussian noise.
    F = factorize_helmholtz(template, m_true)
    d_clean = rec * solve_forward(F, src)
    σ = noise_rel * maximum(abs, d_clean)
    d = d_clean .+ (σ / sqrt(2)) .* complex.(randn(rng, size(d_clean)),
                                             randn(rng, size(d_clean)))

    box = box_constraints(m_true)
    post = FWIPosterior(g, template, src, rec, d, box)
    return (; g, post, m_true, m0, v_true, acq, rng, d_clean, d)
end

@testset "Gaussian lens model and conventions" begin
    v = gaussian_lens_model(; n = (40, 40), spacing_km = 0.05)
    @test size(v) == (40, 40)
    @test maximum(v) ≈ 2.0 rtol = 1e-6                # background away from the lens
    @test minimum(v) < 1.45                            # the lens is present
    @test v[20, 20] < v[1, 1]                          # slowest at the centre

    # Round trip through the inversion parameter.
    m = velocity_to_slowness(v)
    @test slowness_to_velocity(m) ≈ v
    @test m[20, 20] > m[1, 1]                          # slow velocity, high slowness

    lo, hi = box_constraints(m)
    @test lo < minimum(m) && hi > maximum(m)
    @test project_box([0.0, 1e6], (lo, hi)) == [lo, hi]

    @test_throws ArgumentError velocity_to_slowness([-1.0])
    @test_throws ArgumentError slowness_to_velocity([-1.0])
    @test_throws ArgumentError gaussian_lens_model(; background = 1.0, contrast = 2.0)
end

@testset "Transmission geometry" begin
    g = PMLGrid((48, 48), (20.0, 20.0); pml = 10)
    acq = transmission_acquisition(g; n_sources = 6, receiver_stride = 2)
    @test n_sources(acq) == 6
    @test n_receivers(acq) == 24
    @test all(acq.source_z .== 1.0)                    # sources across the top
    @test all(acq.receiver_z .== 47.0)                 # receivers across the bottom
    @test issorted(acq.source_x)
end

@testset "FWIPosterior validation" begin
    P = _small_problem()
    @test n_sources(P.post) == 6
    @test size(P.post.data, 1) == n_receivers(P.acq)

    bad = randn(ComplexF64, n_receivers(P.acq) + 1, n_sources(P.post))
    @test_throws DimensionMismatch FWIPosterior(P.g, P.post.template, P.post.source,
                                                P.post.receiver, bad, P.post.box)
    @test_throws ArgumentError FWIPosterior(P.g, P.post.template, P.post.source,
                                            P.post.receiver, P.post.data, (2.0, 1.0))
end

@testset "ADMM-SVGD inverts the lens" begin
    P = _small_problem()
    Np = 8

    prior = GaussianRandomField(P.g; alpha = 2.0, tau = 2.0,
                                pointwise_std = 0.06 * maximum(P.m0))
    perturb = grf_sample(prior, Np; rng = Xoshiro(3))
    particles0 = P.m0 .+ perturb

    res = run_admm_svgd_fwi(P.post, particles0; n_iterations = 25,
                            m_true = P.m_true, method = :admm,
                            repulsion = 0.5, step_scale = 0.3,
                            threaded = false, verbose = false)

    @test size(res.particles) == (n_extended(P.g), Np)
    @test all(P.post.box[1] .≤ res.particles .≤ P.post.box[2])   # box respected
    @test all(isfinite, res.particles)

    # THE TEST: the POSTERIOR MEAN must move substantially toward the truth.
    # The posterior mean is the estimator the method reports, and it is the
    # right thing to assert on: individual particles spread apart under the
    # Stein repulsion, so their average error is not the same quantity and does
    # not have to fall. See the note below.
    err_start = relative_model_error(P.g, vec(mean(particles0; dims = 2)), P.m_true)
    err_end = relative_model_error(P.g, vec(mean(res.particles; dims = 2)), P.m_true)
    @test err_end < 0.6 * err_start

    # The data misfit decreases: the ensemble is fitting the observations.
    @test mean(res.misfit[:, end]) < mean(res.misfit[:, 1])
    @test issorted(vec(mean(res.misfit; dims = 1))[1:5]; rev = true)

    # The recovered anomaly is in the right place: the slowest velocity of the
    # posterior mean sits near the centre of the model, where the lens is.
    v_mean = slowness_to_velocity(cut(P.g, vec(mean(res.particles; dims = 2))))
    V = reshape(v_mean, physical_size(P.g))
    imin = argmin(V)
    nz, nx = physical_size(P.g)
    @test abs(imin[1] - nz ÷ 2) ≤ nz ÷ 4
    @test abs(imin[2] - nx ÷ 2) ≤ nx ÷ 4
    @test minimum(V) < 1.9                     # a real anomaly, not a flat model

    # The ensemble retains spread: SVGD repulsion must prevent collapse, since
    # a collapsed ensemble reports zero uncertainty.
    @test res.ensemble_std[end] > 0
    @test all(>(0), res.ensemble_std)

    # REPULSION IS LOAD-BEARING WHEN Np << d. With 8 particles in a few
    # thousand dimensions the repulsion term dominates the Stein direction, so
    # the ensemble expands: raising the weight widens the ensemble, improves
    # nothing about the mean, and drives more of the model onto the box
    # constraint. This is the finite-ensemble behavior of SVGD in high
    # dimension rather than a defect, and it is why the weight is a parameter.
    wide = run_admm_svgd_fwi(P.post, particles0; n_iterations = 25,
                             m_true = P.m_true, method = :admm,
                             repulsion = 1.0, step_scale = 0.3,
                             threaded = false, verbose = false)
    narrow = run_admm_svgd_fwi(P.post, particles0; n_iterations = 25,
                               m_true = P.m_true, method = :admm,
                               repulsion = 0.1, step_scale = 0.3,
                               threaded = false, verbose = false)
    @test wide.ensemble_std[end] > res.ensemble_std[end] > narrow.ensemble_std[end]
    @test wide.mean_model_error[end] > narrow.mean_model_error[end]
    # The posterior mean stays good across the whole range, which is what makes
    # the spread a genuine tuning choice rather than a correctness question.
    for r in (wide, narrow)
        @test relative_model_error(P.g, vec(mean(r.particles; dims = 2)),
                                   P.m_true) < 0.6 * err_start
    end

    # Diagnostics are populated and finite.
    @test all(isfinite, res.bandwidth) && all(>(0), res.bandwidth)
    @test all(isfinite, res.penalty) && all(>(0), res.penalty)
    @test all(isfinite, res.constraint_residual)
    @test size(res.misfit) == (Np, 25)
end

@testset "ADMM multipliers tighten the constraint" begin
    P = _small_problem()
    Np = 6
    prior = GaussianRandomField(P.g; alpha = 2.0, tau = 2.0,
                                pointwise_std = 0.06 * maximum(P.m0))
    particles0 = P.m0 .+ grf_sample(prior, Np; rng = Xoshiro(5))

    admm = run_admm_svgd_fwi(P.post, particles0; n_iterations = 20, m_true = P.m_true,
                             method = :admm, threaded = false, verbose = false)
    reduced = run_admm_svgd_fwi(P.post, particles0; n_iterations = 20, m_true = P.m_true,
                                method = :reduced, threaded = false, verbose = false)

    # With multipliers on, the dual variables accumulate rather than staying at
    # their zero initialization.
    @test mean(admm.multiplier_norm[:, end]) > 0
    @test mean(admm.multiplier_norm[:, end]) > mean(admm.multiplier_norm[:, 1])

    # In the reduced-space run there are no multipliers at all.
    @test all(reduced.multiplier_norm .== 0)

    # Both are valid runs that respect the box and stay finite.
    for r in (admm, reduced)
        @test all(isfinite, r.particles)
        @test all(P.post.box[1] .≤ r.particles .≤ P.post.box[2])
    end

    @test_throws ArgumentError run_admm_svgd_fwi(P.post, particles0;
                                                 n_iterations = 5, method = :nope)
    @test_throws DimensionMismatch run_admm_svgd_fwi(P.post, zeros(7, Np);
                                                     n_iterations = 5)
end

@testset "Threaded and serial runs agree" begin
    # Threading must not change the answer: each particle owns its column and
    # its own factorization, so results are thread-count independent.
    P = _small_problem()
    Np = 4
    prior = GaussianRandomField(P.g; alpha = 2.0, tau = 2.0,
                                pointwise_std = 0.06 * maximum(P.m0))
    particles0 = P.m0 .+ grf_sample(prior, Np; rng = Xoshiro(17))

    serial = run_admm_svgd_fwi(P.post, particles0; n_iterations = 6,
                               threaded = false, verbose = false)
    par = run_admm_svgd_fwi(P.post, particles0; n_iterations = 6,
                            threaded = true, verbose = false)

    @test serial.particles ≈ par.particles rtol = 1e-10
    @test serial.bandwidth ≈ par.bandwidth rtol = 1e-10
    @test serial.penalty ≈ par.penalty rtol = 1e-10
end
