# Authors: Ali Siahkoohi, alisk@ucf.edu
# Reusable conditional-inference drivers for the Rosenbrock toy problem.
#
# Factored out of scripts/admm_svgd_conditional_sampling.jl and
# scripts/svgd_conditional_sampling.jl so the same inference can be invoked many
# times (e.g. for simulation-based calibration over independent noise
# realizations). The math is unchanged from those scripts; only the per-instance
# inference is wrapped in a function. Reuses ADMMSVGDSampler / step! /
# svgd_update! / compute_bandwidth from sampling/admm_svgd.jl.
#
# Posterior p(x | y) ∝ p(y | x) p(x); Rosenbrock prior
# p(x) ∝ exp(-a (x₁-μ)² - (x₂-x₁²)²); Gaussian likelihood y = x + N(0, σ²I).
# Constrained reformulation (paper eqs. 29-30): auxiliary z = x₁².

export run_admm_svgd_conditional, run_standard_svgd_conditional

using Random
using Statistics
import Rosenbrock  # qualified Rosenbrock.gradlogpdf only; avoids logpdf/gradlogpdf name clashes

# rng-accepting prior draw for the Rosenbrock distribution. The Rosenbrock package
# only defines `rand(RB, n)` (global RNG); this overload threads an explicit RNG so
# simulation-based calibration can use a thread-local, reproducible RNG without a
# global-RNG data race. The formula is IDENTICAL to Rosenbrock.rand(RB, n) (same
# draw order: x1 first, then x2), so the prior draw is bit-for-bit the same as the
# package's when fed an equivalently-seeded RNG. Extends the module's imported
# `rand` (see `import Random: rand` in SVGDADMMSampler.jl).
function rand(rng::AbstractRNG, RB::Rosenbrock.RosenbrockDistribution{T},
              n_samples::Int) where {T}
    x1 = randn(rng, T, n_samples) / sqrt(2 * RB.a) .+ RB.μ
    x2 = randn(rng, T, n_samples) / sqrt(T(2)) .+ x1 .^ 2
    return vcat(x1', x2')  # 2×n_samples matrix
end

"""
    run_admm_svgd_conditional(y_obs, RB_dist; n_particles, σ, μ, η, n_iterations, h=nothing, rng=nothing)

Run ADMM-SVGD conditional sampling of p(x | y_obs) for the Rosenbrock prior with
the constrained reformulation z = x₁². Returns the final `(2, n_particles)`
particle matrix approximating the posterior. `y_obs` is a length-2 vector.
The closures below are identical to scripts/admm_svgd_conditional_sampling.jl.

`rng` controls only the particle INITIALIZATION (for thread-safe, reproducible
SBC). It defaults to `nothing`, which keeps the original behavior exactly (the
constructor's global-RNG init) so existing callers are unaffected. When an rng is
supplied, the sampler's particles are re-drawn from it after construction and the
bandwidth is recomputed; the ADMM closures and σ²-free z-update are unchanged.
"""
function run_admm_svgd_conditional(y_obs::AbstractVector, RB_dist;
                                   n_particles::Int, σ::Real, μ::Real, η::Real,
                                   n_iterations::Int, h=nothing,
                                   rng::Union{Nothing,AbstractRNG}=nothing)
    σ²   = Float32(σ)^2
    a_rb = Float32(RB_dist.a)
    μ_rb = Float32(RB_dist.μ)
    y1   = Float32(y_obs[1])
    y2   = Float32(y_obs[2])

    sampler = ADMMSVGDSampler(n_particles, 2; μ=Float32(μ), η=Float32(η),
                              h=isnothing(h) ? nothing : Float32(h))

    # Re-initialize particles from the provided rng (thread-local, reproducible)
    # so SBC is deterministic and thread-safe. The constructor's global-RNG init
    # is kept for back-compat; only override when an rng is explicitly given.
    if !isnothing(rng)
        sampler.particles .= randn(rng, Float32, 2, n_particles)
        sampler.h = compute_bandwidth(sampler.particles)
    end

    # z-subproblem closed form: z = (2 x₂ + ε + μ x₁²) / (2 + μ)
    # matches paper eq 30 (σ²-free z-minimizer)
    solve_z_fn = s -> begin
        x1 = s.particles[1, :]
        x2 = s.particles[2, :]
        s.z .= (2 .* x2 .+ s.ε .+ s.μ .* (x1 .^ 2)) ./ (2 + s.μ)
    end

    # ∇log p(x|y) = -∇L, returned as (2, n_particles)
    compute_grad_fn = s -> begin
        x1 = s.particles[1, :]
        x2 = s.particles[2, :]
        dL_dx1 = (x1 .- y1) ./ σ² .+ 2 .* a_rb .* (x1 .- μ_rb) .+
                 2 .* x1 .* (s.ε .- s.μ .* (s.z .- x1 .^ 2))
        dL_dx2 = (x2 .- y2) ./ σ² .+ 2 .* (x2 .- s.z)
        vcat((-dL_dx1)', (-dL_dx2)')
    end

    # dual ascent: ε ← ε + μ (x₁² - z)
    update_multiplier_fn = s -> (s.ε .+= s.μ .* ((s.particles[1, :] .^ 2) .- s.z))

    for _ in 1:n_iterations
        step!(sampler, solve_z_fn, compute_grad_fn, update_multiplier_fn)
    end
    return copy(sampler.particles)
end

"""
    run_standard_svgd_conditional(y_obs, RB_dist; n_particles, σ, η, n_iterations, rng=Random.default_rng())

Run plain SVGD conditional sampling of p(x | y_obs) using the direct posterior
gradient ∇log p(x) + ∇log p(y|x) (no ADMM splitting). Returns `(2, n_particles)`.
Identical update to scripts/svgd_conditional_sampling.jl.

`rng` controls only the particle INITIALIZATION (for thread-safe, reproducible
SBC); it defaults to the global RNG, reproducing the original behavior exactly.
"""
function run_standard_svgd_conditional(y_obs::AbstractVector, RB_dist;
                                       n_particles::Int, σ::Real, η::Real,
                                       n_iterations::Int,
                                       rng::AbstractRNG=Random.default_rng())
    σ²  = Float32(σ)^2
    η32 = Float32(η)
    y   = Float32[Float32(y_obs[1]), Float32(y_obs[2])]

    particles = randn(rng, Float32, 2, n_particles)
    h = compute_bandwidth(particles)
    for _ in 1:n_iterations
        grad_prior = Rosenbrock.gradlogpdf(RB_dist, particles)
        grad_lik   = -(particles .- y) ./ σ²
        svgd_update!(particles, grad_prior .+ grad_lik, η32, h)
        h = compute_bandwidth(particles)
    end
    return copy(particles)
end
