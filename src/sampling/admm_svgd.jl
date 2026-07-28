# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# ADMM-SVGD Sampler with per-particle auxiliary variables (fully vectorized)

export ADMMSVGDSampler, step!

using Statistics
using LinearAlgebra

"""
    ADMMSVGDSampler

Sampler using SVGD with ADMM structure where each particle has its own auxiliary variable z.

Fields:
- particles: (n_dim, n_particles) state samples
- z: (n_particles,) auxiliary variable (one per particle)
- ε: (n_particles,) Lagrange multipliers (one per particle)
- μ: penalty parameter
- η: step size
- h: kernel bandwidth (computed adaptively if not provided)
"""
mutable struct ADMMSVGDSampler
    particles::Matrix{Float32}
    z::Vector{Float32}
    ε::Vector{Float32}
    μ::Float32
    η::Float32
    h::Float32

    function ADMMSVGDSampler(n_particles::Int, n_dim::Int;
                             μ=1.0f0, η=0.01f0, h=nothing)
        # Initialize particles from standard normal
        particles = randn(Float32, n_dim, n_particles)

        # Each particle has its own z and ε
        z = zeros(Float32, n_particles)
        ε = zeros(Float32, n_particles)

        # Compute initial bandwidth if not provided
        h_val = isnothing(h) ? compute_bandwidth(particles) : h

        new(particles, z, ε, μ, η, h_val)
    end
end

"""
    compute_bandwidth(particles)

Compute median heuristic bandwidth using vectorized operations.
"""
function compute_bandwidth(particles::Matrix{Float32})
    n_dim, n_particles = size(particles)
    if n_particles < 2
        return 1.0f0
    end

    # Compute pairwise squared distances using broadcasting
    dists_sq = zeros(Float32, n_particles, n_particles)
    for d in 1:n_dim
        diff = particles[d, :]' .- particles[d, :]
        dists_sq .+= diff .^ 2
    end

    # Extract upper triangular part (excluding diagonal) and take square root
    dists = sqrt.(dists_sq[triu(trues(size(dists_sq)), 1)])

    # Median heuristic: h = median(distances) / sqrt(2 * log(n_particles))
    return Float32(median(dists) / sqrt(2 * log(n_particles)))
end

"""
    svgd_update!(particles, gradients, η, h)

Perform vectorized SVGD update.
"""
function svgd_update!(particles::Matrix{Float32}, gradients::Matrix{Float32},
                      η::Float32, h::Float32)
    n_dim, n_particles = size(particles)

    # Compute pairwise squared distances
    # dists_sq[i, j] = ||x^i - x^j||^2
    dists_sq = zeros(Float32, n_particles, n_particles)
    for d in 1:n_dim
        diff = particles[d, :]' .- particles[d, :]
        dists_sq .+= diff .^ 2
    end

    # Compute kernel matrix: K[i, j] = k(x^i, x^j) = exp(-||x^i - x^j||^2 / (2h^2))
    K = exp.(-dists_sq ./ (2 * h^2))

    # First term: sum_j k(x^i, x^j) * grad^j
    # For each particle i: sum over j of K[i,j] * gradients[:, j]
    term1 = gradients * K'  # (n_dim, n_particles) * (n_particles, n_particles)

    # Second term: sum_j k(x^i, x^j) * (x^i - x^j) / h^2
    # For each particle i and dimension d: sum_j K[i,j] * (x^i_d - x^j_d) / h^2
    term2 = zeros(Float32, n_dim, n_particles)
    for d in 1:n_dim
        diff = particles[d, :]' .- particles[d, :]  # diff[i, j] = x^i_d - x^j_d
        term2[d, :] = sum(K .* diff, dims=2)[:]  # Sum over j for each i
    end
    term2 ./= h^2

    # Compute SVGD direction: phi^i = (term1 - term2) / n_particles
    phi = (term1 .- term2) ./ n_particles

    # Update particles
    particles .+= η .* phi

    return nothing
end

"""
    step!(sampler, solve_z_fn, compute_grad_fn, update_multiplier_fn)

Perform one ADMM-SVGD iteration with fully vectorized SVGD update.

Arguments:
- solve_z_fn: function(sampler) that updates sampler.z
- compute_grad_fn: function(sampler) that returns (n_dim, n_particles) gradients
- update_multiplier_fn: function(sampler) that updates sampler.ε
"""
function step!(sampler::ADMMSVGDSampler,
               solve_z_fn::Function,
               compute_grad_fn::Function,
               update_multiplier_fn::Function)

    # Step 1: Solve for z (per-particle, vectorized)
    solve_z_fn(sampler)

    # Step 2: Compute gradients (vectorized)
    grads = compute_grad_fn(sampler)  # (n_dim, n_particles)

    # Step 3: SVGD update (fully vectorized)
    svgd_update!(sampler.particles, grads, sampler.η, sampler.h)

    # Step 4: Update multipliers (vectorized)
    update_multiplier_fn(sampler)

    # Step 5: Update bandwidth (vectorized)
    sampler.h = compute_bandwidth(sampler.particles)

    return nothing
end