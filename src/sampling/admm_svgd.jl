# ADMM-SVGD Sampler for Constrained Distributions
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export ADMMSVGDSampler, sample!

using LinearAlgebra
using Statistics

"""
    ADMMSVGDSampler

ADMM-based SVGD sampler for distributions with nonlinear constraints.

# Fields
- `n_particles::Int`: Number of particles
- `n_dim::Int`: Dimension of the model space
- `particles::Matrix{Float32}`: Particle positions (n_dim × n_particles)
- `z::Vector{Float32}`: Auxiliary variables (one per particle)
- `ε::Vector{Float32}`: Lagrange multipliers (one per particle)
- `μ::Float32`: Penalty parameter for constraint enforcement
- `η::Float32`: SVGD step size
"""
mutable struct ADMMSVGDSampler{T<:AbstractFloat}
    n_particles::Int
    n_dim::Int
    particles::Matrix{T}
    z::Vector{T}
    ε::Vector{T}
    μ::T
    η::T
end

"""
    ADMMSVGDSampler(n_particles::Int, n_dim::Int; μ::T=1.0f0, η::T=0.01f0) where T

Construct an ADMM-SVGD sampler.

# Arguments
- `n_particles`: Number of particles
- `n_dim`: Dimension of model space
- `μ`: Penalty parameter (default: 1.0)
- `η`: SVGD step size (default: 0.01)

# Returns
- `ADMMSVGDSampler` initialized with random particles
"""
function ADMMSVGDSampler(n_particles::Int, n_dim::Int; μ::T=1.0f0, η::T=0.01f0) where T<:AbstractFloat
    particles = randn(T, n_dim, n_particles)
    z = zeros(T, n_particles)
    ε = zeros(T, n_particles)
    return ADMMSVGDSampler(n_particles, n_dim, particles, z, ε, μ, η)
end

"""
    solve_z!(sampler::ADMMSVGDSampler, a::T, b::T) where T

Solve for auxiliary variable z given current particles and multipliers.

For Rosenbrock: z = (a + b*x₂ + ε + μ*x₁²) / (1 + b + μ)
"""
function solve_z!(sampler::ADMMSVGDSampler{T}, a::T, b::T) where T
    x1 = sampler.particles[1, :]
    x2 = sampler.particles[2, :]

    numerator = a .+ b .* x2 .+ sampler.ε .+ sampler.μ .* (x1 .^ 2)
    denominator = 1 + b + sampler.μ

    sampler.z .= numerator ./ denominator
    return nothing
end

"""
    compute_gradients(sampler::ADMMSVGDSampler, b::T) where T

Compute likelihood gradients for each particle.

Returns 2×n_particles matrix of gradients.
"""
function compute_gradients(sampler::ADMMSVGDSampler{T}, b::T) where T
    x1 = sampler.particles[1, :]
    x2 = sampler.particles[2, :]
    z = sampler.z
    ε = sampler.ε
    μ = sampler.μ

    # ∂L/∂x₁ = -2x₁[ε + μ(z - x₁²)]
    grad_x1 = -2 .* x1 .* (ε .+ μ .* (z .- x1 .^ 2))

    # ∂L/∂x₂ = b(x₂ - z)
    grad_x2 = b .* (x2 .- z)

    # Negative gradient for SVGD (we want to move toward high density)
    return -vcat(grad_x1', grad_x2')  # 2×n_particles
end

"""
    rbf_kernel(X::Matrix{T}, bandwidth::T) where T

Compute RBF kernel matrix and its gradients.

# Arguments
- `X`: Particle positions (n_dim × n_particles)
- `bandwidth`: Kernel bandwidth h

# Returns
- `K`: Kernel matrix (n_particles × n_particles)
- `grad_K`: Gradient of kernel w.r.t. particles (n_dim × n_particles × n_particles)
"""
function rbf_kernel(X::Matrix{T}, bandwidth::T) where T
    n_dim, n_particles = size(X)
    K = zeros(T, n_particles, n_particles)
    grad_K = zeros(T, n_dim, n_particles, n_particles)

    for i in 1:n_particles
        for j in 1:n_particles
            diff = X[:, j] .- X[:, i]  # n_dim vector
            dist_sq = sum(diff .^ 2)

            # Kernel value
            K[i, j] = exp(-dist_sq / (2 * bandwidth^2))

            # Gradient of kernel w.r.t. X[:, j]
            grad_K[:, i, j] = K[i, j] .* diff ./ bandwidth^2
        end
    end

    return K, grad_K
end

"""
    compute_bandwidth(X::Matrix{T}) where T

Compute kernel bandwidth using median heuristic.

h = median(pairwise_distances) / sqrt(2 * log(n_particles))
"""
function compute_bandwidth(X::Matrix{T}) where T
    n_dim, n_particles = size(X)

    # Compute pairwise distances
    distances = T[]
    for i in 1:n_particles
        for j in (i+1):n_particles
            dist = norm(X[:, i] .- X[:, j])
            push!(distances, T(dist))
        end
    end

    if isempty(distances)
        return one(T)
    end

    h = T(median(distances)) / T(sqrt(2 * log(n_particles)))
    return max(h, T(1e-6))  # Avoid zero bandwidth
end

"""
    svgd_update!(sampler::ADMMSVGDSampler, gradients::Matrix{T}) where T

Perform SVGD update on particles.

# Arguments
- `sampler`: The ADMM-SVGD sampler
- `gradients`: Log-posterior gradients (n_dim × n_particles)
"""
function svgd_update!(sampler::ADMMSVGDSampler{T}, gradients::Matrix{T}) where T
    n_dim, n_particles = sampler.n_dim, sampler.n_particles

    # Compute kernel bandwidth
    h = compute_bandwidth(sampler.particles)

    # Compute kernel and its gradients
    K, grad_K = rbf_kernel(sampler.particles, h)

    # Compute SVGD directions for each particle
    phi = zeros(T, n_dim, n_particles)

    for i in 1:n_particles
        # Attractive term: Σⱼ k(xⱼ, xᵢ) ∇log p(xⱼ)
        attractive = sum(K[i, j] .* gradients[:, j] for j in 1:n_particles)

        # Repulsive term: Σⱼ ∇ₓⱼ k(xⱼ, xᵢ)
        repulsive = sum(grad_K[:, i, j] for j in 1:n_particles)

        phi[:, i] = (attractive .+ repulsive) ./ n_particles
    end

    # Update particles
    sampler.particles .+= sampler.η .* phi

    return nothing
end

"""
    update_multipliers!(sampler::ADMMSVGDSampler)

Update Lagrange multipliers via dual ascent.

ε ← ε + (z - x₁²)
"""
function update_multipliers!(sampler::ADMMSVGDSampler)
    x1 = sampler.particles[1, :]
    constraint_residual = sampler.z .- x1 .^ 2
    sampler.ε .+= constraint_residual
    return nothing
end

"""
    sample!(sampler::ADMMSVGDSampler, n_iterations::Int;
            a::T=1.0f0, b::T=100.0f0, verbose::Bool=false) where T

Run ADMM-SVGD sampling for n_iterations.

# Arguments
- `sampler`: The ADMM-SVGD sampler
- `n_iterations`: Number of iterations
- `a`: Rosenbrock parameter (default: 1.0)
- `b`: Rosenbrock parameter (default: 100.0)
- `verbose`: Print progress (default: false)

# Returns
- `history`: Dictionary containing particle history and diagnostics
"""
function sample!(sampler::ADMMSVGDSampler{T}, n_iterations::Int;
                 a::T=1.0f0, b::T=100.0f0, verbose::Bool=false) where T

    # Storage for history
    particle_history = Vector{Matrix{T}}()
    constraint_violations = Vector{T}()

    for iter in 1:n_iterations
        # Step 1: Solve for z
        solve_z!(sampler, a, b)

        # Step 2: Compute gradients
        grads = compute_gradients(sampler, b)

        # Step 3: SVGD update on particles
        svgd_update!(sampler, grads)

        # Step 4: Update multipliers
        update_multipliers!(sampler)

        # Record history
        if iter % 10 == 0 || iter == 1 || iter == n_iterations
            push!(particle_history, copy(sampler.particles))

            # Compute constraint violation
            x1 = sampler.particles[1, :]
            violation = mean(abs.(sampler.z .- x1 .^ 2))
            push!(constraint_violations, violation)

            if verbose
                println("Iteration $iter: Constraint violation = $(round(violation, digits=6))")
            end
        end
    end

    return Dict(
        "particle_history" => particle_history,
        "constraint_violations" => constraint_violations,
        "final_particles" => sampler.particles,
        "final_z" => sampler.z,
        "final_epsilon" => sampler.ε
    )
end