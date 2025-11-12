# ADMM-SVGD Sampler for Constrained Distributions - VECTORIZED VERSION
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export ADMMSVGDSampler, sample!

using LinearAlgebra
using Statistics

"""
    ADMMSVGDSampler

ADMM-based SVGD sampler for distributions with nonlinear constraints.

This is a distribution-agnostic sampler that requires user-provided functions
for constraint solving and gradient computation.

# Fields
- `n_particles::Int`: Number of particles
- `n_dim::Int`: Dimension of the model space
- `particles::Matrix{T}`: Particle positions (n_dim × n_particles)
- `z::Vector{T}`: Auxiliary variables (one per particle)
- `ε::Vector{T}`: Lagrange multipliers (one per particle)
- `μ::T`: Penalty parameter for constraint enforcement
- `η::T`: SVGD step size
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
    ADMMSVGDSampler(n_particles::Int, n_dim::Int; μ::T=0.1f0, η::T=0.001f0, init_particles=nothing) where T

Construct an ADMM-SVGD sampler.

# Arguments
- `n_particles`: Number of particles
- `n_dim`: Dimension of model space
- `μ`: Penalty parameter (default: 0.1)
- `η`: SVGD step size (default: 0.001)
- `init_particles`: Initial particles (n_dim × n_particles), or nothing for random init

# Returns
- `ADMMSVGDSampler` initialized with particles
"""
function ADMMSVGDSampler(n_particles::Int, n_dim::Int; μ::T=0.1f0, η::T=0.001f0, init_particles=nothing) where T<:AbstractFloat
    if init_particles === nothing
        particles = randn(T, n_dim, n_particles)
    else
        particles = T.(init_particles)
        @assert size(particles) == (n_dim, n_particles) "init_particles must be $n_dim × $n_particles"
    end
    z = zeros(T, n_particles)
    ε = zeros(T, n_particles)
    return ADMMSVGDSampler(n_particles, n_dim, particles, z, ε, μ, η)
end

"""
    rbf_kernel_vectorized(X::Matrix{T}, bandwidth::T) where T

Compute RBF kernel matrix and its gradients using VECTORIZED operations.
This is MUCH faster than the nested loop version.

# Arguments
- `X`: Particle positions (n_dim × n_particles)
- `bandwidth`: Kernel bandwidth h

# Returns
- `K`: Kernel matrix (n_particles × n_particles)
- `grad_K`: Gradient of kernel w.r.t. particles (n_dim × n_particles × n_particles)
"""
function rbf_kernel_vectorized(X::Matrix{T}, bandwidth::T) where T
    n_dim, n_particles = size(X)

    # Compute pairwise differences using broadcasting
    # Create views for broadcasting: X[:, i] for all i, and X[:, j] for all j
    # diff[d, i, j] = X[d, j] - X[d, i]

    # Method: use reshape to add singleton dimensions for broadcasting
    X_i = reshape(X, n_dim, n_particles, 1)      # n_dim × n_particles × 1
    X_j = reshape(X, n_dim, 1, n_particles)      # n_dim × 1 × n_particles
    diff = X_j .- X_i                             # Broadcasting: n_dim × n_particles × n_particles

    # Compute squared distances: dist_sq[i, j] = ||X[:, j] - X[:, i]||²
    dist_sq = sum(diff .^ 2, dims=1)[1, :, :]  # Sum over dimension, gives n_particles × n_particles

    # Compute kernel matrix
    K = exp.(-dist_sq ./ (2 * bandwidth^2))

    # Compute gradient of kernel w.r.t. X[:, j]
    # grad_K[d, i, j] = K[i, j] * (X[d, j] - X[d, i]) / h²
    # Broadcasting: K is n_particles × n_particles, need to align with diff
    K_expanded = reshape(K, 1, n_particles, n_particles)  # Add dimension for broadcasting
    grad_K = K_expanded .* diff ./ bandwidth^2

    return K, grad_K
end

"""
    compute_bandwidth(X::Matrix{T}) where T

Compute kernel bandwidth using median heuristic.

h = median(pairwise_distances) / sqrt(2 * log(n_particles))
"""
function compute_bandwidth(X::Matrix{T}) where T
    n_dim, n_particles = size(X)

    # Vectorized pairwise distance computation
    # Use reshape and broadcasting to compute all pairwise differences at once
    X_i = reshape(X, n_dim, n_particles, 1)      # n_dim × n_particles × 1
    X_j = reshape(X, n_dim, 1, n_particles)      # n_dim × 1 × n_particles
    diff = X_j .- X_i                             # n_dim × n_particles × n_particles
    dist_sq = sum(diff .^ 2, dims=1)[1, :, :]    # n_particles × n_particles

    # Extract upper triangular distances (avoid diagonal and duplicates)
    distances = T[]
    for i in 1:n_particles
        for j in (i+1):n_particles
            push!(distances, sqrt(dist_sq[i, j]))
        end
    end

    if isempty(distances)
        return one(T)
    end

    h = T(median(distances)) / T(sqrt(2 * log(n_particles)))
    return max(h, T(1e-6))  # Avoid zero bandwidth
end

"""
    svgd_update_vectorized!(sampler::ADMMSVGDSampler, gradients::Matrix{T}; clip_norm::T=T(10.0)) where T

Perform SVGD update on particles with gradient clipping - VECTORIZED VERSION.
This is MUCH faster than the nested loop version.

# Arguments
- `sampler`: The ADMM-SVGD sampler
- `gradients`: Log-posterior gradients (n_dim × n_particles)
- `clip_norm`: Maximum gradient norm (default: 10.0)
"""
function svgd_update_vectorized!(sampler::ADMMSVGDSampler{T}, gradients::Matrix{T}; clip_norm::T=T(10.0)) where T
    n_dim, n_particles = sampler.n_dim, sampler.n_particles

    # Clip gradients to prevent explosion (vectorized)
    grad_norms = sqrt.(sum(gradients .^ 2, dims=1))  # 1 × n_particles
    scale_factors = min.(one(T), clip_norm ./ grad_norms)
    gradients_clipped = gradients .* scale_factors

    # Compute kernel bandwidth
    h = compute_bandwidth(sampler.particles)

    # Compute kernel and its gradients (vectorized)
    K, grad_K = rbf_kernel_vectorized(sampler.particles, h)

    # Compute SVGD directions using matrix operations
    # Attractive term: Σⱼ k(xⱼ, xᵢ) ∇log p(xⱼ)
    # = gradients_clipped * K'  where K[i,j] is kernel from j to i
    # Result is n_dim × n_particles
    attractive = gradients_clipped * K'  # Matrix multiplication!

    # Repulsive term: Σⱼ ∇ₓⱼ k(xⱼ, xᵢ)
    # grad_K is n_dim × n_particles × n_particles
    # Sum over j (second dimension of n_particles)
    repulsive = sum(grad_K, dims=3)[:, :, 1]  # n_dim × n_particles

    # Combine terms
    phi = (attractive .+ repulsive) ./ n_particles

    # Adaptive step size based on gradient magnitude
    phi_norms = sqrt.(sum(phi .^ 2, dims=1))
    phi_norm_max = maximum(phi_norms)
    effective_eta = phi_norm_max > T(1.0) ? sampler.η / phi_norm_max : sampler.η

    # Update particles (vectorized)
    sampler.particles .+= effective_eta .* phi

    return nothing
end

"""
    sample!(sampler::ADMMSVGDSampler, n_iterations::Int,
            solve_z_fn::Function, compute_grad_fn::Function, update_multiplier_fn::Function;
            verbose::Bool=false, save_every::Int=10)

Run ADMM-SVGD sampling for n_iterations with user-provided functions.

# Arguments
- `sampler`: The ADMM-SVGD sampler
- `n_iterations`: Number of iterations
- `solve_z_fn`: Function with signature `solve_z_fn(sampler) -> nothing` that updates sampler.z
- `compute_grad_fn`: Function with signature `compute_grad_fn(sampler) -> Matrix` returning gradients (n_dim × n_particles)
- `update_multiplier_fn`: Function with signature `update_multiplier_fn(sampler) -> nothing` that updates sampler.ε
- `verbose`: Print progress (default: false)
- `save_every`: Save history every N iterations (default: 10)

# Returns
- `history`: Dictionary containing particle history and diagnostics
"""
function sample!(sampler::ADMMSVGDSampler{T}, n_iterations::Int,
                 solve_z_fn::Function, compute_grad_fn::Function, update_multiplier_fn::Function;
                 verbose::Bool=false, save_every::Int=10) where T

    # Storage for history
    particle_history = Vector{Matrix{T}}()
    constraint_violations = Vector{T}()
    iterations_saved = Vector{Int}()

    for iter in 1:n_iterations
        # Step 1: Solve for z (user-defined)
        solve_z_fn(sampler)

        # Step 2: Compute gradients (user-defined)
        grads = compute_grad_fn(sampler)

        # Step 3: SVGD update on particles (VECTORIZED!)
        svgd_update_vectorized!(sampler, grads)

        # Step 4: Update multipliers (user-defined)
        update_multiplier_fn(sampler)

        # Record history
        if iter % save_every == 0 || iter == 1 || iter == n_iterations
            push!(particle_history, copy(sampler.particles))
            push!(iterations_saved, iter)

            # Compute constraint violation (assumes Rosenbrock structure z = x₁²)
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
        "iterations_saved" => iterations_saved,
        "final_particles" => sampler.particles,
        "final_z" => sampler.z,
        "final_epsilon" => sampler.ε
    )
end