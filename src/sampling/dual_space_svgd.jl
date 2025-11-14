# Dual-Space SVGD Sampler - SVGD on Dual Variables (ε) Only
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export DualSpaceSVGDSampler, sample_dual!

using LinearAlgebra
using Statistics

"""
    DualSpaceSVGDSampler

SVGD sampler for dual space with deterministic primal updates.

Key difference from ADMM-SVGD: SVGD is applied to dual variables (ε) only,
while primal variables (x) follow deterministic gradient descent.

# Fields
- `n_particles::Int`: Number of particles
- `n_dim::Int`: Dimension of the primal space
- `particles::Matrix{T}`: Primal variable positions (n_dim × n_particles)
- `z::Vector{T}`: Auxiliary variables (one per particle)
- `ε::Vector{T}`: Dual variables (one per particle) - SVGD APPLIED HERE
- `μ::T`: Penalty parameter for constraint enforcement
- `α::T`: Step size for primal x updates (deterministic gradient descent)
- `η_ε::T`: Step size for dual ε updates (SVGD)
- `h_ε::T`: RBF kernel bandwidth for ε
"""
mutable struct DualSpaceSVGDSampler{T<:AbstractFloat}
    n_particles::Int
    n_dim::Int
    particles::Matrix{T}
    z::Vector{T}
    ε::Vector{T}
    μ::T
    α::T
    η_ε::T
    h_ε::T
end

"""
    DualSpaceSVGDSampler(n_particles::Int, n_dim::Int; μ::T=0.1f0, α::T=0.01f0, η_ε::T=0.001f0, h_ε::Union{T,Nothing}=nothing, init_particles=nothing) where T

Construct a dual-space SVGD sampler.

# Arguments
- `n_particles`: Number of particles
- `n_dim`: Dimension of primal space
- `μ`: Penalty parameter (default: 0.1)
- `α`: Step size for primal x updates (default: 0.01)
- `η_ε`: Step size for dual ε SVGD updates (default: 0.001)
- `h_ε`: Bandwidth for ε kernel. If `nothing`, computed from initial ε (default: nothing)
- `init_particles`: Initial primal particles (n_dim × n_particles), or nothing for random init

# Returns
- `DualSpaceSVGDSampler` initialized with particles and bandwidth
"""
function DualSpaceSVGDSampler(n_particles::Int, n_dim::Int;
                              μ::T=0.1f0, α::T=0.01f0, η_ε::T=0.001f0,
                              h_ε::Union{Real,Nothing}=nothing,
                              init_particles=nothing) where T<:AbstractFloat
    if init_particles === nothing
        particles = randn(T, n_dim, n_particles)
    else
        particles = T.(init_particles)
        @assert size(particles) == (n_dim, n_particles) "init_particles must be $n_dim × $n_particles"
    end

    z = zeros(T, n_particles)
    ε = T(0.5) .* randn(T, n_particles)

    # Compute or set bandwidth for ε
    if h_ε === nothing
        # Start with reasonable default for scalar ε
        h_ε_val = T(0.5)
        @info "Using default bandwidth for ε: h_ε = $(round(h_ε_val, digits=4))"
    else
        h_ε_val = T(h_ε)
        @info "Using user-specified bandwidth for ε: h_ε = $(round(h_ε_val, digits=4))"
    end

    return DualSpaceSVGDSampler(n_particles, n_dim, particles, z, ε, μ, α, η_ε, h_ε_val)
end

"""
    compute_bandwidth_scalar(values::Vector{T}; minimum::T=T(1e-6)) where T

Compute kernel bandwidth for scalar values using median heuristic.

h = max(median(pairwise_distances) / sqrt(2 * log(n)), minimum)
"""
function compute_bandwidth_scalar(values::Vector{T}; minimum::T=T(1e-6)) where T
    n = length(values)

    # Compute pairwise distances
    distances = T[]
    for i in 1:n
        for j in (i+1):n
            push!(distances, abs(values[i] - values[j]))
        end
    end

    if isempty(distances) || all(distances .== 0)
        return max(one(T), minimum)
    end

    h = T(median(distances)) / T(sqrt(2 * log(n)))
    return max(h, minimum)
end

"""
    rbf_kernel_scalar(values::Vector{T}, bandwidth::T) where T

Compute RBF kernel matrix and its gradients for scalar values (dual variables ε).

# Arguments
- `values`: Scalar values (n_particles,) - the dual variables ε
- `bandwidth`: Kernel bandwidth h_ε

# Returns
- `K`: Kernel matrix (n_particles × n_particles)
- `grad_K`: Gradient of kernel w.r.t. values (n_particles × n_particles)
  where grad_K[i, j] = ∂k(ε_j, ε_i)/∂ε_j
"""
function rbf_kernel_scalar(values::Vector{T}, bandwidth::T) where T
    n = length(values)

    # Compute pairwise differences: diff[i, j] = ε_j - ε_i
    diff = values' .- values  # Broadcasting creates n × n matrix

    # Compute squared distances
    dist_sq = diff .^ 2

    # Compute kernel matrix: K[i, j] = k(ε_j, ε_i)
    K = exp.(-dist_sq ./ (2 * bandwidth^2))

    # Compute gradient: grad_K[i, j] = K[i, j] * (ε_j - ε_i) / h²
    grad_K = K .* diff ./ bandwidth^2

    return K, grad_K
end

"""
    svgd_update_epsilon!(sampler::DualSpaceSVGDSampler, gradients::Vector{T};
                         update_bandwidth::Bool=false) where T

Perform SVGD update on dual variables ε.

# Arguments
- `sampler`: The dual-space SVGD sampler
- `gradients`: Gradient of log-posterior w.r.t. ε (n_particles,)
- `update_bandwidth`: If true, recompute bandwidth from current ε values (default: false)
"""
function svgd_update_epsilon!(sampler::DualSpaceSVGDSampler{T}, gradients::Vector{T};
                              update_bandwidth::Bool=false) where T
    n_particles = sampler.n_particles

    # Optionally update bandwidth
    if update_bandwidth
        h_new = compute_bandwidth_scalar(sampler.ε; minimum=T(0.1))
        sampler.h_ε = h_new
    end

    # Compute kernel and its gradients
    K, grad_K = rbf_kernel_scalar(sampler.ε, sampler.h_ε)

    # Compute SVGD direction
    # Attractive term: Σⱼ k(ε_j, ε_i) * grad_j
    attractive = K' * gradients  # (n_particles,)

    # Repulsive term: Σⱼ ∂k(ε_j, ε_i)/∂ε_j
    repulsive = sum(grad_K, dims=2)[:]  # Sum over j, result is (n_particles,)

    # Combine terms
    phi_ε = (attractive .+ repulsive) ./ n_particles

    # Update dual variables
    sampler.ε .+= sampler.η_ε .* phi_ε

    return nothing
end

"""
    deterministic_update_primal!(sampler::DualSpaceSVGDSampler, gradients::Matrix{T}) where T

Perform deterministic gradient descent update on primal variables x.

# Arguments
- `sampler`: The dual-space SVGD sampler
- `gradients`: Gradient of augmented Lagrangian w.r.t. x (n_dim × n_particles)
"""
function deterministic_update_primal!(sampler::DualSpaceSVGDSampler{T}, gradients::Matrix{T}) where T
    # Simple gradient ascent on log-posterior (which is -∇L)
    sampler.particles .+= sampler.α .* gradients
    return nothing
end

"""
    sample_dual!(sampler::DualSpaceSVGDSampler, n_iterations::Int,
                 solve_z_fn::Function, compute_grad_x_fn::Function,
                 compute_grad_epsilon_fn::Function;
                 verbose::Bool=false, save_every::Int=10,
                 update_bandwidth_every::Union{Int,Nothing}=nothing)

Run dual-space SVGD sampling for n_iterations.

# Arguments
- `sampler`: The dual-space SVGD sampler
- `n_iterations`: Number of iterations
- `solve_z_fn`: Function with signature `solve_z_fn(sampler) -> nothing` that updates sampler.z
- `compute_grad_x_fn`: Function returning gradients w.r.t. x (n_dim × n_particles) for deterministic update
- `compute_grad_epsilon_fn`: Function returning gradients w.r.t. ε (n_particles,) for SVGD update
- `verbose`: Print progress (default: false)
- `save_every`: Save history every N iterations (default: 10)
- `update_bandwidth_every`: Update ε bandwidth every N iterations. If `nothing`, never update (default: nothing)

# Returns
- `history`: Dictionary containing particle history and diagnostics
"""
function sample_dual!(sampler::DualSpaceSVGDSampler{T}, n_iterations::Int,
                      solve_z_fn::Function,
                      compute_grad_x_fn::Function,
                      compute_grad_epsilon_fn::Function;
                      verbose::Bool=false, save_every::Int=10,
                      update_bandwidth_every::Union{Int,Nothing}=nothing) where T

    # Storage for history
    particle_history = Vector{Matrix{T}}()
    epsilon_history = Vector{Vector{T}}()
    constraint_violations = Vector{T}()
    iterations_saved = Vector{Int}()
    bandwidth_history = Vector{T}()
    epsilon_std_history = Vector{T}()

    for iter in 1:n_iterations
        # Step 1: Solve for z (user-defined)
        solve_z_fn(sampler)

        # Step 2: Compute gradients for primal x
        grads_x = compute_grad_x_fn(sampler)

        # Step 3: DETERMINISTIC update for primal x
        deterministic_update_primal!(sampler, grads_x)

        # Step 4: Compute gradients for dual ε
        grads_ε = compute_grad_epsilon_fn(sampler)

        # Step 5: SVGD update for dual ε
        should_update_bandwidth = (update_bandwidth_every !== nothing) && (iter % update_bandwidth_every == 0)
        svgd_update_epsilon!(sampler, grads_ε; update_bandwidth=should_update_bandwidth)

        # Record history
        if iter % save_every == 0 || iter == 1 || iter == n_iterations
            push!(particle_history, copy(sampler.particles))
            push!(epsilon_history, copy(sampler.ε))
            push!(iterations_saved, iter)
            push!(bandwidth_history, sampler.h_ε)
            push!(epsilon_std_history, std(sampler.ε))

            # Compute constraint violation
            x1 = sampler.particles[1, :]
            violation = mean(abs.(sampler.z .- x1 .^ 2))
            push!(constraint_violations, violation)

            if verbose
                println("Iteration $iter: Constraint violation = $(round(violation, digits=6)), " *
                       "ε std = $(round(std(sampler.ε), digits=4)), h_ε = $(round(sampler.h_ε, digits=4))")
            end
        end
    end

    return Dict(
        "particle_history" => particle_history,
        "epsilon_history" => epsilon_history,
        "constraint_violations" => constraint_violations,
        "iterations_saved" => iterations_saved,
        "bandwidth_history" => bandwidth_history,
        "epsilon_std_history" => epsilon_std_history,
        "final_particles" => sampler.particles,
        "final_z" => sampler.z,
        "final_epsilon" => sampler.ε
    )
end