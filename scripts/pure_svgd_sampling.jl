# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# Pure SVGD sampling on Rosenbrock distribution (no ADMM)

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using ProgressMeter
using JLD2
using LinearAlgebra

# Random seed
Random.seed!(42)

# Read configuration from JSON file
args = read_config("pure_svgd_sampling.json")
args = parse_input_args(args)

println("="^60)
println("Pure SVGD Sampling for Rosenbrock Distribution")
println("="^60)
println("Parameters:")
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

# Create Rosenbrock distribution
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# Initialize particles from prior (standard normal)
n_dim = 2  # Rosenbrock is 2D
particles = randn(Float32, n_dim, args["n_particles"])

println("\nInitial particle statistics:")
println("  Mean: ", mean(particles, dims=2)[:])
println("  Std:  ", std(particles, dims=2)[:])

# Function to compute RBF kernel bandwidth using median heuristic
function compute_bandwidth(particles::AbstractMatrix{T}) where T
    n_particles = size(particles, 2)

    # Compute pairwise squared distances using broadcasting
    # For each dimension, compute (x_i - x_j)^2 for all pairs
    dists_sq = zeros(T, n_particles, n_particles)
    for d in 1:size(particles, 1)
        # Broadcasting: particles[d, i] - particles[d, j] for all i, j
        diff = particles[d, :]' .- particles[d, :]
        dists_sq .+= diff .^ 2
    end

    # Extract upper triangular part (excluding diagonal) and take square root
    dists = sqrt.(dists_sq[triu(trues(size(dists_sq)), 1)])

    # Median heuristic: h = median(distances) / sqrt(2 * log(n_particles))
    h = T(median(dists) / sqrt(2 * log(n_particles)))

    return h
end

# Function to compute SVGD update
function svgd_update!(particles::AbstractMatrix{T}, gradients::AbstractMatrix{T}, η::T, h::T) where T
    n_dim, n_particles = size(particles)

    # Compute pairwise squared distances
    # dists_sq[i, j] = ||x^i - x^j||^2
    dists_sq = zeros(T, n_particles, n_particles)
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
    term2 = zeros(T, n_dim, n_particles)
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

# Storage for history
history = Dict(
    "particles" => Vector{Matrix{Float32}}(),
    "mean" => Vector{Vector{Float32}}(),
    "std" => Vector{Vector{Float32}}(),
    "bandwidth" => Vector{Float32}(),
    "iteration" => Vector{Int}()
)

# Initialize bandwidth
if isnothing(args["h"])
    h = compute_bandwidth(particles)
    println("\nInitial bandwidth (h): ", round(h, digits=4))
else
    h = Float32(args["h"])
    println("\nUsing fixed bandwidth (h): ", round(h, digits=4))
end

# Sampling loop
println("\nRunning Pure SVGD sampling...")
@showprogress for iter in 1:args["n_iterations"]
    global h  # Declare as global to avoid scope warning

    # Compute gradients of log-posterior for all particles
    # Use Rosenbrock.gradlogpdf to avoid ambiguity
    gradients = Rosenbrock.gradlogpdf(RB_dist, particles)

    # Update bandwidth if specified
    if isnothing(args["h"]) && !isnothing(args["update_bandwidth_every"])
        if iter % args["update_bandwidth_every"] == 0
            h = compute_bandwidth(particles)
        end
    end

    # SVGD update
    svgd_update!(particles, gradients, Float32(args["eta"]), h)

    # Save history
    if iter % args["save_every"] == 0
        push!(history["particles"], copy(particles))
        push!(history["mean"], mean(particles, dims=2)[:])
        push!(history["std"], std(particles, dims=2)[:])
        push!(history["bandwidth"], h)
        push!(history["iteration"], iter)
    end
end

println("\nFinal particle statistics:")
println("  Mean: ", mean(particles, dims=2)[:])
println("  Std:  ", std(particles, dims=2)[:])
println("  Bandwidth (h): ", round(h, digits=4))

# Save results using DrWatson conventions
results = @dict(
    particles,
    history,
    args
)

# Create directory structure following DrWatson convention
!isdir(datadir(args["sim_name"])) && mkpath(datadir(args["sim_name"]))

# Save with DrWatson using savename
filename = savename(args, "jld2")
filepath = datadir(args["sim_name"], filename)
@tagsave(filepath, results)

println("\n" * "="^60)
println("Results saved to: ", filepath)
println("="^60)