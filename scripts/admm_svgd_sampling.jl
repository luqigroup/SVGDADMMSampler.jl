# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# ADMM-SVGD Sampling for Rosenbrock Distribution - CLEANED UP VERSION

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using ProgressMeter
using JLD2

# Random seed
Random.seed!(42)

# Read configuration from JSON file
args = read_config("admm_svgd_sampling.json")
args = parse_input_args(args)

println("="^60)
println("ADMM-SVGD Sampling for Rosenbrock Distribution")
println("="^60)
println("Parameters:")
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

# Create Rosenbrock distribution using SAME parameters as pure SVGD
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# Initialize sampler
n_dim = 2  # Rosenbrock is 2D
sampler = ADMMSVGDSampler(
    args["n_particles"],
    n_dim;
    μ=Float32(args["mu_penalty"]),
    η=Float32(args["eta"]),
    h=isnothing(args["h"]) ? nothing : Float32(args["h"])
)

println("\nInitial particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])
println("  Bandwidth (h): ", round(sampler.h, digits=4))

# =============================================================================
# Define the three functions needed for ADMM-SVGD
# =============================================================================

# NOTE: Based on MATLAB code analysis, for Rosenbrock we have:
# -log p(x) = (a - x₁)² + b(x₂ - x₁²)²
# where for RosenbrockDistribution(μ, a):
#   - The actual negative log-posterior is: a*(x₁ - μ)² + (x₂ - x₁²)²
#   - In our case: μ=0, a=1, so: -log p(x) = (x₁)² + (x₂ - x₁²)²
#
# We introduce constraint: z = x₁²
# Augmented Lagrangian: L = (x₁)² + (x₂ - z)² - ε(z - x₁²) + (μ/2)(z - x₁²)²

# IMPORTANT: The Rosenbrock distribution has parameter 'a' which appears in the likelihood
# For RosenbrockDistribution(0.0f0, 1.0f0), we have:
#   -log p(x) = 1.0*(x₁ - 0.0)² + (x₂ - x₁²)²
# So the 'b' parameter in the MATLAB code corresponds to 1.0 in our case

# 1. Solve for z given (x, ε)
# From augmented Lagrangian: ∂L/∂z = -2(x₂ - z) - ε + μ(z - x₁²) = 0
# Solving: (2 + μ)z = 2x₂ + ε + μx₁²
# Therefore: z = (2x₂ + ε + μx₁²) / (2 + μ)
function solve_z_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]

    # Using b = 1.0 (implicit in RosenbrockDistribution)
    b = 1.0f0
    numerator = 2 * b .* x2 .+ s.ε .+ s.μ .* (x1 .^ 2)
    denominator = 2 * b + s.μ

    s.z .= numerator ./ denominator
    return nothing
end

# 2. Compute gradients of augmented Lagrangian
# The gradient should be NEGATIVE gradient of L (since SVGD uses gradient of log-posterior)
#
# L(x, z, ε) = a*(x₁ - μ)² + (x₂ - z)² - ε(z - x₁²) + (μ/2)(z - x₁²)²
#
# ∂L/∂x₁ = 2a(x₁ - μ) + 2x₁[ε - μ(z - x₁²)]
# ∂L/∂x₂ = 2(x₂ - z)
#
# Gradient of log-posterior: ∇log p = -∇L
# ∇log p(x₁) = -2a(x₁ - μ) - 2x₁[ε - μ(z - x₁²)]
# ∇log p(x₂) = -2(x₂ - z)
function compute_grad_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]
    z = s.z
    ε = s.ε
    μ_penalty = s.μ

    # Get parameters from RB_dist
    μ_rb = RB_dist.μ  # Should be 0.0
    a = RB_dist.a      # Should be 1.0

    # Gradient of augmented Lagrangian (then negate for log-posterior gradient)
    # ∂L/∂x₁ = 2a(x₁ - μ_rb) + 2x₁[ε - μ_penalty(z - x₁²)]
    dL_dx1 = 2 .* a .* (x1 .- μ_rb) .+ 2 .* x1 .* (ε .- μ_penalty .* (z .- x1 .^ 2))

    # ∂L/∂x₂ = 2(x₂ - z)
    dL_dx2 = 2 .* (x2 .- z)

    # Gradient of log-posterior = -gradient of L
    grad_x1 = -dL_dx1
    grad_x2 = -dL_dx2

    return vcat(grad_x1', grad_x2')  # Return as 2 × n_particles matrix
end

# 3. Update multipliers (dual ascent on ε)
# Dual ascent: ε ← ε + ρ·∂L/∂ε where ∂L/∂ε = -(z - x₁²)
# With ρ = 1: ε ← ε - (z - x₁²) = ε + (x₁² - z)
function update_multiplier_fn(s)
    x1 = s.particles[1, :]
    constraint_residual = x1 .^ 2 .- s.z
    s.ε .+= constraint_residual
    return nothing
end

# =============================================================================
# Run sampling with progress bar
# =============================================================================

history = sample!(
    sampler,
    args["n_iterations"],
    solve_z_fn,
    compute_grad_fn,
    update_multiplier_fn;
    verbose=true,
    save_every=args["save_every"],
    update_bandwidth_every=isnothing(args["update_bandwidth_every"]) ? nothing : args["update_bandwidth_every"]
)

println("\nFinal particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])
println("  Bandwidth (h): ", round(sampler.h, digits=4))

# =============================================================================
# Generate true samples for comparison
# =============================================================================
println("\nGenerating true Rosenbrock samples for comparison...")
true_samples = rand(RB_dist, args["n_particles"])

# =============================================================================
# Compute log-pdf for final particles
# =============================================================================
println("\nComputing log-pdf for final particles...")
final_logpdf = Rosenbrock.logpdf(RB_dist, sampler.particles)
println("  Mean log-pdf: ", round(mean(final_logpdf), digits=4))
println("  Std log-pdf:  ", round(std(final_logpdf), digits=4))

true_logpdf = Rosenbrock.logpdf(RB_dist, true_samples)
println("\nTrue sample log-pdf for comparison:")
println("  Mean log-pdf: ", round(mean(true_logpdf), digits=4))
println("  Std log-pdf:  ", round(std(true_logpdf), digits=4))

# =============================================================================
# Save results using DrWatson conventions
# =============================================================================
println("\nSaving results...")

results = @dict(
    sampler.particles,
    true_samples,
    history,
    final_logpdf,
    true_logpdf,
    args
)

# Create directory structure following DrWatson convention
!isdir(datadir(args["sim_name"])) && mkpath(datadir(args["sim_name"]))

# Save with DrWatson using savename
filename = savename(args, "jld2")
filepath = datadir(args["sim_name"], filename)
@tagsave(filepath, results)

println("Results saved to: $filepath")

println("\n" * "="^60)
println("Sampling complete!")
println("="^60)
println("\nTo visualize results, run:")
println("  julia scripts/admm_svgd_visualization.jl")

upload_to_dropbox(args["sim_name"])