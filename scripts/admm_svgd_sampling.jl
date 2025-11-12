# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using ProgressMeter
using JLD2

# Random seed
Random.seed!(42)

# Set up parameters using DrWatson's @dict macro
args = @dict(
    n_particles = 1000,
    n_iterations = 1000,
    μ = 0.1f0,           # Penalty parameter
    η = 0.001f0,         # SVGD step size (smaller for stability)
    h = nothing,         # Bandwidth: nothing = auto-compute, or specify value (e.g., 0.8f0)
    a = 1.0f0,           # Rosenbrock parameter a
    b = 100.0f0,         # Rosenbrock parameter b
    sim_name = "admm_svgd_rosenbrock"
)

println("="^60)
println("ADMM-SVGD Sampling for Rosenbrock Distribution")
println("="^60)
println("Parameters:")
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

# Initialize sampler with cached bandwidth (computed once at initialization)
n_dim = 2  # Rosenbrock is 2D
sampler = ADMMSVGDSampler(args[:n_particles], n_dim;
                          μ=args[:μ],
                          η=args[:η],
                          h=args[:h])  # Bandwidth computed here and cached

# Create Rosenbrock distribution for gradient computation
RB_dist = RosenbrockDistribution(0.0f0, args[:a])

println("\nInitial particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])
println("  Bandwidth (h): ", round(sampler.h, digits=4))

# Define the three functions needed for ADMM-SVGD

# 1. Solve for z given (x, ε)
# From the corrected LaTeX: z = (2b·x₂ + ε + μ·x₁²) / (2b + μ)
function solve_z_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]

    numerator = 2 * args[:b] .* x2 .+ s.ε .+ s.μ .* (x1 .^ 2)
    denominator = 2 * args[:b] + s.μ

    s.z .= numerator ./ denominator
    return nothing
end

# 2. Compute gradients of augmented Lagrangian
# This is the GRADIENT OF THE LOG-POSTERIOR, which is the NEGATIVE gradient of L
# L(x, z, ε) = (a - x₁)² + b(x₂ - z)² - ε(z - x₁²) + (μ/2)(z - x₁²)²
#
# ∂L/∂x₁ = -2(a - x₁) + 2x₁[ε - μ(z - x₁²)]
# ∂L/∂x₂ = 2b(x₂ - z)
#
# Since we want ∇log p(x) = -∇L(x, z, ε):
# ∇log p(x₁) = 2(a - x₁) + 2x₁[μ(z - x₁²) - ε]
# ∇log p(x₂) = 2b(z - x₂)
function compute_grad_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]
    z = s.z
    ε = s.ε
    μ = s.μ

    # ∇log p(x₁) = 2(a - x₁) + 2x₁[μ(z - x₁²) - ε]
    grad_x1 = 2 .* (args[:a] .- x1) .+ 2 .* x1 .* (μ .* (z .- x1 .^ 2) .- ε)

    # ∇log p(x₂) = 2b(z - x₂)
    grad_x2 = 2 * args[:b] .* (z .- x2)

    return vcat(grad_x1', grad_x2')  # Return as 2 × n_particles matrix
end

# 3. Update multipliers (dual ascent on ε)
# Following equation (33) from corrected LaTeX: ε ← ε + (x₁² - z)
function update_multiplier_fn(s)
    x1 = s.particles[1, :]

    # ∂L/∂ε = -(z - x₁²)
    # Dual ascent: ε ← ε + step_size·∂L/∂ε
    # With step_size = 1: ε ← ε - (z - x₁²) = ε + (x₁² - z)
    constraint_residual = x1 .^ 2 .- s.z
    s.ε .+= constraint_residual
    return nothing
end

# Run sampling with progress bar
history = sample!(
    sampler,
    args[:n_iterations],
    solve_z_fn,
    compute_grad_fn,
    update_multiplier_fn;
    verbose=true,
    save_every=10,
    update_bandwidth_every=50
)

println("\nFinal particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])
println("  Bandwidth (h): ", round(sampler.h, digits=4), " (unchanged)")

# Generate true samples for comparison
println("\nGenerating true Rosenbrock samples for comparison...")
true_samples = rand(RB_dist, args[:n_particles])

# Compute log-pdf for final particles
println("\nComputing log-pdf for final particles...")
final_logpdf = Rosenbrock.logpdf(RB_dist, sampler.particles)
println("  Mean log-pdf: ", round(mean(final_logpdf), digits=4))
println("  Std log-pdf:  ", round(std(final_logpdf), digits=4))

true_logpdf = Rosenbrock.logpdf(RB_dist, true_samples)
println("\nTrue sample log-pdf for comparison:")
println("  Mean log-pdf: ", round(mean(true_logpdf), digits=4))
println("  Std log-pdf:  ", round(std(true_logpdf), digits=4))

# Save results
println("\nSaving results...")

# Update args dict to include the actual bandwidth used
args_with_h = copy(args)
if args[:h] === nothing
    args_with_h[:h] = sampler.h  # Save the auto-computed bandwidth
end

results = merge(
    args_with_h,
    Dict(
        "final_particles" => sampler.particles,
        "true_samples" => true_samples,
        "particle_history" => history["particle_history"],
        "constraint_violations" => history["constraint_violations"],
        "iterations_saved" => history["iterations_saved"],
        "bandwidth_history" => history["bandwidth_history"],
        "final_z" => history["final_z"],
        "final_epsilon" => history["final_epsilon"],
        "final_logpdf" => final_logpdf,
        "true_logpdf" => true_logpdf
    )
)

# Save using DrWatson
save_path = datadir("sims", args[:sim_name], savename(args, "jld2"))
safesave(save_path, results)
println("Results saved to: $save_path")

println("\n" * "="^60)
println("Sampling complete!")
println("="^60)
println("\nTo visualize results, run:")
println("  julia scripts/admm_svgd_visualization.jl")

upload_to_dropbox(args[:sim_name])