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

# Initialize sampler
n_dim = 2  # Rosenbrock is 2D
sampler = ADMMSVGDSampler(args[:n_particles], n_dim; μ=args[:μ], η=args[:η])

# Create Rosenbrock distribution for gradient computation
RB_dist = RosenbrockDistribution(0.0f0, args[:a])

println("\nInitial particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])

# Define the three functions needed for ADMM-SVGD

# 1. Solve for z
function solve_z_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]

    numerator = args[:a] .+ args[:b] .* x2 .+ s.ε .+ s.μ .* (x1 .^ 2)
    denominator = 1 + args[:b] + s.μ

    s.z .= numerator ./ denominator
    return nothing
end

# 2. Compute gradients using Rosenbrock's gradlogpdf
function compute_grad_fn(s)
    # Use the gradlogpdf from Rosenbrock package (explicitly qualified)
    grads_unconstrained = Rosenbrock.gradlogpdf(RB_dist, s.particles)  # 2 × n_particles

    # Add constraint gradient terms from augmented Lagrangian
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]
    z = s.z
    ε = s.ε
    μ = s.μ

    # From L(x, z, ε) = (1/2)(z - a)² + (b/2)(x₂ - z)² - ε(z - x₁²) + (μ/2)(z - x₁²)²
    # We want ∇log p(x) = -∇_x L (since we minimize L)

    # ∂L/∂x₁ = -ε·(-2x₁) + μ(z - x₁²)(-2x₁) = 2x₁[ε - μ(z - x₁²)]
    # So: ∇log p(x₁) = -∂L/∂x₁ = 2x₁[μ(z - x₁²) - ε]
    constraint_grad_x1 = 2 .* x1 .* (μ .* (z .- x1 .^ 2) .- ε)

    # ∂L/∂x₂ = b(x₂ - z)
    # So: ∇log p(x₂) = -∂L/∂x₂ = -b(x₂ - z) = b(z - x₂)
    constraint_grad_x2 = args[:b] .* (z .- x2)

    # Total gradient: unconstrained gradlogpdf + constraint contribution
    grads_total = grads_unconstrained .+ vcat(constraint_grad_x1', constraint_grad_x2')

    return grads_total
end

# 3. Update multipliers (dual ascent on ε)
function update_multiplier_fn(s)
    x1 = s.particles[1, :]
    # Constraint: z = x₁²
    # Residual: z - x₁²
    # Standard ADMM dual ascent: ε ← ε + ρ·(constraint_residual)
    # But we need to check the sign based on our Lagrangian formulation
    # L = ... - ε(z - x₁²) + ...
    # ∂L/∂ε = -(z - x₁²)
    # Dual ascent (maximize): ε ← ε + step_size·∂L/∂ε = ε - step_size·(z - x₁²)
    # OR equivalently, if we wrote L = ... + ε(x₁² - z) + ..., then ascent would be:
    # ε ← ε + step_size·(x₁² - z)

    # Following equation (8d): E ← E + (B - AU)
    # For Rosenbrock: ε ← ε + (x₁² - z) [note the order!]
    constraint_residual = x1 .^ 2 .- s.z  # Changed sign!
    s.ε .+= constraint_residual
    return nothing
end

# Run sampling with progress bar
println("\nRunning ADMM-SVGD sampling...")
history = sample!(
    sampler,
    args[:n_iterations],
    solve_z_fn,
    compute_grad_fn,
    update_multiplier_fn;
    verbose=true,
    save_every=10
)

println("\nFinal particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])

# Generate true samples for comparison
println("\nGenerating true Rosenbrock samples for comparison...")
true_samples = rand(RB_dist, args[:n_particles])

# Compute KL divergence estimate (based on log-pdf)
println("\nComputing log-pdf for final particles...")
final_logpdf = Rosenbrock.logpdf(RB_dist, sampler.particles)
println("  Mean log-pdf: ", round(mean(final_logpdf), digits=4))
println("  Std log-pdf:  ", round(std(final_logpdf), digits=4))

# Save results
println("\nSaving results...")
results = merge(
    args,
    Dict(
        "final_particles" => sampler.particles,
        "true_samples" => true_samples,
        "particle_history" => history["particle_history"],
        "constraint_violations" => history["constraint_violations"],
        "iterations_saved" => history["iterations_saved"],
        "final_z" => history["final_z"],
        "final_epsilon" => history["final_epsilon"],
        "final_logpdf" => final_logpdf
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