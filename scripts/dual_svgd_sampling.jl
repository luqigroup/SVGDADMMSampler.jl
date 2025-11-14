# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# Dual-Space SVGD Sampling for Rosenbrock Distribution
# SVGD on dual variables (ε), deterministic updates on primal variables (x)

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
args = read_config("dual_svgd_sampling.json")
args = parse_input_args(args)

println("="^60)
println("Dual-Space SVGD Sampling for Rosenbrock Distribution")
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
sampler = DualSpaceSVGDSampler(
    args["n_particles"],
    n_dim;
    μ=Float32(args["mu_penalty"]),
    α=Float32(args["alpha"]),
    η_ε=Float32(args["eta_epsilon"]),
    h_ε=isnothing(args["h_epsilon"]) ? nothing : Float32(args["h_epsilon"])
)

println("\nInitial particle statistics:")
println("  Mean x: ", mean(sampler.particles, dims=2)[:])
println("  Std x:  ", std(sampler.particles, dims=2)[:])
println("  Mean ε: ", mean(sampler.ε))
println("  Std ε:  ", std(sampler.ε))
println("  Bandwidth (h_ε): ", round(sampler.h_ε, digits=4))

# =============================================================================
# Define the three functions needed for Dual-Space SVGD
# =============================================================================

# 1. Solve for z given (x, ε) - SAME AS BEFORE
function solve_z_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]

    b = 1.0f0
    numerator = 2 * b .* x2 .+ s.ε .+ s.μ .* (x1 .^ 2)
    denominator = 2 * b + s.μ

    s.z .= numerator ./ denominator
    return nothing
end

# 2. Compute gradients w.r.t. x (for DETERMINISTIC update)
# This is the SAME gradient as before, but now used for gradient descent
function compute_grad_x_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]
    z = s.z
    ε = s.ε
    μ_penalty = s.μ

    # Get parameters from RB_dist
    μ_rb = RB_dist.μ
    a = RB_dist.a

    # Gradient of augmented Lagrangian
    dL_dx1 = 2 .* a .* (x1 .- μ_rb) .+ 2 .* x1 .* (ε .- μ_penalty .* (z .- x1 .^ 2))
    dL_dx2 = 2 .* (x2 .- z)

    # Gradient of log-posterior = -gradient of L
    grad_x1 = -dL_dx1
    grad_x2 = -dL_dx2

    return vcat(grad_x1', grad_x2')  # Return as 2 × n_particles matrix
end

# 3. Compute gradients w.r.t. ε (for SVGD update)
# This is NEW - gradient of augmented Lagrangian w.r.t. ε
function compute_grad_epsilon_fn(s)
    x1 = s.particles[1, :]
    z = s.z

    # ∂L/∂ε = -(z - x₁²)
    # ∇log p(ε) = -∂L/∂ε = (z - x₁²)
    grad_ε = z .- x1 .^ 2

    return grad_ε  # Return as vector (n_particles,)
end

# =============================================================================
# Run sampling with progress bar
# =============================================================================

history = sample_dual!(
    sampler,
    args["n_iterations"],
    solve_z_fn,
    compute_grad_x_fn,
    compute_grad_epsilon_fn;
    verbose=true,
    save_every=args["save_every"],
    update_bandwidth_every=isnothing(args["update_bandwidth_every"]) ? nothing : args["update_bandwidth_every"]
)

println("\nFinal particle statistics:")
println("  Mean x: ", mean(sampler.particles, dims=2)[:])
println("  Std x:  ", std(sampler.particles, dims=2)[:])
println("  Mean ε: ", mean(sampler.ε))
println("  Std ε:  ", std(sampler.ε))
println("  Bandwidth (h_ε): ", round(sampler.h_ε, digits=4))

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
    particles = sampler.particles,
    true_samples = true_samples,
    history = history,
    final_logpdf = final_logpdf,
    true_logpdf = true_logpdf,
    args = args
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
println("  julia scripts/dual_svgd_visualization.jl")

upload_to_dropbox(args["sim_name"])