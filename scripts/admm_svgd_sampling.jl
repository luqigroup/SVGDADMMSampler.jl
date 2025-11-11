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
    μ = 1.0f0,           # Penalty parameter
    η = 0.001f0,          # SVGD step size
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

println("\nInitial particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])

# Run sampling with progress bar
println("\nRunning ADMM-SVGD sampling...")
history = sample!(
    sampler,
    args[:n_iterations];
    a=args[:a],
    b=args[:b],
    verbose=true
)

println("\nFinal particle statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])

# Generate true samples for comparison
println("\nGenerating true Rosenbrock samples for comparison...")
RB_dist = RosenbrockDistribution(0.0f0, args[:a])
true_samples = rand(RB_dist, args[:n_particles])

# Compute KL divergence estimate (based on log-pdf)
println("\nComputing log-pdf for final particles...")
final_logpdf = logpdf(RB_dist, sampler.particles)
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