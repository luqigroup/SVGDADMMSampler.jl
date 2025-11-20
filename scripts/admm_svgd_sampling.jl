# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# ADMM-SVGD Sampling for Rosenbrock Distribution

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using ProgressMeter
using JLD2

# Random seed
Random.seed!(42)

# Read configuration
args = read_config("admm_svgd_sampling.json")
args = parse_input_args(args)

println("="^60)
println("ADMM-SVGD Sampling for Rosenbrock")
println("="^60)
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

# Create Rosenbrock distribution
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# Initialize sampler
sampler = ADMMSVGDSampler(
    args["n_particles"],
    2;  # Rosenbrock is 2D
    μ=Float32(args["mu_penalty"]),
    η=Float32(args["eta"]),
    h=isnothing(args["h"]) ? nothing : Float32(args["h"])
)

println("\nInitial statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])
println("  Bandwidth: ", round(sampler.h, digits=4))

# Define ADMM functions for Rosenbrock with constraint z = x₁²
# -log p(x) = a*(x₁ - μ)² + (x₂ - x₁²)²
# Constraint: z = x₁²
# Augmented Lagrangian: L = a*(x₁-μ)² + (x₂-z)² - ε(z-x₁²) + (μ/2)(z-x₁²)²

# Solve for z: z = (2*x₂ + ε + μ*x₁²) / (2 + μ)
function solve_z_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]
    s.z .= (2 .* x2 .+ s.ε .+ s.μ .* (x1 .^ 2)) ./ (2 + s.μ)
end

# Compute gradients: ∇log p = -∇L
# ∇L/∂x₁ = 2a(x₁-μ) + 2x₁[ε - μ(z-x₁²)]
# ∇L/∂x₂ = 2(x₂-z)
function compute_grad_fn(s)
    x1 = s.particles[1, :]
    x2 = s.particles[2, :]
    a = RB_dist.a
    μ_rb = RB_dist.μ

    dL_dx1 = 2 .* a .* (x1 .- μ_rb) .+ 2 .* x1 .* (s.ε .- s.μ .* (s.z .- x1 .^ 2))
    dL_dx2 = 2 .* (x2 .- s.z)

    # Return negative gradient (for log-posterior)
    return vcat((-dL_dx1)', (-dL_dx2)')
end

# Update multipliers: ε ← ε + μ(x₁² - z)
function update_multiplier_fn(s)
    x1 = s.particles[1, :]
    s.ε .+= s.μ * ((x1 .^ 2) .- s.z)
end

# Storage for history
history = Dict(
    "particles" => Vector{Matrix{Float32}}(),
    "mean" => Vector{Vector{Float32}}(),
    "std" => Vector{Vector{Float32}}(),
    "bandwidth" => Vector{Float32}(),
    "avg_logpdf" => Vector{Float32}(),
    "constraint_residual" => Vector{Float32}(),
    "iteration" => Vector{Int}()
)

# Run sampling with enhanced progress reporting
println("\nRunning ADMM-SVGD...")
prog = Progress(args["n_iterations"]; dt=1.0, desc="ADMM-SVGD: ")
for iter in 1:args["n_iterations"]
    step!(sampler, solve_z_fn, compute_grad_fn, update_multiplier_fn)

    # Compute log-pdf for progress reporting
    logpdf_vals = Rosenbrock.logpdf(RB_dist, sampler.particles)
    avg_logpdf = mean(logpdf_vals)

    # Compute constraint residual: ||z - x₁²||
    x1 = sampler.particles[1, :]
    constraint_res = mean(abs.(sampler.z .- x1 .^ 2))

    # Update progress with statistics
    next!(prog; showvalues = [
        (:iteration, iter),
        (:bandwidth_h, round(sampler.h, digits=4)),
        (:avg_logpdf, round(avg_logpdf, digits=4)),
        (:constraint_res, round(constraint_res, digits=6)),
        (:mean_x1, round(mean(sampler.particles[1, :]), digits=4)),
        (:mean_x2, round(mean(sampler.particles[2, :]), digits=4))
    ])

    # Save history
    if iter % args["save_every"] == 0
        push!(history["particles"], copy(sampler.particles))
        push!(history["mean"], mean(sampler.particles, dims=2)[:])
        push!(history["std"], std(sampler.particles, dims=2)[:])
        push!(history["bandwidth"], sampler.h)
        push!(history["avg_logpdf"], avg_logpdf)
        push!(history["constraint_residual"], constraint_res)
        push!(history["iteration"], iter)
    end
end

println("\nFinal statistics:")
println("  Mean: ", mean(sampler.particles, dims=2)[:])
println("  Std:  ", std(sampler.particles, dims=2)[:])
println("  Bandwidth (h): ", round(sampler.h, digits=4))

# Save results using DrWatson conventions
results = @dict(
    particles=sampler.particles,
    z=sampler.z,
    eps=sampler.ε,
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

upload_to_dropbox(args["sim_name"])