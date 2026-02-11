# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# ADMM-SVGD Conditional Sampling for Rosenbrock Distribution

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using ProgressMeter
using JLD2
using LinearAlgebra

# Random seed
Random.seed!(11)

# Read configuration
args = read_config("admm_svgd_conditional_sampling.json")
args = parse_input_args(args)

println("="^60)
println("ADMM-SVGD Conditional Sampling for Rosenbrock")
println("="^60)
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

# Create Rosenbrock distribution
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# Generate test data
test_num = args["test_num"]
n_particles = args["n_particles"]

# Generate true samples from prior (Rosenbrock)
X_test_2d = rand(RB_dist, n_particles)  # 2×n_particles
X_test = reshape(X_test_2d, 1, 1, 2, n_particles)  # 1×1×2×n_particles

# Generate noisy observations: Y = X + noise
Y_test = X_test + args["sigma"] * randn(Float32, 1, 1, 2, n_particles)

# Select fixed test instances for conditional sampling
X_fixed_2d = rand(RB_dist, test_num)
X_fixed = reshape(X_fixed_2d, 1, 1, 2, test_num)
Y_fixed = X_fixed + args["sigma"] * randn(Float32, 1, 1, 2, test_num)

pair_wise_dist = 1.0f-1
while pair_wise_dist < 4.0f0
    global X_fixed_2d = rand(RB_dist, test_num)  # 2×test_num
    global X_fixed = reshape(X_fixed_2d, 1, 1, 2, test_num)
    global Y_fixed = X_fixed + args["sigma"] * randn(Float32, 1, 1, 2, test_num)
    global pair_wise_dist = 1.0f0
    for j = 1:test_num
        for k = 1:test_num
            if j != k
                global pair_wise_dist *= norm(Y_fixed[1, 1, :, j] - Y_fixed[1, 1, :, k], 1)
            end
        end
    end
    global pair_wise_dist = pair_wise_dist^(1.0f0 / test_num / (test_num - 1))
end


println("\nTest data statistics:")
println("  True samples X:")
println("    Mean x₁: ", mean(X_test[1, 1, 1, :]))
println("    Mean x₂: ", mean(X_test[1, 1, 2, :]))
println("    Std x₁:  ", std(X_test[1, 1, 1, :]))
println("    Std x₂:  ", std(X_test[1, 1, 2, :]))
println("  Noisy observations Y:")
println("    Mean y₁: ", mean(Y_test[1, 1, 1, :]))
println("    Mean y₂: ", mean(Y_test[1, 1, 2, :]))
println("    Std y₁:  ", std(Y_test[1, 1, 1, :]))
println("    Std y₂:  ", std(Y_test[1, 1, 2, :]))

# Storage for posterior samples and convergence history
X_post = zeros(Float32, 1, 1, 2, n_particles, test_num)
n_iters = args["n_iterations"]
hist_constraint_res = zeros(Float32, n_iters, test_num)
hist_logpdf = zeros(Float32, n_iters, test_num)
hist_bandwidth = zeros(Float32, n_iters, test_num)
hist_mean = zeros(Float32, n_iters, 2, test_num)
hist_std = zeros(Float32, n_iters, 2, test_num)

# Run ADMM-SVGD for each fixed observation
println("\nRunning ADMM-SVGD conditional sampling...")
for j = 1:test_num
    println("\n" * "="^60)
    println("Processing observation $j / $test_num")
    println("  Observed y: [$(Y_fixed[1,1,1,j]), $(Y_fixed[1,1,2,j])]")
    println("="^60)

    # Current observation - extract scalar values
    y_obs = Y_fixed[:, :, :, j]
    y1_obs = Float32(Y_fixed[1, 1, 1, j])
    y2_obs = Float32(Y_fixed[1, 1, 2, j])

    # Extract parameters that will be used in inner functions
    σ² = Float32(args["sigma"]^2)
    a_rb = Float32(RB_dist.a)
    μ_rb = Float32(RB_dist.μ)

    # Initialize ADMM-SVGD sampler for this observation
    sampler = ADMMSVGDSampler(
        n_particles,
        2;  # Rosenbrock is 2D
        μ=Float32(args["mu_penalty"]),
        η=Float32(args["eta"]),
        h=isnothing(args["h"]) ? nothing : Float32(args["h"])
    )

    # Define conditional ADMM functions
    # Posterior: p(x|y) ∝ p(y|x) * p(x)
    # -log p(x|y) = -log p(y|x) - log p(x) + const
    #             = (1/2σ²)||y-x||² + a*(x₁-μ)² + (x₂-x₁²)²
    # With constraint: z = x₁²
    # Augmented Lagrangian: L = (1/2σ²)||y-x||² + a*(x₁-μ)² + (x₂-z)² - ε(z-x₁²) + (μ/2)(z-x₁²)²

    # Solve for z: z = (2*x₂ + σ²*ε + σ²*μ*x₁²) / (2 + σ²*μ)
    function solve_z_fn(s)
        x1 = s.particles[1, :]
        x2 = s.particles[2, :]
        numerator = 2 .* x2 .+ σ² .* s.ε .+ σ² .* s.μ .* (x1 .^ 2)
        denominator = 2 + σ² * s.μ
        s.z .= numerator ./ denominator
    end

    # Compute gradients: ∇log p(x|y) = -∇L
    # ∇L/∂x₁ = (1/σ²)(x₁-y₁) + 2a(x₁-μ) + 2x₁[ε - μ(z-x₁²)]
    # ∇L/∂x₂ = (1/σ²)(x₂-y₂) + 2(x₂-z)
    function compute_grad_fn(s)
        x1 = s.particles[1, :]
        x2 = s.particles[2, :]

        # Likelihood gradient
        dL_likelihood_dx1 = (x1 .- y1_obs) ./ σ²
        dL_likelihood_dx2 = (x2 .- y2_obs) ./ σ²

        # Prior gradient (Rosenbrock part)
        dL_prior_dx1 = 2 .* a_rb .* (x1 .- μ_rb) .+ 2 .* x1 .* (s.ε .- s.μ .* (s.z .- x1 .^ 2))
        dL_prior_dx2 = 2 .* (x2 .- s.z)

        # Total gradient
        dL_dx1 = dL_likelihood_dx1 .+ dL_prior_dx1
        dL_dx2 = dL_likelihood_dx2 .+ dL_prior_dx2

        # Return negative gradient (for log-posterior)
        return vcat((-dL_dx1)', (-dL_dx2)')
    end

    # Update multipliers: ε ← ε + μ(x₁² - z)
    function update_multiplier_fn(s)
        x1 = s.particles[1, :]
        s.ε .+= s.μ * ((x1 .^ 2) .- s.z)
    end

    # Run sampling with progress bar
    prog = Progress(args["n_iterations"]; dt=1.0, desc="  ADMM-SVGD [$j/$test_num]: ")
    for iter in 1:args["n_iterations"]
        step!(sampler, solve_z_fn, compute_grad_fn, update_multiplier_fn)

        # Compute log-pdf for progress reporting
        x_2d = sampler.particles  # Already 2 × n_particles
        logpdf_prior = Rosenbrock.logpdf(RB_dist, x_2d)

        # Log-likelihood: p(y|x) = N(y; x, σ²I)
        diff_x = sampler.particles .- reshape([y1_obs, y2_obs], 2, 1)
        logpdf_likelihood = -0.5f0 .* sum(diff_x .^ 2, dims=1)' ./ σ²
        logpdf_posterior = logpdf_prior .+ logpdf_likelihood
        avg_logpdf = mean(logpdf_posterior)

        # Compute constraint residual
        x1 = sampler.particles[1, :]
        constraint_res = mean(abs.(sampler.z .- x1 .^ 2))

        # Save convergence history
        hist_constraint_res[iter, j] = constraint_res
        hist_logpdf[iter, j] = avg_logpdf
        hist_bandwidth[iter, j] = sampler.h
        hist_mean[iter, 1, j] = mean(sampler.particles[1, :])
        hist_mean[iter, 2, j] = mean(sampler.particles[2, :])
        hist_std[iter, 1, j] = std(sampler.particles[1, :])
        hist_std[iter, 2, j] = std(sampler.particles[2, :])

        # Update progress
        next!(prog; showvalues = [
            (:iteration, iter),
            (:bandwidth_h, round(sampler.h, digits=4)),
            (:avg_logpdf_post, round(avg_logpdf, digits=4)),
            (:constraint_res, round(constraint_res, digits=6)),
            (:mean_x1, round(mean(sampler.particles[1, :]), digits=4)),
            (:mean_x2, round(mean(sampler.particles[2, :]), digits=4))
        ])

        # # Update bandwidth periodically
        # if !isnothing(args["update_bandwidth_every"]) &&
        #    iter % args["update_bandwidth_every"] == 0 &&
        #    iter < args["n_iterations"]
        #     update_bandwidth!(sampler)
        # end
    end

    # Store final particles as posterior samples
    X_post[:, :, :, :, j] = reshape(sampler.particles, 1, 1, 2, n_particles)

    println("\nFinal statistics for observation $j:")
    println("  Posterior mean: [$(mean(sampler.particles[1,:])), $(mean(sampler.particles[2,:]))]")
    println("  Posterior std:  [$(std(sampler.particles[1,:])), $(std(sampler.particles[2,:]))]")
    println("  True value:     [$(X_fixed[1,1,1,j]), $(X_fixed[1,1,2,j])]")
end

# Save results
println("\n" * "="^60)
println("Saving results...")
!isdir(datadir(args["sim_name"])) && mkpath(datadir(args["sim_name"]))

filename = savename(args, "jld2")
filepath = datadir(args["sim_name"], filename)

save_dict = Dict{String,Any}()
for (key, val) in args
    save_dict[key] = val
end

save_dict = merge(
    save_dict,
    Dict(
        "X_test" => X_test,
        "Y_test" => Y_test,
        "X_fixed" => X_fixed,
        "Y_fixed" => Y_fixed,
        "X_post" => X_post,
        "hist_constraint_res" => hist_constraint_res,
        "hist_logpdf" => hist_logpdf,
        "hist_bandwidth" => hist_bandwidth,
        "hist_mean" => hist_mean,
        "hist_std" => hist_std,
    ),
)

@tagsave(filepath, save_dict; safe = true)
println("Results saved to: $filepath")
println("="^60)

upload_to_dropbox(args["sim_name"])