# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Feb 2026
# Plain SVGD Conditional Sampling for Rosenbrock Distribution (no ADMM)

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using ProgressMeter
using JLD2
using LinearAlgebra

# Random seed (same as ADMM script for particle initialization)
Random.seed!(7)

# Read configuration
args = read_config("svgd_conditional_sampling.json")
args = parse_input_args(args)

println("="^60)
println("Plain SVGD Conditional Sampling for Rosenbrock")
println("="^60)
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

# Create Rosenbrock distribution
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# Load test instances from ADMM results to ensure identical Y_fixed/X_fixed
println("\nLoading test instances from ADMM results...")
admm_args = read_config("admm_svgd_conditional_sampling.json")
admm_args = parse_input_args(admm_args)

admm_keys = load_experiment(
    admm_args,
    ["X_test", "Y_test", "X_fixed", "Y_fixed"],
)

X_test = admm_keys["X_test"]
Y_test = admm_keys["Y_test"]
X_fixed = admm_keys["X_fixed"]
Y_fixed = admm_keys["Y_fixed"]

test_num = args["test_num"]
n_particles = args["n_particles"]

println("  Loaded X_fixed: $(size(X_fixed))")
println("  Loaded Y_fixed: $(size(Y_fixed))")

# Storage for posterior samples and convergence history
X_post = zeros(Float32, 1, 1, 2, n_particles, test_num)
n_iters = args["n_iterations"]
hist_logpdf = zeros(Float32, n_iters, test_num)
hist_bandwidth = zeros(Float32, n_iters, test_num)
hist_mean = zeros(Float32, n_iters, 2, test_num)
hist_std = zeros(Float32, n_iters, 2, test_num)

# Extract parameters
σ² = Float32(args["sigma"]^2)
a_rb = Float32(RB_dist.a)
μ_rb = Float32(RB_dist.μ)
η = Float32(args["eta"])

# Run plain SVGD for each fixed observation
println("\nRunning plain SVGD conditional sampling...")
for j = 1:test_num
    println("\n" * "="^60)
    println("Processing observation $j / $test_num")
    println("  Observed y: [$(Y_fixed[1,1,1,j]), $(Y_fixed[1,1,2,j])]")
    println("="^60)

    # Current observation
    y1_obs = Float32(Y_fixed[1, 1, 1, j])
    y2_obs = Float32(Y_fixed[1, 1, 2, j])
    y_obs = Float32[y1_obs, y2_obs]  # 2-element vector

    # Initialize particles from standard normal
    particles = randn(Float32, 2, n_particles)

    # Initial bandwidth
    h = SVGDADMMSampler.compute_bandwidth(particles)

    # Run plain SVGD
    prog = Progress(n_iters; dt=1.0, desc="  SVGD [$j/$test_num]: ")
    for iter in 1:n_iters
        # Compute gradient of log-posterior: ∇log p(x|y) = ∇log p(x) + ∇log p(y|x)
        # Prior gradient via Rosenbrock module (2×n_particles matrix)
        grad_prior = Rosenbrock.gradlogpdf(RB_dist, particles)

        # Likelihood gradient: ∇log p(y|x) = -(x - y) / σ²
        grad_likelihood = -(particles .- y_obs) ./ σ²

        # Total log-posterior gradient
        gradients = grad_prior .+ grad_likelihood

        # SVGD update
        SVGDADMMSampler.svgd_update!(particles, gradients, η, h)

        # Update bandwidth
        h = SVGDADMMSampler.compute_bandwidth(particles)

        # Compute log-posterior for tracking
        logpdf_prior = Rosenbrock.logpdf(RB_dist, particles)
        diff_x = particles .- y_obs
        logpdf_likelihood = -0.5f0 .* sum(diff_x .^ 2, dims=1)' ./ σ²
        logpdf_posterior = logpdf_prior .+ logpdf_likelihood
        avg_logpdf = mean(logpdf_posterior)

        # Save convergence history
        hist_logpdf[iter, j] = avg_logpdf
        hist_bandwidth[iter, j] = h
        hist_mean[iter, 1, j] = mean(particles[1, :])
        hist_mean[iter, 2, j] = mean(particles[2, :])
        hist_std[iter, 1, j] = std(particles[1, :])
        hist_std[iter, 2, j] = std(particles[2, :])

        # Update progress
        next!(prog; showvalues = [
            (:iteration, iter),
            (:bandwidth_h, round(h, digits=4)),
            (:avg_logpdf_post, round(avg_logpdf, digits=4)),
            (:mean_x1, round(mean(particles[1, :]), digits=4)),
            (:mean_x2, round(mean(particles[2, :]), digits=4))
        ])
    end

    # Store final particles as posterior samples
    X_post[:, :, :, :, j] = reshape(particles, 1, 1, 2, n_particles)

    println("\nFinal statistics for observation $j:")
    println("  Posterior mean: [$(mean(particles[1,:])), $(mean(particles[2,:]))]")
    println("  Posterior std:  [$(std(particles[1,:])), $(std(particles[2,:]))]")
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
