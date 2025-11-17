# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# ADMM-SVGD Conditional Sampling Visualization for Rosenbrock Distribution

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using PyPlot
using Seaborn
using Printf
using LinearAlgebra
using ProgressMeter

# Set plot configurations
font_prop = set_plot_configs()[1]

# Read configuration
args = read_config("admm_svgd_conditional_visualization.json")
args = parse_input_args(args)

println("="^60)
println("ADMM-SVGD Conditional Visualization")
println("="^60)
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

# Load saved results
println("\nLoading results...")
loaded_keys = load_experiment(
    args,
    ["X_test", "Y_test", "X_fixed", "Y_fixed", "X_post"]
)

X_test = loaded_keys["X_test"]
Y_test = loaded_keys["Y_test"]
X_fixed = loaded_keys["X_fixed"]
Y_fixed = loaded_keys["Y_fixed"]
X_post = loaded_keys["X_post"]

test_num = size(X_fixed, 4)
test_size = size(X_post, 4)

println("Data loaded successfully!")
println("  Test instances: $test_num")
println("  Posterior samples per instance: $test_size")

# Create save directory
save_path = plotsdir(args["sim_name"], savename(args))
!isdir(save_path) && mkpath(save_path)
println("  Saving plots to: $save_path")

# Create Rosenbrock distribution for reference
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# ============================================================================
# Generate MCMC samples for comparison (if requested)
# ============================================================================
println("\nComputing MCMC samples for comparison...")

X_sgld = zeros(Float32, 1, 1, 2, test_size, test_num)
max_itr = 20000

# Objective function: -log p(x|y) = -log p(y|x) - log p(x)
obj(x, y) = begin
    # Reshape x from 1×1×2×n to 2×n for Rosenbrock
    x_2d = reshape(x, 2, size(x, 4))

    # Data term: (1/2σ²)||y-x||²
    data_term = (1.0f0 / (2.0f0 * args["sigma"]^2.0f0)) * sum((x .- y) .^ 2.0f0)

    # Prior term: -log p(x)
    prior_term = -sum(Rosenbrock.logpdf(RB_dist, x_2d))

    return data_term + prior_term
end

prog = Progress(test_num; desc="MCMC sampling: ")
for j = 1:test_num
    f(x) = obj(x, Y_fixed[:, :, :, j:j])

    # Run MCMC sampler
    X_sgld[:, :, :, :, j] = MCMC_sampler(
        max_itr,
        randn(Float32, 1, 1, 2, 1),
        f;
        lr = 5.0f0,
        lr_final = 1.0f-1,
        thinning = 1,
    )[:, :, :, end - test_size + 1:end]
    next!(prog)
end
println("MCMC sampling complete!")

# ============================================================================
# Figure 1: Prior distribution (true samples from Rosenbrock)
# ============================================================================
println("\nGenerating visualizations...")
println("  Creating Figure 1: Prior distribution...")

fig = figure("prior", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
scatter(X_test[1, 1, 1, :], X_test[1, 1, 2, :], s = 0.5, color = "#000000", alpha = 0.35)
scatter(
    X_fixed[1, 1, 1, :],
    X_fixed[1, 1, 2, :],
    s = 50.0,
    color = "#D68D96",
    marker = "^",
    label = "Test instances (true)",
)
grid(false)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.legend(loc = "upper right")
ax.set_title("Prior distribution (Rosenbrock)")
wsave(joinpath(save_path, "prior.png"), fig)
close(fig)
println("  ✓ Saved prior.png")

# ============================================================================
# Figure 2: Data distribution (noisy observations)
# ============================================================================
println("  Creating Figure 2: Data distribution...")

fig = figure("data", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
scatter(Y_test[1, 1, 1, :], Y_test[1, 1, 2, :], s = 0.5, color = "#000000", alpha = 0.35)
scatter(
    Y_fixed[1, 1, 1, :],
    Y_fixed[1, 1, 2, :],
    s = 50.0,
    color = "#819FB3",
    marker = "v",
    label = "Test instances (observed)",
)
grid(false)
ax.legend(loc = "upper right")
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$y_2$")
ax.set_xlabel(L"$y_1$")
ax.set_title("Data distribution (noisy)")
wsave(joinpath(save_path, "data.png"), fig)
close(fig)
println("  ✓ Saved data.png")

# ============================================================================
# Figure 3: ADMM-SVGD posterior samples
# ============================================================================
println("  Creating Figure 3: ADMM-SVGD posterior samples...")

fig = figure("admm-svgd-posterior", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
scatter(
    X_post[1, 1, 1, :, :],
    X_post[1, 1, 2, :, :],
    s = 0.5,
    color = "#D68D96",
    alpha = 0.35,
    label = "ADMM-SVGD posterior",
)
grid(true)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.set_title("ADMM-SVGD Conditional Posterior")
ax.legend()
wsave(joinpath(save_path, "admm-svgd-posterior.png"), fig)
close(fig)
println("  ✓ Saved admm-svgd-posterior.png")

# ============================================================================
# Figure 4: MCMC posterior samples (if computed)
# ============================================================================
println("  Creating Figure 4: MCMC posterior samples...")

fig = figure("mcmc-posterior", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
scatter(
    X_sgld[1, 1, 1, :, :],
    X_sgld[1, 1, 2, :, :],
    s = 0.5,
    color = "#819FB3",
    alpha = 0.35,
    label = "MCMC",
)
grid(true)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.set_title("MCMC Posterior")
ax.legend()
wsave(joinpath(save_path, "mcmc-posterior.png"), fig)
close(fig)
println("  ✓ Saved mcmc-posterior.png")

# ============================================================================
# Figure 5: Q-Q plots for each test instance
# ============================================================================
println("  Creating Q-Q plots...")

rc("font", family = "serif", size = 16)
font_prop = matplotlib.font_manager.FontProperties(
    family = "serif",
    style = "normal",
    size = 18
)
dq = 1.0f-2

for j = 1:test_num
    # Q-Q plot for x₁ component
    mcmc_quantile = quantile(X_sgld[1, 1, 1, :, j], dq:dq:1-dq)
    admm_quantile = quantile(X_post[1, 1, 1, :, j], dq:dq:1-dq)

    fig = figure("qq1-$j", figsize = (5, 5))
    plot(mcmc_quantile, admm_quantile, "o", lw = 2.0, color = "k", alpha = 0.6)
    ref_line = range(mcmc_quantile[1], mcmc_quantile[end], length = 100)
    plot(ref_line, ref_line, lw = 1.0, color = "k", alpha = 1)
    title("Q-Q plot for " * L"$x_1$" * " (instance $j)")
    xlabel("MCMC quantiles")
    ylabel("ADMM-SVGD quantiles")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    wsave(joinpath(save_path, "qq1-$j.png"), fig)
    close(fig)

    # Q-Q plot for x₂ component
    mcmc_quantile = quantile(X_sgld[1, 1, 2, :, j], dq:dq:1-dq)
    admm_quantile = quantile(X_post[1, 1, 2, :, j], dq:dq:1-dq)

    fig = figure("qq2-$j", figsize = (5, 5))
    plot(mcmc_quantile, admm_quantile, "o", lw = 2.0, color = "k", alpha = 0.6)
    ref_line = range(mcmc_quantile[1], mcmc_quantile[end], length = 100)
    plot(ref_line, ref_line, lw = 1.0, color = "k", alpha = 1)
    title("Q-Q plot for " * L"$x_2$" * " (instance $j)")
    xlabel("MCMC quantiles")
    ylabel("ADMM-SVGD quantiles")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    wsave(joinpath(save_path, "qq2-$j.png"), fig)
    close(fig)
end
println("  ✓ Saved Q-Q plots for all instances")

# ============================================================================
# Figure 6: Individual conditional posteriors (one per test instance)
# ============================================================================
println("  Creating individual conditional posterior plots...")

for j = 1:test_num
    fig = figure("conditional-$j", figsize = (6, 6))
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor("white")

    # Background: prior samples
    scatter(X_test[1, 1, 1, :], X_test[1, 1, 2, :],
            s = 0.5, color = "#000000", alpha = 0.35, label = "Prior")

    # Posterior samples for this instance
    scatter(X_post[1, 1, 1, :, j], X_post[1, 1, 2, :, j],
            s = 2.0, color = "#D68D96", alpha = 0.3, label = "Posterior")

    # Observed data point
    scatter([Y_fixed[1, 1, 1, j]], [Y_fixed[1, 1, 2, j]],
            s = 100.0, color = "#819FB3", marker = "v",
            label = "Observation", edgecolors = "black", linewidths = 1.5)

    # True underlying value
    scatter([X_fixed[1, 1, 1, j]], [X_fixed[1, 1, 2, j]],
            s = 100.0, color = "#D68D96", marker = "^",
            label = "True value", edgecolors = "black", linewidths = 1.5)

    grid(true, alpha = 0.3)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-2.5, 7])
    ax.set_ylabel(L"$x_2$")
    ax.set_xlabel(L"$x_1$")
    ax.set_title(@sprintf("Posterior Samples (Instance %d)", j))
    ax.legend(loc = "upper right")
    wsave(joinpath(save_path, "conditional-$j.png"), fig)
    close(fig)
end
println("  ✓ Saved individual conditional plots")

# ============================================================================
# Figure 7: Summary statistics comparison
# ============================================================================
println("  Creating summary statistics plot...")

fig = figure("summary", figsize = (12, 5))

# Subplot 1: Mean comparison
subplot(1, 2, 1)
for j = 1:test_num
    # ADMM-SVGD means
    mean_x1_admm = mean(X_post[1, 1, 1, :, j])
    mean_x2_admm = mean(X_post[1, 1, 2, :, j])

    # True values
    true_x1 = X_fixed[1, 1, 1, j]
    true_x2 = X_fixed[1, 1, 2, j]

    scatter([true_x1], [mean_x1_admm], color = "#D68D96", alpha = 0.6, s = 50)
end
plot([-3, 3], [-3, 3], "k--", lw = 1.0, alpha = 0.5)
xlabel(L"True $x_1$")
ylabel(L"Posterior Mean $x_1$")
title("Mean Accuracy (x₁)")
grid(true, alpha = 0.3)
xlim([-3, 3])
ylim([-3, 3])

# Subplot 2: Standard deviation
subplot(1, 2, 2)
std_vals = [std(X_post[1, 1, :, :, j]) for j = 1:test_num]
bar(1:test_num, std_vals, color = "#819FB3", alpha = 0.7)
xlabel("Test Instance")
ylabel("Posterior Std")
title("Posterior Uncertainty")
grid(true, alpha = 0.3, axis = "y")

tight_layout()
wsave(joinpath(save_path, "summary-statistics.png"), fig)
close(fig)
println("  ✓ Saved summary-statistics.png")

# ============================================================================
# Print summary statistics
# ============================================================================
println("\n" * "="^60)
println("Summary Statistics")
println("="^60)

println("\nTrue values (X_fixed):")
for j = 1:test_num
    println(@sprintf("  Instance %d: [%.3f, %.3f]",
                    j, X_fixed[1,1,1,j], X_fixed[1,1,2,j]))
end

println("\nObserved values (Y_fixed):")
for j = 1:test_num
    println(@sprintf("  Instance %d: [%.3f, %.3f]",
                    j, Y_fixed[1,1,1,j], Y_fixed[1,1,2,j]))
end

println("\nPosterior means (ADMM-SVGD):")
for j = 1:test_num
    mean_x1 = mean(X_post[1, 1, 1, :, j])
    mean_x2 = mean(X_post[1, 1, 2, :, j])
    println(@sprintf("  Instance %d: [%.3f, %.3f]", j, mean_x1, mean_x2))
end

println("\nPosterior standard deviations:")
for j = 1:test_num
    std_x1 = std(X_post[1, 1, 1, :, j])
    std_x2 = std(X_post[1, 1, 2, :, j])
    println(@sprintf("  Instance %d: [%.3f, %.3f]", j, std_x1, std_x2))
end

# Compute error metrics
println("\nError metrics (||posterior_mean - true||):")
for j = 1:test_num
    mean_x1 = mean(X_post[1, 1, 1, :, j])
    mean_x2 = mean(X_post[1, 1, 2, :, j])
    true_x1 = X_fixed[1, 1, 1, j]
    true_x2 = X_fixed[1, 1, 2, j]
    error = sqrt((mean_x1 - true_x1)^2 + (mean_x2 - true_x2)^2)
    println(@sprintf("  Instance %d: %.4f", j, error))
end

println("\n" * "="^60)
println("Visualization complete!")
println("Plots saved to: ", save_path)
println("="^60)

upload_to_dropbox(args["sim_name"])