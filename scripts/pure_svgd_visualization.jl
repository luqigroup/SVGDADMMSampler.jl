# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# Visualization for Pure SVGD sampling on Rosenbrock distribution

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using JLD2
using PyPlot
using Statistics
using Printf

# Set up plotting
font_prop = set_plot_configs()[1]

# Read configuration from JSON file
args = read_config("pure_svgd_visualization.json")
args = parse_input_args(args)

println("="^60)
println("Pure SVGD Visualization")
println("="^60)

# Load results using DrWatson conventions
filename = savename(args, "jld2")
filepath = datadir(args["sim_name"], filename)

if !isfile(filepath)
    error("Results file not found: $filepath\nPlease run pure_svgd_sampling.jl first.")
end

println("Loading results from: ", filepath)
results = wload(filepath)

particles = results["particles"]
history = results["history"]
saved_args = results["args"]

println("\nLoaded parameters:")
for (key, val) in saved_args
    println("  $key: $val")
end
println("="^60)

# Create Rosenbrock distribution for reference
RB_dist = RosenbrockDistribution(saved_args["mu_rb"], saved_args["a"])

# Generate ground truth samples for comparison
n_true_samples = 10000
true_samples = rand(RB_dist, n_true_samples)

println("\nGenerating visualizations...")

# Create output directory using DrWatson convention
save_path = plotsdir(args["sim_name"], savename(args))
!isdir(save_path) && mkpath(save_path)

# ============================================================================
# Figure 1: Final particle distribution vs true distribution
# ============================================================================
fig1 = figure(figsize=(14, 6))

# Plot 1: Scatter plot comparison
subplot(1, 2, 1)
# True samples (background)
scatter(true_samples[1, :], true_samples[2, :],
        alpha=0.2, s=1, c="gray", label="True Rosenbrock")
# SVGD particles (foreground)
scatter(particles[1, :], particles[2, :],
        alpha=0.6, s=10, c="red", label="SVGD Particles")
xlabel(L"x_1")
ylabel(L"x_2")
title("Final Distribution (Iteration $(saved_args["n_iterations"]))")
legend(loc="upper right")
grid(true, alpha=0.3)

# Plot 2: 2D histogram comparison
subplot(1, 2, 2)
# Create 2D histogram for SVGD particles
plt.hist2d(particles[1, :], particles[2, :], bins=50, cmap="Reds",
       density=true, alpha=0.7)
colorbar(label="SVGD Density")
xlabel(L"x_1")
ylabel(L"x_2")
title("SVGD Sample Density")
grid(true, alpha=0.3)

tight_layout()

# Save figure
wsave(joinpath(save_path, "final_distribution.png"), fig1)
close(fig1)
println("  ✓ Saved final_distribution.png")

# ============================================================================
# Figure 2: Marginal distributions comparison
# ============================================================================
fig2 = figure(figsize=(14, 6))

# x₁ marginal
subplot(1, 2, 1)
hist(true_samples[1, :], bins=50, density=true, alpha=0.5,
     label="True", color="gray")
hist(particles[1, :], bins=50, density=true, alpha=0.7,
     label="SVGD", color="red")
xlabel(L"x_1")
ylabel("Density")
title(L"Marginal Distribution: $x_1$")
legend()
grid(true, alpha=0.3)

# x₂ marginal
subplot(1, 2, 2)
hist(true_samples[2, :], bins=50, density=true, alpha=0.5,
     label="True", color="gray")
hist(particles[2, :], bins=50, density=true, alpha=0.7,
     label="SVGD", color="red")
xlabel(L"x_2")
ylabel("Density")
title(L"Marginal Distribution: $x_2$")
legend()
grid(true, alpha=0.3)

tight_layout()
wsave(joinpath(save_path, "marginal_distributions.png"), fig2)
close(fig2)
println("  ✓ Saved marginal_distributions.png")

# ============================================================================
# Figure 3: Convergence statistics over iterations
# ============================================================================
fig3 = figure(figsize=(14, 10))

# Mean convergence
subplot(2, 2, 1)
mean_x1 = [m[1] for m in history["mean"]]
mean_x2 = [m[2] for m in history["mean"]]
plot(history["iteration"], mean_x1, label=L"x_1", linewidth=2)
plot(history["iteration"], mean_x2, label=L"x_2", linewidth=2)
axhline(y=mean(true_samples[1, :]), color="blue", linestyle="--",
        alpha=0.5, label=L"True mean $x_1$")
axhline(y=mean(true_samples[2, :]), color="orange", linestyle="--",
        alpha=0.5, label=L"True mean $x_2$")
xlabel("Iteration")
ylabel("Mean")
title("Convergence of Mean")
legend()
grid(true, alpha=0.3)

# Standard deviation convergence
subplot(2, 2, 2)
std_x1 = [s[1] for s in history["std"]]
std_x2 = [s[2] for s in history["std"]]
plot(history["iteration"], std_x1, label=L"x_1", linewidth=2)
plot(history["iteration"], std_x2, label=L"x_2", linewidth=2)
axhline(y=std(true_samples[1, :]), color="blue", linestyle="--",
        alpha=0.5, label=L"True std $x_1$")
axhline(y=std(true_samples[2, :]), color="orange", linestyle="--",
        alpha=0.5, label=L"True std $x_2$")
xlabel("Iteration")
ylabel("Standard Deviation")
title("Convergence of Standard Deviation")
legend()
grid(true, alpha=0.3)

# Bandwidth evolution
subplot(2, 2, 3)
plot(history["iteration"], history["bandwidth"], linewidth=2, color="purple")
xlabel("Iteration")
ylabel("Bandwidth (h)")
title("SVGD Bandwidth Evolution")
grid(true, alpha=0.3)

# Log-likelihood evolution (using average over particles)
subplot(2, 2, 4)
avg_logpdf = Float32[]
for (idx, iter) in enumerate(history["iteration"])
    particles_at_iter = history["particles"][idx]
    # Use Rosenbrock.logpdf to avoid ambiguity
    lp = Rosenbrock.logpdf(RB_dist, particles_at_iter)
    push!(avg_logpdf, mean(lp))
end
plot(history["iteration"], avg_logpdf, linewidth=2, color="green")
xlabel("Iteration")
ylabel("Average Log-Density")
title("Average Log-Density Over Particles")
grid(true, alpha=0.3)

tight_layout()
wsave(joinpath(save_path, "convergence_statistics.png"), fig3)
close(fig3)
println("  ✓ Saved convergence_statistics.png")

# ============================================================================
# Figure 4: Evolution animation frames (selected iterations)
# ============================================================================
# Select 9 evenly spaced iterations to show evolution
n_frames = min(9, length(history["iteration"]))
frame_indices = round.(Int, range(1, length(history["iteration"]), length=n_frames))

fig4 = figure(figsize=(15, 10))

for (plot_idx, hist_idx) in enumerate(frame_indices)
    iter = history["iteration"][hist_idx]
    particles_at_iter = history["particles"][hist_idx]

    subplot(3, 3, plot_idx)

    # Background: true distribution
    scatter(true_samples[1, :], true_samples[2, :],
            alpha=0.1, s=0.5, c="gray")

    # Foreground: SVGD particles at this iteration
    scatter(particles_at_iter[1, :], particles_at_iter[2, :],
            alpha=0.6, s=4, c="red")

    xlabel(L"x_1")
    ylabel(L"x_2")
    title(@sprintf("Iteration %d", iter))
    grid(true, alpha=0.3)

    # Set consistent axis limits
    xlim(minimum(true_samples[1, :]) - 0.5, maximum(true_samples[1, :]) + 0.5)
    ylim(minimum(true_samples[2, :]) - 0.5, maximum(true_samples[2, :]) + 0.5)
end

suptitle("SVGD Evolution Over Iterations", fontsize=14)
tight_layout()
wsave(joinpath(save_path, "evolution_frames.png"), fig4)
close(fig4)
println("  ✓ Saved evolution_frames.png")

# ============================================================================
# Print summary statistics
# ============================================================================
println("\n" * "="^60)
println("Summary Statistics")
println("="^60)

println("\nTrue Distribution:")
println("  Mean x₁: ", @sprintf("%.4f", mean(true_samples[1, :])))
println("  Mean x₂: ", @sprintf("%.4f", mean(true_samples[2, :])))
println("  Std  x₁: ", @sprintf("%.4f", std(true_samples[1, :])))
println("  Std  x₂: ", @sprintf("%.4f", std(true_samples[2, :])))

println("\nSVGD Final Distribution:")
println("  Mean x₁: ", @sprintf("%.4f", mean(particles[1, :])))
println("  Mean x₂: ", @sprintf("%.4f", mean(particles[2, :])))
println("  Std  x₁: ", @sprintf("%.4f", std(particles[1, :])))
println("  Std  x₂: ", @sprintf("%.4f", std(particles[2, :])))

println("\nAbsolute Errors:")
println("  Mean x₁: ", @sprintf("%.4f", abs(mean(particles[1, :]) - mean(true_samples[1, :]))))
println("  Mean x₂: ", @sprintf("%.4f", abs(mean(particles[2, :]) - mean(true_samples[2, :]))))
println("  Std  x₁: ", @sprintf("%.4f", abs(std(particles[1, :]) - std(true_samples[1, :]))))
println("  Std  x₂: ", @sprintf("%.4f", abs(std(particles[2, :]) - std(true_samples[2, :]))))

println("\n" * "="^60)
println("All visualizations saved to: ", save_path)
println("="^60)

upload_to_dropbox(args["sim_name"])