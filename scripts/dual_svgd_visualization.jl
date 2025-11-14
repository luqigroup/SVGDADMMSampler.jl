# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025
# Visualization for Dual-Space SVGD sampling on Rosenbrock distribution

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
args = read_config("dual_svgd_visualization.json")
args = parse_input_args(args)

println("="^60)
println("Dual-Space SVGD Visualization")
println("="^60)

# Load results using DrWatson conventions
filename = savename(args, "jld2")
filepath = datadir(args["sim_name"], filename)

if !isfile(filepath)
    error("Results file not found: $filepath\nPlease run dual_svgd_sampling.jl first.")
end

println("Loading results from: ", filepath)
results = wload(filepath)

particles = results["particles"]
history = results["history"]
true_samples = results["true_samples"]
saved_args = results["args"]

println("\nLoaded parameters:")
for (key, val) in saved_args
    println("  $key: $val")
end
println("="^60)

# Create Rosenbrock distribution for reference
RB_dist = RosenbrockDistribution(saved_args["mu_rb"], saved_args["a"])

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
# Dual-SVGD particles (foreground)
scatter(particles[1, :], particles[2, :],
        alpha=0.6, s=10, c="red", label="Dual-SVGD Particles")
xlabel(L"x_1")
ylabel(L"x_2")
title("Final Distribution (Iteration $(saved_args["n_iterations"]))")
legend(loc="upper right")
grid(true, alpha=0.3)

# Plot 2: 2D histogram comparison
subplot(1, 2, 2)
plt.hist2d(particles[1, :], particles[2, :], bins=50, cmap="Reds",
       density=true, alpha=0.7)
colorbar(label="Dual-SVGD Density")
xlabel(L"x_1")
ylabel(L"x_2")
title("Dual-SVGD Sample Density")
grid(true, alpha=0.3)

tight_layout()
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
     label="Dual-SVGD", color="red")
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
     label="Dual-SVGD", color="red")
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
# Figure 3: Convergence diagnostics (including ε)
# ============================================================================
fig3 = figure(figsize=(14, 12))

# Constraint violation
subplot(3, 2, 1)
plot(history["iterations_saved"], history["constraint_violations"],
     linewidth=2, color="#4a4a4a")
xlabel("Iteration")
ylabel("Mean Constraint Violation")
title(L"Constraint Violation: $|z - x_1^2|$")
grid(true, alpha=0.3)
yscale("log")

# Bandwidth evolution for ε
subplot(3, 2, 2)
plot(history["iterations_saved"], history["bandwidth_history"],
     linewidth=2, color="purple")
xlabel("Iteration")
ylabel(L"Bandwidth ($h_\varepsilon$)")
title(L"SVGD Bandwidth Evolution for $\varepsilon$")
grid(true, alpha=0.3)

# Mean convergence for x
subplot(3, 2, 3)
particle_history = history["particle_history"]
mean_x1 = [mean(p[1, :]) for p in particle_history]
mean_x2 = [mean(p[2, :]) for p in particle_history]
plot(history["iterations_saved"], mean_x1, label=L"x_1", linewidth=2)
plot(history["iterations_saved"], mean_x2, label=L"x_2", linewidth=2)
axhline(y=mean(true_samples[1, :]), color="blue", linestyle="--",
        alpha=0.5, label=L"True mean $x_1$")
axhline(y=mean(true_samples[2, :]), color="orange", linestyle="--",
        alpha=0.5, label=L"True mean $x_2$")
xlabel("Iteration")
ylabel("Mean")
title("Convergence of Mean (Primal x)")
legend()
grid(true, alpha=0.3)

# Standard deviation convergence for x
subplot(3, 2, 4)
std_x1 = [std(p[1, :]) for p in particle_history]
std_x2 = [std(p[2, :]) for p in particle_history]
plot(history["iterations_saved"], std_x1, label=L"x_1", linewidth=2)
plot(history["iterations_saved"], std_x2, label=L"x_2", linewidth=2)
axhline(y=std(true_samples[1, :]), color="blue", linestyle="--",
        alpha=0.5, label=L"True std $x_1$")
axhline(y=std(true_samples[2, :]), color="orange", linestyle="--",
        alpha=0.5, label=L"True std $x_2$")
xlabel("Iteration")
ylabel("Standard Deviation")
title("Convergence of Std (Primal x)")
legend()
grid(true, alpha=0.3)

# NEW: Mean and Std of ε
subplot(3, 2, 5)
epsilon_history = history["epsilon_history"]
mean_ε = [mean(e) for e in epsilon_history]
plot(history["iterations_saved"], mean_ε, linewidth=2, color="green")
xlabel("Iteration")
ylabel(L"Mean($\varepsilon$)")
title(L"Mean of Dual Variable $\varepsilon$")
grid(true, alpha=0.3)

subplot(3, 2, 6)
std_ε = history["epsilon_std_history"]
plot(history["iterations_saved"], std_ε, linewidth=2, color="green")
xlabel("Iteration")
ylabel(L"Std($\varepsilon$)")
title(L"Std of Dual Variable $\varepsilon$ (Diversity)")
grid(true, alpha=0.3)

tight_layout()
wsave(joinpath(save_path, "convergence_diagnostics.png"), fig3)
close(fig3)
println("  ✓ Saved convergence_diagnostics.png")

# ============================================================================
# Figure 4: Evolution animation frames (selected iterations)
# ============================================================================
n_frames = min(9, length(history["iterations_saved"]))
frame_indices = round.(Int, range(1, length(history["iterations_saved"]), length=n_frames))

fig4 = figure(figsize=(15, 10))

for (plot_idx, hist_idx) in enumerate(frame_indices)
    iter = history["iterations_saved"][hist_idx]
    particles_at_iter = particle_history[hist_idx]

    subplot(3, 3, plot_idx)

    # Background: true distribution
    scatter(true_samples[1, :], true_samples[2, :],
            alpha=0.1, s=0.5, c="gray")

    # Foreground: Dual-SVGD particles at this iteration
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

suptitle("Dual-SVGD Evolution Over Iterations", fontsize=14)
tight_layout()
wsave(joinpath(save_path, "evolution_frames.png"), fig4)
close(fig4)
println("  ✓ Saved evolution_frames.png")

# ============================================================================
# Figure 5: Dual variable ε distribution and evolution
# ============================================================================
fig5 = figure(figsize=(14, 6))

# Histogram of final ε values
subplot(1, 2, 1)
final_epsilon = epsilon_history[end]
hist(final_epsilon, bins=50, color="green", alpha=0.7, edgecolor="black")
xlabel(L"$\varepsilon$")
ylabel("Count")
title(L"Distribution of Final Dual Variable $\varepsilon$")
grid(true, alpha=0.3)

# Evolution of ε values over time (sample of particles)
subplot(1, 2, 2)
n_particles_to_plot = min(50, length(final_epsilon))
for i in 1:n_particles_to_plot
    epsilon_trajectory = [e[i] for e in epsilon_history]
    plot(history["iterations_saved"], epsilon_trajectory,
         alpha=0.3, linewidth=0.5, color="green")
end
xlabel("Iteration")
ylabel(L"$\varepsilon$")
title(L"Evolution of Dual Variables $\varepsilon$ (50 particles)")
grid(true, alpha=0.3)

tight_layout()
wsave(joinpath(save_path, "epsilon_analysis.png"), fig5)
close(fig5)
println("  ✓ Saved epsilon_analysis.png")

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

println("\nDual-SVGD Final Distribution:")
println("  Mean x₁: ", @sprintf("%.4f", mean(particles[1, :])))
println("  Mean x₂: ", @sprintf("%.4f", mean(particles[2, :])))
println("  Std  x₁: ", @sprintf("%.4f", std(particles[1, :])))
println("  Std  x₂: ", @sprintf("%.4f", std(particles[2, :])))

println("\nDual Variables ε:")
println("  Mean ε: ", @sprintf("%.4f", mean(final_epsilon)))
println("  Std  ε: ", @sprintf("%.4f", std(final_epsilon)))
println("  Min  ε: ", @sprintf("%.4f", minimum(final_epsilon)))
println("  Max  ε: ", @sprintf("%.4f", maximum(final_epsilon)))

println("\nAbsolute Errors:")
println("  Mean x₁: ", @sprintf("%.4f", abs(mean(particles[1, :]) - mean(true_samples[1, :]))))
println("  Mean x₂: ", @sprintf("%.4f", abs(mean(particles[2, :]) - mean(true_samples[2, :]))))
println("  Std  x₁: ", @sprintf("%.4f", abs(std(particles[1, :]) - std(true_samples[1, :]))))
println("  Std  x₂: ", @sprintf("%.4f", abs(std(particles[2, :]) - std(true_samples[2, :]))))

println("\nFinal Constraint Violation:")
println("  Mean |z - x₁²|: ", @sprintf("%.6f", history["constraint_violations"][end]))

println("\nLog-PDF Statistics:")
if haskey(results, "final_logpdf") && haskey(results, "true_logpdf")
    println("  Dual-SVGD mean: ", @sprintf("%.4f", mean(results["final_logpdf"])))
    println("  True samples mean: ", @sprintf("%.4f", mean(results["true_logpdf"])))
    println("  Difference: ", @sprintf("%.4f", mean(results["true_logpdf"]) - mean(results["final_logpdf"])))
end

println("\n" * "="^60)
println("All visualizations saved to: ", save_path)
println("="^60)

upload_to_dropbox(args["sim_name"])