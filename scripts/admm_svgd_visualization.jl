# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

using DrWatson
@quickactivate :SVGDADMMSampler

using Rosenbrock
using JLD2
using PyPlot
using Statistics
using LaTeXStrings

# Load results
args = @dict(
    n_particles = 1000,
    n_iterations = 1000,
    μ = 1.0f0,
    η = 0.01f0,
    a = 1.0f0,
    b = 100.0f0,
    sim_name = "admm_svgd_rosenbrock"
)

println("="^60)
println("ADMM-SVGD Visualization")
println("="^60)

# Load saved results
load_path = datadir("sims", args[:sim_name], savename(args, "jld2"))
println("Loading results from: $load_path")
results = load(load_path)

final_particles = results["final_particles"]
true_samples = results["true_samples"]
particle_history = results["particle_history"]
constraint_violations = results["constraint_violations"]

println("Loaded $(size(final_particles, 2)) final particles")
println("Loaded $(length(particle_history)) snapshots from history")

# Create output directory
save_dir = plotsdir(args[:sim_name], savename(args))
mkpath(save_dir)

# =====================================================
# Figure 1: Final samples comparison
# =====================================================
println("\nCreating Figure 1: Final samples comparison...")

fig = figure("samples_comparison", figsize=(12, 5))

# True samples
subplot(1, 2, 1)
scatter(
    true_samples[1, :],
    true_samples[2, :],
    s=0.5,
    color="#819FB3",
    alpha=0.5,
    label="True samples"
)
xlim(-3.5, 3.5)
ylim(-2.5, 8)
xlabel(L"x_1")
ylabel(L"x_2")
title("True Rosenbrock Samples")
grid(true, alpha=0.3)
legend()

# ADMM-SVGD samples
subplot(1, 2, 2)
scatter(
    final_particles[1, :],
    final_particles[2, :],
    s=0.5,
    color="#D68D96",
    alpha=0.5,
    label="ADMM-SVGD samples"
)
xlim(-3.5, 3.5)
ylim(-2.5, 8)
xlabel(L"x_1")
ylabel(L"x_2")
title("ADMM-SVGD Samples")
grid(true, alpha=0.3)
legend()

tight_layout()
savefig(joinpath(save_dir, "samples_comparison.png"), dpi=250, bbox_inches="tight")
close(fig)
println("  Saved: samples_comparison.png")

# =====================================================
# Figure 2: Constraint violation over iterations
# =====================================================
println("\nCreating Figure 2: Constraint violation...")

fig = figure("constraint_violation", figsize=(8, 5))
plot(constraint_violations, linewidth=2, color="#4a4a4a")
xlabel("Iteration (every 10)")
ylabel("Mean Constraint Violation")
title(L"Constraint Violation: $|z - x_1^2|$")
grid(true, alpha=0.3)
yscale("log")
tight_layout()
savefig(joinpath(save_dir, "constraint_violation.png"), dpi=250, bbox_inches="tight")
close(fig)
println("  Saved: constraint_violation.png")

# =====================================================
# Figure 3: Particle evolution
# =====================================================
println("\nCreating Figure 3: Particle evolution...")

n_snapshots = min(6, length(particle_history))
snapshot_indices = round.(Int, range(1, length(particle_history), length=n_snapshots))

fig = figure("particle_evolution", figsize=(15, 10))

for (idx, snap_idx) in enumerate(snapshot_indices)
    subplot(2, 3, idx)
    particles = particle_history[snap_idx]

    scatter(
        particles[1, :],
        particles[2, :],
        s=0.5,
        color="#D68D96",
        alpha=0.5
    )

    xlim(-3.5, 3.5)
    ylim(-2.5, 8)
    xlabel(L"x_1")
    ylabel(L"x_2")

    # Calculate actual iteration number
    iter_num = snap_idx * 10
    if snap_idx == 1
        iter_num = 1
    elseif snap_idx == length(particle_history)
        iter_num = args[:n_iterations]
    end

    title("Iteration $iter_num")
    grid(true, alpha=0.3)
end

suptitle("Particle Evolution", fontsize=16)
tight_layout()
savefig(joinpath(save_dir, "particle_evolution.png"), dpi=250, bbox_inches="tight")
close(fig)
println("  Saved: particle_evolution.png")

# =====================================================
# Figure 4: 2D Histograms
# =====================================================
println("\nCreating Figure 4: 2D histograms...")

fig = figure("histograms", figsize=(12, 5))

# True samples histogram
subplot(1, 2, 1)
hist2d(
    true_samples[1, :],
    true_samples[2, :],
    bins=75,
    cmap="viridis",
    density=true
)
colorbar(label="Density")
xlim(-3.5, 3.5)
ylim(-2.5, 8)
xlabel(L"x_1")
ylabel(L"x_2")
title("True Sample Density")

# ADMM-SVGD samples histogram
subplot(1, 2, 2)
hist2d(
    final_particles[1, :],
    final_particles[2, :],
    bins=75,
    cmap="viridis",
    density=true
)
colorbar(label="Density")
xlim(-3.5, 3.5)
ylim(-2.5, 8)
xlabel(L"x_1")
ylabel(L"x_2")
title("ADMM-SVGD Sample Density")

tight_layout()
savefig(joinpath(save_dir, "histograms.png"), dpi=250, bbox_inches="tight")
close(fig)
println("  Saved: histograms.png")

# =====================================================
# Figure 5: Marginal distributions
# =====================================================
println("\nCreating Figure 5: Marginal distributions...")

fig = figure("marginals", figsize=(12, 5))

# x1 marginal
subplot(1, 2, 1)
hist(true_samples[1, :], bins=50, alpha=0.6, color="#819FB3",
     label="True", density=true, edgecolor="black", linewidth=0.5)
hist(final_particles[1, :], bins=50, alpha=0.6, color="#D68D96",
     label="ADMM-SVGD", density=true, edgecolor="black", linewidth=0.5)
xlabel(L"x_1")
ylabel("Density")
title(L"Marginal Distribution of $x_1$")
legend()
grid(true, alpha=0.3)

# x2 marginal
subplot(1, 2, 2)
hist(true_samples[2, :], bins=50, alpha=0.6, color="#819FB3",
     label="True", density=true, edgecolor="black", linewidth=0.5)
hist(final_particles[2, :], bins=50, alpha=0.6, color="#D68D96",
     label="ADMM-SVGD", density=true, edgecolor="black", linewidth=0.5)
xlabel(L"x_2")
ylabel("Density")
title(L"Marginal Distribution of $x_2$")
legend()
grid(true, alpha=0.3)

tight_layout()
savefig(joinpath(save_dir, "marginals.png"), dpi=250, bbox_inches="tight")
close(fig)
println("  Saved: marginals.png")

# =====================================================
# Print summary statistics
# =====================================================
println("\n" * "="^60)
println("Summary Statistics")
println("="^60)

println("\nTrue samples:")
println("  x₁: mean = $(round(mean(true_samples[1, :]), digits=3)), std = $(round(std(true_samples[1, :]), digits=3))")
println("  x₂: mean = $(round(mean(true_samples[2, :]), digits=3)), std = $(round(std(true_samples[2, :]), digits=3))")

println("\nADMM-SVGD samples:")
println("  x₁: mean = $(round(mean(final_particles[1, :]), digits=3)), std = $(round(std(final_particles[1, :]), digits=3))")
println("  x₂: mean = $(round(mean(final_particles[2, :]), digits=3)), std = $(round(std(final_particles[2, :]), digits=3))")

println("\nFinal constraint violation: $(round(constraint_violations[end], digits=6))")

println("\n" * "="^60)
println("Visualization complete!")
println("="^60)
println("Figures saved to: $save_dir")