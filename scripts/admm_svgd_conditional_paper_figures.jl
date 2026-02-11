# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Feb 2026
# Generate paper-quality figures for Rosenbrock conditional inference

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
font_prop, sfmt = set_plot_configs(; fontsize = 10)

# Read configuration
args = read_config("admm_svgd_conditional_sampling.json")
args = parse_input_args(args)

# Load saved results
println("Loading results...")
loaded_keys = load_experiment(
    args,
    [
        "X_test", "Y_test", "X_fixed", "Y_fixed", "X_post",
        "hist_constraint_res", "hist_logpdf", "hist_bandwidth",
        "hist_mean", "hist_std",
    ],
)

X_test = loaded_keys["X_test"]
Y_test = loaded_keys["Y_test"]
X_fixed = loaded_keys["X_fixed"]
Y_fixed = loaded_keys["Y_fixed"]
X_post = loaded_keys["X_post"]
hist_constraint_res = loaded_keys["hist_constraint_res"]
hist_logpdf = loaded_keys["hist_logpdf"]
hist_bandwidth = loaded_keys["hist_bandwidth"]
hist_mean = loaded_keys["hist_mean"]
hist_std = loaded_keys["hist_std"]

n_particles = size(X_post, 4)
n_iters = size(hist_constraint_res, 1)

# Select 4 instances that fit within the view [-3,3]×[-2.5,7]
# Instance 4 (true≈[3.1, 9.6]) is outside the view — skip it
inst_idx = [1, 2, 3, 5]
n_inst = length(inst_idx)

println("  Selected instances: $inst_idx (skipping instance 4, outside view)")
println("  Posterior samples per instance: $n_particles")
println("  Iterations: $n_iters")

# Create Rosenbrock distribution for reference
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# Save directories
paper_figs = joinpath(
    homedir(), "Documents", "paper-GEO2026svgd", "figs", "rosenbrock-conditional",
)
!isdir(paper_figs) && mkpath(paper_figs)

plot_save = plotsdir(args["sim_name"], savename(args))
!isdir(plot_save) && mkpath(plot_save)

println("  Paper figures: $paper_figs")
println("  Plot directory: $plot_save")

# Color palette
c_prior = "#555555"
c_svgd = "#D68D96"
c_mala = "#819FB3"
c_obs = "#2C3E50"
c_true = "#E74C3C"

# Instance colors for convergence/Q-Q
inst_colors = ["#E24A33", "#348ABD", "#988ED5", "#8EBA42"]
inst_labels = ["Instance $k" for k = 1:n_inst]

# ==========================================================================
# Run MALA for all selected instances (needed for combined plots and Q-Q)
# ==========================================================================
println("\nRunning MALA sampling...")

σ² = Float32(args["sigma"]^2)
a_rb = Float32(RB_dist.a)
μ_rb = Float32(RB_dist.μ)

function nlogpost(x, y1, y2)
    data = ((x[1] - y1)^2 + (x[2] - y2)^2) / (2 * σ²)
    prior = a_rb * (x[1] - μ_rb)^2 + (x[2] - x[1]^2)^2
    return data + prior
end

function grad_nlogpost(x, y1, y2)
    x1, x2 = x[1], x[2]
    dx1 = (x1 - y1) / σ² + 2 * a_rb * (x1 - μ_rb) - 4 * x1 * (x2 - x1^2)
    dx2 = (x2 - y2) / σ² + 2 * (x2 - x1^2)
    return Float32[dx1, dx2]
end

n_mcmc = n_particles  # Match SVGD sample count
max_itr = 50000
X_mala = zeros(Float32, 2, n_mcmc, n_inst)

prog = Progress(n_inst; desc = "MALA sampling: ")
for (k, j) in enumerate(inst_idx)
    y1_j = Float32(Y_fixed[1, 1, 1, j])
    y2_j = Float32(Y_fixed[1, 1, 2, j])
    U(x) = nlogpost(x, y1_j, y2_j)
    ∇U(x) = grad_nlogpost(x, y1_j, y2_j)
    samples = MALA_sampler(
        max_itr, randn(Float32, 2), U, ∇U;
        τ = 3.0f-2, thinning = 5, burn_in_frac = 0.5f0,
    )
    n_avail = size(samples, 2)
    X_mala[:, :, k] = samples[:, max(1, n_avail - n_mcmc + 1):n_avail]
    next!(prog)
end

# ==========================================================================
# Figure 1a: Prior distribution
# ==========================================================================
println("\nGenerating Figure 1a: Prior distribution...")

fig = figure(figsize = (4, 4))
ax = fig.add_subplot(111)
scatter(
    X_test[1, 1, 1, :], X_test[1, 1, 2, :],
    s = 0.3, color = c_prior, alpha = 0.3, rasterized = true,
)
for (k, j) in enumerate(inst_idx)
    scatter(
        [X_fixed[1, 1, 1, j]], [X_fixed[1, 1, 2, j]],
        s = 60.0, color = c_true, marker = "^",
        edgecolors = "black", linewidths = 0.8, zorder = 10,
    )
end
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_xlabel(L"$x_1$")
ax.set_ylabel(L"$x_2$")
tight_layout()
for path in [joinpath(paper_figs, "prior.png"), joinpath(plot_save, "prior.png")]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 1b: Data distribution (noisy observations)
# ==========================================================================
println("Generating Figure 1b: Data distribution...")

fig = figure(figsize = (4, 4))
ax = fig.add_subplot(111)
scatter(
    Y_test[1, 1, 1, :], Y_test[1, 1, 2, :],
    s = 0.3, color = c_prior, alpha = 0.3, rasterized = true,
)
for (k, j) in enumerate(inst_idx)
    scatter(
        [Y_fixed[1, 1, 1, j]], [Y_fixed[1, 1, 2, j]],
        s = 60.0, color = c_mala, marker = "v",
        edgecolors = "black", linewidths = 0.8, zorder = 10,
    )
end
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_xlabel(L"$y_1$")
ax.set_ylabel(L"$y_2$")
tight_layout()
for path in [joinpath(paper_figs, "data.png"), joinpath(plot_save, "data.png")]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 2: ADMM-SVGD conditional posteriors (2×2 panel)
# ==========================================================================
println("Generating Figure 2: ADMM-SVGD conditional posteriors (2×2)...")

fig = figure(figsize = (8, 8))
for (k, j) in enumerate(inst_idx)
    local ax = fig.add_subplot(2, 2, k)

    # Background: prior samples
    ax.scatter(
        X_test[1, 1, 1, :], X_test[1, 1, 2, :],
        s = 0.3, color = c_prior, alpha = 0.25, rasterized = true,
    )

    # SVGD posterior samples
    ax.scatter(
        X_post[1, 1, 1, :, j], X_post[1, 1, 2, :, j],
        s = 1.5, color = c_svgd, alpha = 0.3, rasterized = true,
    )

    # Observed value
    ax.scatter(
        [Y_fixed[1, 1, 1, j]], [Y_fixed[1, 1, 2, j]],
        s = 80.0, color = c_obs, marker = "v",
        edgecolors = "black", linewidths = 1.0, zorder = 10,
    )

    # True value
    ax.scatter(
        [X_fixed[1, 1, 1, j]], [X_fixed[1, 1, 2, j]],
        s = 80.0, color = c_true, marker = "^",
        edgecolors = "black", linewidths = 1.0, zorder = 10,
    )

    ax.set_xlim([-3, 3])
    ax.set_ylim([-2.5, 7])
    ax.set_xlabel(L"$x_1$")
    ax.set_ylabel(L"$x_2$")
    ax.set_title("Instance $k")
end
tight_layout()
for path in [
    joinpath(paper_figs, "svgd-posteriors.png"),
    joinpath(plot_save, "svgd-posteriors.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 3: MALA conditional posteriors (2×2 panel)
# ==========================================================================
println("Generating Figure 3: MALA conditional posteriors (2×2)...")

fig = figure(figsize = (8, 8))
for (k, j) in enumerate(inst_idx)
    local ax = fig.add_subplot(2, 2, k)

    # Background: prior samples
    ax.scatter(
        X_test[1, 1, 1, :], X_test[1, 1, 2, :],
        s = 0.3, color = c_prior, alpha = 0.25, rasterized = true,
    )

    # MALA posterior samples
    ax.scatter(
        X_mala[1, :, k], X_mala[2, :, k],
        s = 1.5, color = c_mala, alpha = 0.3, rasterized = true,
    )

    # Observed value
    ax.scatter(
        [Y_fixed[1, 1, 1, j]], [Y_fixed[1, 1, 2, j]],
        s = 80.0, color = c_obs, marker = "v",
        edgecolors = "black", linewidths = 1.0, zorder = 10,
    )

    # True value
    ax.scatter(
        [X_fixed[1, 1, 1, j]], [X_fixed[1, 1, 2, j]],
        s = 80.0, color = c_true, marker = "^",
        edgecolors = "black", linewidths = 1.0, zorder = 10,
    )

    ax.set_xlim([-3, 3])
    ax.set_ylim([-2.5, 7])
    ax.set_xlabel(L"$x_1$")
    ax.set_ylabel(L"$x_2$")
    ax.set_title("Instance $k")
end
tight_layout()
for path in [
    joinpath(paper_figs, "mala-posteriors.png"),
    joinpath(plot_save, "mala-posteriors.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 4: Combined SVGD + MALA posteriors (2×2 panel)
# ==========================================================================
println("Generating Figure 4: Combined SVGD + MALA posteriors (2×2)...")

fig = figure(figsize = (8, 8))
for (k, j) in enumerate(inst_idx)
    local ax = fig.add_subplot(2, 2, k)

    # Background: prior samples
    ax.scatter(
        X_test[1, 1, 1, :], X_test[1, 1, 2, :],
        s = 0.3, color = c_prior, alpha = 0.15, rasterized = true,
    )

    # MALA posterior samples
    ax.scatter(
        X_mala[1, :, k], X_mala[2, :, k],
        s = 1.5, color = c_mala, alpha = 0.25, rasterized = true,
        label = k == 1 ? "MALA" : nothing,
    )

    # SVGD posterior samples
    ax.scatter(
        X_post[1, 1, 1, :, j], X_post[1, 1, 2, :, j],
        s = 1.5, color = c_svgd, alpha = 0.25, rasterized = true,
        label = k == 1 ? "ADMM-SVGD" : nothing,
    )

    # Observed value
    ax.scatter(
        [Y_fixed[1, 1, 1, j]], [Y_fixed[1, 1, 2, j]],
        s = 80.0, color = c_obs, marker = "v",
        edgecolors = "black", linewidths = 1.0, zorder = 10,
        label = k == 1 ? "Observation" : nothing,
    )

    # True value
    ax.scatter(
        [X_fixed[1, 1, 1, j]], [X_fixed[1, 1, 2, j]],
        s = 80.0, color = c_true, marker = "^",
        edgecolors = "black", linewidths = 1.0, zorder = 10,
        label = k == 1 ? "True" : nothing,
    )

    ax.set_xlim([-3, 3])
    ax.set_ylim([-2.5, 7])
    ax.set_xlabel(L"$x_1$")
    ax.set_ylabel(L"$x_2$")
    ax.set_title("Instance $k")
end

# Shared legend from first panel
handles, labels = fig.axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc = "lower center", ncol = 4, fontsize = 9, frameon = false)
tight_layout(rect = [0, 0.04, 1, 1])
for path in [
    joinpath(paper_figs, "combined-posteriors.png"),
    joinpath(plot_save, "combined-posteriors.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 5: Convergence diagnostics (6-panel, 4 instances)
# ==========================================================================
println("Generating Figure 5: Convergence diagnostics...")

iters = 1:n_iters

fig = figure(figsize = (12, 6.5))

# Panel (a): Constraint residual (log scale)
ax1 = fig.add_subplot(2, 3, 1)
for (k, j) in enumerate(inst_idx)
    ax1.semilogy(collect(iters), Vector{Float64}(hist_constraint_res[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax1.set_xlabel("Iteration")
ax1.set_ylabel(L"$|z - x_1^2|$")
ax1.set_title("(a) Constraint residual")

# Panel (b): Average log-posterior
ax2 = fig.add_subplot(2, 3, 2)
for (k, j) in enumerate(inst_idx)
    ax2.plot(collect(iters), Vector{Float64}(hist_logpdf[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax2.set_xlabel("Iteration")
ax2.set_ylabel(L"$\langle \log p(\mathbf{x}|\mathbf{y}) \rangle$")
ax2.set_title("(b) Avg. log-posterior")

# Panel (c): Bandwidth
ax3 = fig.add_subplot(2, 3, 3)
for (k, j) in enumerate(inst_idx)
    ax3.plot(collect(iters), Vector{Float64}(hist_bandwidth[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax3.set_xlabel("Iteration")
ax3.set_ylabel(L"$h$")
ax3.set_title("(c) Kernel bandwidth")

# Panel (d): Particle mean x₁
ax4 = fig.add_subplot(2, 3, 4)
for (k, j) in enumerate(inst_idx)
    ax4.plot(collect(iters), Vector{Float64}(hist_mean[:, 1, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax4.axhline(
        y = Float64(X_fixed[1, 1, 1, j]), color = inst_colors[k],
        linestyle = ":", alpha = 0.6, lw = 1.0,
    )
end
ax4.set_xlabel("Iteration")
ax4.set_ylabel(L"$\langle x_1 \rangle$")
ax4.set_title(L"(d) Particle mean $x_1$" * " (dotted = true)")

# Panel (e): Particle mean x₂
ax5 = fig.add_subplot(2, 3, 5)
for (k, j) in enumerate(inst_idx)
    ax5.plot(collect(iters), Vector{Float64}(hist_mean[:, 2, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax5.axhline(
        y = Float64(X_fixed[1, 1, 2, j]), color = inst_colors[k],
        linestyle = ":", alpha = 0.6, lw = 1.0,
    )
end
ax5.set_xlabel("Iteration")
ax5.set_ylabel(L"$\langle x_2 \rangle$")
ax5.set_title(L"(e) Particle mean $x_2$" * " (dotted = true)")

# Panel (f): Posterior std (x₁ solid, x₂ dashed)
ax6 = fig.add_subplot(2, 3, 6)
for (k, j) in enumerate(inst_idx)
    ax6.plot(collect(iters), Vector{Float64}(hist_std[:, 1, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax6.plot(collect(iters), Vector{Float64}(hist_std[:, 2, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2, linestyle = "--")
end
ax6.plot([], [], color = "gray", lw = 1.2, linestyle = "-", label = L"$\sigma_{x_1}$")
ax6.plot([], [], color = "gray", lw = 1.2, linestyle = "--", label = L"$\sigma_{x_2}$")
ax6.set_xlabel("Iteration")
ax6.set_ylabel(L"$\sigma$")
ax6.set_title("(f) Posterior std. dev.")
ax6.legend(fontsize = 8, loc = "upper right")

# Shared legend for instances
handles = [matplotlib.patches.Patch(facecolor = inst_colors[k], label = inst_labels[k]) for k = 1:n_inst]
fig.legend(handles = handles, loc = "lower center", ncol = n_inst, fontsize = 9, frameon = false)

tight_layout(rect = [0, 0.05, 1, 1])
for path in [
    joinpath(paper_figs, "convergence.png"),
    joinpath(plot_save, "convergence.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 6: Q-Q plots vs MALA (2×4 grid for 4 instances)
# ==========================================================================
println("Generating Figure 6: Q-Q plots vs MALA...")

dq = 1.0f-2
fig, axes = subplots(2, n_inst; figsize = (3 * n_inst, 6))

for (k, j) in enumerate(inst_idx)
    for (row, dim, label) in [(1, 1, L"$x_1$"), (2, 2, L"$x_2$")]
        local ax = axes[row, k]
        mcmc_q = quantile(vec(X_mala[dim, :, k]), dq:dq:1-dq)
        admm_q = quantile(vec(X_post[1, 1, dim, :, j]), dq:dq:1-dq)
        ax.plot(mcmc_q, admm_q, "o", ms = 2.0, color = inst_colors[k], alpha = 0.6)
        ref = range(
            min(minimum(mcmc_q), minimum(admm_q)),
            max(maximum(mcmc_q), maximum(admm_q)),
            length = 100,
        )
        ax.plot(ref, ref, "k-", lw = 0.8, alpha = 0.5)
        ax.set_aspect("equal")
        if row == 2
            ax.set_xlabel("MALA")
        end
        if k == 1
            ax.set_ylabel("ADMM-SVGD")
        end
        if row == 1
            ax.set_title("Inst. $k ($label)")
        else
            ax.set_title("($label)")
        end
    end
end
tight_layout()
for path in [
    joinpath(paper_figs, "qq-plots.png"),
    joinpath(plot_save, "qq-plots.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Print summary
# ==========================================================================
println("\n" * "="^60)
println("Summary Statistics (4 selected instances)")
println("="^60)
for (k, j) in enumerate(inst_idx)
    mx1 = mean(X_post[1, 1, 1, :, j])
    mx2 = mean(X_post[1, 1, 2, :, j])
    sx1 = std(X_post[1, 1, 1, :, j])
    sx2 = std(X_post[1, 1, 2, :, j])
    tx1 = X_fixed[1, 1, 1, j]
    tx2 = X_fixed[1, 1, 2, j]
    err = sqrt((mx1 - tx1)^2 + (mx2 - tx2)^2)
    cres = hist_constraint_res[end, j]
    @printf(
        "  Instance %d (idx=%d): true=[%6.3f,%6.3f] mean=[%6.3f,%6.3f] std=[%.3f,%.3f] err=%.4f cres=%.2e\n",
        k, j, tx1, tx2, mx1, mx2, sx1, sx2, err, cres,
    )
end
println("\nFigures saved to:")
println("  Paper: $paper_figs")
println("  Plots: $plot_save")
println("="^60)
