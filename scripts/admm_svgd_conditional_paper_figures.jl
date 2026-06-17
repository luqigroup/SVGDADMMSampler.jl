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

# Set plot configurations
font_prop, sfmt = set_plot_configs(; fontsize = 10)

# Read configuration
args = read_config("admm_svgd_conditional_sampling.json")
args = parse_input_args(args)

# Load saved results
println("Loading ADMM-SVGD results...")
loaded_keys = load_experiment(
    args,
    [
        "X_test", "Y_test", "X_fixed", "Y_fixed", "X_post",
        "hist_constraint_res", "hist_logpdf", "hist_bandwidth",
        "hist_mean", "hist_std", "hist_multiplier", "hist_penalty",
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
hist_multiplier = loaded_keys["hist_multiplier"]
hist_penalty = loaded_keys["hist_penalty"]

n_particles = size(X_post, 4)
n_iters = size(hist_constraint_res, 1)

# Load plain SVGD results
println("Loading plain SVGD results...")
svgd_args = read_config("svgd_conditional_sampling.json")
svgd_args = parse_input_args(svgd_args)

svgd_keys = load_experiment(
    svgd_args,
    [
        "X_post", "hist_logpdf", "hist_bandwidth",
        "hist_mean", "hist_std",
    ],
)

X_post_plain = svgd_keys["X_post"]
hist_logpdf_plain = svgd_keys["hist_logpdf"]
hist_bandwidth_plain = svgd_keys["hist_bandwidth"]
hist_mean_plain = svgd_keys["hist_mean"]
hist_std_plain = svgd_keys["hist_std"]

n_iters_plain = size(hist_logpdf_plain, 1)

# Select 4 instances that fit within the view [-3,3]×[-2.5,7]
# Instance 4 (true≈[3.1, 9.6]) is outside the view — skip it
inst_idx = [1, 2, 3, 5]
n_inst = length(inst_idx)

println("  Selected instances: $inst_idx (skipping instance 4, outside view)")
println("  Posterior samples per instance: $n_particles")
println("  ADMM-SVGD iterations: $n_iters")
println("  Plain SVGD iterations: $n_iters_plain")

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
c_plain_svgd = "#4E9A6D"
c_obs = "#2C3E50"
c_true = "#E74C3C"

# Instance colors for convergence/Q-Q
inst_colors = ["#E24A33", "#348ABD", "#988ED5", "#8EBA42"]
inst_labels = ["Instance $k" for k = 1:n_inst]

# ==========================================================================
# Figure 1a: Prior distribution
# ==========================================================================
println("\nGenerating Figure 1a: Prior distribution...")

fig = figure(figsize = (4, 4))
ax = fig.add_subplot(111)
scatter(
    X_test[1, 1, 1, :], X_test[1, 1, 2, :],
    s = 1.0, color = c_prior, alpha = 0.4, rasterized = true,
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
    s = 1.0, color = c_prior, alpha = 0.4, rasterized = true,
)
for (k, j) in enumerate(inst_idx)
    scatter(
        [Y_fixed[1, 1, 1, j]], [Y_fixed[1, 1, 2, j]],
        s = 60.0, color = c_obs, marker = "v",
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
        s = 1.0, color = c_prior, alpha = 0.25, rasterized = true,
    )

    # ADMM-SVGD posterior samples
    ax.scatter(
        X_post[1, 1, 1, :, j], X_post[1, 1, 2, :, j],
        s = 6.0, color = c_svgd, alpha = 0.4, rasterized = true,
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
# Figure 2b: Plain SVGD conditional posteriors (2×2 panel)
# ==========================================================================
println("Generating Figure 2b: Plain SVGD conditional posteriors (2×2)...")

fig = figure(figsize = (8, 8))
for (k, j) in enumerate(inst_idx)
    local ax = fig.add_subplot(2, 2, k)

    # Background: prior samples
    ax.scatter(
        X_test[1, 1, 1, :], X_test[1, 1, 2, :],
        s = 1.0, color = c_prior, alpha = 0.25, rasterized = true,
    )

    # Plain SVGD posterior samples
    ax.scatter(
        X_post_plain[1, 1, 1, :, j], X_post_plain[1, 1, 2, :, j],
        s = 6.0, color = c_plain_svgd, alpha = 0.4, rasterized = true,
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
    joinpath(paper_figs, "plain-svgd-posteriors.png"),
    joinpath(plot_save, "plain-svgd-posteriors.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 3: Combined ADMM-SVGD + plain SVGD posteriors (2×2 panel)
# ==========================================================================
println("Generating Figure 3: Combined ADMM-SVGD + SVGD posteriors (2×2)...")

fig = figure(figsize = (8, 8))
for (k, j) in enumerate(inst_idx)
    local ax = fig.add_subplot(2, 2, k)

    # Background: prior samples
    ax.scatter(
        X_test[1, 1, 1, :], X_test[1, 1, 2, :],
        s = 1.0, color = c_prior, alpha = 0.25, rasterized = true,
    )

    # ADMM-SVGD posterior samples
    ax.scatter(
        X_post[1, 1, 1, :, j], X_post[1, 1, 2, :, j],
        s = 6.0, color = c_svgd, alpha = 0.35, rasterized = true,
        label = k == 1 ? "ADMM-SVGD" : nothing,
    )

    # Plain SVGD posterior samples
    ax.scatter(
        X_post_plain[1, 1, 1, :, j], X_post_plain[1, 1, 2, :, j],
        s = 6.0, color = c_plain_svgd, alpha = 0.35, rasterized = true,
        label = k == 1 ? "SVGD" : nothing,
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
    joinpath(paper_figs, "combined-all-posteriors.png"),
    joinpath(plot_save, "combined-all-posteriors.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 4: ADMM-SVGD Convergence diagnostics (8-panel, 4 instances)
#   Panels (a)-(c) are the ADMM augmented-Lagrangian diagnostics — constraint
#   residual, dual variable (multiplier) ε, and quadratic penalty term — showing
#   how the ADMM components evolve and how the constraint is progressively
#   enforced. Panels (d)-(h) are the SVGD/posterior diagnostics.
# ==========================================================================
println("Generating Figure 4: ADMM-SVGD Convergence diagnostics...")

iters = 1:n_iters

fig = figure(figsize = (16, 6.5))

# Panel (a): Constraint residual (log scale)
ax1 = fig.add_subplot(2, 4, 1)
for (k, j) in enumerate(inst_idx)
    ax1.semilogy(collect(iters), Vector{Float64}(hist_constraint_res[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax1.set_xlabel("Iteration")
ax1.set_ylabel(L"$|z - x_1^2|$")
ax1.set_title("(a) Constraint residual")

# Panel (b): Multiplier (dual variable) ε — ensemble mean
ax2 = fig.add_subplot(2, 4, 2)
for (k, j) in enumerate(inst_idx)
    ax2.plot(collect(iters), Vector{Float64}(hist_multiplier[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax2.set_xlabel("Iteration")
ax2.set_ylabel(L"$\langle \varepsilon \rangle$")
ax2.set_title("(b) Multiplier")

# Panel (c): Penalty term (μ/2)(z - x₁²)² — ensemble mean (log scale)
ax3 = fig.add_subplot(2, 4, 3)
for (k, j) in enumerate(inst_idx)
    ax3.semilogy(collect(iters), Vector{Float64}(hist_penalty[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax3.set_xlabel("Iteration")
ax3.set_ylabel(L"$\langle \frac{\mu}{2}(z - x_1^2)^2 \rangle$")
ax3.set_title("(c) Penalty term")

# Panel (d): Average log-posterior
ax4 = fig.add_subplot(2, 4, 4)
for (k, j) in enumerate(inst_idx)
    ax4.plot(collect(iters), Vector{Float64}(hist_logpdf[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax4.set_xlabel("Iteration")
ax4.set_ylabel(L"$\langle \log p(\mathbf{x}|\mathbf{y}) \rangle$")
ax4.set_title("(d) Avg. log-posterior")

# Panel (e): Bandwidth
ax5 = fig.add_subplot(2, 4, 5)
for (k, j) in enumerate(inst_idx)
    ax5.plot(collect(iters), Vector{Float64}(hist_bandwidth[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax5.set_xlabel("Iteration")
ax5.set_ylabel(L"$h$")
ax5.set_title("(e) Kernel bandwidth")

# Panel (f): Particle mean x₁
ax6 = fig.add_subplot(2, 4, 6)
for (k, j) in enumerate(inst_idx)
    ax6.plot(collect(iters), Vector{Float64}(hist_mean[:, 1, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax6.axhline(
        y = Float64(X_fixed[1, 1, 1, j]), color = inst_colors[k],
        linestyle = ":", alpha = 0.6, lw = 1.0,
    )
end
ax6.set_xlabel("Iteration")
ax6.set_ylabel(L"$\langle x_1 \rangle$")
ax6.set_title(L"(f) Particle mean $x_1$" * " (dotted = true)")

# Panel (g): Particle mean x₂
ax7 = fig.add_subplot(2, 4, 7)
for (k, j) in enumerate(inst_idx)
    ax7.plot(collect(iters), Vector{Float64}(hist_mean[:, 2, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax7.axhline(
        y = Float64(X_fixed[1, 1, 2, j]), color = inst_colors[k],
        linestyle = ":", alpha = 0.6, lw = 1.0,
    )
end
ax7.set_xlabel("Iteration")
ax7.set_ylabel(L"$\langle x_2 \rangle$")
ax7.set_title(L"(g) Particle mean $x_2$" * " (dotted = true)")

# Panel (h): Posterior std (x₁ solid, x₂ dashed)
ax8 = fig.add_subplot(2, 4, 8)
for (k, j) in enumerate(inst_idx)
    ax8.plot(collect(iters), Vector{Float64}(hist_std[:, 1, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax8.plot(collect(iters), Vector{Float64}(hist_std[:, 2, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2, linestyle = "--")
end
ax8.plot([], [], color = "gray", lw = 1.2, linestyle = "-", label = L"$\sigma_{x_1}$")
ax8.plot([], [], color = "gray", lw = 1.2, linestyle = "--", label = L"$\sigma_{x_2}$")
ax8.set_xlabel("Iteration")
ax8.set_ylabel(L"$\sigma$")
ax8.set_title("(h) Posterior std. dev.")
ax8.legend(fontsize = 8, loc = "upper right")

# Shared legend for instances
handles = [matplotlib.patches.Patch(facecolor = inst_colors[k], label = inst_labels[k]) for k = 1:n_inst]
fig.legend(handles = handles, loc = "lower center", ncol = n_inst, fontsize = 9, frameon = false)

tight_layout(rect = [0, 0.05, 1, 1])
for path in [
    joinpath(paper_figs, "convergence.png"),
    joinpath(plot_save, "convergence.png"),
]
    _wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 4b: Plain SVGD convergence diagnostics (5-panel, no constraint)
# ==========================================================================
println("Generating Figure 4b: Plain SVGD convergence diagnostics...")

iters_plain = 1:n_iters_plain

fig = figure(figsize = (12, 6.5))

# Panel (a): Average log-posterior
ax1 = fig.add_subplot(2, 3, 1)
for (k, j) in enumerate(inst_idx)
    ax1.plot(collect(iters_plain), Vector{Float64}(hist_logpdf_plain[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax1.set_xlabel("Iteration")
ax1.set_ylabel(L"$\langle \log p(\mathbf{x}|\mathbf{y}) \rangle$")
ax1.set_title("(a) Avg. log-posterior")

# Panel (b): Bandwidth
ax2 = fig.add_subplot(2, 3, 2)
for (k, j) in enumerate(inst_idx)
    ax2.plot(collect(iters_plain), Vector{Float64}(hist_bandwidth_plain[:, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
end
ax2.set_xlabel("Iteration")
ax2.set_ylabel(L"$h$")
ax2.set_title("(b) Kernel bandwidth")

# Panel (c): Particle mean x₁
ax3 = fig.add_subplot(2, 3, 3)
for (k, j) in enumerate(inst_idx)
    ax3.plot(collect(iters_plain), Vector{Float64}(hist_mean_plain[:, 1, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax3.axhline(
        y = Float64(X_fixed[1, 1, 1, j]), color = inst_colors[k],
        linestyle = ":", alpha = 0.6, lw = 1.0,
    )
end
ax3.set_xlabel("Iteration")
ax3.set_ylabel(L"$\langle x_1 \rangle$")
ax3.set_title(L"(c) Particle mean $x_1$" * " (dotted = true)")

# Panel (d): Particle mean x₂
ax4 = fig.add_subplot(2, 3, 4)
for (k, j) in enumerate(inst_idx)
    ax4.plot(collect(iters_plain), Vector{Float64}(hist_mean_plain[:, 2, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax4.axhline(
        y = Float64(X_fixed[1, 1, 2, j]), color = inst_colors[k],
        linestyle = ":", alpha = 0.6, lw = 1.0,
    )
end
ax4.set_xlabel("Iteration")
ax4.set_ylabel(L"$\langle x_2 \rangle$")
ax4.set_title(L"(d) Particle mean $x_2$" * " (dotted = true)")

# Panel (e): Posterior std (x₁ solid, x₂ dashed)
ax5 = fig.add_subplot(2, 3, 5)
for (k, j) in enumerate(inst_idx)
    ax5.plot(collect(iters_plain), Vector{Float64}(hist_std_plain[:, 1, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2)
    ax5.plot(collect(iters_plain), Vector{Float64}(hist_std_plain[:, 2, j]), color = inst_colors[k], alpha = 0.8, lw = 1.2, linestyle = "--")
end
ax5.plot([], [], color = "gray", lw = 1.2, linestyle = "-", label = L"$\sigma_{x_1}$")
ax5.plot([], [], color = "gray", lw = 1.2, linestyle = "--", label = L"$\sigma_{x_2}$")
ax5.set_xlabel("Iteration")
ax5.set_ylabel(L"$\sigma$")
ax5.set_title("(e) Posterior std. dev.")
ax5.legend(fontsize = 8, loc = "upper right")

# Shared legend for instances
handles = [matplotlib.patches.Patch(facecolor = inst_colors[k], label = inst_labels[k]) for k = 1:n_inst]
fig.legend(handles = handles, loc = "lower center", ncol = n_inst, fontsize = 9, frameon = false)

tight_layout(rect = [0, 0.05, 1, 1])
for path in [
    joinpath(paper_figs, "convergence-plain-svgd.png"),
    joinpath(plot_save, "convergence-plain-svgd.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 5: Q-Q plots SVGD vs ADMM-SVGD (2×4 grid for 4 instances)
# ==========================================================================
println("Generating Figure 5: Q-Q plots SVGD vs ADMM-SVGD...")

dq = 1.0f-2
fig, axes = subplots(2, n_inst; figsize = (3 * n_inst, 6))

for (k, j) in enumerate(inst_idx)
    for (row, dim, label) in [(1, 1, L"$x_1$"), (2, 2, L"$x_2$")]
        local ax = axes[row, k]
        admm_q = quantile(vec(X_post[1, 1, dim, :, j]), dq:dq:1-dq)
        svgd_q = quantile(vec(X_post_plain[1, 1, dim, :, j]), dq:dq:1-dq)
        ax.plot(admm_q, svgd_q, "o", ms = 2.0, color = inst_colors[k], alpha = 0.6)
        ref = range(
            min(minimum(admm_q), minimum(svgd_q)),
            max(maximum(admm_q), maximum(svgd_q)),
            length = 100,
        )
        ax.plot(ref, ref, "k-", lw = 0.8, alpha = 0.5)
        ax.set_aspect("equal")
        if row == 2
            ax.set_xlabel("ADMM-SVGD")
        end
        if k == 1
            ax.set_ylabel("SVGD")
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
    joinpath(paper_figs, "qq-plots-svgd-vs-admm.png"),
    joinpath(plot_save, "qq-plots-svgd-vs-admm.png"),
]
    wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure 6: Analytic true conditional posterior + samples overlay (2×2 panel)
# ==========================================================================
# For a fixed observation y, the conditional posterior is
#   p(x|y) ∝ p(x) p(y|x),  Rosenbrock prior p(x) ∝ exp(-a(x₁-μ₀)² - (x₂-x₁²)²),
#   Gaussian likelihood y = x + N(0, σ²I)  ⇒  p(y|x) ∝ exp(-‖y-x‖²/(2σ²)).
# Negative log density (up to an additive constant):
#   U(x) = a(x₁-μ₀)² + (x₂-x₁²)² + ‖y-x‖²/(2σ²),   p(x|y) ∝ exp(-U(x)).
# (Matches the prior in Rosenbrock.logpdf and the σ²-Gaussian likelihood used in
#  the ADMM/SVGD samplers; a, μ₀, σ are read from the same args/RB_dist below.)
println("Generating Figure 6: Analytic posterior + samples overlay (2×2)...")

a_post = Float64(RB_dist.a)        # scaling parameter a
mu0_post = Float64(RB_dist.μ)      # prior mean μ₀ (mu_rb)
sigma_post = Float64(args["sigma"]) # observation noise std σ
n_grid = 400                       # grid resolution per axis

fig = figure(figsize = (8, 8))
for (k, j) in enumerate(inst_idx)
    local ax = fig.add_subplot(2, 2, k)

    # Observation y = (y₁, y₂) for this instance
    local y1 = Float64(Y_fixed[1, 1, 1, j])
    local y2 = Float64(Y_fixed[1, 1, 2, j])

    # Grid spanning the sample range (both methods + observation + true value),
    # padded by ~10% on each axis.
    local x1_vals = vcat(
        vec(Float64.(X_post[1, 1, 1, :, j])),
        vec(Float64.(X_post_plain[1, 1, 1, :, j])),
        [y1, Float64(X_fixed[1, 1, 1, j])],
    )
    local x2_vals = vcat(
        vec(Float64.(X_post[1, 1, 2, :, j])),
        vec(Float64.(X_post_plain[1, 1, 2, :, j])),
        [y2, Float64(X_fixed[1, 1, 2, j])],
    )
    local x1_lo, x1_hi = minimum(x1_vals), maximum(x1_vals)
    local x2_lo, x2_hi = minimum(x2_vals), maximum(x2_vals)
    local pad1 = 0.1 * (x1_hi - x1_lo)
    local pad2 = 0.1 * (x2_hi - x2_lo)
    x1_lo -= pad1; x1_hi += pad1
    x2_lo -= pad2; x2_hi += pad2

    local g1 = range(x1_lo, x1_hi; length = n_grid)
    local g2 = range(x2_lo, x2_hi; length = n_grid)

    # Evaluate the unnormalized density exp(-U) on the grid.
    # Rows index x₂ (g2), columns index x₁ (g1) for pcolormesh(X1, X2, P).
    local X1 = [x1 for _ in g2, x1 in g1]
    local X2 = [x2 for x2 in g2, _ in g1]
    local U = a_post .* (X1 .- mu0_post) .^ 2 .+ (X2 .- X1 .^ 2) .^ 2 .+
              ((y1 .- X1) .^ 2 .+ (y2 .- X2) .^ 2) ./ (2 * sigma_post^2)
    local P = exp.(-(U .- minimum(U)))  # subtract min for numerical stability

    # Background: analytic true posterior density
    ax.pcolormesh(
        collect(g1), collect(g2), P,
        cmap = "viridis", shading = "auto", rasterized = true,
    )

    # ADMM-SVGD posterior samples
    ax.scatter(
        X_post[1, 1, 1, :, j], X_post[1, 1, 2, :, j],
        s = 6.0, color = c_svgd, alpha = 0.35, rasterized = true,
        label = k == 1 ? "ADMM-SVGD" : nothing,
    )

    # Plain SVGD posterior samples
    ax.scatter(
        X_post_plain[1, 1, 1, :, j], X_post_plain[1, 1, 2, :, j],
        s = 6.0, color = c_plain_svgd, alpha = 0.35, rasterized = true,
        label = k == 1 ? "SVGD" : nothing,
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

    ax.set_xlim([x1_lo, x1_hi])
    ax.set_ylim([x2_lo, x2_hi])
    ax.set_xlabel(L"$x_1$")
    ax.set_ylabel(L"$x_2$")
    ax.set_title("Instance $k")
end

# Shared legend from first panel
handles, labels = fig.axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc = "lower center", ncol = 4, fontsize = 9, frameon = false)
tight_layout(rect = [0, 0.04, 1, 1])
for path in [
    joinpath(paper_figs, "true-posterior-overlay.png"),
    joinpath(plot_save, "true-posterior-overlay.png"),
]
    _wsave(path, fig)
end
close(fig)

# ==========================================================================
# Print summary
# ==========================================================================
println("\n" * "="^60)
println("Summary Statistics — ADMM-SVGD (4 selected instances)")
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

println("\n" * "="^60)
println("Summary Statistics — Plain SVGD (4 selected instances)")
println("="^60)
for (k, j) in enumerate(inst_idx)
    mx1 = mean(X_post_plain[1, 1, 1, :, j])
    mx2 = mean(X_post_plain[1, 1, 2, :, j])
    sx1 = std(X_post_plain[1, 1, 1, :, j])
    sx2 = std(X_post_plain[1, 1, 2, :, j])
    tx1 = X_fixed[1, 1, 1, j]
    tx2 = X_fixed[1, 1, 2, j]
    err = sqrt((mx1 - tx1)^2 + (mx2 - tx2)^2)
    @printf(
        "  Instance %d (idx=%d): true=[%6.3f,%6.3f] mean=[%6.3f,%6.3f] std=[%.3f,%.3f] err=%.4f\n",
        k, j, tx1, tx2, mx1, mx2, sx1, sx2, err,
    )
end

println("\nFigures saved to:")
println("  Paper: $paper_figs")
println("  Plots: $plot_save")
println("="^60)
