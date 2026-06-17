# Authors: Ali Siahkoohi, alisk@ucf.edu
# Generate paper-quality calibration figures (SBC rank histograms + reliability
# curve) for the Rosenbrock conditional sampler, comparing ADMM-SVGD vs plain
# SVGD. Loads the results saved by scripts/rosenbrock_calibration.jl.

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
args = read_config("rosenbrock_calibration.json")
args = parse_input_args(args)

# Load saved results
println("Loading Rosenbrock calibration results...")
loaded_keys = load_experiment(
    args,
    [
        "admm_ranks_x1", "admm_ranks_x2", "admm_cm", "admm_sd", "admm_theta",
        "svgd_ranks_x1", "svgd_ranks_x2", "svgd_cm", "svgd_sd", "svgd_theta",
        "selftest_pval",
        "sbc_L", "sbc_N", "sbc_bins", "reliability_bins",
    ],
)

admm_ranks_x1 = loaded_keys["admm_ranks_x1"]
admm_ranks_x2 = loaded_keys["admm_ranks_x2"]
admm_cm = loaded_keys["admm_cm"]
admm_sd = loaded_keys["admm_sd"]
admm_theta = loaded_keys["admm_theta"]
svgd_ranks_x1 = loaded_keys["svgd_ranks_x1"]
svgd_ranks_x2 = loaded_keys["svgd_ranks_x2"]
svgd_cm = loaded_keys["svgd_cm"]
svgd_sd = loaded_keys["svgd_sd"]
svgd_theta = loaded_keys["svgd_theta"]
selftest_pval = loaded_keys["selftest_pval"]

println("  SBC self-test p-value: ", round(selftest_pval, digits = 4))
println("  SBC replications L: ", args["sbc_L"])
println("  SBC bins: ", args["sbc_bins"])
println("  SBC N (thinned particles): ", args["sbc_N"])
println("  Reliability bins: ", args["reliability_bins"])

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
c_admm = "#D68D96"   # ADMM-SVGD (pink)
c_svgd = "#4E9A6D"   # plain SVGD (green)
c_ref = "#555555"    # reference / uniform expectation (gray)

# ==========================================================================
# Figure A: SBC rank histograms (2×2: rows = {x₁, x₂}, cols = {ADMM, SVGD})
# ==========================================================================
println("\nGenerating Figure A: SBC rank histograms...")

# Uniform expectation and (1-α) binomial band for the rank histogram.
band = sbc_uniform_band(args["sbc_L"], args["sbc_bins"])
rank_edges = range(0, args["sbc_N"]; length = args["sbc_bins"] + 1)

# (row, col) → (rank vector, color, method label, coordinate label, panel tag)
panels = [
    (1, 1, admm_ranks_x1, c_admm, "ADMM-SVGD", L"$x_1$", "(a)"),
    (1, 2, svgd_ranks_x1, c_svgd, "SVGD", L"$x_1$", "(b)"),
    (2, 1, admm_ranks_x2, c_admm, "ADMM-SVGD", L"$x_2$", "(c)"),
    (2, 2, svgd_ranks_x2, c_svgd, "SVGD", L"$x_2$", "(d)"),
]

fig, axes = subplots(2, 2; figsize = (8, 6))
for (row, col, ranks, color, method, coord, tag) in panels
    local ax = axes[row, col]

    ax.hist(
        Vector{Float64}(ranks), bins = collect(rank_edges),
        color = color, alpha = 0.75, edgecolor = "white", linewidth = 0.5,
    )

    # Uniform expectation (horizontal line at L/n_bins) and binomial band.
    ax.axhline(y = band.expected, color = c_ref, linestyle = "--", lw = 1.0, alpha = 0.8)
    ax.fill_between(
        [0, args["sbc_N"]], band.lower, band.upper,
        color = c_ref, alpha = 0.15, linewidth = 0.0,
    )

    ax.set_xlim([0, args["sbc_N"]])
    ax.set_xlabel("rank")
    ax.set_ylabel("count")
    ax.set_title(string(tag, " ", method, ", ") * coord)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end
tight_layout()
for path in [
    joinpath(paper_figs, "sbc-rank-histograms.png"),
    joinpath(plot_save, "sbc-rank-histograms.png"),
]
    _wsave(path, fig)
end
close(fig)

# ==========================================================================
# Figure B: reliability curve (RMSE-vs-std, with y=x reference)
# ==========================================================================
# CRITICAL: plot RMSE-vs-std against y=x — NOT |error|-vs-std. A calibrated
# Gaussian has E[|error|] = √(2/π)·σ ≈ 0.8σ, which sits below y=x and would be
# misread as over-confident. RMSE = √(E[error²]) = σ for a calibrated Gaussian,
# so a calibrated ensemble falls ON the y=x line.
println("\nGenerating Figure B: reliability curve (RMSE-vs-std)...")

coords = [(1, L"$x_1$"), (2, L"$x_2$")]
methods = [
    (admm_cm, admm_sd, admm_theta, c_admm, "ADMM-SVGD"),
    (svgd_cm, svgd_sd, svgd_theta, c_svgd, "SVGD"),
]

fig, axes = subplots(1, 2; figsize = (8, 4))
for (panel, (c, coord)) in enumerate(coords)
    local ax = axes[panel]

    lo = Inf
    hi = -Inf
    for (cm, sd, theta, color, label) in methods
        error = cm[:, c] .- theta[:, c]
        bin_std, bin_mse =
            binned_calibration(sd[:, c], error .^ 2; n_bins = args["reliability_bins"])
        rmse = sqrt.(bin_mse)
        ax.plot(
            Vector{Float64}(bin_std), Vector{Float64}(rmse),
            "-o", ms = 4.0, lw = 1.2, color = color, label = label,
        )
        lo = min(lo, minimum(bin_std), minimum(rmse))
        hi = max(hi, maximum(bin_std), maximum(rmse))
    end

    # y = x reference spanning the axis range.
    ref = range(lo, hi; length = 100)
    ax.plot(collect(ref), collect(ref), linestyle = "--", color = c_ref, lw = 1.0, alpha = 0.8)

    ax.set_xlim([lo, hi])
    ax.set_ylim([lo, hi])
    ax.set_aspect("equal")
    ax.set_xlabel("posterior std")
    ax.set_ylabel("RMSE")
    ax.set_title(coord)
    ax.legend(loc = "upper left", fontsize = 9, frameon = false)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end
tight_layout()
for path in [
    joinpath(paper_figs, "calibration-reliability.png"),
    joinpath(plot_save, "calibration-reliability.png"),
]
    _wsave(path, fig)
end
close(fig)

println("\nFigures saved to:")
println("  Paper: $paper_figs")
println("  Plots: $plot_save")
