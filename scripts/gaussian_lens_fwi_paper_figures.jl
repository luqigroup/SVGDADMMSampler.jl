# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Publication figures for the Gaussian-lens run, one vector PDF per plot.
#
#   julia --project=. scripts/gaussian_lens_fwi_paper_figures.jl
#
# Render only: reads the cached JLD2, never recomputes. Conventions follow the
# paper's Gaussian-anomaly figures so these drop in alongside them -- jet
# velocity maps with distance on the top axis, a white-to-dark-red ramp for the
# standard deviation, boxed axes, and per-particle curves colored by particle
# number with the ensemble mean dashed black over them.
#
# Output goes to plots/<sim>/<savename>/paper/.

using DrWatson
@quickactivate :SVGDADMMSampler

using JLD2
using Statistics
using LinearAlgebra
using PyPlot
using Printf

const IDENTITY_KEYS = ["sim_name", "nz", "nx", "freq", "n_sources", "n_particles",
                       "n_iterations", "method", "repulsion", "step_scale",
                       "prior_std_frac", "noise_rel", "seed"]

# Semantic palette, held fixed across every figure.
const C_TRUTH = "#E8A33D"      # ground truth, dashed
const C_POST = "#D62728"       # posterior mean, solid
const C_PRIOR = "#1F4E9C"      # prior mean, dotted
const C_PARTICLE = "#9A9A9A"   # prior particles
const C_MEAN = "black"         # ensemble mean over per-particle curves

args = parse_input_args(read_config("gaussian_lens_fwi.json"))
filepath = joinpath(datadir(args["sim_name"]),
                    savename(args, "jld2"; accesses = IDENTITY_KEYS))
isfile(filepath) ||
    error("no cached run at $filepath -- run scripts/gaussian_lens_fwi.jl first")
R = wload(filepath)
println("Loaded ", filepath)

outdir = plotsdir(args["sim_name"], savename(args; accesses = IDENTITY_KEYS), "paper")
!isdir(outdir) && mkpath(outdir)

# --------------------------------------------------------------------------
# Publication rc. Type-42 fonts embed the glyphs so the PDF is editable and
# passes journal preflight; unicode_minus off avoids a glyph most serif fonts
# lack. Set before any figure is created.
# --------------------------------------------------------------------------
matplotlib.use("Agg")
rc("font", family = "serif", size = 9)
rc("mathtext", fontset = "cm")
rc("pdf", fonttype = 42)
rc("ps", fonttype = 42)
rc("axes", unicode_minus = false, linewidth = 0.7, grid = true)
rc("grid", linestyle = ":", linewidth = 0.4, alpha = 0.6, color = "0.5")
rc("xtick", direction = "in", top = true)
rc("ytick", direction = "in", right = true)
rc("legend", frameon = true, framealpha = 0.9, edgecolor = "0.7", fontsize = 8)

"""White to cyan to blue to black, matching the density ramp of the paper."""
const CMAP_PDF = matplotlib.colors.LinearSegmentedColormap.from_list(
    "pdf_ramp", [(1.0, 1.0, 1.0), (0.65, 0.95, 0.98), (0.10, 0.55, 0.85),
                 (0.03, 0.12, 0.45), (0.0, 0.0, 0.0)])

# Figures the README embeds also get a PNG copy under figs/, since GitHub does
# not render PDF and hand-converted images would drift from the run.
const README_FIGS = ("01-true-velocity", "03-posterior-mean", "04-posterior-std",
                     "06a-profile-horizontal", "06b-profile-vertical")
const README_DIR = projectdir("figs", "gaussian-lens")

function save_pdf(fig, name)
    path = joinpath(outdir, name)
    fig.savefig(path, bbox_inches = "tight", pad_inches = 0.02)
    stem = first(splitext(name))
    if stem in README_FIGS
        !isdir(README_DIR) && mkpath(README_DIR)
        fig.savefig(joinpath(README_DIR, stem * ".png"); dpi = 200,
                    bbox_inches = "tight", pad_inches = 0.02)
    end
    close(fig)
    println("  ", path)
end

# --------------------------------------------------------------------------
# Load and reduce.
# --------------------------------------------------------------------------
n = (Int(R["nz"]), Int(R["nx"]))
hm = Float64(R["h"])
grid = PMLGrid(n, (hm, hm); pml = Int(R["pml"]))

v_true = R["v_true"]
V_post = 1 ./ sqrt.(max.(cut(grid, R["particles"]), eps()))
V_prior = 1 ./ sqrt.(max.(cut(grid, R["particles_initial"]), eps()))

v_mean = reshape(vec(mean(V_post; dims = 2)), n)
v_std = reshape(vec(std(V_post; dims = 2)), n)
v_prior_mean = reshape(vec(mean(V_prior; dims = 2)), n)

Lx = n[2] * hm / 1000
Lz = n[1] * hm / 1000
extent = [0.0, Lx, Lz, 0.0]
Np = size(V_post, 2)
n_iter = length(R["bandwidth"])
iters = collect(1:n_iter)

# Acquisition, rebuilt from the config so the setup figure shows the true
# geometry rather than a redrawing of it.
acq = transmission_acquisition(grid; n_sources = Int(R["n_sources"]),
                               receiver_stride = Int(R["receiver_stride"]))

"""Map panel with the paper's axis convention: distance on top, depth down."""
function map_axes(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.grid(false)
    return ax
end

function velocity_map(field, name; cmap = "jet", label = "(km/s)",
                      vmin = nothing, vmax = nothing)
    fig = figure(figsize = (3.1, 3.1))
    ax = map_axes(fig)
    im = ax.imshow(field; extent = extent, cmap = cmap, aspect = "equal",
                   vmin = vmin, vmax = vmax, interpolation = "bilinear")
    cb = fig.colorbar(im, ax = ax, orientation = "horizontal", pad = 0.10,
                      fraction = 0.05, shrink = 0.92)
    cb.set_label(label)
    cb.ax.tick_params(labelsize = 8)
    save_pdf(fig, name)
end

println("Writing figures:")

# --- 1. True model with the acquisition geometry ---------------------------
let
    fig = figure(figsize = (3.1, 3.1))
    ax = map_axes(fig)
    im = ax.imshow(v_true; extent = extent, cmap = "jet", aspect = "equal",
                   interpolation = "bilinear")
    sx = (acq.source_x .- 1) .* hm ./ 1000
    sz = (acq.source_z .- 1) .* hm ./ 1000
    rx = (acq.receiver_x .- 1) .* hm ./ 1000
    rz = (acq.receiver_z .- 1) .* hm ./ 1000
    # No legend: it would sit on top of the model. Sources and receivers are
    # identified in the caption, as in the paper. Receivers are drawn as a line
    # with subsampled markers so that 200 of them read as an array rather than
    # as a solid black bar.
    ax.plot([0, Lx], [rz[1], rz[1]]; color = "black", lw = 0.8, zorder = 4)
    ax.scatter(rx[1:8:end], rz[1:8:end]; s = 9, c = "black", marker = "v",
               zorder = 5, clip_on = false)
    ax.scatter(sx, sz; s = 22, c = "white", marker = "*", edgecolors = "black",
               linewidths = 0.35, zorder = 6, clip_on = false)
    ax.set_xlim(0, Lx); ax.set_ylim(Lz, 0)
    cb = fig.colorbar(im, ax = ax, orientation = "horizontal", pad = 0.10,
                      fraction = 0.05, shrink = 0.92)
    cb.set_label("(km/s)")
    cb.ax.tick_params(labelsize = 8)
    save_pdf(fig, "01-true-velocity.pdf")
end

# --- 2-4. Prior mean, posterior mean, posterior standard deviation ---------
vlo, vhi = extrema(v_true)
velocity_map(v_prior_mean, "02-prior-mean.pdf"; vmin = vlo, vmax = vhi)
velocity_map(v_mean, "03-posterior-mean.pdf"; vmin = vlo, vmax = vhi)
velocity_map(v_std, "04-posterior-std.pdf"; cmap = "Reds", label = "Std. dev. (km/s)")

# --- 5. Two representative prior particles --------------------------------
for (k, j) in enumerate((1, Np ÷ 2))
    velocity_map(reshape(V_prior[:, j], n), @sprintf("05%c-prior-particle.pdf", 'a' + k - 1);
                 vmin = vlo, vmax = vhi)
end

# --------------------------------------------------------------------------
# 6. Normalized posterior density along a profile through the lens.
#
# Forty particles make a per-column histogram visibly lumpy, so the density is
# a Gaussian kernel estimate at Silverman bandwidth, each column normalized to
# its own maximum: the figure reads as a shape rather than as an amplitude that
# varies with local spread.
# --------------------------------------------------------------------------
function profile_density(orientation::Symbol, name)
    # Assigned closures, not `slice(M) = ...`: Julia lowers two same-signature
    # definitions in the two branches into one local function, so the lexically
    # last would win whichever branch ran.
    horizontal = orientation === :horizontal
    idx = clamp(round(Int, 1.0 * 1000 / hm), 1, horizontal ? n[1] : n[2])
    slice = horizontal ? (M -> vec(reshape(M, n)[idx, :])) :
                         (M -> vec(reshape(M, n)[:, idx]))
    coord = horizontal ? collect(range(0, Lx; length = n[2])) :
                         collect(range(0, Lz; length = n[1]))
    axis_label = horizontal ? "Distance (km)" : "Depth (km)"
    cut_label = @sprintf("%s = %.1f km", horizontal ? "z" : "x", (idx - 1) * hm / 1000)

    post = reduce(hcat, [slice(V_post[:, j]) for j in 1:Np])
    prior = reduce(hcat, [slice(V_prior[:, j]) for j in 1:Np])

    vgrid = collect(range(1.0, 2.6; length = 400))
    dens = zeros(length(vgrid), length(coord))
    for i in eachindex(coord)
        smp = @view post[i, :]
        bw = max(1.06 * std(smp) * Np^(-1 / 5), 0.01)
        @inbounds for (iv, vv) in enumerate(vgrid)
            acc = 0.0
            for k in 1:Np
                acc += exp(-0.5 * ((vv - smp[k]) / bw)^2)
            end
            dens[iv, i] = acc
        end
        m = maximum(@view dens[:, i])
        m > 0 && (dens[:, i] ./= m)
    end

    fig = figure(figsize = (4.6, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(dens; extent = [0, coord[end], vgrid[1], vgrid[end]],
                   origin = "lower", aspect = "auto", cmap = CMAP_PDF,
                   vmin = 0, vmax = 1)
    for j in 1:Np
        ax.plot(coord, prior[:, j]; color = C_PARTICLE, lw = 0.3, alpha = 0.75,
                zorder = 2, label = j == 1 ? "Prior particles" : "")
    end
    ax.plot(coord, slice(vec(mean(V_prior; dims = 2))); color = C_PRIOR, lw = 1.3,
            ls = ":", zorder = 3, label = "Prior mean")
    ax.plot(coord, slice(vec(v_true)); color = C_TRUTH, lw = 1.8, ls = "--",
            zorder = 4, label = "Ground truth")
    ax.plot(coord, slice(vec(v_mean)); color = C_POST, lw = 1.6, zorder = 5,
            label = "Posterior mean")
    ax.set_xlabel(axis_label); ax.set_ylabel("Velocity (km/s)")
    ax.set_xlim(0, coord[end]); ax.set_ylim(1.0, 2.6)
    ax.text(0.03, 0.93, cut_label; transform = ax.transAxes, fontsize = 8, va = "top")
    ax.legend(loc = "lower right", fontsize = 7)
    ax.grid(false)
    cb = fig.colorbar(im, ax = ax, pad = 0.02, fraction = 0.045)
    cb.set_label("Normalized PDF")
    cb.ax.tick_params(labelsize = 8)
    save_pdf(fig, name)
end

profile_density(:horizontal, "06a-profile-horizontal.pdf")
profile_density(:vertical, "06b-profile-vertical.pdf")

# --------------------------------------------------------------------------
# 7-8. Ensemble convergence: model error and normalized data residual.
# --------------------------------------------------------------------------
function ensemble_curve(Y, ylabel, name; logy = false, normalize = false)
    Ym = vec(mean(Y; dims = 1))
    scale = normalize ? Ym[1] : 1.0
    fig = figure(figsize = (3.4, 2.5))
    ax = fig.add_subplot(1, 1, 1)
    plotf = logy ? ax.semilogy : ax.plot
    for j in axes(Y, 1)
        plotf(iters, Vector{Float64}(Y[j, :] ./ scale); lw = 0.35, alpha = 0.45,
              color = get_cmap("jet")((j - 1) / max(size(Y, 1) - 1, 1)), zorder = 2)
    end
    plotf(iters, Vector{Float64}(Ym ./ scale); lw = 1.8, color = C_MEAN, ls = "--",
          zorder = 4, label = "Ensemble mean")
    ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
    ax.set_xlim(1, n_iter)
    ax.legend(loc = "best", fontsize = 7)
    # Particle-number colorbar, so the per-particle curves can be read as an
    # ordered ensemble rather than as undifferentiated spaghetti.
    sm = matplotlib.cm.ScalarMappable(
        cmap = "jet", norm = matplotlib.colors.Normalize(vmin = 1, vmax = size(Y, 1)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax = ax, pad = 0.02, fraction = 0.04)
    cb.set_label("Particle no.", fontsize = 8)
    cb.ax.tick_params(labelsize = 7)
    save_pdf(fig, name)
end

ensemble_curve(R["model_error"], "RME (%)", "07-model-error.pdf")
ensemble_curve(R["misfit"], "Data residual (normalized)", "08-data-residual.pdf";
               normalize = true)
ensemble_curve(R["constraint_residual"], L"$\|A(m)u-b\|_F$",
               "09-constraint-residual.pdf"; logy = true)
ensemble_curve(R["multiplier_norm"], L"$\|\varepsilon\|_F$",
               "10-multiplier-norm.pdf"; logy = true)

# --------------------------------------------------------------------------
# 11. Posterior-mean model error, the estimator the method actually reports.
# --------------------------------------------------------------------------
let
    fig = figure(figsize = (3.4, 2.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iters, Vector{Float64}(R["mean_model_error"]); lw = 1.5,
            color = C_POST, label = "Per-particle mean")
    ax.axhline(relative_model_error(grid, vec(mean(R["particles"]; dims = 2)),
                                    R["m_true"]);
               color = C_MEAN, ls = "--", lw = 1.3, label = "Posterior mean, final")
    ax.axhline(relative_model_error(grid, vec(mean(R["particles_initial"]; dims = 2)),
                                    R["m_true"]);
               color = C_PRIOR, ls = ":", lw = 1.3, label = "Prior mean")
    ax.set_xlabel("Iteration"); ax.set_ylabel("RME (%)")
    ax.set_xlim(1, n_iter); ax.legend(loc = "best", fontsize = 7)
    save_pdf(fig, "11-model-error-summary.pdf")
end

println("\n", length(readdir(outdir)), " figures in ", outdir)
