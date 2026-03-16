# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Mar 2026
# Compare standard RBF kernel vs wavefield kernel on Rosenbrock ADMM-SVGD.
#
# Forward map: F(x) = (α*x₁, x₂ - x₁²)
# Wavefield distance: d_w = α²(x₁-x₁')² + ((x₂-x₁²) - (x₂'-x₁'²))²
# Jacobian: J_F = [[α, 0], [-2x₁, 1]]
#
# The wavefield distance captures the banana geometry:
# - Along the ridge (x₂ = x₁²), d_w ≈ α²(Δx₁)²
# - Perpendicular to the ridge, d_w ≈ (Δx₂)²
# Under Born approximation: d_w ≈ δm^T H_w δm with H_w = J^T J

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using ProgressMeter
using JLD2
using PyPlot
using LinearAlgebra

# Random seed
Random.seed!(42)

# Read configuration
args = read_config("wavefield_kernel_rosenbrock.json")
args = parse_input_args(args)

println("=" ^ 60)
println("Wavefield Kernel vs RBF: Rosenbrock ADMM-SVGD")
println("=" ^ 60)
for (key, val) in args
    println("  $key: $val")
end
println("=" ^ 60)

# Rosenbrock distribution: p(x) ∝ exp[-a(x₁-μ)² - (x₂-x₁²)²]
RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

α = Float32(args["alpha"])

# ---------------------------------------------------------------
# Forward map F(x) = (α*x₁, x₂ - x₁²) and its Jacobian transpose
# ---------------------------------------------------------------
function forward_map(particles::Matrix{Float32})
    x1 = particles[1, :]
    x2 = particles[2, :]
    F = zeros(Float32, 2, size(particles, 2))
    F[1, :] = α .* x1
    F[2, :] = x2 .- x1 .^ 2
    return F
end

# J_F(x)^T * v for each particle: J^T = [[α, -2x₁], [0, 1]]
function jt_product(particles::Matrix{Float32}, V::Matrix{Float32})
    x1 = particles[1, :]
    result = zeros(Float32, 2, size(particles, 2))
    result[1, :] = α .* V[1, :] .- 2 .* x1 .* V[2, :]
    result[2, :] = V[2, :]
    return result
end

# ---------------------------------------------------------------
# ADMM functions for Rosenbrock (same for both kernels)
# Constraint: z = x₁²
# ---------------------------------------------------------------
function make_admm_fns(dist)
    a = dist.a
    μ_rb = dist.μ

    function solve_z(s)
        x1 = s.particles[1, :]
        x2 = s.particles[2, :]
        s.z .= (2 .* x2 .+ s.ε .+ s.μ .* (x1 .^ 2)) ./ (2 + s.μ)
    end

    function compute_grad(s)
        x1 = s.particles[1, :]
        x2 = s.particles[2, :]
        dL_dx1 = 2 .* a .* (x1 .- μ_rb) .+ 2 .* x1 .* (s.ε .- s.μ .* (s.z .- x1 .^ 2))
        dL_dx2 = 2 .* (x2 .- s.z)
        return vcat((-dL_dx1)', (-dL_dx2)')
    end

    function update_mult(s)
        x1 = s.particles[1, :]
        s.ε .+= s.μ * ((x1 .^ 2) .- s.z)
    end

    return solve_z, compute_grad, update_mult
end

solve_z_fn, compute_grad_fn, update_multiplier_fn = make_admm_fns(RB_dist)

# ---------------------------------------------------------------
# Step function for wavefield kernel ADMM-SVGD
# ---------------------------------------------------------------
function step_wavefield!(sampler::ADMMSVGDSampler,
                          solve_z_fn::Function,
                          compute_grad_fn::Function,
                          update_multiplier_fn::Function,
                          forward_map::Function,
                          jt_product::Function)
    # Step 1: auxiliary variables
    solve_z_fn(sampler)

    # Step 2: gradients
    grads = compute_grad_fn(sampler)

    # Step 3: compute forward map and wavefield kernel SVGD update
    F = forward_map(sampler.particles)
    svgd_update_wavefield!(sampler.particles, grads, F, jt_product,
                            sampler.η, sampler.h)

    # Step 4: multiplier update
    update_multiplier_fn(sampler)

    # Step 5: bandwidth in wavefield space
    F_new = forward_map(sampler.particles)
    sampler.h = compute_wavefield_bandwidth(F_new)

    return nothing
end

# ---------------------------------------------------------------
# Run both samplers from the same initial particles
# ---------------------------------------------------------------
n_particles = args["n_particles"]
n_iterations = args["n_iterations"]

# Shared initial particles
init_particles = randn(Float32, 2, n_particles)

function run_sampler(kernel_type::Symbol, init_particles)
    sampler = ADMMSVGDSampler(
        n_particles, 2;
        μ=Float32(args["mu_penalty"]),
        η=Float32(args["eta"]),
        h=nothing
    )
    sampler.particles .= copy(init_particles)

    # Recompute initial bandwidth for the right space
    if kernel_type == :wavefield
        F0 = forward_map(sampler.particles)
        sampler.h = compute_wavefield_bandwidth(F0)
    else
        sampler.h = SVGDADMMSampler.compute_bandwidth(sampler.particles)
    end

    history = Dict(
        "particles" => Vector{Matrix{Float32}}(),
        "mean" => Vector{Vector{Float32}}(),
        "std" => Vector{Vector{Float32}}(),
        "bandwidth" => Vector{Float32}(),
        "avg_logpdf" => Vector{Float32}(),
        "constraint_residual" => Vector{Float32}(),
        "iteration" => Vector{Int}()
    )

    prog = Progress(n_iterations; dt=1.0,
                    desc="ADMM-SVGD ($kernel_type): ")
    for iter in 1:n_iterations
        if kernel_type == :wavefield
            step_wavefield!(sampler, solve_z_fn, compute_grad_fn,
                            update_multiplier_fn, forward_map, jt_product)
        else
            step!(sampler, solve_z_fn, compute_grad_fn, update_multiplier_fn)
        end

        logpdf_vals = Rosenbrock.logpdf(RB_dist, sampler.particles)
        avg_logpdf = mean(logpdf_vals)
        x1 = sampler.particles[1, :]
        constraint_res = mean(abs.(sampler.z .- x1 .^ 2))

        next!(prog; showvalues = [
            (:iteration, iter),
            (:bandwidth_h, round(sampler.h, digits=4)),
            (:avg_logpdf, round(avg_logpdf, digits=4)),
            (:constraint_res, round(constraint_res, digits=6))
        ])

        if iter % args["save_every"] == 0
            push!(history["particles"], copy(sampler.particles))
            push!(history["mean"], mean(sampler.particles, dims=2)[:])
            push!(history["std"], std(sampler.particles, dims=2)[:])
            push!(history["bandwidth"], sampler.h)
            push!(history["avg_logpdf"], avg_logpdf)
            push!(history["constraint_residual"], constraint_res)
            push!(history["iteration"], iter)
        end
    end

    return sampler, history
end

println("\n--- Running RBF kernel ---")
sampler_rbf, history_rbf = run_sampler(:rbf, init_particles)

println("\n--- Running Wavefield kernel (α=$α) ---")
sampler_wf, history_wf = run_sampler(:wavefield, init_particles)

# ---------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------
println("\n" * "=" ^ 60)
println("Final comparison:")
println("-" ^ 60)

for (name, s) in [("RBF", sampler_rbf), ("Wavefield", sampler_wf)]
    m = mean(s.particles, dims=2)[:]
    sd = std(s.particles, dims=2)[:]
    avg_lp = mean(Rosenbrock.logpdf(RB_dist, s.particles))
    cres = mean(abs.(s.z .- s.particles[1, :] .^ 2))
    println("  $name:")
    println("    Mean:    [$(round(m[1], digits=4)), $(round(m[2], digits=4))]")
    println("    Std:     [$(round(sd[1], digits=4)), $(round(sd[2], digits=4))]")
    println("    Avg logpdf:         $(round(avg_lp, digits=4))")
    println("    Constraint residual: $(round(cres, digits=6))")
    println("    Bandwidth (h):      $(round(s.h, digits=4))")
end
println("=" ^ 60)

# ---------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------
println("\nGenerating comparison plots...")

# Generate reference samples via dense grid evaluation
x1_grid = range(-3, 3, length=200)
x2_grid = range(-2, 8, length=200)
log_density = zeros(length(x2_grid), length(x1_grid))
for (i, x1) in enumerate(x1_grid)
    for (j, x2) in enumerate(x2_grid)
        log_density[j, i] = -RB_dist.a * (x1 - RB_dist.μ)^2 - (x2 - x1^2)^2
    end
end
density = exp.(log_density .- maximum(log_density))

fig, axes = subplots(1, 3, figsize=(18, 5))

# Panel 1: RBF particles
ax = axes[1]
ax.contour(collect(x1_grid), collect(x2_grid), density, levels=6,
           colors="gray", linewidths=0.5, alpha=0.7)
ax.scatter(sampler_rbf.particles[1, 1:min(500, end)],
           sampler_rbf.particles[2, 1:min(500, end)],
           s=3, alpha=0.4, c="tab:blue")
ax.set_title("Standard RBF kernel")
ax.set_xlabel(L"$x_1$")
ax.set_ylabel(L"$x_2$")
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 8)

# Panel 2: Wavefield particles
ax = axes[2]
ax.contour(collect(x1_grid), collect(x2_grid), density, levels=6,
           colors="gray", linewidths=0.5, alpha=0.7)
ax.scatter(sampler_wf.particles[1, 1:min(500, end)],
           sampler_wf.particles[2, 1:min(500, end)],
           s=3, alpha=0.4, c="tab:orange")
ax.set_title("Wavefield kernel (α=$α)")
ax.set_xlabel(L"$x_1$")
ax.set_ylabel(L"$x_2$")
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 8)

# Panel 3: Convergence comparison (avg log-pdf)
ax = axes[3]
iters_rbf = history_rbf["iteration"]
iters_wf = history_wf["iteration"]
ax.plot(iters_rbf, history_rbf["avg_logpdf"], label="RBF", color="tab:blue")
ax.plot(iters_wf, history_wf["avg_logpdf"], label="Wavefield", color="tab:orange")
ax.set_xlabel("Iteration")
ax.set_ylabel("Avg log p(x)")
ax.set_title("Convergence")
ax.legend()

tight_layout()

# Save plot
plot_dir = plotsdir(args["sim_name"])
!isdir(plot_dir) && mkpath(plot_dir)
plot_path = joinpath(plot_dir, "comparison_alpha_$(α).png")
savefig(plot_path, dpi=150)
println("Plot saved to: $plot_path")
close(fig)

# ---------------------------------------------------------------
# Save results
# ---------------------------------------------------------------
results = @dict(
    particles_rbf=sampler_rbf.particles,
    particles_wf=sampler_wf.particles,
    history_rbf,
    history_wf,
    args,
    alpha=α
)

save_dir = datadir(args["sim_name"])
!isdir(save_dir) && mkpath(save_dir)
filename = savename(args, "jld2")
filepath = joinpath(save_dir, filename)
@tagsave(filepath, results)
println("Results saved to: $filepath")
