# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Mar 2026
# 1D Helmholtz FWI: wavefield kernel vs standard RBF kernel.
#
# True model: homogeneous v=2km/s + Gaussian low-velocity anomaly at depth.
# This is the 1D analog of the Gaussian anomaly experiment in the parent paper.

using DrWatson
@quickactivate :SVGDADMMSampler
using Random
using Statistics
using ProgressMeter
using JLD2
using PyPlot
using LinearAlgebra

Random.seed!(42)

# ---------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------
N = 50            # grid points
N_p = 30          # particles
n_iterations = 2000
η = 2f-3          # step size

println("=" ^ 60)
println("1D Helmholtz FWI: Wavefield Kernel vs RBF")
println("  N=$N grid points, N_p=$N_p particles, $n_iterations iterations")
println("=" ^ 60)

# Best settings from earlier: 5Hz, 5src, 30rec, 2% noise, broad prior
prob = setup_helmholtz1d(
    N=N, L=1f0, freq=5f0,
    n_src=5, n_rec=30, n_pml=5,
    σ=0.02f0, v_bg=2f0,
    anomaly_center=0.5f0, anomaly_width=0.15f0,
    anomaly_dv=-0.15f0, corr_length=0.1f0,
    v_prior_std=0.3f0, seed=42
)

N_s = length(prob.src_idx)
v_true = prob.v_true
z = range(0f0, prob.dz * (N - 1), length=N) |> collect

println("  Sources at indices: ", prob.src_idx)
println("  Receivers: $(length(prob.rec_idx)) from $(prob.rec_idx[1]) to $(prob.rec_idx[end])")
println("  True velocity: [$(round(minimum(v_true), digits=3)), $(round(maximum(v_true), digits=3))] km/s")

# ---------------------------------------------------------------
# Run both samplers from same initial particles
# ---------------------------------------------------------------
function run_helmholtz(kernel_type::Symbol, prob, N_p, n_iterations, η; seed=123)
    sampler = HelmholtzSVGD(prob, N_p; η=η, seed=seed)

    history = Dict(
        "mean_v" => Vector{Vector{Float32}}(),
        "std_v" => Vector{Vector{Float32}}(),
        "data_misfit" => Vector{Float32}(),
        "rme" => Vector{Float32}(),
        "iteration" => Vector{Int}()
    )

    use_wf = (kernel_type == :wavefield)
    prog = Progress(n_iterations; dt=1.0,
                    desc="SVGD ($kernel_type): ")

    for iter in 1:n_iterations
        helmholtz_svgd_step!(sampler, prob; use_wavefield_kernel=use_wf)

        if iter % 10 == 0
            mean_v = mean(sampler.particles, dims=2)[:]
            std_v = std(sampler.particles, dims=2)[:]

            # Data misfit (sample a few particles)
            misfit = 0f0
            n_eval = min(N_p, 10)
            for j in 1:n_eval
                u_j = forward_solve(sampler.particles[:, j], prob)
                for i in 1:length(prob.src_idx)
                    d_pred = u_j[i][prob.rec_idx]
                    d_max = maximum(abs.(prob.d_obs[i]))
                    misfit += sum(abs2.(prob.d_obs[i] .- d_pred)) / (prob.σ * d_max)^2
                end
            end
            misfit /= n_eval

            rme = norm(mean_v .- prob.v_true) / norm(prob.v_true) * 100

            push!(history["mean_v"], mean_v)
            push!(history["std_v"], std_v)
            push!(history["data_misfit"], misfit)
            push!(history["rme"], rme)
            push!(history["iteration"], iter)

            next!(prog; showvalues = [
                (:iteration, iter),
                (:data_misfit, round(misfit, digits=4)),
                (:rme_pct, round(rme, digits=2)),
                (:bandwidth, round(sampler.h, sigdigits=3))
            ])
        else
            next!(prog)
        end
    end

    return sampler, history
end

println("\n--- Running RBF kernel ---")
sampler_rbf, history_rbf = run_helmholtz(:rbf, prob, N_p, n_iterations, η)

println("\n--- Running Wavefield kernel ---")
sampler_wf, history_wf = run_helmholtz(:wavefield, prob, N_p, n_iterations, η)

# ---------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------
println("\nGenerating plots...")

mean_rbf = mean(sampler_rbf.particles, dims=2)[:]
std_rbf = std(sampler_rbf.particles, dims=2)[:]
mean_wf = mean(sampler_wf.particles, dims=2)[:]
std_wf = std(sampler_wf.particles, dims=2)[:]

fig, axes = subplots(2, 2, figsize=(14, 10))

# Panel 1: RBF mean ± std
ax = axes[1, 1]
ax.fill_betweenx(z, mean_rbf .- 2 .* std_rbf, mean_rbf .+ 2 .* std_rbf,
                  alpha=0.3, color="tab:blue", label="95% CI")
ax.fill_betweenx(z, mean_rbf .- std_rbf, mean_rbf .+ std_rbf,
                  alpha=0.5, color="tab:blue", label="68% CI")
ax.plot(mean_rbf, z, color="tab:blue", linewidth=2, label="Mean")
ax.plot(v_true, z, "k--", linewidth=1.5, label="True")
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Depth (km)")
ax.set_title("Standard RBF kernel")
ax.invert_yaxis()
ax.legend(fontsize=8)

# Panel 2: Wavefield mean ± std
ax = axes[1, 2]
ax.fill_betweenx(z, mean_wf .- 2 .* std_wf, mean_wf .+ 2 .* std_wf,
                  alpha=0.3, color="tab:orange", label="95% CI")
ax.fill_betweenx(z, mean_wf .- std_wf, mean_wf .+ std_wf,
                  alpha=0.5, color="tab:orange", label="68% CI")
ax.plot(mean_wf, z, color="tab:orange", linewidth=2, label="Mean")
ax.plot(v_true, z, "k--", linewidth=1.5, label="True")
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Depth (km)")
ax.set_title("Wavefield kernel")
ax.invert_yaxis()
ax.legend(fontsize=8)

# Panel 3: Pointwise std comparison
ax = axes[2, 1]
ax.plot(std_rbf, z, color="tab:blue", linewidth=2, label="RBF")
ax.plot(std_wf, z, color="tab:orange", linewidth=2, label="Wavefield")
ax.set_xlabel("Pointwise Std (km/s)")
ax.set_ylabel("Depth (km)")
ax.set_title("Uncertainty comparison")
ax.invert_yaxis()
ax.legend()

# Panel 4: Convergence
ax = axes[2, 2]
ax.semilogy(history_rbf["iteration"], history_rbf["data_misfit"],
            color="tab:blue", label="RBF")
ax.semilogy(history_wf["iteration"], history_wf["data_misfit"],
            color="tab:orange", label="Wavefield")
ax.set_xlabel("Iteration")
ax.set_ylabel("Data misfit")
ax.set_title("Convergence")
ax.legend()

tight_layout()

plot_dir = plotsdir("wavefield_kernel_helmholtz1d")
!isdir(plot_dir) && mkpath(plot_dir)
plot_path = joinpath(plot_dir, "comparison_N$(N)_Np$(N_p).png")
savefig(plot_path, dpi=150)
println("Plot saved to: $plot_path")
close(fig)

# Save results
results = @dict(
    particles_rbf=sampler_rbf.particles,
    particles_wf=sampler_wf.particles,
    history_rbf, history_wf,
    v_true=prob.v_true,
    N, N_p, n_iterations, η
)
save_dir = datadir("wavefield_kernel_helmholtz1d")
!isdir(save_dir) && mkpath(save_dir)
filepath = joinpath(save_dir, "results_N$(N)_Np$(N_p).jld2")
@tagsave(filepath, results)
println("Results saved to: $filepath")

# Upload plots
rclone_dir = "MyDropbox:wavefield-kernel/helmholtz1d/"
try
    run(`rclone copy $plot_dir $rclone_dir`)
    println("Uploaded to $rclone_dir")
catch e
    println("rclone upload skipped: $e")
end
