using DrWatson
@quickactivate :SVGDADMMSampler
using SVGDADMMSampler: compute_gradient_and_wavefields, compute_wavefield_bandwidth,
    build_helmholtz, adjoint_solve, CF32
using Statistics, LinearAlgebra

prob = setup_helmholtz1d(N=50, L=1f0, freq=5f0, n_src=2, n_rec=15, n_pml=5,
    σ=0.05f0, v_bg=2f0, anomaly_dv=-0.3f0, corr_length=0.1f0, v_prior_std=0.15f0)

sampler = HelmholtzSVGD(prob, 30; η=1f-2, seed=123)
N = prob.N; N_s = length(prob.src_idx); N_p = 30
ω = prob.ω

# Compute gradients and wavefields
gradients = zeros(Float32, N, N_p)
all_wf = Vector{Vector{Vector{SVGDADMMSampler.CF32}}}(undef, N_p)
for j in 1:N_p
    g, wf = compute_gradient_and_wavefields(sampler.particles[:, j], prob)
    gradients[:, j] = g
    all_wf[j] = wf
end

println("=== Gradient statistics ===")
println("  Likelihood+prior grad: mean|g|=$(round(mean(abs.(gradients)), sigdigits=3))")
println("  max|g|=$(round(maximum(abs.(gradients)), sigdigits=3))")

# --- RBF ---
h_rbf = SVGDADMMSampler.compute_bandwidth(sampler.particles)
dists_sq = zeros(Float32, N_p, N_p)
for d in 1:N
    diff = sampler.particles[d:d, :]' .- sampler.particles[d:d, :]
    dists_sq .+= diff[:,:] .^ 2
end
K_rbf = exp.(-dists_sq ./ (2 * h_rbf^2))

term1_rbf = gradients * K_rbf'
term2_rbf = zeros(Float32, N, N_p)
for d in 1:N
    diff = sampler.particles[d:d, :]' .- sampler.particles[d:d, :]
    term2_rbf[d:d, :] = sum(K_rbf .* diff[:,:], dims=2)'
end
term2_rbf ./= h_rbf^2
phi_rbf = (term1_rbf .- term2_rbf) ./ N_p

println("\n=== RBF Kernel ===")
println("  h = $(round(h_rbf, sigdigits=3))")
println("  K: min=$(round(minimum(K_rbf), digits=4)), mean_offdiag=$(round((sum(K_rbf)-N_p)/(N_p*(N_p-1)), digits=4))")
println("  |term1| (attract) = $(round(mean(abs.(term1_rbf)), sigdigits=3))")
println("  |term2| (repulse) = $(round(mean(abs.(term2_rbf)), sigdigits=3))")
println("  attract/repulse   = $(round(mean(abs.(term1_rbf))/mean(abs.(term2_rbf)), digits=2))")
println("  |phi|             = $(round(mean(abs.(phi_rbf)), sigdigits=3))")
println("  η*|phi| (step)    = $(round(0.01f0*mean(abs.(phi_rbf)), sigdigits=3))")

# --- Wavefield ---
CF = SVGDADMMSampler.CF32
n_fwd = 2 * N * N_s
F_particles = zeros(Float32, n_fwd, N_p)
for j in 1:N_p, i in 1:N_s
    idx = (i-1)*2N
    F_particles[idx+1:idx+N, j] = Float32.(real.(all_wf[j][i]))
    F_particles[idx+N+1:idx+2N, j] = Float32.(imag.(all_wf[j][i]))
end

h_wf = compute_wavefield_bandwidth(F_particles)
dists_sq_wf = zeros(Float32, N_p, N_p)
for d in 1:n_fwd
    diff = F_particles[d:d, :]' .- F_particles[d:d, :]
    dists_sq_wf .+= diff[:,:] .^ 2
end
K_wf = exp.(-dists_sq_wf ./ (2 * h_wf^2))
term1_wf = gradients * K_wf'

# w = kernel-weighted wavefield difference
w = zeros(Float32, n_fwd, N_p)
for d in 1:n_fwd
    diff = F_particles[d:d, :]' .- F_particles[d:d, :]
    w[d:d, :] = sum(K_wf .* diff[:,:], dims=2)'
end

# J^T * w (wavefield kernel repulsion)
term2_wf = zeros(Float32, N, N_p)
for j in 1:N_p
    v_j = sampler.particles[:, j]
    A_j = build_helmholtz(v_j, prob)
    F_j = lu(A_j)
    for i in 1:N_s
        idx = (i-1)*2N
        v_re = w[idx+1:idx+N, j]
        v_im = w[idx+N+1:idx+2N, j]
        v_complex = CF.(v_re) .+ im .* CF.(v_im)
        Ainv_v = SVGDADMMSampler.adjoint_solve(F_j, v_complex)
        u_i = all_wf[j][i]
        jt_m = Float32.(-ω^2 .* real.(conj.(u_i) .* Ainv_v))
        term2_wf[:, j] .+= jt_m .* (-2f0 ./ v_j.^3)
    end
end
term2_wf ./= h_wf^2
phi_wf = (term1_wf .- term2_wf) ./ N_p

println("\n=== Wavefield Kernel ===")
println("  h_wf = $(round(h_wf, sigdigits=3))")
println("  K_wf: min=$(round(minimum(K_wf), digits=4)), mean_offdiag=$(round((sum(K_wf)-N_p)/(N_p*(N_p-1)), digits=4))")
println("  |term1| (attract) = $(round(mean(abs.(term1_wf)), sigdigits=3))")
println("  |term2| (J^T*w/h²) = $(round(mean(abs.(term2_wf)), sigdigits=3))")
println("  attract/repulse   = $(round(mean(abs.(term1_wf))/mean(abs.(term2_wf)), digits=2))")
println("  |phi|             = $(round(mean(abs.(phi_wf)), sigdigits=3))")
println("  η*|phi| (step)    = $(round(0.01f0*mean(abs.(phi_wf)), sigdigits=3))")

# Distance statistics
model_dists = sqrt.(dists_sq[triu(trues(N_p, N_p), 1)])
wf_dists = sqrt.(dists_sq_wf[triu(trues(N_p, N_p), 1)])
println("\n=== Distance statistics ===")
println("  Model-space: median=$(round(median(model_dists), sigdigits=3))")
println("  Wavefield:   median=$(round(median(wf_dists), sigdigits=3))")
println("  Ratio wf/model: $(round(median(wf_dists)/median(model_dists), sigdigits=3))")

println("\n=== Recommended step sizes ===")
# Target: η*|phi| ≈ 0.001 (small fraction of v_bg=2)
target_step = 0.001f0
println("  RBF:      η = $(round(target_step / mean(abs.(phi_rbf)), sigdigits=2))")
println("  Wavefield: η = $(round(target_step / mean(abs.(phi_wf)), sigdigits=2))")
