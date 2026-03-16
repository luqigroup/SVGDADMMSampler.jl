# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Mar 2026
# 1D Helmholtz forward modeling and SVGD for FWI toy problem.
# Particles are velocities (km/s). Helmholtz uses m = 1/v².
# Reduced-space SVGD (adjoint-state gradient, no ADMM splitting).

export Helmholtz1D, HelmholtzSVGD
export build_helmholtz, forward_solve, setup_helmholtz1d
export helmholtz_svgd_step!

using LinearAlgebra
using Statistics
using Random

const CF32 = ComplexF32

# ---------------------------------------------------------------
# Problem setup (all internal units: km, km/s, Hz)
# ---------------------------------------------------------------

struct Helmholtz1D
    N::Int
    dz::Float32                     # grid spacing (km)
    ω::Float32                      # angular frequency (rad/s)
    src_idx::Vector{Int}
    rec_idx::Vector{Int}
    b::Vector{Vector{CF32}}         # source vectors
    d_obs::Vector{Vector{CF32}}     # observed data
    σ::Float32                      # relative noise level
    v_true::Vector{Float32}         # true velocity (km/s)
    v_prior_mean::Vector{Float32}   # prior mean velocity (km/s)
    v_prior_std::Float32            # prior marginal std (km/s)
    corr_length::Float32            # correlation length (km)
    pml::Vector{CF32}
end

function build_helmholtz(v::Vector{Float32}, prob::Helmholtz1D)
    N = prob.N; dz = prob.dz; ω = prob.ω
    m = 1f0 ./ v.^2  # squared slowness
    diag_main = CF32.(ω^2 .* m) .+ ω^2 .* prob.pml .- 2f0 / dz^2
    diag_off = fill(CF32(1f0 / dz^2), N - 1)
    return Tridiagonal(diag_off, diag_main, diag_off)
end

function forward_solve(v::Vector{Float32}, prob::Helmholtz1D)
    A = build_helmholtz(v, prob)
    F = lu(A)
    return [F \ bi for bi in prob.b]
end

function adjoint_solve(A_lu, rhs::Vector{CF32})
    return conj(A_lu \ conj(rhs))
end

function make_pml(N::Int, dz::Float32, n_pml::Int, ω::Float32)
    pml = zeros(CF32, N)
    for i in 1:n_pml
        s = Float32((n_pml - i + 1) / n_pml)^2
        pml[i] += im * 5f0 * s / ω
        pml[N - i + 1] += im * 5f0 * s / ω
    end
    return pml
end

function setup_helmholtz1d(;
    N::Int = 100, L::Float32 = 1f0, freq::Float32 = 5f0,
    n_src::Int = 3, n_rec::Int = 20, n_pml::Int = 10,
    σ::Float32 = 0.05f0, v_bg::Float32 = 2f0,
    anomaly_center::Float32 = 0.5f0, anomaly_width::Float32 = 0.1f0,
    anomaly_dv::Float32 = -0.3f0,
    corr_length::Float32 = 0.1f0,
    v_prior_std::Float32 = 0.15f0,
    seed::Int = 42
)
    rng = MersenneTwister(seed)
    dz = L / (N - 1)
    ω = 2f0 * Float32(π) * freq
    z = range(0f0, L, length=N) |> collect

    # True velocity: background + Gaussian anomaly
    v_true = fill(v_bg, N)
    z_center = anomaly_center * L
    for i in 1:N
        v_true[i] += anomaly_dv * exp(-0.5f0 * ((z[i] - z_center) / (anomaly_width * L))^2)
    end

    v_prior_mean = fill(v_bg, N)
    pml = make_pml(N, dz, n_pml, ω)

    # Sources near surface
    src_idx = [max(n_pml + 1, round(Int, n_pml + 2 + (i - 1) * 3)) for i in 1:n_src]
    b = Vector{Vector{CF32}}(undef, n_src)
    for i in 1:n_src
        bi = zeros(CF32, N)
        bi[src_idx[i]] = CF32(1f0 / dz)
        b[i] = bi
    end

    # Receivers
    rec_start = n_pml + 1
    rec_end = N - n_pml
    rec_idx = round.(Int, range(rec_start, rec_end, length=n_rec)) |> collect

    # Generate data
    prob_tmp = Helmholtz1D(N, dz, ω, src_idx, rec_idx, b,
                           [zeros(CF32, n_rec) for _ in 1:n_src],
                           σ, v_true, v_prior_mean, v_prior_std,
                           corr_length, pml)
    u_true = forward_solve(v_true, prob_tmp)
    d_obs = Vector{Vector{CF32}}(undef, n_src)
    for i in 1:n_src
        d_clean = u_true[i][rec_idx]
        noise_std = σ * maximum(abs.(d_clean))
        noise = noise_std / sqrt(2f0) .* (randn(rng, Float32, n_rec) .+
                                           im .* randn(rng, Float32, n_rec))
        d_obs[i] = d_clean .+ CF32.(noise)
    end

    return Helmholtz1D(N, dz, ω, src_idx, rec_idx, b, d_obs, σ,
                       v_true, v_prior_mean, v_prior_std, corr_length, pml)
end

# ---------------------------------------------------------------
# SVGD sampler (particles = velocity in km/s)
# ---------------------------------------------------------------

mutable struct HelmholtzSVGD
    particles::Matrix{Float32}  # (N, N_p) velocities
    η::Float32
    h::Float32
end

function HelmholtzSVGD(prob::Helmholtz1D, N_p::Int; η::Float32 = 0.01f0,
                        seed::Int = 123)
    rng = MersenneTwister(seed)
    N = prob.N
    particles = zeros(Float32, N, N_p)

    # Sample from Gaussian prior with spatial correlation
    # Use simple exponential correlation in spatial domain
    for j in 1:N_p
        # Generate correlated noise via convolution with Gaussian kernel
        white = randn(rng, Float32, N)
        z = range(0f0, prob.dz * (N - 1), length=N)
        # Smoothing kernel
        n_kern = max(1, round(Int, 3 * prob.corr_length / prob.dz))
        kernel = Float32[exp(-0.5f0 * (k * prob.dz / prob.corr_length)^2)
                         for k in -n_kern:n_kern]
        kernel ./= sum(kernel)

        # Convolve (simple, padded)
        smooth = zeros(Float32, N)
        for i in 1:N
            for (ki, k) in enumerate(-n_kern:n_kern)
                idx = clamp(i + k, 1, N)
                smooth[i] += kernel[ki] * white[idx]
            end
        end
        # Scale to desired std
        smooth .*= prob.v_prior_std / std(smooth)
        particles[:, j] = prob.v_prior_mean .+ smooth
    end

    h = compute_bandwidth(particles)
    return HelmholtzSVGD(particles, η, h)
end

"""
Prior score ∇_v log p(v) for isotropic Gaussian prior on velocity.
Simple: treats each grid point independently with shared std.
"""
function prior_score_v(v::Vector{Float32}, prob::Helmholtz1D)
    return -(v .- prob.v_prior_mean) ./ prob.v_prior_std^2
end

"""
    compute_gradient_and_wavefields(v, prob)

Gradient of log-posterior w.r.t. velocity v, plus forward wavefields.
Uses chain rule: dg/dv = dg/dm * dm/dv = g_m * (-2/v³).
"""
function compute_gradient_and_wavefields(v::Vector{Float32}, prob::Helmholtz1D)
    N = prob.N
    ω = prob.ω
    N_s = length(prob.src_idx)

    A = build_helmholtz(v, prob)
    F = lu(A)

    g_m = zeros(Float32, N)       # gradient w.r.t. squared slowness
    source_energy = zeros(Float32, N)
    wavefields = Vector{Vector{CF32}}(undef, N_s)

    for i in 1:N_s
        u_i = F \ prob.b[i]
        wavefields[i] = u_i

        d_pred = u_i[prob.rec_idx]
        residual = d_pred .- prob.d_obs[i]

        # Noise variance (per-source)
        d_max = maximum(abs.(prob.d_obs[i]))
        noise_var = (prob.σ * d_max)^2

        rhs_adj = zeros(CF32, N)
        for (k, ri) in enumerate(prob.rec_idx)
            rhs_adj[ri] = residual[k] / noise_var
        end

        λ_i = adjoint_solve(F, rhs_adj)

        g_m .+= Float32.(real.(conj.(u_i) .* λ_i))
        source_energy .+= Float32.(real.(conj.(u_i) .* u_i))
    end

    # Normalized gradient w.r.t. m
    safe_energy = max.(source_energy, 1f-10)
    g_m_norm = -(1f0 / ω^2) .* g_m ./ safe_energy

    # Chain rule: dm/dv = -2/v³, so dL/dv = dL/dm * (-2/v³)
    g_v_like = g_m_norm .* (-2f0 ./ v.^3)

    # Prior score in velocity space
    g_v_prior = prior_score_v(v, prob)

    gradient = g_v_like .+ g_v_prior
    return gradient, wavefields
end

"""
    helmholtz_svgd_step!(sampler, prob; use_wavefield_kernel=false)
"""
function helmholtz_svgd_step!(sampler::HelmholtzSVGD, prob::Helmholtz1D;
                               use_wavefield_kernel::Bool = false)
    N = prob.N
    N_s = length(prob.src_idx)
    N_p = size(sampler.particles, 2)
    ω = prob.ω

    gradients = zeros(Float32, N, N_p)
    all_wavefields = Vector{Vector{Vector{CF32}}}(undef, N_p)

    for j in 1:N_p
        v_j = sampler.particles[:, j]
        grad_j, wf_j = compute_gradient_and_wavefields(v_j, prob)
        gradients[:, j] = grad_j
        all_wavefields[j] = wf_j
    end

    if use_wavefield_kernel
        # Stack real/imag of wavefields as forward map
        n_fwd = 2 * N * N_s
        F_particles = zeros(Float32, n_fwd, N_p)
        for j in 1:N_p
            idx = 1
            for i in 1:N_s
                F_particles[idx:idx+N-1, j] = Float32.(real.(all_wavefields[j][i]))
                F_particles[idx+N:idx+2N-1, j] = Float32.(imag.(all_wavefields[j][i]))
                idx += 2N
            end
        end

        # J_F^T * V: Jacobian of wavefield w.r.t. velocity
        # du/dv = du/dm * dm/dv = J_m * (-2/v³)
        # J_m = -ω² A^{-1} diag(u_i)
        # So du/dv_n = -ω² A^{-1} diag(u_i) * (-2/v_n³) e_n
        #            = 2ω² / v_n³ * A^{-1} (u_i ∘ e_n)
        # J_F^T v = Σ_i Re(J_m_i^H (v_re_i + i v_im_i)) * (-2/v³)
        function jt_product(particles, V)
            result = zeros(Float32, N, N_p)
            for j in 1:N_p
                v_j = particles[:, j]
                A_j = build_helmholtz(v_j, prob)
                F_j = lu(A_j)
                for i in 1:N_s
                    idx = (i - 1) * 2N
                    v_re = V[idx+1:idx+N, j]
                    v_im = V[idx+N+1:idx+2N, j]
                    v_complex = CF32.(v_re) .+ im .* CF32.(v_im)

                    Ainv_v = adjoint_solve(F_j, v_complex)
                    u_i = all_wavefields[j][i]
                    # J_m^H v = -ω² conj(diag(u_i)) A^{-H} v
                    jt_m = Float32.(-ω^2 .* real.(conj.(u_i) .* Ainv_v))
                    # Chain rule: dm/dv = -2/v³
                    result[:, j] .+= jt_m .* (-2f0 ./ v_j.^3)
                end
            end
            return result
        end

        h_wf = compute_wavefield_bandwidth(F_particles)
        svgd_update_wavefield!(sampler.particles, gradients, F_particles,
                                jt_product, sampler.η, h_wf)
        sampler.h = h_wf
    else
        svgd_update!(sampler.particles, gradients, sampler.η, sampler.h)
        sampler.h = compute_bandwidth(sampler.particles)
    end

    # Clamp velocities to physical range
    clamp!(sampler.particles, 0.5f0, 5f0)

    # Clip maximum per-particle step (prevents oscillation)
    # Already applied in particles update above, so clip post-hoc
    return nothing
end
