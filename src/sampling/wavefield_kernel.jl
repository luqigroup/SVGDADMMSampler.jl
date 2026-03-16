# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Mar 2026
# Wavefield kernel for SVGD: K_w(m,m') = exp(-d_w(m,m')/2h²)
# where d_w = ||F(m) - F(m')||² for a forward map F.

export compute_wavefield_bandwidth, svgd_update_wavefield!

using Statistics
using LinearAlgebra

"""
    compute_wavefield_bandwidth(F_particles)

Median heuristic bandwidth in wavefield space.
F_particles: (n_fwd, n_particles) forward map values at each particle.
"""
function compute_wavefield_bandwidth(F_particles::Matrix{Float32})
    n_fwd, n_particles = size(F_particles)
    if n_particles < 2
        return 1.0f0
    end

    # Pairwise wavefield squared distances
    dists_sq = zeros(Float32, n_particles, n_particles)
    for d in 1:n_fwd
        diff = F_particles[d, :]' .- F_particles[d, :]
        dists_sq .+= diff .^ 2
    end

    dists = sqrt.(dists_sq[triu(trues(size(dists_sq)), 1)])
    h = Float32(median(dists) / sqrt(2 * log(n_particles)))
    return max(h, 1f-6)
end

"""
    svgd_update_wavefield!(particles, gradients, F_particles, jt_product, η, h)

SVGD update with wavefield kernel K_w(m,m') = exp(-||F(m)-F(m')||²/2h²).

Uses the convention from the wavefield kernel paper: the kernel gradient
for the repulsion term is ∇_{m^(j)} K_w(m^(j), m^(ℓ)), evaluated at the
particle being updated. This allows factoring J_F(m^(j)) out of the sum
over ℓ (in FWI, this enables LU reuse).

Arguments:
- particles: (n_dim, n_particles)
- gradients: (n_dim, n_particles) ∇log p at each particle
- F_particles: (n_fwd, n_particles) forward map values F(m^(j))
- jt_product: function(particles, V) -> (n_dim, n_particles)
    Computes J_F(m^(j))^T * V[:, j] for each particle j.
    V is (n_fwd, n_particles).
- η: step size
- h: bandwidth
"""
function svgd_update_wavefield!(particles::Matrix{Float32},
                                 gradients::Matrix{Float32},
                                 F_particles::Matrix{Float32},
                                 jt_product::Function,
                                 η::Float32, h::Float32)
    n_dim, n_particles = size(particles)
    n_fwd = size(F_particles, 1)

    # 1. Pairwise wavefield squared distances
    dists_sq = zeros(Float32, n_particles, n_particles)
    for d in 1:n_fwd
        diff = F_particles[d, :]' .- F_particles[d, :]
        dists_sq .+= diff .^ 2
    end

    # 2. Wavefield kernel matrix
    K = exp.(-dists_sq ./ (2 * h^2))

    # 3. Attraction: term1[:, j] = Σ_ℓ K[j,ℓ] * gradients[:, ℓ]
    term1 = gradients * K'

    # 4. Repulsion via ∇_{m^(j)} K_w(m^(j), m^(ℓ))
    #    = K_w^(jℓ)/h² * J_F(m^(j))^T * (F(m^(ℓ)) - F(m^(j)))
    #
    #    Summing over ℓ, J_F(m^(j)) is constant so:
    #    Σ_ℓ ∇_{m^(j)} K_w^(jℓ) = (1/h²) * J_F(m^(j))^T * w_j
    #    where w_j = Σ_ℓ K_w^(jℓ) * (F(m^(ℓ)) - F(m^(j)))
    #
    #    This is the kernel-weighted wavefield difference (û in the paper).

    # Compute w_j = Σ_ℓ K[j,ℓ] * (F[:, ℓ] - F[:, j])
    w = zeros(Float32, n_fwd, n_particles)
    for d in 1:n_fwd
        diff = F_particles[d, :]' .- F_particles[d, :]  # diff[j,ℓ] = F[d,ℓ] - F[d,j]
        w[d, :] = sum(K .* diff, dims=2)[:]
    end

    # Apply J_F(m^(j))^T to w_j for each j
    term2 = jt_product(particles, w) ./ h^2

    # 5. SVGD direction: φ = (term1 - term2) / n_particles
    #    (term2 = Σ ∇_{m^(j)} K, and SVGD subtracts it, same as standard code)
    phi = (term1 .- term2) ./ n_particles

    # 6. Update particles
    particles .+= η .* phi

    return nothing
end
