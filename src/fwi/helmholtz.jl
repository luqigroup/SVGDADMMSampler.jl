# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Frequency-domain Helmholtz operator: optimal rotated 9-point differences with
# a perfectly matched layer,
#
#     A(m) = DEL + omega^2 C diag(m) B,
#
# with DEL the PML-stretched angle-blended Laplacian, C the diagonal of
# stretching factors, B the anti-lumped mass (Chen et al. 2013, eqs 2.3, 3.1).
#
# Only `diag(m)` depends on the model, which is why this file splits into a
# template and an assembly step: DEL, C and B are built once per (grid,
# frequency, stencil), and each particle's operator is then O(nnz) rather than
# a full re-assembly.
#
# UNITS. Velocity is km/s so m = 1/v^2 is s^2/km^2, while h is metres; the 1e-6
# in `omega2` converts km^-2 to m^-2 so mass and stiffness are commensurate.

export HelmholtzTemplate, helmholtz_template, helmholtz_operator,
       model_derivative, angular_frequency

using SparseArrays
using LinearAlgebra

"""
    HelmholtzTemplate

Model-independent part of the Helmholtz operator at one frequency, assembled
once and reused for every model.

Fields `DEL`, `B`, and `cvec` are the three cached pieces of equation (*);
`omega2` is `1e-6 (2 pi f)^2`, the unit-consistent squared angular frequency.
Build with [`helmholtz_template`](@ref) and apply with
[`helmholtz_operator`](@ref).
"""
struct HelmholtzTemplate
    grid::PMLGrid
    stencil::NinePointStencil
    freq::Float64
    omega2::Float64
    DEL::SparseMatrixCSC{ComplexF64,Int}
    B::SparseMatrixCSC{Float64,Int}
    cvec::Vector{ComplexF64}
    CB::SparseMatrixCSC{ComplexF64,Int}   # C * B, precomputed for the model derivative
end

Base.show(io::IO, t::HelmholtzTemplate) = print(
    io, "HelmholtzTemplate(", t.freq, " Hz, ", t.grid.ne[1], "x", t.grid.ne[2],
    ", nnz=", nnz(t.DEL), ")")

"""
    angular_frequency(freq)

Squared angular frequency in the operator's units, `1e-6 (2 pi f)^2`. The
factor reconciles the km/s velocity convention with the metre grid spacing.
"""
angular_frequency(freq::Real) = 1e-6 * (2π * freq)^2

"""
    helmholtz_template(grid, stencil, pml, freq)

Assemble the model-independent part of the Helmholtz operator at `freq` Hz.

This is the expensive assembly — do it once per (grid, stencil, frequency) and
reuse the result for every particle.
"""
function helmholtz_template(g::PMLGrid, stencil::NinePointStencil,
                            pml::PMLProfile, freq::Real)
    freq > 0 || throw(ArgumentError("frequency must be positive, got $freq"))
    nze, nxe = g.ne
    beta = pml_beta(pml, freq)
    γ = stretch_axes(g, beta)

    # --- Stretched second derivatives along each axis (Chen et al. eq. 3.1).
    # Row i reads (1/h^2) [ gamma^-1_{i-1/2} u_{i-1}
    #                       - (gamma^-1_{i-1/2} + gamma^-1_{i+1/2}) u_i
    #                       + gamma^-1_{i+1/2} u_{i+1} ].
    Dzz = _stretched_second_derivative(γ.z_minus, γ.z_plus, g.h[1])
    Dxx = _stretched_second_derivative(γ.x_minus, γ.x_plus, g.h[2])

    # --- Angle-blended Laplacian.
    # The axis-aligned part carries the cell-centred stretching factor of the
    # OTHER axis; the rotated part carries the same factor evaluated one node
    # over, which is what produces the 45-degree stencil arms.
    Iz = _diag_sparse(γ.z)
    Ix = _diag_sparse(γ.x)
    Zup = _shifted_diag(γ.z, nze, +1)
    Zdn = _shifted_diag(γ.z, nze, -1)
    Xup = _shifted_diag(γ.x, nxe, +1)
    Xdn = _shifted_diag(γ.x, nxe, -1)

    b = stencil.b
    w = (1 - b) / 2
    DEL = b * (kron(Dxx, Iz) + kron(Ix, Dzz)) +
          w * (kron(Dxx, Zup) + kron(Xup, Dzz)) +
          w * (kron(Dxx, Zdn) + kron(Xdn, Dzz))

    # --- Coordinate-stretching diagonal C, in column-major (z fastest) order.
    cvec = Vector{ComplexF64}(undef, nze * nxe)
    @inbounds for ix in 1:nxe
        gx = γ.x[ix]
        col = (ix - 1) * nze
        for iz in 1:nze
            cvec[col+iz] = γ.z[iz] * gx
        end
    end

    # --- Anti-lumped mass operator B = c I + d I0 + e I45 (Chen et al. eq 2.3).
    B = _mass_operator(nze, nxe, stencil)

    CB = Diagonal(cvec) * B

    return HelmholtzTemplate(g, stencil, Float64(freq), angular_frequency(freq),
                             DEL, B, cvec, CB)
end

"""
    helmholtz_operator(template, m)

Assemble `A(m) = DEL + omega^2 C diag(m) B` for the squared-slowness model `m`
given on the PML-extended grid (length `n_extended(grid)`).

Costs `O(nnz(B))` because everything except `diag(m)` is cached in `template`.
"""
function helmholtz_operator(t::HelmholtzTemplate, m::AbstractVector)
    length(m) == n_extended(t.grid) ||
        throw(DimensionMismatch("model has $(length(m)) entries, expected " *
                                "$(n_extended(t.grid)) on the extended grid"))
    scale = Diagonal(t.omega2 .* t.cvec .* m)
    return t.DEL + scale * t.B
end

(t::HelmholtzTemplate)(m::AbstractVector) = helmholtz_operator(t, m)

"""
    model_derivative(template, u)

Apply the diagonal of the operator's model derivative to a wavefield:

    model_derivative(t, u)[i] = d(A(m) u)[i] / d m[i] = omega^2 (C B u)[i].

In the frequency domain this is the second time derivative of `u` scaled by
the medium term, which is why the FWI gradient reduces to a zero-lag
correlation of this quantity with the adjoint wavefield. `u` may be a vector
or an `Ne x ns` matrix of one wavefield per source.
"""
model_derivative(t::HelmholtzTemplate, u::AbstractVecOrMat) = t.omega2 .* (t.CB * u)

# ---------------------------------------------------------------------------
# Assembly helpers. All are internal; none are exported.
# ---------------------------------------------------------------------------

# Tridiagonal stretched second derivative on one axis. `gm[i]` and `gp[i]` are
# gamma at the faces i-1/2 and i+1/2; the stencil uses their reciprocals.
function _stretched_second_derivative(gm::Vector{ComplexF64}, gp::Vector{ComplexF64},
                                      h::Float64)
    n = length(gm)
    inv_m = 1 ./ gm
    inv_p = 1 ./ gp
    inv_h2 = 1 / h^2

    I = Int[]
    J = Int[]
    V = ComplexF64[]
    sizehint!(I, 3n); sizehint!(J, 3n); sizehint!(V, 3n)

    @inbounds for i in 1:n
        if i > 1
            push!(I, i); push!(J, i - 1); push!(V, inv_m[i] * inv_h2)
        end
        push!(I, i); push!(J, i); push!(V, -(inv_m[i] + inv_p[i]) * inv_h2)
        if i < n
            push!(I, i); push!(J, i + 1); push!(V, inv_p[i] * inv_h2)
        end
    end
    return sparse(I, J, V, n, n)
end

_diag_sparse(v::Vector{ComplexF64}) = spdiagm(0 => v)

# Sparse matrix with `v` evaluated at the TARGET node on the k-th diagonal:
# entry (i, i+k) equals v[i+k]. This places the stretching factor at the node
# the rotated stencil arm reaches, not at the node it leaves.
function _shifted_diag(v::Vector{ComplexF64}, n::Int, k::Int)
    len = n - abs(k)
    len ≤ 0 && return spzeros(ComplexF64, n, n)
    I = Vector{Int}(undef, len)
    J = Vector{Int}(undef, len)
    V = Vector{ComplexF64}(undef, len)
    idx = 0
    @inbounds for i in 1:n
        j = i + k
        (1 ≤ j ≤ n) || continue
        idx += 1
        I[idx] = i; J[idx] = j; V[idx] = v[j]
    end
    return sparse(I, J, V, n, n)
end

# Shift operator on one axis: ones on the k-th diagonal.
function _shift(n::Int, k::Int)
    len = n - abs(k)
    len ≤ 0 && return spzeros(Float64, n, n)
    return spdiagm(k => ones(len))
end

# Anti-lumped mass B = c I + d I0 + e I45, with I0 the average of the four
# axial neighbours and I45 the average of the four diagonal ones.
function _mass_operator(nze::Int, nxe::Int, s::NinePointStencil)
    N = nze * nxe
    shift2(kx, kz) = kron(_shift(nxe, kx), _shift(nze, kz))

    B = s.c * sparse(1.0I, N, N)
    if s.d != 0
        I0 = 0.25 * (shift2(-1, 0) + shift2(1, 0) + shift2(0, -1) + shift2(0, 1))
        B += s.d * I0
    end
    if s.e != 0
        I45 = 0.25 * (shift2(-1, 1) + shift2(1, -1) + shift2(-1, -1) + shift2(1, 1))
        B += s.e * I45
    end
    return B
end
