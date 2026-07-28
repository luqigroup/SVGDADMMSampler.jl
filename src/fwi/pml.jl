# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Perfectly matched layer: complex coordinate stretching for the Helmholtz
# operator. Along each axis
#
#     gamma(x) = 1 - i beta (d_near/L)^2 - i beta (d_far/L)^2,
#     beta     = 2 pi a0 f0 / f,
#
# with `d_near`/`d_far` the depths into each absorbing layer (zero outside) and
# `L` the normalizing thickness. Quadratic in depth, so damping and its first
# derivative vanish at the interface and no reflection is introduced there.
#
# Chen, Cheng, Feng & Wu (2013), IJNAM 10(2).

export PMLProfile, stretch, stretch_axis

"""
    PMLProfile(; a0 = 1.79, f0 = 1.0)

Damping parameters of the absorbing layer: `a0` is the damping constant and
`f0` the reference frequency (Hz) that scales the profile. The frequency
dependence enters through `beta = 2 pi a0 f0 / f`, so a fixed `PMLProfile`
absorbs consistently across a frequency sweep. `f0` is conventionally set near
the dominant frequency of the source wavelet.
"""
Base.@kwdef struct PMLProfile
    a0::Float64 = 1.79
    f0::Float64 = 1.0
end

"""
    pml_beta(profile, freq)

Frequency-dependent damping strength `beta = 2 pi a0 f0 / freq`.
"""
function pml_beta(p::PMLProfile, freq::Real)
    freq > 0 || throw(ArgumentError("frequency must be positive, got $freq"))
    return 2π * p.a0 * p.f0 / freq
end

"""
    stretch(coord, n_ext, npml_near, npml_far, beta)

Complex stretching function `gamma` evaluated at grid coordinate `coord`, which
may be half-integer — the operator needs `gamma` at cell centres and at cell
faces `i ± 1/2`.

`n_ext` is the extended axis length, and `npml_near`/`npml_far` the layer
thicknesses at the low and high ends. Both layers are normalized by
`max(npml_near, npml_far)`, matching the reference implementation.
"""
@inline function stretch(coord::Real, n_ext::Integer, npml_near::Integer,
                         npml_far::Integer, beta::Real)
    L = max(npml_near, npml_far)
    L == 0 && return complex(1.0, 0.0)
    Lf = float(L)

    d_near = npml_near - coord + 0.5
    d_far = coord - 0.5 - (n_ext - npml_far)

    damping = 0.0
    d_near > 0 && (damping += (d_near / Lf)^2)
    d_far > 0 && (damping += (d_far / Lf)^2)

    return complex(1.0, -beta * damping)
end

"""
    stretch_axis(n_ext, npml_near, npml_far, beta; offset = 0.0)

Vector of `gamma` values along one axis at coordinates `(1:n_ext) .+ offset`.
Use `offset = 0` for cell centres and `offset = ±1/2` for the staggered faces
that the second-derivative stencil needs.
"""
function stretch_axis(n_ext::Integer, npml_near::Integer, npml_far::Integer,
                      beta::Real; offset::Real = 0.0)
    return ComplexF64[stretch(i + offset, n_ext, npml_near, npml_far, beta)
                      for i in 1:n_ext]
end

"""
    stretch_axes(grid, beta)

Named tuple of the six stretching vectors the 9-point operator consumes:
cell-centred `z`/`x` profiles and the two staggered face profiles per axis.
"""
function stretch_axes(g::PMLGrid, beta::Real)
    nze, nxe = g.ne
    return (
        z = stretch_axis(nze, g.pml_top, g.pml_bottom, beta),
        z_minus = stretch_axis(nze, g.pml_top, g.pml_bottom, beta; offset = -0.5),
        z_plus = stretch_axis(nze, g.pml_top, g.pml_bottom, beta; offset = 0.5),
        x = stretch_axis(nxe, g.pml_left, g.pml_right, beta),
        x_minus = stretch_axis(nxe, g.pml_left, g.pml_right, beta; offset = -0.5),
        x_plus = stretch_axis(nxe, g.pml_left, g.pml_right, beta; offset = 0.5),
    )
end
