# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Acquisition geometry: source injection `S` (Ne x ns) and receiver sampling
# `P` (nr x Ne), so a forward experiment is `d = P * (A(m) \ (w .* S))`.
#
# Positions are given in PHYSICAL grid coordinates and shifted into the
# extended grid here, so callers never handle the PML offset. Off-grid
# positions use Hicks (2002) Kaiser-windowed sinc interpolation; placing an
# off-grid source at the nearest node instead injects an aliased wavelet.

export Acquisition, source_operator, receiver_operator, n_sources, n_receivers,
       ricker_weight

using SparseArrays

"""
    Acquisition(grid; source_z, source_x, receiver_z, receiver_x,
                hicks_radius = 4, hicks_beta = 6.31)

Source and receiver positions in physical grid coordinates (1-based, and not
necessarily integer).

`hicks_radius` and `hicks_beta` are the half-width and Kaiser shape parameter
of the windowed-sinc interpolation used for off-grid positions; they are
ignored for positions that land exactly on a node.
"""
struct Acquisition
    grid::PMLGrid
    source_z::Vector{Float64}
    source_x::Vector{Float64}
    receiver_z::Vector{Float64}
    receiver_x::Vector{Float64}
    hicks_radius::Int
    hicks_beta::Float64
end

function Acquisition(g::PMLGrid;
                     source_z::AbstractVector, source_x::AbstractVector,
                     receiver_z::AbstractVector, receiver_x::AbstractVector,
                     hicks_radius::Int = 4, hicks_beta::Real = 6.31)
    length(source_z) == length(source_x) ||
        throw(ArgumentError("source_z and source_x must have equal length, got " *
                            "$(length(source_z)) and $(length(source_x))"))
    length(receiver_z) == length(receiver_x) ||
        throw(ArgumentError("receiver_z and receiver_x must have equal length, got " *
                            "$(length(receiver_z)) and $(length(receiver_x))"))
    hicks_radius ≥ 1 || throw(ArgumentError("hicks_radius must be at least 1"))

    nz, nx = g.n
    for (zs, xs, what) in ((source_z, source_x, "source"),
                           (receiver_z, receiver_x, "receiver"))
        for (z, x) in zip(zs, xs)
            (1 ≤ z ≤ nz) ||
                throw(ArgumentError("$what depth $z outside the physical grid 1:$nz"))
            (1 ≤ x ≤ nx) ||
                throw(ArgumentError("$what offset $x outside the physical grid 1:$nx"))
        end
    end

    return Acquisition(g, collect(float.(source_z)), collect(float.(source_x)),
                       collect(float.(receiver_z)), collect(float.(receiver_x)),
                       hicks_radius, Float64(hicks_beta))
end

n_sources(a::Acquisition) = length(a.source_z)
n_receivers(a::Acquisition) = length(a.receiver_z)

Base.show(io::IO, a::Acquisition) = print(
    io, "Acquisition(", n_sources(a), " sources, ", n_receivers(a), " receivers)")

"""
    source_operator(acq)

Sparse `Ne x ns` injection operator; column `j` places source `j` on the
extended grid.
"""
function source_operator(a::Acquisition)
    return _placement_matrix(a, a.source_z, a.source_x)
end

"""
    receiver_operator(acq)

Sparse `nr x Ne` sampling operator; row `i` reads receiver `i` off the extended
grid. This is the transpose of the same placement matrix used for sources, so
that sampling and injection are exact adjoints of one another.
"""
function receiver_operator(a::Acquisition)
    return sparse(_placement_matrix(a, a.receiver_z, a.receiver_x)')
end

"""
    ricker_weight(freq, f_dominant)

Spectral weight of a Ricker-like source wavelet at `freq`, proportional to
`f^2 exp(-(f / f_dom)^2)`. Applied as a scalar multiplier on the source
operator at each inversion frequency.
"""
function ricker_weight(freq::Real, f_dominant::Real)
    f_dominant > 0 || throw(ArgumentError("dominant frequency must be positive"))
    return freq^2 * exp(-(freq / f_dominant)^2)
end

# ---------------------------------------------------------------------------
# Placement: build an Ne x k sparse matrix whose columns place each (z, x)
# position on the extended grid, using a delta for on-node positions and a
# Hicks Kaiser-windowed sinc otherwise.
# ---------------------------------------------------------------------------
function _placement_matrix(a::Acquisition, zs::Vector{Float64}, xs::Vector{Float64})
    g = a.grid
    nze, nxe = g.ne
    k = length(zs)

    I = Int[]
    J = Int[]
    V = Float64[]

    for j in 1:k
        # Shift into extended-grid coordinates.
        ze = zs[j] + g.pml_top
        xe = xs[j] + g.pml_left

        wz, iz0 = _placement_weights(ze, nze, a.hicks_radius, a.hicks_beta)
        wx, ix0 = _placement_weights(xe, nxe, a.hicks_radius, a.hicks_beta)

        for (bx, ix) in enumerate(ix0)
            wxb = wx[bx]
            wxb == 0 && continue
            col = (ix - 1) * nze
            for (bz, iz) in enumerate(iz0)
                wzb = wz[bz]
                wzb == 0 && continue
                push!(I, col + iz)
                push!(J, j)
                push!(V, wzb * wxb)
            end
        end
    end

    return sparse(I, J, V, nze * nxe, k)
end

# Weights and node indices for one axis. Integer positions collapse to a
# single unit weight; fractional ones get the Hicks stencil, clipped to the
# grid and renormalized so the total injected amplitude is preserved.
function _placement_weights(pos::Float64, n::Int, radius::Int, beta::Float64)
    nearest = round(Int, pos)
    if isapprox(pos, nearest; atol = 1e-12)
        i = clamp(nearest, 1, n)
        return ([1.0], [i])
    end

    base = floor(Int, pos)
    idx = Int[]
    w = Float64[]
    for off in (-radius+1):radius
        i = base + off
        (1 ≤ i ≤ n) || continue
        wi = _hicks_weight(i - pos, radius, beta)
        wi == 0 && continue
        push!(idx, i)
        push!(w, wi)
    end

    isempty(idx) && throw(ArgumentError("position $pos has no support on 1:$n"))
    s = sum(w)
    s == 0 && throw(ArgumentError("degenerate interpolation weights at position $pos"))
    w ./= s
    return (w, idx)
end

# Hicks (2002) monopole weight: a sinc truncated by a Kaiser window of
# half-width `radius` and shape `beta`.
function _hicks_weight(x::Float64, radius::Int, beta::Float64)
    abs(x) < radius || return 0.0
    kaiser = _besseli0(beta * sqrt(1 - (x / radius)^2)) / _besseli0(beta)
    return sinc(x) * kaiser
end

# Modified Bessel function of the first kind, order zero.
# Abramowitz & Stegun 9.8.1 / 9.8.2; relative error below 2e-7, which is far
# tighter than the accuracy needed for source-injection weights. Implemented
# here rather than pulled from SpecialFunctions to keep the dependency set of
# this package unchanged.
function _besseli0(x::Float64)
    ax = abs(x)
    if ax < 3.75
        t = (x / 3.75)^2
        return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 +
               t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    else
        t = 3.75 / ax
        poly = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 +
               t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 +
               t * (-0.01647633 + t * 0.00392377)))))))
        return exp(ax) / sqrt(ax) * poly
    end
end
