# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# PML-extended grid, and the maps between the physical model and the
# absorbing-layer-extended model the Helmholtz operator lives on.
#
# `pad` extends by replicating boundary values outward, which keeps the medium
# constant through the PML as the absorbing boundary assumes; `cut` extracts
# the interior. So `cut(pad(m)) == m` but not the converse: the pair is a
# section/retraction, NOT adjoints. The true adjoint of `pad` is `pad_adjoint`.
#
# Both are index vectors, so applying them is a gather rather than a sparse
# matrix product.

export PMLGrid, pad, pad!, cut, cut!, pad_adjoint, physical_size,
       extended_size, n_physical, n_extended

"""
    PMLGrid(n, h; pml_top, pml_bottom, pml_left, pml_right)

Two-dimensional grid with a perfectly matched layer, in `(z, x)` order.

`n = (nz, nx)` is the physical model size and `h = (hz, hx)` the grid spacing
in metres. The four PML thicknesses are given in grid points and default to a
symmetric `pml` on every side.

Arrays are stored column-major with `z` varying fastest, i.e. linear index
`i = (ix - 1) * nz + iz`, matching the reference MATLAB layout so that the
Kronecker products in `helmholtz.jl` carry over unchanged.

!!! note "Corner convention"
    The reference MATLAB code carries the four thicknesses in a 2x2 array whose
    row/column meaning is read one way by `getA9_PML` (rows = near/far,
    columns = z/x) and the opposite way by `paddingmtx` (`[left bottom; right
    top]`). The two readings agree only when all four thicknesses are equal,
    which is the case for every published example. Here each side is named, so
    an asymmetric PML is unambiguous.
"""
struct PMLGrid
    n::Tuple{Int,Int}            # physical size (nz, nx)
    ne::Tuple{Int,Int}           # extended size (nz + top + bottom, nx + left + right)
    h::Tuple{Float64,Float64}    # spacing (hz, hx) in metres
    pml_top::Int                 # z-near
    pml_bottom::Int              # z-far
    pml_left::Int                # x-near
    pml_right::Int               # x-far
    pad_idx::Vector{Int}         # length Ne; physical index feeding each extended point
    cut_idx::Vector{Int}         # length N;  extended index of each physical point
end

function PMLGrid(n::Tuple{Int,Int}, h::Tuple{<:Real,<:Real};
                 pml::Int = 20,
                 pml_top::Int = pml, pml_bottom::Int = pml,
                 pml_left::Int = pml, pml_right::Int = pml)
    nz, nx = n
    (nz > 0 && nx > 0) ||
        throw(ArgumentError("grid size must be positive, got $n"))
    all(≥(0), (pml_top, pml_bottom, pml_left, pml_right)) ||
        throw(ArgumentError("PML thicknesses must be non-negative"))

    nze = nz + pml_top + pml_bottom
    nxe = nx + pml_left + pml_right

    # Edge-extension: clamp each extended coordinate back into the physical box.
    pad_idx = Vector{Int}(undef, nze * nxe)
    @inbounds for ixe in 1:nxe
        ix = clamp(ixe - pml_left, 1, nx)
        base = (ix - 1) * nz
        col = (ixe - 1) * nze
        for ize in 1:nze
            iz = clamp(ize - pml_top, 1, nz)
            pad_idx[col+ize] = base + iz
        end
    end

    cut_idx = Vector{Int}(undef, nz * nx)
    @inbounds for ix in 1:nx
        col = (ix + pml_left - 1) * nze
        base = (ix - 1) * nz
        for iz in 1:nz
            cut_idx[base+iz] = col + iz + pml_top
        end
    end

    return PMLGrid((nz, nx), (nze, nxe), (Float64(h[1]), Float64(h[2])),
                   pml_top, pml_bottom, pml_left, pml_right, pad_idx, cut_idx)
end

PMLGrid(nz::Int, nx::Int, h::Tuple{<:Real,<:Real}; kwargs...) =
    PMLGrid((nz, nx), h; kwargs...)

physical_size(g::PMLGrid) = g.n
extended_size(g::PMLGrid) = g.ne
n_physical(g::PMLGrid) = prod(g.n)
n_extended(g::PMLGrid) = prod(g.ne)

function Base.show(io::IO, g::PMLGrid)
    print(io, "PMLGrid(", g.n[1], "x", g.n[2], " -> ", g.ne[1], "x", g.ne[2],
          ", h=(", g.h[1], ", ", g.h[2], ") m, pml=(t", g.pml_top,
          ", b", g.pml_bottom, ", l", g.pml_left, ", r", g.pml_right, "))")
end

"""
    pad(grid, m)

Extend a physical field onto the PML grid by replicating its boundary values.
Accepts a length-`N` vector or an `N x k` matrix of `k` fields (columns are
extended independently); returns the corresponding `Ne`-row array.
"""
pad(g::PMLGrid, m::AbstractVector) = m[g.pad_idx]
pad(g::PMLGrid, M::AbstractMatrix) = M[g.pad_idx, :]

"""
    pad!(out, grid, m)

In-place [`pad`](@ref), writing into a preallocated `Ne`-element vector.
"""
function pad!(out::AbstractVector, g::PMLGrid, m::AbstractVector)
    length(out) == n_extended(g) ||
        throw(DimensionMismatch("output has $(length(out)) entries, expected $(n_extended(g))"))
    length(m) == n_physical(g) ||
        throw(DimensionMismatch("input has $(length(m)) entries, expected $(n_physical(g))"))
    @inbounds for i in eachindex(out)
        out[i] = m[g.pad_idx[i]]
    end
    return out
end

"""
    cut(grid, me)

Restrict a PML-grid field to the physical grid by extracting the interior
block. Accepts a length-`Ne` vector or an `Ne x k` matrix.
"""
cut(g::PMLGrid, me::AbstractVector) = me[g.cut_idx]
cut(g::PMLGrid, Me::AbstractMatrix) = Me[g.cut_idx, :]

"""
    cut!(out, grid, me)

In-place [`cut`](@ref), writing into a preallocated `N`-element vector.
"""
function cut!(out::AbstractVector, g::PMLGrid, me::AbstractVector)
    length(out) == n_physical(g) ||
        throw(DimensionMismatch("output has $(length(out)) entries, expected $(n_physical(g))"))
    length(me) == n_extended(g) ||
        throw(DimensionMismatch("input has $(length(me)) entries, expected $(n_extended(g))"))
    @inbounds for i in eachindex(out)
        out[i] = me[g.cut_idx[i]]
    end
    return out
end

"""
    pad_adjoint(grid, me)

Apply the true adjoint of [`pad`](@ref): scatter-add each extended entry back
onto the physical point it was replicated from, so that
`dot(pad_adjoint(g, y), x) == dot(y, pad(g, x))` for all `x`, `y`.

This is *not* what [`cut`](@ref) does, and is *not* what the reference MATLAB
gradient path applies. It is provided for callers that need a genuine
transpose — for example when verifying a gradient against a directional
derivative taken on the extended grid.
"""
function pad_adjoint(g::PMLGrid, me::AbstractVector{T}) where {T}
    out = zeros(T, n_physical(g))
    @inbounds for i in eachindex(me)
        out[g.pad_idx[i]] += me[i]
    end
    return out
end

function pad_adjoint(g::PMLGrid, Me::AbstractMatrix{T}) where {T}
    out = zeros(T, n_physical(g), size(Me, 2))
    @inbounds for j in axes(Me, 2), i in axes(Me, 1)
        out[g.pad_idx[i], j] += Me[i, j]
    end
    return out
end

"""
    grid_axes(grid)

Physical-grid coordinate vectors `(z, x)` in metres, measured from the first
grid point.
"""
grid_axes(g::PMLGrid) =
    (range(0.0; step = g.h[1], length = g.n[1]),
     range(0.0; step = g.h[2], length = g.n[2]))
