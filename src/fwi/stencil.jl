# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Optimal rotated 9-point stencil coefficients (Chen, Cheng, Feng & Wu 2013,
# rule 3.4).
#
# The scheme blends the axis-aligned and 45-degree Laplacians with weights `b`
# and `1 - b`, and replaces the lumped mass by an anti-lumped average with
# weights `(c, d, e)` summing to one. The four numbers are fitted to the
# discrete dispersion relation over the propagation angles and grid densities a
# run actually spans, subject to `-1 <= b, d, e <= 1`.
#
# That bound-constrained least squares is a convex QP in three unknowns, so its
# minimizer lies on a face of the box. Enumerating all 27 faces and keeping the
# feasible one with the smallest residual solves it exactly, with no optimizer
# dependency (the reference calls MATLAB `lsqlin`).

export NinePointStencil, optimal_stencil, standard_stencil, grid_density_range

"""
    NinePointStencil(b, c, d, e)

Coefficients of the optimal rotated 9-point Helmholtz stencil.

  * `b`   — weight on the axis-aligned Laplacian; `1 - b` goes to the rotated one.
  * `c`, `d`, `e` — anti-lumped mass weights on the centre, the four axial
    neighbours, and the four diagonal neighbours. They satisfy `c + d + e = 1`,
    which is what makes the mass operator consistent (it reproduces constants).
"""
struct NinePointStencil
    b::Float64
    c::Float64
    d::Float64
    e::Float64
end

function NinePointStencil(b::Real, c::Real, d::Real, e::Real)
    s = c + d + e
    isapprox(s, 1.0; atol = 1e-8) ||
        throw(ArgumentError("mass weights must satisfy c + d + e == 1, got $s"))
    return NinePointStencil(Float64(b), Float64(c), Float64(d), Float64(e))
end

"""
    standard_stencil()

The plain, unoptimized five-point scheme (`b = 1`, lumped mass). Useful as a
reference in tests and when a dispersion-optimized stencil is not wanted.
"""
standard_stencil() = NinePointStencil(1.0, 1.0, 0.0, 0.0)

Base.show(io::IO, s::NinePointStencil) = print(
    io, "NinePointStencil(b=", round(s.b; digits = 6), ", c=", round(s.c; digits = 6),
    ", d=", round(s.d; digits = 6), ", e=", round(s.e; digits = 6), ")")

"""
    grid_density_range(v_min, v_max, h, f_min, f_max)

Points per wavelength `G = v / (h f)` at the two extremes of a run: the
smallest density (slowest medium, highest frequency) and the largest (fastest
medium, lowest frequency). Velocities are in km/s and `h` in metres, so the
factor of 1000 converts to a consistent length unit.

Returns `(G_min, G_max)`.
"""
function grid_density_range(v_min::Real, v_max::Real, h::Real,
                            f_min::Real, f_max::Real)
    (v_min > 0 && v_max > 0) || throw(ArgumentError("velocities must be positive"))
    (f_min > 0 && f_max > 0) || throw(ArgumentError("frequencies must be positive"))
    h > 0 || throw(ArgumentError("grid spacing must be positive"))
    return (1000 * v_min / (h * f_max), 1000 * v_max / (h * f_min))
end

"""
    optimal_stencil(G_min, G_max; n_theta, n_density)

Fit `(b, d, e)` to the discrete dispersion relation sampled over propagation
angles in `[0, pi/4]` and grid densities in `[G_min, G_max]`, subject to
`-1 <= b, d, e <= 1`.

When `G_min == G_max` (a single frequency and a homogeneous medium) the density
axis collapses to a point and the angular sampling is refined instead, matching
the reference implementation's `l = 500, r = 1` versus `l = 10, r = 100` split.
"""
function optimal_stencil(G_min::Real, G_max::Real;
                         n_theta::Union{Nothing,Int} = nothing,
                         n_density::Union{Nothing,Int} = nothing)
    (G_min > 0 && G_max > 0) ||
        throw(ArgumentError("grid densities must be positive, got ($G_min, $G_max)"))
    G_min ≤ G_max ||
        throw(ArgumentError("G_min must not exceed G_max, got ($G_min, $G_max)"))

    degenerate = G_min ≈ G_max
    nθ = something(n_theta, degenerate ? 500 : 10)
    nG = something(n_density, degenerate ? 1 : 100)
    nθ ≥ 2 || throw(ArgumentError("need at least two angles, got $nθ"))

    θ = range(0, π / 4; length = nθ)
    # Sample uniformly in inverse density: that is the variable the dispersion
    # error is smooth in, and it is what the reference sweep uses.
    invG = nG == 1 ? [1 / G_max] :
           collect(range(1 / G_max, 1 / G_min; length = nG))

    rows = nθ * nG
    A = Matrix{Float64}(undef, rows, 3)
    y = Vector{Float64}(undef, rows)

    k = 0
    @inbounds for iG in 1:nG
        G = 1 / invG[iG]
        w = 2π / G
        for iθ in 1:nθ
            ct, st = cos(θ[iθ]), sin(θ[iθ])
            P = cos(w * ct)
            Q = cos(w * st)
            k += 1
            A[k, 1] = 2 * G^2 * (1 - P - Q + P * Q)
            A[k, 2] = π^2 * (2 - P - Q)
            A[k, 3] = 2 * π^2 * (1 - P * Q)
            y[k] = 2 * π^2 + G^2 * (2 * P * Q - P - Q)
        end
    end

    b, d, e = _bounded_lsq3(A, y, -1.0, 1.0)
    return NinePointStencil(b, 1 - d - e, d, e)
end

"""
    optimal_stencil(grid, velocities, frequencies)

Convenience wrapper: derive the density range from a velocity model (km/s) and
the frequencies (Hz) actually used, then fit the stencil.
"""
function optimal_stencil(g::PMLGrid, v::AbstractArray, f::AbstractVector)
    G_min, G_max = grid_density_range(minimum(v), maximum(v), g.h[1],
                                      minimum(f), maximum(f))
    return optimal_stencil(G_min, G_max)
end

# ---------------------------------------------------------------------------
# Exact bound-constrained least squares in three variables.
#
# minimize ||A x - y||_2^2  subject to  lo <= x_i <= hi.
#
# Convex QP over a box: the minimizer lies on a face of the box, and on that
# face it is the unconstrained minimizer over the free coordinates. Enumerate
# all 3^3 assignments of {free, at lo, at hi}, solve the reduced normal
# equations for the free block, discard candidates that violate the bounds, and
# keep the smallest residual. Exact, allocation-light, and dependency-free.
# ---------------------------------------------------------------------------
function _bounded_lsq3(A::AbstractMatrix, y::AbstractVector, lo::Float64, hi::Float64)
    size(A, 2) == 3 || throw(ArgumentError("expected three unknowns, got $(size(A, 2))"))
    G = Symmetric(A' * A)          # 3x3 Gram matrix
    c = A' * y

    best_x = fill(NaN, 3)
    best_obj = Inf
    states = (:free, :lo, :hi)

    for s1 in states, s2 in states, s3 in states
        s = (s1, s2, s3)
        x = zeros(3)
        free = Int[]
        for i in 1:3
            if s[i] === :lo
                x[i] = lo
            elseif s[i] === :hi
                x[i] = hi
            else
                push!(free, i)
            end
        end

        if !isempty(free)
            # Normal equations restricted to the free block, with the fixed
            # coordinates moved to the right-hand side.
            rhs = c[free] .- (Matrix(G)[free, :] * x)
            Gff = Matrix(G)[free, free]
            local xf
            try
                xf = Gff \ rhs
            catch
                continue                     # singular face; a coarser face covers it
            end
            all(isfinite, xf) || continue
            all(v -> lo - 1e-12 ≤ v ≤ hi + 1e-12, xf) || continue
            x[free] .= clamp.(xf, lo, hi)
        end

        obj = sum(abs2, A * x .- y)
        if obj < best_obj
            best_obj = obj
            best_x = x
        end
    end

    isfinite(best_obj) ||
        error("bound-constrained least squares failed to find a feasible point")
    return best_x[1], best_x[2], best_x[3]
end
