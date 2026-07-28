# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Generic Stein variational gradient descent (Liu & Wang 2016),
#
#     phi(x_j) = (1/Np) sum_i [ k(x_i, x_j) score(x_i) + grad_{x_i} k(x_i, x_j) ],
#
# an attraction term toward high density plus a repulsion term that keeps
# particles apart. For the RBF kernel the gradient term is
# `k(x_i, x_j) (x_j - x_i) / h^2`.
#
# Independent of any target: it takes particles and scores, returns a
# direction. `sampling/admm_svgd.jl` keeps a separate Float32, `(d, Np)`-layout
# implementation for the Rosenbrock experiment, which is frozen against
# published figures; this one is general and uses the `(Np, d)` layout the FWI
# ensemble needs.

export median_bandwidth, stein_direction, adagrad_step!, AdaGradState

using LinearAlgebra
using Statistics

"""
    pairwise_sqdist(X)

Matrix of squared distances between the rows of `X`, computed from the Gram
matrix rather than by explicit differencing: for an ensemble of `Np` particles
in `d` dimensions this costs `O(Np^2 d)` in one matrix product instead of
`Np^2` vector subtractions, which matters when `d` is a model grid.

Small negative values from cancellation are clamped to zero, and the diagonal
is set exactly to zero.
"""
function pairwise_sqdist(X::AbstractMatrix)
    G = X * X'
    sq = diag(G)
    D2 = sq .+ sq' .- 2 .* G
    @inbounds for i in eachindex(D2)
        D2[i] = max(D2[i], zero(eltype(D2)))
    end
    @inbounds for i in axes(D2, 1)
        D2[i, i] = zero(eltype(D2))
    end
    return D2
end

"""
    median_bandwidth(D2; include_diagonal = true)

RBF bandwidth from the median heuristic,
`h = sqrt( median(D2) / (2 log(Np + 1)) )`, given the squared-distance matrix.

`include_diagonal` selects which population the median is taken over. With
`true` (the default, and the convention of the reference FWI implementation)
the `Np` zero self-distances are included; with `false` the median is taken
over the strict upper triangle only, which is the convention used elsewhere in
this package. The two differ by a bias that shrinks as `Np` grows — with `Np`
zeros among `Np^2` entries the included diagonal pulls the median down — so the
choice is explicit rather than implied.
"""
function median_bandwidth(D2::AbstractMatrix; include_diagonal::Bool = true)
    Np = size(D2, 1)
    Np < 2 && return one(eltype(D2))
    pop = include_diagonal ? vec(D2) : [D2[i, j] for j in 2:Np for i in 1:(j-1)]
    med = median(pop)
    med ≤ 0 && return eps(float(eltype(D2)))^(1 / 4)   # degenerate ensemble
    return sqrt(med / (2 * log(Np + 1)))
end

"""
    stein_direction(X, scores; h = nothing, repulsion = 1.0,
                    include_diagonal = true)

Stein direction for an ensemble laid out with one particle per ROW.

  * `X` — `Np x d` particle matrix.
  * `scores` — `Np x d` matrix of `grad log p` at each particle.
  * `h` — kernel bandwidth; recomputed by the median heuristic when omitted.
  * `repulsion` — weight on the kernel-gradient term. The Stein direction
    proper takes `1`; smaller values deliberately under-disperse the ensemble.

Returns `(phi, h)` with `phi` the `Np x d` direction.

!!! note "Repulsion strength"
    For `k(x, y) = exp(-||x - y||^2 / (2 h^2))` the kernel gradient carries
    `1 / h^2`, which is what this function uses. The reference FWI
    implementation divides by `2 h^2` instead, halving the repulsion; pass
    `repulsion = 0.5` to reproduce that behavior exactly. A weaker repulsion
    yields a narrower ensemble, so the distinction matters for any statement
    about posterior spread.
"""
function stein_direction(X::AbstractMatrix, scores::AbstractMatrix;
                         h::Union{Nothing,Real} = nothing,
                         repulsion::Real = 1.0,
                         include_diagonal::Bool = true)
    size(X) == size(scores) ||
        throw(DimensionMismatch("particles are $(size(X)) but scores are $(size(scores))"))
    Np = size(X, 1)
    Np ≥ 1 || throw(ArgumentError("need at least one particle"))

    D2 = pairwise_sqdist(X)
    hval = something(h, median_bandwidth(D2; include_diagonal = include_diagonal))
    hval > 0 || throw(ArgumentError("bandwidth must be positive, got $hval"))

    K = exp.(-D2 ./ (2 * hval^2))

    # Attraction: sum_i k(x_i, x_j) score(x_i).
    attract = K * scores

    # Repulsion: sum_i k(x_i, x_j) (x_j - x_i) / h^2, formed as
    # (rowsum(K) .* X - K * X) / h^2 so it is one matrix product, not Np^2
    # vector operations.
    sumK = vec(sum(K; dims = 2))
    repel = (sumK .* X .- K * X) ./ hval^2

    phi = (attract .+ repulsion .* repel) ./ Np
    return phi, hval
end

"""
    AdaGradState(shape)

Running average of squared Stein directions, for the optional AdaGrad-scaled
step. Construct once per run and pass to [`adagrad_step!`](@ref) on every
iteration so the history actually accumulates.

!!! note
    AdaGrad is an implementation-level step-size adaptation, not part of the
    algorithm as published, which keeps the step size fixed. It is off by
    default in the FWI driver.
"""
mutable struct AdaGradState
    history::Union{Nothing,Matrix{Float64}}
    decay::Float64
    fudge::Float64
end

AdaGradState(; decay::Real = 0.9, fudge::Real = 1e-6) =
    AdaGradState(nothing, Float64(decay), Float64(fudge))

"""
    adagrad_step!(state, phi)

Rescale a Stein direction by the running root-mean-square of its own history,
returning the adapted direction and updating `state` in place.
"""
function adagrad_step!(s::AdaGradState, phi::AbstractMatrix)
    if isnothing(s.history)
        s.history = Float64.(phi) .^ 2
    else
        s.history .= s.decay .* s.history .+ (1 - s.decay) .* (Float64.(phi) .^ 2)
    end
    return phi ./ (s.fudge .+ sqrt.(s.history))
end
