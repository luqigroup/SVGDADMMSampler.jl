# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Factorized Helmholtz solves and the reduced (data-space) operator.
#
# Everything the ADMM-SVGD iteration needs from the physics comes from ONE
# sparse LU per particle per outer iteration:
#
#   W = A^-H P^H      reduced sampling operator, Ne x nr   (nr solves)
#   Q = W^H W         data-space Gram matrix,    nr x nr
#   L = W z           adjoint wavefield,         Ne x ns
#   u = A^-1 r        forward wavefield,         Ne x ns
#
# Julia's sparse `lu` owns its permutations, so the reference's four
# hand-applied factors collapse to `F \ x` and `F' \ x`.
#
# Q is Hermitian positive semidefinite by construction. Saying so in the type
# gets the symmetric eigensolver, real eigenvalues and a unitary basis, and
# spares the penalty search from cleaning up spurious eigenvalues.
#
# MEMORY. W is the largest object in the iteration (~92 MB per particle for the
# lens model), so peak memory scales with the number of particles factorized
# CONCURRENTLY -- that is, with thread count.

export HelmholtzFactorization, factorize_helmholtz, solve_forward,
       solve_adjoint, wave_residual, reduced_operator, reduced_gram,
       gram_eigen, shifted_gram_solve

using SparseArrays
using LinearAlgebra

"""
    HelmholtzFactorization(A)

Sparse LU factorization of a Helmholtz operator, holding the operator itself
alongside the factors so that residuals `A u - b` can be formed without
re-assembly.

Build with [`factorize_helmholtz`](@ref).
"""
struct HelmholtzFactorization{TA<:AbstractMatrix,TF}
    A::TA
    F::TF
end

"""
    factorize_helmholtz(A)
    factorize_helmholtz(template, m)

Factorize the Helmholtz operator, either from an assembled matrix or directly
from a template and a model on the extended grid.

Thread safety: each call produces an independent factorization object, and
UMFPACK solves on distinct objects may run concurrently. Do not share one
`HelmholtzFactorization` across threads.
"""
factorize_helmholtz(A::AbstractMatrix) = HelmholtzFactorization(A, lu(A))

factorize_helmholtz(t::HelmholtzTemplate, m::AbstractVector) =
    factorize_helmholtz(helmholtz_operator(t, m))

Base.show(io::IO, f::HelmholtzFactorization) =
    print(io, "HelmholtzFactorization(", size(f.A, 1), "x", size(f.A, 2), ")")

"""
    solve_forward(F, b)

Solve `A u = b`. `b` may be a vector or an `Ne x k` matrix of right-hand sides.
"""
solve_forward(f::HelmholtzFactorization, b::AbstractVecOrMat) = f.F \ b

"""
    solve_adjoint(F, y)

Solve `A^H x = y`, the adjoint (conjugate-transpose) system.
"""
solve_adjoint(f::HelmholtzFactorization, y::AbstractVecOrMat) = f.F' \ y

"""
    wave_residual(F, u, b)

Wave-equation residual `A u - b`, the quantity the ADMM multiplier update
ascends on. Uses the stored operator, so it costs one sparse matrix product
and no solve.
"""
wave_residual(f::HelmholtzFactorization, u::AbstractVecOrMat, b::AbstractVecOrMat) =
    f.A * u - b

"""
    reduced_operator(F, P)

Reduced sampling operator `W = A^-H P^H`, of size `Ne x nr`.

`W^H x` applies `P A^-1 x`, i.e. models data from a source; `W y` applies
`A^-H P^H y`, i.e. back-projects a data-space vector into an adjoint
wavefield. Forming `W` costs `nr` adjoint solves and is the dominant cost of
one particle's iteration.
"""
function reduced_operator(f::HelmholtzFactorization, P::AbstractMatrix)
    size(P, 2) == size(f.A, 1) ||
        throw(DimensionMismatch("receiver operator has $(size(P, 2)) columns, " *
                                "expected $(size(f.A, 1))"))
    return solve_adjoint(f, Matrix{ComplexF64}(P'))
end

"""
    reduced_gram(W)

Data-space Gram matrix `Q = W^H W = P A^-1 A^-H P^H` from the reduced operator
`W`, typed as `Hermitian` because it is one by construction.

`W` is not blocked away here on purpose: the iteration needs it again for
`W^H b` and `W z`, so forming `Q` from a stored `W` is both the cheapest option
in solve count and the only one that avoids recomputing it. If peak memory
becomes the binding constraint, reduce the number of particles factorized
concurrently rather than recomputing `W`.
"""
reduced_gram(W::AbstractMatrix) = Hermitian(W' * W)

"""
    gram_eigen(Q)

Eigendecomposition of a Hermitian positive-semidefinite Gram matrix, returning
`(values, vectors)` with real, ascending eigenvalues and a unitary eigenvector
matrix.

Small negative eigenvalues arising from rounding are clamped to zero, which is
the mathematically correct projection for a matrix that is PSD by construction.
"""
function gram_eigen(Q::AbstractMatrix)
    H = Q isa Hermitian ? Q : Hermitian((Q + Q') / 2)
    E = eigen(H)
    λ = max.(real.(E.values), 0.0)
    return λ, E.vectors
end

"""
    shifted_gram_solve(λ, V, rhs, μ)

Solve `(Q + μ I) x = rhs` reusing the eigendecomposition `Q = V diag(λ) V^H`,
as `x = V diag(1 / (λ + μ)) V^H rhs`.

The penalty parameter is chosen by scanning many candidate values of `μ`, and
each candidate would otherwise require its own factorization of `Q + μ I`.
Reusing the single eigendecomposition already computed for the penalty search
turns that scan into a diagonal rescaling.
"""
function shifted_gram_solve(λ::AbstractVector, V::AbstractMatrix,
                            rhs::AbstractVecOrMat, μ::Real)
    μ ≥ 0 || throw(ArgumentError("shift must be non-negative, got $μ"))
    return V * ((1 ./ (λ .+ μ)) .* (V' * rhs))
end
