# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# ADMM-SVGD for constrained Bayesian FWI (Algorithm 2 of the paper). The wave
# equation A(m)u = b is relaxed through an augmented Lagrangian, so a
# dual-space posterior can be sampled with SVGD instead of the PDE being
# enforced exactly at every step. Each outer iteration is four blocks:
#
#   1. AUXILIARY.   Per particle, solve for the adjoint wavefield lambda and
#      the forward wavefield u given the current multiplier. The penalty is
#      chosen per particle per iteration by the residual whiteness principle.
#   2. GRADIENT.    Read off the dual iteration as a zero-lag correlation of u
#      and lambda -- no separate adjoint-state solve. Preconditioned by the
#      diagonal pseudo-Hessian and clipped so no step can leave the box.
#   3. SVGD.        Particles move along the Stein direction.
#   4. MULTIPLIERS. Relaxed dual ascent on the wave-equation residual,
#      progressively tightening the constraint.
#
# Blocks 1, 2 and 4 are per-particle and independent, and are threaded. The
# unit of work is one sparse LU plus `nr` adjoint solves, which dominates
# everything else; see the memory note on `run_admm_svgd_fwi`.

export FWIPosterior, ADMMSVGDFWIResult, run_admm_svgd_fwi, relative_model_error

using LinearAlgebra
using SparseArrays
using Statistics
using Printf

"""
    FWIPosterior(grid, template, acquisition, source, receiver, data, box;
                 prior = nothing)

Everything the sampler needs about one inverse problem at one frequency: the
discretization (`grid`, `template`), the geometry (`acquisition`, and the
assembled `source`/`receiver` operators), the observed `data` (`nr x ns`), and
the box constraint on squared slowness.

`prior` is an optional [`GaussianRandomField`](@ref) on the extended grid whose
score is added to the likelihood gradient. It defaults to `nothing`, matching
the reference implementation, in which the prior enters only through the
initial ensemble and the box.
"""
struct FWIPosterior{TP,TS}
    grid::PMLGrid
    template::HelmholtzTemplate
    source::TS                       # Ne x ns, already weighted by the wavelet
    receiver::TP                     # nr x Ne
    data::Matrix{ComplexF64}         # nr x ns
    box::Tuple{Float64,Float64}
    prior::Union{Nothing,GaussianRandomField}
end

function FWIPosterior(grid::PMLGrid, template::HelmholtzTemplate,
                      source::AbstractMatrix, receiver::AbstractMatrix,
                      data::AbstractMatrix, box::Tuple{<:Real,<:Real};
                      prior::Union{Nothing,GaussianRandomField} = nothing)
    size(data, 1) == size(receiver, 1) ||
        throw(DimensionMismatch("data has $(size(data, 1)) rows but there are " *
                                "$(size(receiver, 1)) receivers"))
    size(data, 2) == size(source, 2) ||
        throw(DimensionMismatch("data has $(size(data, 2)) columns but there are " *
                                "$(size(source, 2)) sources"))
    box[1] < box[2] || throw(ArgumentError("box must satisfy lo < hi, got $box"))
    if !isnothing(prior)
        size(prior) == extended_size(grid) ||
            throw(DimensionMismatch("prior is $(size(prior)) but the extended grid " *
                                    "is $(extended_size(grid))"))
    end
    return FWIPosterior(grid, template, source, receiver,
                        Matrix{ComplexF64}(data), (Float64(box[1]), Float64(box[2])),
                        prior)
end

n_sources(p::FWIPosterior) = size(p.source, 2)
n_receivers(p::FWIPosterior) = size(p.receiver, 1)

"""
    ADMMSVGDFWIResult

Final ensemble and per-iteration diagnostics of a run.

`particles` is `Ne x Np` on the extended grid. The histories are indexed by
outer iteration; `misfit` and `model_error` are per particle, the rest are
ensemble scalars.
"""
struct ADMMSVGDFWIResult
    particles::Matrix{Float64}
    particles_initial::Matrix{Float64}
    misfit::Matrix{Float64}              # Np x n_iter
    model_error::Matrix{Float64}         # Np x n_iter, percent
    constraint_residual::Matrix{Float64} # Np x n_iter, ||A u - b||
    multiplier_norm::Matrix{Float64}     # Np x n_iter
    penalty::Matrix{Float64}             # Np x n_iter, the RWP-selected mu
    bandwidth::Vector{Float64}
    ensemble_std::Vector{Float64}
    mean_model_error::Vector{Float64}
end

"""
    relative_model_error(grid, m, m_true)

Model error in percent on the PHYSICAL grid, `100 ||cut(m - m_true)|| /
||cut(m_true)||`. Restricting to the physical grid keeps the replicated PML
values, which carry no independent information, from inflating or deflating
the number.
"""
function relative_model_error(g::PMLGrid, m::AbstractVector, m_true::AbstractVector)
    return 100 * norm(cut(g, m) - cut(g, m_true)) / norm(cut(g, m_true))
end

# Per-particle physics: one factorization, the reduced operator, the RWP
# penalty, and the two wavefields. Returns the physical-grid gradient of the
# log likelihood together with the diagnostics the caller records.
function _particle_gradient(p::FWIPosterior, m::AbstractVector,
                            eps_j::AbstractMatrix, use_admm::Bool,
                            rwp_scan::Int)
    F = factorize_helmholtz(p.template, m)
    W = reduced_operator(F, p.receiver)

    Q = reduced_gram(W)
    delta_d = p.data - W' * p.source
    misfit = norm(delta_d)
    use_admm && (delta_d -= W' * eps_j)

    λ, V = gram_eigen(Q)
    μ = rwp_penalty(λ, V, delta_d; n_scan = rwp_scan)

    # Adjoint wavefield (paper eq. 21). The eigendecomposition already computed
    # for the penalty search is reused for the shifted solve.
    lambda = W * shifted_gram_solve(λ, V, delta_d, μ)

    # Forward wavefield (paper eq. 22).
    rhs = use_admm ? p.source + lambda + eps_j : p.source + lambda
    u = solve_forward(F, rhs)

    # Gradient (paper eq. 24): zero-lag correlation of the model derivative of
    # the wavefield with the adjoint wavefield, preconditioned by the diagonal
    # pseudo-Hessian sum_s |u_tt|^2. The correlation is summed over sources.
    u_tt = model_derivative(p.template, u)
    num = cut(p.grid, real.(vec(sum(conj.(u_tt) .* lambda; dims = 2))))
    den = cut(p.grid, real.(vec(sum(abs2, u_tt; dims = 2))))
    grad = -num ./ den

    return (grad = grad, u = u, misfit = misfit, penalty = μ, factorization = F)
end

"""
    run_admm_svgd_fwi(posterior, particles0; n_iterations, m_true = nothing,
                      method = :admm, repulsion = 1.0, step_scale = 1.0,
                      relaxation_offset = 5, adagrad = false, rwp_scan = 48,
                      threaded = true, verbose = true)

Run Algorithm 2 and return an [`ADMMSVGDFWIResult`](@ref).

  * `particles0` — `Ne x Np` initial ensemble on the extended grid.
  * `method` — `:admm` (multiplier updates on), `:penalty` (penalty term but no
    multiplier), or `:reduced` (neither; standard reduced-space SVGD).
  * `repulsion` — weight on the Stein repulsion term; see [`stein_direction`](@ref).
  * `step_scale` — multiplies the per-particle adaptive step size, which is set
    from the ratio of the gradient norm to the Stein-direction norm so that
    particles with weak local gradients are not stalled by the repulsion.
  * `relaxation_offset` — the `c` in the dual relaxation `k / (k + c)`, which
    ramps the multiplier update toward full strength.
  * `m_true` — reference model for the model-error diagnostic; optional.

!!! note "Memory and threads"
    Each concurrently processed particle holds a reduced operator of size
    `Ne x nr` in complex double precision, plus its LU factors. For the lens
    model that is roughly 90 MB per particle before the factorization, so peak
    memory scales with the thread count, not the ensemble size. Reduce
    `JULIA_NUM_THREADS` rather than the ensemble if memory binds.
"""
function run_admm_svgd_fwi(p::FWIPosterior, particles0::AbstractMatrix;
                           n_iterations::Int,
                           m_true::Union{Nothing,AbstractVector} = nothing,
                           method::Symbol = :admm,
                           repulsion::Real = 1.0,
                           step_scale::Real = 1.0,
                           relaxation_offset::Real = 5,
                           adagrad::Bool = false,
                           rwp_scan::Int = 48,
                           threaded::Bool = true,
                           verbose::Bool = true)
    method in (:admm, :penalty, :reduced) ||
        throw(ArgumentError("method must be :admm, :penalty or :reduced, got $method"))
    n_iterations ≥ 1 || throw(ArgumentError("n_iterations must be positive"))
    size(particles0, 1) == n_extended(p.grid) ||
        throw(DimensionMismatch("particles have $(size(particles0, 1)) rows, " *
                                "expected $(n_extended(p.grid))"))

    g = p.grid
    Ne, Np = size(particles0)
    N = n_physical(g)
    ns = n_sources(p)
    use_admm = method === :admm
    use_constraint = method !== :reduced

    m_particles = Matrix{Float64}(particles0)
    m_initial = copy(m_particles)
    project_box!(m_particles, p.box)

    # Scaled Lagrange multipliers, one Ne x ns block per particle.
    multipliers = [zeros(ComplexF64, Ne, ns) for _ in 1:Np]

    misfit = zeros(Np, n_iterations)
    model_error = zeros(Np, n_iterations)
    constraint_residual = zeros(Np, n_iterations)
    multiplier_norm = zeros(Np, n_iterations)
    penalty = zeros(Np, n_iterations)
    bandwidth = zeros(n_iterations)
    ensemble_std = zeros(n_iterations)
    mean_model_error = zeros(n_iterations)

    grad_ext = zeros(Float64, Ne, Np)
    wavefields = Vector{Matrix{ComplexF64}}(undef, Np)
    factorizations = Vector{Any}(undef, Np)
    adagrad_state = adagrad ? AdaGradState() : nothing

    for iter in 1:n_iterations
        # ---- Steps 1 and 2: per-particle physics and gradient.
        _foreach_particle(threaded, Np) do j
            m_j = view(m_particles, :, j)
            out = _particle_gradient(p, m_j, multipliers[j], use_constraint, rwp_scan)

            # Clip the step so a single update cannot leave the box, then lift
            # back to the extended grid.
            grad_ext[:, j] .= pad(g, out.grad)
            @inbounds for i in 1:Ne
                lo = p.box[1] - m_particles[i, j]
                hi = p.box[2] - m_particles[i, j]
                grad_ext[i, j] = clamp(grad_ext[i, j], lo, hi)
            end

            wavefields[j] = out.u
            factorizations[j] = out.factorization
            misfit[j, iter] = out.misfit
            penalty[j, iter] = out.penalty
            isnothing(m_true) ||
                (model_error[j, iter] = relative_model_error(g, m_j, m_true))
        end

        # Optional prior score, evaluated on the extended grid where the field
        # is defined, then folded into the same gradient.
        if !isnothing(p.prior)
            grad_ext .+= grf_score(p.prior, m_particles)
        end

        mean_model_error[iter] = mean(view(model_error, :, iter))

        # ---- Step 3: SVGD update, on the physical grid.
        X = permutedims(cut(g, m_particles))              # Np x N
        S = permutedims(cut(g, grad_ext))                 # Np x N
        phi, h = stein_direction(X, S; repulsion = repulsion)
        bandwidth[iter] = h

        # Per-particle step size: the ratio of gradient norm to Stein-direction
        # norm, so the update advances at the scale the gradient asks for
        # regardless of how strongly the repulsion happens to act locally.
        step = Vector{Float64}(undef, Np)
        @inbounds for j in 1:Np
            pn = norm(view(phi, j, :))
            step[j] = pn > 0 ? step_scale * norm(view(S, j, :)) / pn : 0.0
        end

        direction = isnothing(adagrad_state) ? phi : adagrad_step!(adagrad_state, phi)
        X .+= step .* direction

        Xp = permutedims(X)                                # N x Np
        project_box!(Xp, p.box)
        m_particles = pad(g, Xp)

        ensemble_std[iter] = mean(std(m_particles; dims = 2))

        # ---- Step 4: multiplier update by relaxed dual ascent.
        if use_admm
            relax = iter / (iter + relaxation_offset)
            _foreach_particle(threaded, Np) do j
                # The residual uses the updated model with the wavefield from
                # this iteration, as in the reference implementation. Only a
                # matrix product is needed, not a new factorization.
                A = helmholtz_operator(p.template, view(m_particles, :, j))
                r = A * wavefields[j] - p.source
                multipliers[j] .= relax .* (multipliers[j] .- r)
                constraint_residual[j, iter] = norm(r)
                multiplier_norm[j, iter] = norm(multipliers[j])
            end
        else
            _foreach_particle(threaded, Np) do j
                r = factorizations[j].A * wavefields[j] - p.source
                constraint_residual[j, iter] = norm(r)
            end
        end

        if verbose
            @printf("iter %3d/%3d  RME %6.2f%%  misfit %.4e  h %.3e  resid %.3e\n",
                    iter, n_iterations, mean_model_error[iter],
                    mean(view(misfit, :, iter)), bandwidth[iter],
                    mean(view(constraint_residual, :, iter)))
            flush(stdout)
        end
    end

    return ADMMSVGDFWIResult(m_particles, m_initial, misfit, model_error,
                             constraint_residual, multiplier_norm, penalty,
                             bandwidth, ensemble_std, mean_model_error)
end

# Particle loop, threaded or serial. Each iteration touches only column `j` of
# the shared arrays and its own factorization, so there is nothing to guard.
function _foreach_particle(body, threaded::Bool, Np::Int)
    if threaded && Threads.nthreads() > 1
        Threads.@threads for j in 1:Np
            body(j)
        end
    else
        for j in 1:Np
            body(j)
        end
    end
    return nothing
end
