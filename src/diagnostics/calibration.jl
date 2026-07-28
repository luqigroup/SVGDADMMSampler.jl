# Authors: Ali Siahkoohi, alisk@ucf.edu
# Simulation-based calibration (Talts et al., 2018) and a binned error-vs-std
# reliability diagnostic for the Rosenbrock conditional sampler. These quantify
# (i) whether the particle ensembles are calibrated, and (ii) whether ADMM-SVGD
# on the constrained reformulation RECOVERS the pure-SVGD posterior (equivalence).
#
# IMPORTANT INTERPRETATION: SBC here (with thinning) tests EMPIRICAL-CDF
# calibration of the particle ENSEMBLE — i.e. whether the ensemble's empirical
# CDF matches the true posterior CDF — NOT whether the samples are i.i.d. A
# uniform rank histogram certifies that the ensemble distribution is calibrated;
# it does NOT certify that SVGD produces independent samples (SVGD particles are
# deterministically coupled). Random thinning to sbc_N particles mitigates the
# within-ensemble dependence for the rank statistic; it does not make the draws
# independent. The claim being validated is ensemble calibration, not i.i.d.-ness.

export sbc_rank, run_sbc, sbc_uniform_band, sbc_selftest, binned_calibration

using Random
using Statistics
using Distributions
using ProgressMeter

"""
    sbc_rank(theta_star, samples) -> Int

Rank of the scalar ground truth `theta_star` among posterior `samples`:
r = #{s ∈ samples : s < theta_star} ∈ {0, …, length(samples)}. Under exact
inference the rank is discrete-uniform on these length(samples)+1 values
(Talts et al., 2018).
"""
sbc_rank(theta_star::Real, samples::AbstractVector) = count(<(theta_star), samples)

"""
    run_sbc(infer_fn, RB_dist; L, σ, sbc_N, seed=0) -> NamedTuple

Simulation-based calibration over `L` independent replications. For each ℓ:
build a per-replication thread-local RNG `rng = Xoshiro(seed+ℓ)` (reproducible and
independent across ℓ), draw θ* ~ prior, simulate y = θ* + σ·noise, run
`infer_fn(y, rng)` -> `(2, n_particles)` posterior particles, thin to `sbc_N`
randomly-chosen particles, and record the per-coordinate rank of θ*. Also records
the full-ensemble posterior mean/std (for the reliability curve) and θ*. Returns
(ranks_x1, ranks_x2, cm, sd, theta).

The replications are INDEPENDENT and run with `Threads.@threads`. Each ℓ owns a
distinct thread-local `Xoshiro(seed+ℓ)` (no shared/global RNG ⇒ no data race) and
writes only its own row/index of every output (distinct indices ⇒ no race), so the
results are independent of thread scheduling: the SAME `seed` yields IDENTICAL
ranks on every run, with any number of threads. Because θ*, y, and the particle
initialization are all drawn from `Xoshiro(seed+ℓ)` in the same order, the
across-method comparison is exactly paired when the same `seed` is used (identical
θ* per ℓ). SVGD particles are NOT i.i.d.; random thinning is the pragmatic
mitigation for the rank test (a documented limitation).

What a uniform rank histogram certifies: that the particle ENSEMBLE's empirical
CDF is calibrated against the true posterior — NOT that SVGD draws are
independent. This is an empirical-CDF calibration test of the ensemble, not an
i.i.d.-ness test.
"""
function run_sbc(infer_fn::Function, RB_dist; L::Int, σ::Real, sbc_N::Int, seed::Int=0)
    ranks_x1 = zeros(Int, L)
    ranks_x2 = zeros(Int, L)
    cm    = zeros(Float32, L, 2)
    sd    = zeros(Float32, L, 2)
    theta = zeros(Float32, L, 2)

    # Thread-safe progress: an atomic counter printed occasionally. The print is
    # informational only and must NOT gate correctness, so racing prints are fine.
    done = Threads.Atomic{Int}(0)
    println("  SBC: 0/$L  ($(Threads.nthreads()) threads)")

    # Parallel over INDEPENDENT replications. Each ℓ uses its OWN thread-local
    # Xoshiro(seed+ℓ) (no shared RNG) and writes only distinct indices, so there
    # is no data race and the result is deterministic w.r.t. thread scheduling.
    Threads.@threads for ℓ in 1:L
        rng = Xoshiro(seed + ℓ)                              # per-replication, thread-local
        θ = Float32.(vec(rand(rng, RB_dist, 1)))             # length-2 prior draw
        y = θ .+ Float32(σ) .* randn(rng, Float32, 2)        # simulated data
        post = infer_fn(y, rng)                              # (2, n_particles)

        Np  = size(post, 2)
        idx = randperm(rng, Np)[1:min(sbc_N, Np)]            # thin to ~independent draws
        ranks_x1[ℓ] = sbc_rank(θ[1], view(post, 1, idx))
        ranks_x2[ℓ] = sbc_rank(θ[2], view(post, 2, idx))

        cm[ℓ, 1] = mean(view(post, 1, :)); cm[ℓ, 2] = mean(view(post, 2, :))
        sd[ℓ, 1] = std(view(post, 1, :));  sd[ℓ, 2] = std(view(post, 2, :))
        theta[ℓ, 1] = θ[1]; theta[ℓ, 2] = θ[2]

        n = Threads.atomic_add!(done, 1) + 1
        (n == L || n % max(1, L ÷ 10) == 0) && println("  SBC: $n/$L")
    end
    return (ranks_x1=ranks_x1, ranks_x2=ranks_x2, cm=cm, sd=sd, theta=theta)
end

"""
    sbc_uniform_band(L, n_bins; α=0.01) -> (expected, lower, upper)

Expected per-bin count and (1-α) binomial confidence band for an SBC rank
histogram with `n_bins` equal-width bins over `L` replications, under uniformity.
"""
function sbc_uniform_band(L::Int, n_bins::Int; α::Float64=0.01)
    d = Binomial(L, 1 / n_bins)
    return (expected = L / n_bins,
            lower = quantile(d, α / 2),
            upper = quantile(d, 1 - α / 2))
end

"""
    sbc_selftest(; N=99, L=2000, seed=0) -> pval

Validate the rank machinery itself. With an exact posterior (prior = posterior =
N(0,1), N exact i.i.d. draws) the rank histogram must be uniform; returns a χ²
goodness-of-fit p-value against uniform. A large p-value (≳0.05) means the SBC
implementation is unbiased before it is applied to SVGD.
"""
function sbc_selftest(; N::Int=99, L::Int=2000, seed::Int=0)
    Random.seed!(seed)
    n_bins = min(20, N + 1)
    obs = zeros(Int, n_bins)
    for _ in 1:L
        θ = randn()
        r = count(<(θ), randn(N))
        b = clamp(floor(Int, r * n_bins / (N + 1)) + 1, 1, n_bins)
        obs[b] += 1
    end
    e  = L / n_bins
    χ2 = sum((obs .- e) .^ 2 ./ e)
    return 1 - cdf(Chisq(n_bins - 1), χ2)
end

"""
    binned_calibration(x_vals, y_vals; n_bins=12) -> (bin_x, bin_y)

True equal-count (quantile) binning of (x_vals, y_vals) by x_vals; returns the
per-bin means `(bin_x, bin_y)`. Always produces exactly `n_bins` balanced bins:
the points are sorted by `x_vals` and split at `round.(Int, range(...))` indices
so that bin sizes differ by at most 1. For the reliability curve, x = posterior
std and y = squared error (the caller takes √ of `bin_y` for an RMSE curve).
Equal-count binning follows the postmem reference (`_binned_calibration`).
"""
function binned_calibration(x_vals::AbstractVector, y_vals::AbstractVector; n_bins::Int=12)
    order = sortperm(x_vals)
    n = length(order)
    nb = min(n_bins, n)                                   # can't have more bins than points
    edges = round.(Int, range(0, n; length=nb + 1))      # nb balanced split points; sizes differ by ≤1
    bin_x = Float32[]
    bin_y = Float32[]
    for b in 1:nb
        cc = order[(edges[b] + 1):edges[b + 1]]
        push!(bin_x, mean(x_vals[cc]))
        push!(bin_y, mean(y_vals[cc]))
    end
    return bin_x, bin_y
end
