# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Feb 2026

export MALA_sampler

"""
    MALA_sampler(max_itr, x0, nlog_density, grad_nlog_density; τ, thinning, burn_in_frac)

Metropolis-Adjusted Langevin Algorithm (MALA) sampler.

Proposes via a Langevin step and corrects with Metropolis-Hastings
accept/reject, yielding asymptotically exact samples.

# Arguments
- `max_itr::Int`: Total number of MALA iterations.
- `x0::Vector{Float32}`: Initial state (length-d vector).
- `nlog_density::Function`: Negative log-density U(x), returns scalar.
- `grad_nlog_density::Function`: Gradient ∇U(x), returns vector same size as x.
- `τ::Float32`: Step size (default 1e-2).
- `thinning::Int`: Keep every `thinning`-th sample (default 1).
- `burn_in_frac::Float32`: Fraction of samples to discard as burn-in (default 0.5).
"""
function MALA_sampler(
    max_itr::Int,
    x0::Vector{Float32},
    nlog_density::Function,
    grad_nlog_density::Function;
    τ::Float32 = 1.0f-2,
    thinning::Int = 1,
    burn_in_frac::Float32 = 0.5f0,
)
    d = length(x0)
    x = copy(x0)

    # Pre-allocate: d × max_itr
    x_samples = zeros(Float32, d, max_itr)

    n_accept = 0
    U_x = nlog_density(x)
    grad_x = grad_nlog_density(x)

    p = Progress(max_itr)
    for j = 1:max_itr

        # Propose: x' = x - (τ/2) ∇U(x) + √τ ξ
        ξ = randn(Float32, d)
        mean_fwd = x .- (τ / 2) .* grad_x
        x_prop = mean_fwd .+ sqrt(τ) .* ξ

        # Evaluate at proposed point
        U_xp = nlog_density(x_prop)
        grad_xp = grad_nlog_density(x_prop)

        # Reverse proposal mean
        mean_bwd = x_prop .- (τ / 2) .* grad_xp

        # Log acceptance ratio
        log_q_fwd = -sum((x_prop .- mean_fwd) .^ 2) / (2 * τ)
        log_q_bwd = -sum((x .- mean_bwd) .^ 2) / (2 * τ)
        log_α = -U_xp + U_x + log_q_bwd - log_q_fwd

        # Accept/reject
        if log(rand(Float32)) < Float32(log_α)
            x = x_prop
            U_x = U_xp
            grad_x = grad_xp
            n_accept += 1
        end

        x_samples[:, j] .= x

        ProgressMeter.next!(
            p;
            showvalues = [
                ("MALA iteration", j),
                ("NLL", U_x),
                ("step size τ", τ),
                ("accept rate", round(n_accept / j, digits=3)),
            ],
        )
    end

    # Burn-in
    burn_in = max(1, round(Int, burn_in_frac * max_itr))
    x_samples = x_samples[:, burn_in:end]

    # Thinning
    x_samples = x_samples[:, 1:thinning:end]

    return x_samples
end
