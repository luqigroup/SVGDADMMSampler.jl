# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# ADMM-SVGD posterior sampling for frequency-domain FWI on the Gaussian-lens
# model. Julia port of the reference `Demo_SVGD_Gaussian_model.m`.
#
#   julia --project=. -t auto scripts/gaussian_lens_fwi.jl
#
# Every config key is a CLI override, so a fast smoke run is
#
#   julia --project=. -t 4 scripts/gaussian_lens_fwi.jl \
#         --nz 64 --nx 64 --h 31.25 --n_particles 8 --n_iterations 8 --n_sources 12
#
# MEMORY scales with the THREAD COUNT, not the ensemble size: each concurrently
# processed particle holds an `Ne x nr` reduced operator plus LU factors,
# roughly 90 MB plus factors at the default size. Lower `-t` before `--n_particles`.

using DrWatson
@quickactivate :SVGDADMMSampler

using Random
using Statistics
using LinearAlgebra
using JLD2

# The per-particle work is threaded over the ensemble, and each particle's
# inner work is dominated by dense complex GEMM against the reduced operator.
# Leaving BLAS multithreaded underneath would oversubscribe the machine by the
# square of the thread count, so BLAS is pinned to one thread per particle
# task. This changes scheduling only, never the arithmetic.
BLAS.set_num_threads(1)

args = parse_input_args(read_config("gaussian_lens_fwi.json"))

# --------------------------------------------------------------------------
# Problem setup.
# --------------------------------------------------------------------------
n = (Int(args["nz"]), Int(args["nx"]))
h = Float64(args["h"])
freq = Float64(args["freq"])

v_true = gaussian_lens_model(; n = n,
                             background = args["background"],
                             contrast = args["contrast"],
                             width_x = args["width_x"],
                             width_z = args["width_z"],
                             spacing_km = h / 1000)
v0 = fill(Float64(args["background"]), n)

grid = PMLGrid(n, (h, h); pml = Int(args["pml"]))

# Dispersion-optimal stencil for the grid densities this run actually spans.
stencil = optimal_stencil(grid_density_range(minimum(v_true), maximum(v_true),
                                             h, freq, freq)...)
pml = PMLProfile(a0 = args["a0_pml"], f0 = args["f0_pml"])
template = helmholtz_template(grid, stencil, pml, freq)

acq = transmission_acquisition(grid; n_sources = Int(args["n_sources"]),
                               receiver_stride = Int(args["receiver_stride"]))

m_true = pad(grid, vec(velocity_to_slowness(v_true)))
m0 = pad(grid, vec(velocity_to_slowness(v0)))
box = box_constraints(m_true; lower_factor = args["box_lower"],
                      upper_factor = args["box_upper"])

source = ricker_weight(freq, args["fdom"]) .*
         Matrix{ComplexF64}(source_operator(acq))
receiver = receiver_operator(acq)

println("grid       ", grid)
println("stencil    ", stencil)
println("geometry   ", acq, "  at ", freq, " Hz")
println("box        ", box, "   (v in [",
        round(1 / sqrt(box[2]); digits = 3), ", ",
        round(1 / sqrt(box[1]); digits = 3), "] km/s)")

# --------------------------------------------------------------------------
# Observed data: forward model the truth, then add complex Gaussian noise
# scaled to the data amplitude.
# --------------------------------------------------------------------------
rng = Xoshiro(Int(args["seed"]))
F_true = factorize_helmholtz(template, m_true)
d_clean = receiver * solve_forward(F_true, source)
σ_noise = args["noise_rel"] * maximum(abs, d_clean)
noise = (σ_noise / sqrt(2)) .* complex.(randn(rng, size(d_clean)),
                                        randn(rng, size(d_clean)))
data = d_clean .+ noise
println("data       ", size(data), "   noise level ", round(σ_noise; sigdigits = 3))

# --------------------------------------------------------------------------
# Initial ensemble: the constant starting model plus Gaussian-random-field
# perturbations. `prior_std_frac` is the marginal standard deviation of the
# perturbation as a fraction of the starting model, which is the physically
# interpretable scale; the default reproduces the spread of the reference
# implementation's ensemble.
# --------------------------------------------------------------------------
prior = GaussianRandomField(grid; alpha = args["prior_alpha"],
                            tau = args["prior_tau"],
                            pointwise_std = args["prior_std_frac"] * maximum(m0))
Np = Int(args["n_particles"])
particles0 = m0 .+ grf_sample(prior, Np; rng = rng)

posterior = FWIPosterior(grid, template, source, receiver, data, box;
                         prior = args["prior_in_gradient"] == 1 ? prior : nothing)

println("ensemble   ", Np, " particles, prior std ",
        round(grf_pointwise_std(prior); sigdigits = 3),
        "   threads ", Threads.nthreads())
println("initial RME ",
        round(relative_model_error(grid, vec(mean(particles0; dims = 2)), m_true);
              digits = 2), "%")

# --------------------------------------------------------------------------
# Sample.
# --------------------------------------------------------------------------
elapsed = @elapsed result = run_admm_svgd_fwi(
    posterior, particles0;
    n_iterations = Int(args["n_iterations"]),
    m_true = m_true,
    method = Symbol(args["method"]),
    repulsion = args["repulsion"],
    step_scale = args["step_scale"],
    relaxation_offset = args["relaxation_offset"],
    adagrad = args["adagrad"] == 1,
    rwp_scan = Int(args["rwp_scan"]),
    threaded = true,
    verbose = true,
)

m_mean = vec(mean(result.particles; dims = 2))
println("final RME  ", round(relative_model_error(grid, m_mean, m_true); digits = 2),
        "%   in ", round(elapsed / 60; digits = 1), " min")

# --------------------------------------------------------------------------
# Persist. The config is the experiment's identity, so the filename carries it.
# --------------------------------------------------------------------------
save_dict = Dict{String,Any}(args)
save_dict["v_true"] = v_true
save_dict["m_true"] = m_true
save_dict["m0"] = m0
save_dict["particles"] = result.particles
save_dict["particles_initial"] = result.particles_initial
save_dict["misfit"] = result.misfit
save_dict["model_error"] = result.model_error
save_dict["constraint_residual"] = result.constraint_residual
save_dict["multiplier_norm"] = result.multiplier_norm
save_dict["penalty"] = result.penalty
save_dict["bandwidth"] = result.bandwidth
save_dict["ensemble_std"] = result.ensemble_std
save_dict["mean_model_error"] = result.mean_model_error
save_dict["data"] = data
save_dict["data_clean"] = d_clean
save_dict["noise_std"] = σ_noise
save_dict["elapsed_seconds"] = elapsed

# The full config carries 30-odd keys, and `savename` over all of them exceeds
# the 255-byte filename limit. Name the run by the keys that actually
# distinguish one experiment from another; every key still lives inside the
# file, so nothing is lost and `load_experiment` can still match on all of them.
const IDENTITY_KEYS = ["sim_name", "nz", "nx", "freq", "n_sources", "n_particles",
                       "n_iterations", "method", "repulsion", "step_scale",
                       "prior_std_frac", "noise_rel", "seed"]

outdir = datadir(args["sim_name"])
!isdir(outdir) && mkpath(outdir)
filepath = joinpath(outdir, savename(args, "jld2"; accesses = IDENTITY_KEYS))
@tagsave(filepath, save_dict; safe = true)
println("Saved to ", filepath)
