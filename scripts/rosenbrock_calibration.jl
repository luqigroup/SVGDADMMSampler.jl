# Authors: Ali Siahkoohi, alisk@ucf.edu
# Simulation-based calibration (SBC) + reliability diagnostics comparing the
# Rosenbrock conditional ADMM-SVGD sampler against plain SVGD.
#
# Purpose (addresses Reviewer 1, comment 1): the Rosenbrock problem is directly
# sampleable by pure SVGD; we build an EQUIVALENT constrained reformulation
# (z = x₁²) and show ADMM-SVGD RECOVERS the same, calibrated posterior — an
# equivalence/validation, not a superiority claim. SBC tests full-posterior
# calibration; the reliability curve tests the mean±kσ summary the paper reports.

using DrWatson
@quickactivate :SVGDADMMSampler
using Rosenbrock
using Random
using Statistics
using JLD2
using LinearAlgebra

# SBC parallelizes INDEPENDENT replications with Threads.@threads (one replication
# per Julia thread). The inner SVGD kernels (svgd_update!, compute_bandwidth) do
# dense n_particles×n_particles matmuls that otherwise dispatch to multithreaded
# BLAS; with N Julia threads each spawning BLAS threads on N cores that is N²-way
# oversubscription. Pin BLAS to 1 thread so each replication gets exactly one core
# (outer task parallelism, inner serial BLAS). This changes ONLY CPU scheduling —
# not the math, RNG, ranks, or figures.
BLAS.set_num_threads(1)

# Read configuration
args = read_config("rosenbrock_calibration.json")
args = parse_input_args(args)

println("="^60)
println("Rosenbrock calibration: SBC + reliability (ADMM-SVGD vs SVGD)")
println("="^60)
for (key, val) in args
    println("  $key: $val")
end
println("="^60)

RB_dist = RosenbrockDistribution(args["mu_rb"], args["a"])

# Per-method inference closures: (y, rng) -> (2, n_particles) posterior particles.
# The rng is the per-replication thread-local RNG from run_sbc; it seeds particle
# initialization so SBC is thread-safe and deterministic (see run_sbc docstring).
admm_infer = (y, rng) -> run_admm_svgd_conditional(y, RB_dist;
    n_particles=args["n_particles"], σ=args["sigma"], μ=args["admm_mu_penalty"],
    η=args["admm_eta"], n_iterations=args["admm_n_iterations"], h=args["h"], rng=rng)
svgd_infer = (y, rng) -> run_standard_svgd_conditional(y, RB_dist;
    n_particles=args["n_particles"], σ=args["sigma"], η=args["svgd_eta"],
    n_iterations=args["svgd_n_iterations"], rng=rng)

# SBC self-test: the rank machinery must be uniform on exact i.i.d. draws.
selftest_pval = sbc_selftest(N=args["sbc_N"], L=2000, seed=args["seed"])
println("\nSBC self-test p-value (want ≳ 0.05): ", round(selftest_pval, digits=4))

# Paired SBC for both methods: the SAME seed gives identical (θ*, y) streams and
# particle initialization, so the two methods are evaluated on identical
# instances (a paired equivalence test).
println("\nADMM-SVGD SBC...  (threads: $(Threads.nthreads()))")
admm_time = @elapsed (global admm = run_sbc(admm_infer, RB_dist; L=args["sbc_L"], σ=args["sigma"],
               sbc_N=args["sbc_N"], seed=args["seed"]))
println("  ADMM-SVGD SBC loop wall-clock: $(round(admm_time, digits=2)) s")
println("\nStandard SVGD SBC...")
svgd_time = @elapsed (global svgd = run_sbc(svgd_infer, RB_dist; L=args["sbc_L"], σ=args["sigma"],
               sbc_N=args["sbc_N"], seed=args["seed"]))
println("  Standard SVGD SBC loop wall-clock: $(round(svgd_time, digits=2)) s")

@assert admm.theta ≈ svgd.theta "paired SBC expects identical θ* across methods"

# Save results.
println("\n" * "="^60)
println("Saving results...")
!isdir(datadir(args["sim_name"])) && mkpath(datadir(args["sim_name"]))
filename = savename(args, "jld2")
filepath = datadir(args["sim_name"], filename)

save_dict = Dict{String,Any}()
for (key, val) in args
    save_dict[key] = val
end
save_dict = merge(save_dict, Dict(
    "selftest_pval" => selftest_pval,
    "admm_ranks_x1" => admm.ranks_x1, "admm_ranks_x2" => admm.ranks_x2,
    "admm_cm" => admm.cm, "admm_sd" => admm.sd, "admm_theta" => admm.theta,
    "svgd_ranks_x1" => svgd.ranks_x1, "svgd_ranks_x2" => svgd.ranks_x2,
    "svgd_cm" => svgd.cm, "svgd_sd" => svgd.sd, "svgd_theta" => svgd.theta,
))
@tagsave(filepath, save_dict; safe=true)
println("Results saved to: $filepath")
println("="^60)
