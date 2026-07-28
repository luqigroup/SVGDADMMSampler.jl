# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

module SVGDADMMSampler

using DrWatson
using JLD2
using JSON
using ArgParse
using Random
using DataFrames
using LinearAlgebra
using SparseArrays
using Distributions
using Statistics
using ProgressMeter
using PyPlot
using Seaborn

import DrWatson: _wsave
import Random: rand

# Utilities.
include("./utils/load_experiment.jl")
include("./utils/savefig.jl")
include("./utils/config.jl")

# Frequency-domain wave physics (Helmholtz + PML), one concept per file.
# Order matters: grid -> pml -> stencil -> helmholtz, since each layer builds
# on the types introduced by the previous one.
include("./fwi/grid.jl")
include("./fwi/pml.jl")
include("./fwi/stencil.jl")
include("./fwi/helmholtz.jl")
include("./fwi/acquisition.jl")
include("./fwi/model.jl")
include("./fwi/solver.jl")
include("./fwi/rwp.jl")

# Priors.
include("./priors/grf.jl")

# Sampling.
include("./sampling/svgd.jl")
include("./sampling/admm_svgd.jl")
include("./sampling/conditional_inference.jl")
include("./sampling/admm_svgd_fwi.jl")

# Diagnostics.
include("./diagnostics/calibration.jl")

end
