# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Test entry point.
#
# Run with:  julia --project=. test/runtests.jl
# or from the package REPL:  ] test

using Test

@testset "SVGDADMMSampler" begin
    include("test_fwi_physics.jl")
    include("test_grf.jl")
    include("test_svgd_rwp.jl")
    include("test_admm_svgd_fwi.jl")
end
