# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Verification of the frequency-domain wave physics.
#
# These are behavioral tests against properties the discretization must satisfy,
# not regression tests against stored numbers:
#
#   * the padding maps are a section/retraction pair, and `pad_adjoint` is a
#     genuine adjoint;
#   * the bound-constrained stencil fit matches an exhaustive search;
#   * the PML profile is exactly transparent in the physical region;
#   * a PLANE WAVE at the correct wavenumber nearly annihilates the operator in
#     the interior -- this is the test that validates the stencil coefficients
#     AND the unit convention together, since a wrong factor of 1e-6 or a wrong
#     h would put the wavenumber off by orders of magnitude;
#   * the PML actually absorbs: the interior solution is insensitive to how far
#     away the absorbing boundary is placed;
#   * source injection and receiver sampling are exact adjoints.

using Test
using LinearAlgebra
using SparseArrays
using SVGDADMMSampler

const S = SVGDADMMSampler

@testset "PMLGrid padding maps" begin
    g = PMLGrid((7, 5), (10.0, 10.0); pml_top = 2, pml_bottom = 3,
                pml_left = 1, pml_right = 4)

    @test extended_size(g) == (7 + 2 + 3, 5 + 1 + 4)
    @test n_physical(g) == 35
    @test n_extended(g) == 12 * 10

    m = randn(n_physical(g))
    @test cut(g, pad(g, m)) ≈ m                     # retraction
    @test length(pad(g, m)) == n_extended(g)

    # Edge extension: every padded value equals some physical value, and the
    # corners replicate the corresponding corner of the physical model.
    mp = pad(g, m)
    M = reshape(m, g.n)
    MP = reshape(mp, g.ne)
    @test MP[1, 1] == M[1, 1]
    @test MP[end, end] == M[end, end]
    @test MP[1, end] == M[1, end]
    @test all(v -> v in m, mp)

    # Matrix methods act column-wise.
    Mk = randn(n_physical(g), 3)
    @test pad(g, Mk) ≈ hcat((pad(g, Mk[:, j]) for j in 1:3)...)
    @test cut(g, pad(g, Mk)) ≈ Mk

    # pad_adjoint is the true transpose of pad; cut deliberately is not.
    x = randn(n_physical(g))
    y = randn(n_extended(g))
    @test dot(pad_adjoint(g, y), x) ≈ dot(y, pad(g, x))
    @test !isapprox(dot(cut(g, y), x), dot(y, pad(g, x)))

    @test_throws DimensionMismatch pad!(zeros(3), g, x)
end

@testset "Optimal stencil coefficients" begin
    s = optimal_stencil(35.0, 50.0)
    @test s.c + s.d + s.e ≈ 1
    @test all(v -> -1 - 1e-9 ≤ v ≤ 1 + 1e-9, (s.b, s.d, s.e))

    # The blend should stay close to the axis-aligned Laplacian for these well
    # sampled grids, and the mass operator close to lumped.
    @test 0.5 < s.b ≤ 1.0

    @test standard_stencil().b == 1.0
    @test standard_stencil().c == 1.0

    # Degenerate (single frequency, homogeneous) range must not divide by zero.
    sd = optimal_stencil(40.0, 40.0)
    @test isfinite(sd.b) && isfinite(sd.d) && isfinite(sd.e)

    @test_throws ArgumentError optimal_stencil(50.0, 35.0)
    @test_throws ArgumentError optimal_stencil(-1.0, 5.0)

    # The bound-constrained solve must agree with brute force on a problem
    # whose unconstrained optimum lies outside the box.
    A = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    y = [5.0, -3.0, 0.25]
    b, d, e = S._bounded_lsq3(A, y, -1.0, 1.0)
    @test (b, d, e) == (1.0, -1.0, 0.25)

    # Random dense instances against an exhaustive grid search.
    for trial in 1:5
        Ar = randn(20, 3)
        yr = randn(20)
        xopt = collect(S._bounded_lsq3(Ar, yr, -1.0, 1.0))
        fopt = sum(abs2, Ar * xopt - yr)
        grid = range(-1, 1; length = 41)
        fbest = minimum(sum(abs2, Ar * [a, c, d] - yr)
                        for a in grid, c in grid, d in grid)
        @test fopt ≤ fbest + 1e-8
    end

    @test grid_density_range(1.4, 2.0, 10.0, 4.0, 4.0) == (35.0, 50.0)
end

@testset "PML stretching profile" begin
    p = PMLProfile(a0 = 1.79, f0 = 1.0)
    beta = S.pml_beta(p, 4.0)
    @test beta ≈ 2π * 1.79 * 1.0 / 4.0

    n_ext, npml = 60, 10
    γ = stretch_axis(n_ext, npml, npml, beta)

    # Transparent in the physical interior: gamma is exactly 1 there.
    @test all(γ[(npml+1):(n_ext-npml)] .== 1)

    # Damping grows monotonically into each layer and is symmetric.
    damp = -imag.(γ)
    @test all(diff(damp[1:npml]) .< 0)                     # decays toward the interior
    @test all(diff(damp[(n_ext-npml+1):end]) .> 0)          # grows toward the far edge
    @test damp[1] ≈ damp[end] rtol = 1e-12
    @test all(damp .≥ 0)

    # A zero-thickness layer must be the identity, not a division by zero.
    @test all(stretch_axis(10, 0, 0, beta) .== 1)
end

@testset "Helmholtz operator: plane-wave dispersion" begin
    # A homogeneous medium with the PML switched off (a0 = 0) admits an exact
    # plane wave exp(i k . x) with |k| = omega / v. The discrete operator must
    # nearly annihilate it in the interior; the residual IS the grid dispersion
    # error the optimal stencil minimizes.
    #
    # This test is the unit check as much as the stencil check: m is in s^2/km^2
    # while h is in metres, and the operator's 1e-6 factor reconciles them. Any
    # error there moves the resonant wavenumber by orders of magnitude and this
    # residual becomes O(1).
    v = 2.0                      # km/s
    freq = 4.0                   # Hz
    h = 10.0                     # m
    n = (80, 80)
    g = PMLGrid(n, (h, h); pml = 0)
    pml = PMLProfile(a0 = 0.0, f0 = 1.0)      # no damping: pure dispersion test

    G = 1000 * v / (h * freq)                  # points per wavelength
    stencil = optimal_stencil(G, G)
    t = helmholtz_template(g, stencil, pml, freq)

    m = fill(1 / v^2, n_extended(g))
    A = helmholtz_operator(t, m)

    k = 2π * freq / (v * 1000)                 # rad/m
    nz, nx = g.ne
    for θ in (0.0, π / 8, π / 4)
        kz, kx = k * cos(θ), k * sin(θ)
        u = ComplexF64[cis(kz * (iz - 1) * h + kx * (ix - 1) * h)
                       for ix in 1:nx for iz in 1:nz]
        r = A * u

        # Compare the interior residual against the magnitude of the individual
        # terms, which is the meaningful scale: each term is O(k^2) but they
        # cancel to the dispersion error.
        interior = falses(nz, nx)
        interior[3:(nz-2), 3:(nx-2)] .= true
        scale = maximum(abs, t.omega2 .* t.cvec .* m .* (t.B * u))
        rel = maximum(abs, r[vec(interior)]) / scale
        @test rel < 5e-3
    end

    # The unoptimized five-point stencil must be strictly worse at the same
    # sampling -- if it is not, the optimization is not doing anything.
    t5 = helmholtz_template(g, standard_stencil(), pml, freq)
    A5 = helmholtz_operator(t5, m)
    u0 = ComplexF64[cis(k * cos(π / 4) * (iz - 1) * h + k * sin(π / 4) * (ix - 1) * h)
                    for ix in 1:nx for iz in 1:nz]
    interior = falses(nz, nx)
    interior[3:(nz-2), 3:(nx-2)] .= true
    res_opt = maximum(abs, (A * u0)[vec(interior)])
    res_std = maximum(abs, (A5 * u0)[vec(interior)])
    @test res_opt < res_std
end

@testset "Helmholtz operator: structure and caching" begin
    g = PMLGrid((20, 24), (10.0, 10.0); pml = 5)
    t = helmholtz_template(g, optimal_stencil(35.0, 50.0), PMLProfile(), 4.0)

    m1 = fill(0.25, n_extended(g))
    m2 = fill(0.30, n_extended(g))
    A1 = helmholtz_operator(t, m1)
    A2 = helmholtz_operator(t, m2)

    @test size(A1) == (n_extended(g), n_extended(g))
    @test eltype(A1) == ComplexF64

    # Only the mass term moves with the model, so the difference is exactly the
    # cached scaling applied to B -- this is the property that lets the template
    # be reused across particles.
    ΔA = A2 - A1
    expected = Diagonal(t.omega2 .* t.cvec .* (m2 .- m1)) * t.B
    @test maximum(abs, ΔA - expected) < 1e-12 * maximum(abs, A1)

    # Nine-point connectivity: no row couples more than nine nodes.
    @test maximum(diff(A1.colptr)) ≤ 9

    # The model derivative is the diagonal of d(Au)/dm, so a finite difference
    # in m must match it.
    u = randn(ComplexF64, n_extended(g))
    δ = 1e-7
    j = 137
    mp = copy(m1); mp[j] += δ
    fd = (helmholtz_operator(t, mp) * u - A1 * u) / δ
    md = model_derivative(t, u)
    @test abs(fd[j] - md[j]) < 1e-4 * abs(md[j])
    @test maximum(abs, fd[setdiff(1:length(fd), j)]) < 1e-6 * abs(md[j])

    @test_throws DimensionMismatch helmholtz_operator(t, zeros(3))
    @test_throws ArgumentError helmholtz_template(g, standard_stencil(), PMLProfile(), 0.0)
end

@testset "PML absorbs: interior solution is boundary-independent" begin
    # The strongest available check on the absorbing layer. Solve the same point
    # source in a homogeneous medium on two domains of different physical width
    # and compare the wavefield on the region they share. If the PML reflected,
    # the two solutions would differ by the reflected arrival, which moves when
    # the boundary moves.
    v, freq, h = 2.0, 6.0, 10.0
    pml = PMLProfile(a0 = 1.79, f0 = 1.0)
    stencil = optimal_stencil(grid_density_range(v, v, h, freq, freq)...)

    function point_source_field(npad_extra::Int)
        n = (60 + 2 * npad_extra, 60 + 2 * npad_extra)
        g = PMLGrid(n, (h, h); pml = 20)
        t = helmholtz_template(g, stencil, pml, freq)
        m = fill(1 / v^2, n_extended(g))
        F = factorize_helmholtz(t, m)

        zs = 30.0 + npad_extra
        xs = 30.0 + npad_extra
        acq = Acquisition(g; source_z = [zs], source_x = [xs],
                          receiver_z = [zs], receiver_x = [xs])
        b = Matrix{ComplexF64}(source_operator(acq))
        u = solve_forward(F, b)

        # Return the field on a fixed 30x30 window centred on the source.
        U = reshape(u[:, 1], g.ne)
        cz = Int(zs) + g.pml_top
        cx = Int(xs) + g.pml_left
        return U[(cz-14):(cz+15), (cx-14):(cx+15)]
    end

    U_small = point_source_field(0)
    U_large = point_source_field(15)

    rel = norm(U_small - U_large) / norm(U_large)
    @test rel < 5e-2
end

@testset "Acquisition operators" begin
    g = PMLGrid((30, 40), (10.0, 10.0); pml_top = 3, pml_bottom = 4,
                pml_left = 5, pml_right = 6)
    acq = Acquisition(g; source_z = [1.0, 10.0], source_x = [2.0, 20.0],
                      receiver_z = [29.0, 29.0, 29.0],
                      receiver_x = [1.0, 20.0, 40.0])

    @test n_sources(acq) == 2
    @test n_receivers(acq) == 3

    Sop = source_operator(acq)
    Pop = receiver_operator(acq)
    @test size(Sop) == (n_extended(g), 2)
    @test size(Pop) == (3, n_extended(g))

    # On-node placement is a delta at the PML-shifted index.
    @test nnz(Sop[:, 1]) == 1
    idx = findfirst(!iszero, Sop[:, 1])
    expected_idx = (2 + g.pml_left - 1) * g.ne[1] + (1 + g.pml_top)
    @test idx == expected_idx
    @test Sop[idx, 1] == 1.0

    # Sampling and injection are exact adjoints.
    u = randn(ComplexF64, n_extended(g))
    y = randn(ComplexF64, 3)
    @test dot(Pop * u, y) ≈ dot(u, Pop' * y)

    # Off-grid positions spread over a Hicks stencil with unit total weight and
    # a symmetric split at the exact midpoint.
    acq_frac = Acquisition(g; source_z = [10.5], source_x = [20.0],
                           receiver_z = [29.0], receiver_x = [1.0])
    Sf = source_operator(acq_frac)
    w = nonzeros(Sf[:, 1])
    @test sum(w) ≈ 1.0
    @test length(w) > 1
    @test sort(w) ≈ sort(reverse(w))                    # symmetric about the midpoint

    @test_throws ArgumentError Acquisition(g; source_z = [1.0], source_x = [1.0, 2.0],
                                           receiver_z = [1.0], receiver_x = [1.0])
    @test_throws ArgumentError Acquisition(g; source_z = [99.0], source_x = [1.0],
                                           receiver_z = [1.0], receiver_x = [1.0])

    @test ricker_weight(4.0, 6.0) ≈ 16 * exp(-(4 / 6)^2)
end

@testset "Factorized solves and the reduced operator" begin
    v, freq, h = 2.0, 5.0, 10.0
    g = PMLGrid((40, 40), (h, h); pml = 10)
    t = helmholtz_template(g, optimal_stencil(grid_density_range(v, v, h, freq, freq)...),
                           PMLProfile(), freq)
    m = fill(1 / v^2, n_extended(g)) .+ 0.01 .* randn(n_extended(g))
    A = helmholtz_operator(t, m)
    F = factorize_helmholtz(A)

    b = randn(ComplexF64, n_extended(g), 2)
    u = solve_forward(F, b)
    @test norm(A * u - b) < 1e-8 * norm(b)

    y = randn(ComplexF64, n_extended(g), 2)
    x = solve_adjoint(F, y)
    @test norm(A' * x - y) < 1e-8 * norm(y)

    @test norm(wave_residual(F, u, b)) < 1e-8 * norm(b)

    acq = Acquisition(g; source_z = [2.0], source_x = [20.0],
                      receiver_z = fill(39.0, 8), receiver_x = collect(range(2, 39; length = 8)))
    P = receiver_operator(acq)

    W = reduced_operator(F, P)
    @test size(W) == (n_extended(g), 8)

    # W is A^-H P^H, so W^H acts as P A^-1: applying it to a source must give
    # the same data as solving then sampling.
    src = Matrix{ComplexF64}(source_operator(acq))
    @test W' * src ≈ P * solve_forward(F, src) rtol = 1e-8

    Q = reduced_gram(W)
    @test Q isa Hermitian
    λ, V = gram_eigen(Q)
    @test all(λ .≥ 0)
    @test issorted(λ)
    @test V' * V ≈ I rtol = 1e-10
    @test V * Diagonal(λ) * V' ≈ Matrix(Q) rtol = 1e-8

    # The shifted solve must agree with an explicit factorization.
    rhs = randn(ComplexF64, 8, 3)
    μ = 0.37 * maximum(λ)
    @test shifted_gram_solve(λ, V, rhs, μ) ≈ (Matrix(Q) + μ * I) \ rhs rtol = 1e-8

    @test_throws ArgumentError shifted_gram_solve(λ, V, rhs, -1.0)
    @test_throws DimensionMismatch reduced_operator(F, P[:, 1:10])
end
