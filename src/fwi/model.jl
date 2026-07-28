# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Jul 2026
# Velocity models and the model-parameter conventions of the FWI experiments.
#
# The inversion variable is SQUARED SLOWNESS `m = 1/v^2`, the parameter the
# Helmholtz operator is linear in -- which is what makes the model derivative
# diagonal and the gradient a zero-lag correlation. Models are stated in
# velocity (km/s) for readability and converted at the boundary.

export gaussian_lens_model, velocity_to_slowness, slowness_to_velocity,
       box_constraints, project_box, project_box!, transmission_acquisition

"""
    gaussian_lens_model(; n = (200, 200), background = 2.0, contrast = 0.6,
                        width_x = 0.25, width_z = 0.5, spacing_km = 0.01)

Smooth synthetic velocity model: a constant background with a single Gaussian
low-velocity lens at its centre, in km/s.

    v(x, z) = background - contrast * exp( -((x - xc)/width_x)^2
                                           -((z - zc)/width_z)^2 )

The lens is the anomaly the inversion has to recover, and because it is smooth
and isolated it isolates the sampler's behavior from any structural complexity
of the medium. Grid coordinates run from `spacing_km` to `n * spacing_km` in
kilometres, so the defaults give a 2 km by 2 km model on a 10 m grid, with the
lens centred at (1 km, 1 km).
"""
function gaussian_lens_model(; n::Tuple{Int,Int} = (200, 200),
                             background::Real = 2.0, contrast::Real = 0.6,
                             width_x::Real = 0.25, width_z::Real = 0.5,
                             spacing_km::Real = 0.01)
    nz, nx = n
    (nz > 0 && nx > 0) || throw(ArgumentError("grid size must be positive, got $n"))
    (width_x > 0 && width_z > 0) || throw(ArgumentError("lens widths must be positive"))
    background > contrast ||
        throw(ArgumentError("contrast $contrast would make the velocity non-positive " *
                            "against background $background"))

    z = collect(spacing_km:spacing_km:(nz*spacing_km))
    x = collect(spacing_km:spacing_km:(nx*spacing_km))
    zc = nz * spacing_km / 2
    xc = nx * spacing_km / 2

    v = Matrix{Float64}(undef, nz, nx)
    @inbounds for j in 1:nx, i in 1:nz
        v[i, j] = background -
                  contrast * exp(-((x[j] - xc) / width_x)^2 - ((z[i] - zc) / width_z)^2)
    end
    return v
end

"""
    velocity_to_slowness(v)

Squared slowness `m = 1 / v^2` from velocity in km/s.
"""
function velocity_to_slowness(v::AbstractArray)
    any(≤(0), v) && throw(ArgumentError("velocity must be strictly positive"))
    return @. 1 / v^2
end

"""
    slowness_to_velocity(m)

Velocity from squared slowness. Negative entries would be unphysical; they are
an error rather than a silently complex result.
"""
function slowness_to_velocity(m::AbstractArray)
    any(≤(0), m) &&
        throw(ArgumentError("squared slowness must be strictly positive; " *
                            "a particle left the physical box"))
    return @. 1 / sqrt(m)
end

"""
    box_constraints(m_true; lower_factor = 0.95, upper_factor = 1.20)

Physically plausible box `[m_min, m_max]` for squared slowness, taken as a
margin around the range of a reference model.

The box serves two purposes in the sampler: the likelihood gradient is clipped
so no single step can leave it, and particles are projected back onto it after
the Stein update. Without it the ensemble can wander into velocities that make
the Helmholtz operator singular or the medium non-physical.
"""
function box_constraints(m_true::AbstractArray; lower_factor::Real = 0.95,
                         upper_factor::Real = 1.20)
    lower_factor ≤ upper_factor ||
        throw(ArgumentError("lower_factor must not exceed upper_factor"))
    return (lower_factor * minimum(m_true), upper_factor * maximum(m_true))
end

"""
    project_box(x, box)
    project_box!(x, box)

Clamp onto `box = (lo, hi)`, elementwise.
"""
project_box(x::AbstractArray, box::Tuple{<:Real,<:Real}) = clamp.(x, box[1], box[2])

function project_box!(x::AbstractArray, box::Tuple{<:Real,<:Real})
    @inbounds for i in eachindex(x)
        x[i] = clamp(x[i], box[1], box[2])
    end
    return x
end

"""
    transmission_acquisition(grid; n_sources = 50, source_depth = 1,
                             receiver_depth = nothing, receiver_stride = 1)

Transmission-style geometry for the lens experiment: a line of sources across
the top of the model and a line of receivers across the bottom, so energy
crosses the anomaly rather than reflecting off it.

`receiver_depth` defaults to one node above the bottom edge, keeping receivers
off the outermost row.
"""
function transmission_acquisition(g::PMLGrid; n_sources::Int = 50,
                                  source_depth::Int = 1,
                                  receiver_depth::Union{Nothing,Int} = nothing,
                                  receiver_stride::Int = 1)
    nz, nx = g.n
    n_sources ≥ 1 || throw(ArgumentError("need at least one source"))
    rz = something(receiver_depth, nz - 1)

    sx = collect(range(2, nx - 2; length = n_sources))
    rx = collect(1:receiver_stride:nx)

    return Acquisition(g;
                       source_z = fill(float(source_depth), n_sources),
                       source_x = sx,
                       receiver_z = fill(float(rz), length(rx)),
                       receiver_x = float.(rx))
end
