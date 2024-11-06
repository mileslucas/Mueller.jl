
# Various formulae for polarization components
"""
    rotation([T=Float64], theta)

Generate a rotation matrix with the given angle, in radians, `theta`. This can be used to rotate the axes of polarization components arbitrarily. For convenience, the [`rotate`](@ref) method will rotate a component, without having to generate this matrix, itself.

# Examples

Rotate a linear polarizer by 90 degrees counter-clockwise
```jldoctest
julia> M = linear_polarizer();

julia> r = rotation(π/4);

julia> Mr = r' * M * r;

julia> Mr ≈ linear_polarizer(π/4)
true
```

# See also
[`rotate`](@ref)
"""
function rotation(T, theta)
    sin2t, cos2t = sincos(2 * theta)
    return SA{T}[1 0 0 0
              0 cos2t sin2t 0
              0 -sin2t cos2t 0
              0 0 0 1]
end
rotation(theta) = rotation(Float64, theta)

@doc raw"""
    rotate(M, theta)

Rotates the component represented by the Mueller matrix `M` counter-clockwise by angle `theta` (in radians).

This is accomplished by the matrix multiplication of two [`Mueller.rotation`](@ref) matrices

```math
\mathbf{M^\prime} = \mathbf{T}(-\theta)\cdot\mathbf{M}\cdot\mathbf{T}(\theta)
```

# Examples
```jldoctest
julia> M = linear_polarizer();

julia> Mr = rotate(M, π/2);

julia> Mr ≈ linear_polarizer(π/2)
true
```

# See also
[`Mueller.rotation`](@ref)
"""
function rotate(mat::AbstractMatrix{T}, theta) where T
    r = rotation(T, theta)
    return r' * mat * r
end

"""
    linear_polarizer([T=Float64], theta=0; p=1)

A linear polarizer with the throughput axis given by `theta`, in radians, by default horizontal. The partial polarization can be given with the `p` keyword argument, which changes the intensity by a factor of `p^2/2`.

# Examples

```jldoctest
julia> M = linear_polarizer()
4×4 StaticArrays.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 0.5  0.5  0.0  0.0
 0.5  0.5  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> S = [1, 0, 0, 0]; # I, Q, U, V

julia> M * S # only horizontal component (+Q) remains
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
 0.5
 0.5
 0.0
 0.0
```
"""
function linear_polarizer(T::Type, theta=0; p=1)
    I = T(p^2 / 2)
    sin2t, cos2t = sincos(2 * theta)
    M = I * SA{T}[1 cos2t sin2t 0
                 cos2t cos2t^2 cos2t * sin2t 0
                 sin2t cos2t * sin2t sin2t^2 0
                 0 0 0 0]
    return M
end
linear_polarizer(theta=0; kwargs...) = linear_polarizer(Float64, theta; kwargs...)

"""
    wollaston([T=Float64], ordinary=true)

A Wollaston prism. The matrix for the ordinary beam is returned if `ordinary` is true, otherwise the matrix for the extra-ordinary beam is returned.

Note this is just a convenience wrapper around [`linear_polarizer`](@ref)

# Examples
```jldoctest
julia> M = wollaston()
4×4 StaticArraysCore.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 0.5  0.5  0.0  0.0
 0.5  0.5  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> M = wollaston(false)
4×4 StaticArraysCore.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
  0.5          -0.5           6.12323e-17  0.0
 -0.5           0.5          -6.12323e-17  0.0
  6.12323e-17  -6.12323e-17   7.4988e-33   0.0
  0.0           0.0           0.0          0.0

# See also
[`linear_polarizer`](@ref)
```
"""
wollaston(T::Type, ordinary=true) =ifelse(ordinary, linear_polarizer(T, 0), linear_polarizer(T, π/2))
wollaston(ordinary=true) = wollaston(Float64, ordinary)

# wave plates

"""
    waveplate([T=Float64], theta=0; delta=0)

A generic phase retarder (waveplate) with fast axis aligned with angle `theta`, in radians, 
and phase delay of `delta`, in radians, along the slow axis.

# Examples
```jldoctest
julia> M = waveplate(delta=π);

julia> M ≈ hwp()
true

julia> M = waveplate(π/4, delta=π/2);

julia> M ≈ qwp(π/4)
true
```

# See also
[`hwp`](@ref), [`qwp`](@ref), [`mirror`](@ref)
"""
function waveplate(T::Type, theta=0; delta=0)
    sin2t, cos2t = sincos(2 * theta)
    sind, cosd = sincos(delta)
    return SA{T}[1 0 0 0
                 0 cos2t^2 + sin2t^2*cosd cos2t * sin2t * (1 - cosd) sin2t * sind
                 0 cos2t*sin2t*(1-cosd) cos2t^2*cosd + sin2t^2 -cos2t * sind
                 0 -sin2t * sind cos2t * sind cosd]
end
waveplate(theta=0; kwargs...) = waveplate(Float64, theta; kwargs...)

"""
    hwp([T=Float64], theta=0)

A half-wave plate (HWP) with fast axis oriented at angle `theta`, in radians.

# Examples
```jldoctest
julia> M = hwp()
4×4 StaticArrays.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 1.0   0.0   0.0           0.0
 0.0   1.0   0.0           0.0
 0.0   0.0  -1.0          -1.22465e-16
 0.0  -0.0   1.22465e-16  -1.0

julia> S = [1, 1, 0, 0]; # I, Q, U, V

julia> M * S # allow +Q through unchanged
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
 1.0
 1.0
 0.0
 0.0

julia> rotate(M, π/8) * S # switch +Q to +U
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
  1.0
  1.9967346175427393e-16
  1.0
 -8.659560562354932e-17

```

# See also
[`waveplate`](@ref), [`qwp`](@ref)
"""
hwp(T::Type, theta=0) = waveplate(T, theta; delta=π)
hwp(theta=0) = hwp(Float64, theta)


"""
    qwp([T=Float64], theta=0)

A quarter-wave plate (QWP) with fast axis oriented at angle `theta`, in radians.

# Examples
```jldoctest
julia> M = qwp()
4×4 StaticArrays.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 1.0   0.0  0.0           0.0
 0.0   1.0  0.0           0.0
 0.0   0.0  6.12323e-17  -1.0
 0.0  -0.0  1.0           6.12323e-17

julia> S = [1, 1, 0, 0]; # I, Q, U, V

julia> M * S # allow +Q through unchanged
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
 1.0
 1.0
 0.0
 0.0

julia> qwp(-π/4) * S # switch +Q to +V
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
  1.0
  6.123233995736766e-17
 -6.123233995736765e-17
  1.0
```

# See also
[`waveplate`](@ref), [`hwp`](@ref)
"""
qwp(T::Type, theta=0) = waveplate(T, theta; delta=π/2)
qwp(theta=0) = qwp(Float64, theta)

"""
    mirror([T=Float64], theta=0, delta=π, r=1)

A reflective mirror with reflectance `r`, oriented at angle `theta`, in radians, compared to the reference frame of the light, and with phase shift `δ`, in radians. An ideal mirror will have perfect reflectance (1) and a half-wave phase shift (π).

# Examples
```jldoctest
julia> M = mirror()
4×4 StaticArrays.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 1.0  0.0   0.0           0.0
 0.0  1.0   0.0          -0.0
 0.0  0.0  -1.0           1.22465e-16
 0.0  0.0  -1.22465e-16  -1.0

julia> S = [1, 1, 0, 0]; # I, Q, U, V

julia> M * S # no change
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
 1.0
 1.0
 0.0
 0.0

julia> mirror(π/4) * S # rotates polarized light
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
  1.0
 -1.0
  1.2246467991473532e-16
  1.2246467991473532e-16
```
"""
function mirror(T::Type, theta=0; delta=π, r=1)
    a = 0.5 * (r + 1)
    b = 0.5 * (r - 1)
    sin2t, cos2t = sincos(2 * theta)
    sin4t = sin(4 * theta)
    sind, cosd = sincos(delta)
    sqrm = sqrt(r)
    M = SA{T}[a          b * cos2t                            b * sin2t                           0
              b * cos2t  a * cos2t^2 + sqrm * cosd * sin2t^2 (a - sqrm * cosd) * sin4t * 0.5     -sqrm * sind * sin2t
              b * sin2t (a - sqrm * cosd) * sin4t * 0.5       a * sin2t^2 + sqrm * cosd * cos2t^2 sqrm * sind * cos2t
              0          sqrm * sind * sin2t                 -sqrm * sind * cos2t                 sqrm * cosd]
    return M
end
mirror(theta=0; kwargs...) = mirror(Float64, theta; kwargs...)
