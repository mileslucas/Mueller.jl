# Mueller.jl

[![Build Status](https://github.com/JuliaPhysics/Mueller.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaPhysics/Mueller.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/M/Mueller.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)
[![Coverage](https://codecov.io/gh/JuliaPhysics/Mueller.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaPhysics/Mueller.jl)
[![License](https://img.shields.io/github/license/JuliaPhysics/Mueller.jl?color=yellow)](LICENSE)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaphysics.github.io/Mueller.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaphysics.github.io/Mueller.jl/dev)

[Mueller matrices](https://en.wikipedia.org/wiki/Mueller_calculus) for common optical components such as polarizers, phase retarders, and attenuating filters. The matrices are built using [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for speed and can be arbitrarily rotated.

## Installation

**Note** not yet registered, please add the repo manually

```julia
julia>] add Mueller
```

## Usage

Import the library like any other Julia package

```julia
julia> using Mueller
```

Mueller.jl provides building blocks for commonn components. Here I generate the Mueller matrix for an optical system comprising three linear polarizers, each rotated 45 degrees from the one prior.

```julia
julia> M = linear_polarizer(0) * linear_polarizer(π/4) * linear_polarizer(π/2)
4×4 StaticArrays.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 0.125  -0.125  1.53081e-17  0.0
 0.125  -0.125  1.53081e-17  0.0
 0.0     0.0    0.0          0.0
 0.0     0.0    0.0          0.0
```

you'll notice some roundoff due to the finite precision of `π/4`, you can avoid this by using [Unitful.jl](https://github.com/PainterQubits/Unitful.jl)

```julia
julia> using Unitful: °

julia> M = linear_polarizer(0°) * linear_polarizer(45°) * linear_polarizer(90°)
4×4 StaticArrays.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 0.125  -0.125  0.0  0.0
 0.125  -0.125  0.0  0.0
 0.0     0.0    0.0  0.0
 0.0     0.0    0.0  0.0
```

let's see what happens when completely unpolarized light passes through these filters. We can represent light using the [Stokes vector](https://en.wikipedia.org/wiki/Stokes_parameters)

```julia
julia> S = [1, 0, 0, 0] # I, Q, U, V

julia> Sp = M * S
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
 0.125
 0.125
 0.0
 0.0
```

the output vector has 1/8 the total intensity of the original light, and it is 1/8 polarized in the +Q direction. This demonstrates the somewhat paradoxical quantum behavior of light ([Bell's Theroem](https://en.wikipedia.org/wiki/Bell%27s_theorem), inspired by [this video](https://www.youtube.com/watch?v=zcqZHYo7ONs)): even though the light passes through two orthogonal linear polarizers (the 0° and 90° ones) because the wave equation operates probabilistically, 50% passes through the first polarizer, 50% of that light passes through the 45° polarizer, and then 50% of the remaining light passes through the final polarizer, combining to 1/8 of the original light.

## Contributing and Support

If you would like to contribute, feel free to open a [pull request](https://github.com/JuliaPhysics/Mueller.jl/pulls). If you want to discuss something before contributing, head over to [discussions](https://github.com/JuliaPhysics/Mueller.jl/discussions) and join or open a new topic. If you're having problems with something, please open an [issue](https://github.com/JuliaPhysics/Mueller.jl/issues).
