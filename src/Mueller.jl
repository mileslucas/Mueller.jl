module Mueller

using StaticArrays

export rotation, rotate, linear_polarizer, hwp, qwp, mirror, waveplate, wollaston

include("components.jl")
include("plots.jl")

end # module
