module FluxExtensions
using Flux

include("layers/resdense.jl")
include("layers/layerbuilder.jl")
include("utils.jl")
include("learn.jl")

export ResDense
end # module
