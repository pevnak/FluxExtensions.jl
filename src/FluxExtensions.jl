module FluxExtensions
using Flux

include("layers/resdense.jl")
include("utils.jl")

export ResDense, adapt	
end # module
