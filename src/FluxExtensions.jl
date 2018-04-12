module FluxExtensions
using Adapt
using Flux

include("layers/resdense.jl")
include("layers/layerbuilder.jl")
include("utils.jl")
include("learn.jl")
include("plot.jl")


freeze(m) = Flux.mapleaves(Flux.Tracker.data,m)

function regcov(x)
	xx = x .- mean(x,2)
	mean(xx * xx')
end


export ResDense
end # module
