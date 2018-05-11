module FluxExtensions
using Adapt
using Flux
import Adapt: adapt

include("layers/resdense.jl")
include("layers/layerbuilder.jl")
include("utils.jl")
include("learn.jl")
include("plot.jl")
include("sparse.jl")
include("search/evaluation.jl")


freeze(m) = Flux.mapleaves(Flux.Tracker.data,m)

function regcov(x)
	xx = x .- mean(x,2)
	mean(xx * xx')
end


restoremodel!(m,p) = foreach(a -> copy!(Flux.data(a[1]),a[2]),zip(Flux.params(m),p))


export ResDense
end # module
