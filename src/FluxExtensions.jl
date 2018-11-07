module FluxExtensions
using Adapt, Flux, Statistics
import Adapt: adapt
using LinearAlgebra

include("layers/resdense.jl")
include("layers/layerbuilder.jl")
include("utils.jl")
include("sparse.jl")
include("pdfs.jl")
include("sumnondiagonal.jl")
include("scatter.jl")
include("lbfgs.jl")
include("search/evaluation.jl")
include("triangularloss.jl")
include("learn.jl")
include("productlayer.jl")

freeze(m) = Flux.mapleaves(Flux.Tracker.data,m)

function regcov(x,dim::Int=2)
	xx = x .- mean(x,dim)
	xx = xx * xx' / size(x,dim);
	mean(xx.^2)
end

function regcovmd(x,dim::Int=2)
	xx = (x .- mean(x,dim))./(std(x,dim))
	xx = xx * xx' / size(x,dim);
	xx = xx - diagm(diag(xx))
	mean(xx.^2)
end

function corr(x)
  xx = x .- mean(x,2)
  c = xx*xx'/size(xx,2)
  s = sqrt.(diag(c) + 1f-6)
  c./(s*s')
end



restoremodel!(m,p) = foreach(a -> copy!(Flux.data(a[1]),a[2]),zip(Flux.params(m),p))


export ResDense, restoremodel, layerbuilder, ProductLayer, to32
export logit_cross_entropy, weighted_logit_cross_entropy, classweightvector
end # module
