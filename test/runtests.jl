using Test
using Flux, NNlib, FluxExtensions
using Flux.Tracker, NNlib
using Flux.Tracker: TrackedReal, gradcheck


# include("utils.jl")
include("layers/resdense.jl")
# include("triangularloss.jl")


@testset "sumnondiagonal" begin
	x = randn(4,4)
	@test abs(FluxExtensions.sumnondiagonal(x) .- (sum(x) - sum(diag(x)))) < 1e-14
	@test gradcheck(FluxExtensions.sumnondiagonal,x)
end


@testset "scatter" begin
	x = randn(4,4)
	@test gradcheck((x) -> sum(FluxExtensions.scatter(x,4)),x)
	@test gradcheck((x) -> sum(FluxExtensions.scatter(x,[1,2,3,4])),x)
end
