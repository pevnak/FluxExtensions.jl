using FluxExtensions
using Base.Test
using Flux.Tracker, Base.Test, NNlib
using Flux.Tracker: TrackedReal, gradcheck


include("layers/utils.jl")
include("layers/resdense.jl")


@testset "sumnondiagonal" begin 
	x = randn(4,4) 
	@test abs(FluxExtensions.sumnondiagonal(x) .- (sum(x) - sum(diag(x)))) < 1e-14
	@test gradcheck(FluxExtensions.sumnondiagonal,x)
end