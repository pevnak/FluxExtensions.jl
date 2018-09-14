using Adapt:adapt
@testset "testing adapt to Float32" begin
	@test typeof(adapt(Float32,Flux.param(randn(3,3,3)))) <: Flux.TrackedArray{Float32,3}
	@test typeof(adapt(Float32,Flux.param(randn(3,3)))) <: Flux.TrackedArray{Float32,2}
	@test typeof(adapt(Float32,Flux.param(randn(3)))) <: Flux.TrackedArray{Float32,1}
	@test typeof(adapt(Float32,Flux.Dense(3,3)).W) <: Flux.TrackedArray{Float32,2}
	@test typeof(adapt(Float32,Flux.Dense(3,3)).b) <: Flux.TrackedArray{Float32,1}
	@test typeof(adapt(Float32, FluxExtensions.ResDense(3, 3)).a.W) <: Flux.TrackedArray{Float32,2}
end
