@testset "Testing ResDense" begin
	@test size(FluxExtensions.ResDense(3,3,NNlib.relu)(randn(3,4))) == (3,4)
	@test size(FluxExtensions.ResDense(5,3,NNlib.relu)(randn(5,4))) == (3,4)
	@test length(Flux.params(FluxExtensions.ResDense(5,3,NNlib.relu))) == 4
	@test length(Flux.params(FluxExtensions.ResDense(3,3,NNlib.relu))) == 3
end
