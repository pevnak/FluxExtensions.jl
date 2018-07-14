@testset "Testing ResDense" begin 
	@test size(ResDense(3,3,NNlib.relu)(randn(3,4))) == (3,4)
	@test size(ResDense(5,3,NNlib.relu)(randn(5,4))) == (3,4)
	@test length(Flux.params(ResDense(5,3,NNlib.relu))) == 4
	@test length(Flux.params(ResDense(3,3,NNlib.relu))) == 3
end

