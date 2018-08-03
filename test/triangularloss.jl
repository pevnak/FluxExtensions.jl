@testset "FluxExtensions.triangularloss" begin
	d = rand(10)
	y = rand(1:3,10)
	g = Flux.Tracker.ngradient((d) -> FluxExtensions.triangularloss(d,y), d)[1];
	@test all( g - FluxExtensions.triangularloss_back(d, y, 1.0) .== 0 )
	d = Flux.param(d)
	Flux.back!(FluxExtensions.triangularloss(d,y))
	@test all(d.grad - g .== 0)
end
