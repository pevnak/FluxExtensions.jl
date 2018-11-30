using FluxExtensions, Test, Flux

# we want to verify that threaded and non-threaded versions of
# learning returns the same function value and gradient
@testset "support for threaded learning" begin
	datas = [randn(3, 10) for i in 1:3];
	model = Dense(3, 1);
	loss(model, data) = sum(model(data));
	f1 = 0.0
	for i in 1:length(datas)
		f = loss(model, datas[i])
		@show f
		f1 += Flux.data(f)
		Flux.back!(f)
	end
	f1 /= length(datas)
	FluxExtensions.scalegrads!(model, 1/length(datas));

	models = [deepcopy(model) for _ in 1:length(datas)];
	f2 = FluxExtensions.∇loss!(loss, models, datas, nothing);
	@test f1 ≈ f2
	g1 = map( p -> Flux.Tracker.grad(p), params(model))
	g2 = map( p -> Flux.Tracker.grad(p), params(models[1]))
	@test g1 ≈ g2
end