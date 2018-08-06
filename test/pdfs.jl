using Distributions, Base.Test, FluxExtensions, Distances
using Flux: param
import FluxExtensions: pdf_normal, pairwisel2, scaled_pairwisel2

@testset "testing pairwise function" begin 
	x =  [-0.0953253   1.3719  -0.61826; -0.0734501  -1.4168   0.258718];
	y =  [0.515695   1.6764    0.64819   -0.348698;	 0.781103  -0.339036  0.899459  -0.718019];
	o =  [1.10361  3.20955  1.49937  0.479667;  5.56386  1.2543   5.88881  3.44875;  1.55874  5.62277  2.01444  1.02668;]
	σ =  [ 0.148651  0.982446  0.903678  0.421709;  0.3799    0.192967  0.884023  0.841242]	
	@test all(abs.(pairwisel2(x,y) - o) .< 1e-5)
	@test all(abs.(pairwisel2(x, y) - scaled_pairwisel2(x, y, fill(1,size(x)))) .< 1e-5)

	oo = hcat([pairwise(SqMahalanobis(diagm(1./σ[:,i].^2)), x,y[:,i:i]) for i in 1:size(y,2)]...)
	@test all(abs.(scaled_pairwisel2(y, x ,σ)' .- oo) .< 1e-10)

	@test all(abs.(Flux.data(scaled_pairwisel2(param(y), param(x), param(σ))) .- scaled_pairwisel2(y, x, σ)) .< 1e-10)
	@test Flux.Tracker.gradcheck(x -> sum(scaled_pairwisel2(y, x, σ)), x)
	@test Flux.Tracker.gradcheck(y -> sum(scaled_pairwisel2(y, x, σ)), y)
	@test Flux.Tracker.gradcheck(σ -> sum(scaled_pairwisel2(y, x, σ)), σ)
end

@testset "testing pdf of a multivariate normal distribution with a scalar σ" begin
	x = randn(2,3)
	c = randn(2,4)
	σ2 = 1.0
	o = pdf_normal(x,c,σ2)
	for i in 1:size(c,2)
		@test all(abs.(pdf(MvNormal(c[:,i],fill(1,2)),x) .- o[i,:]) .< 1e-8)
	end

	σ2 = 0.5
	o = pdf_normal(x,c,σ2)
	for i in 1:size(c,2)
		@test all(abs.(pdf(MvNormal(c[:,i],fill(σ2,2)),x) .- o[i,:]) .< 1e-8)
	end
	@test Flux.Tracker.gradcheck((x,c) -> sum(scaled_pairwisel2(c, x, σ)), x, c)
end

@testset "testing pdf of multivariate normal distribution with a vector σ" begin
	x = randn(2,3)
	c = randn(2,4)
	σ2 = rand(2)
	o = pdf_normal(x, c, σ2)
	for i in 1:size(c,2)
		@test all(abs.(pdf(MvNormal(c[:,i],σ2),x) .- o[i,:]) .< 1e-8)
	end
	@test Flux.Tracker.gradcheck((x,c,σ) -> sum(scaled_pairwisel2(c, x, σ)), x, c, σ2)
end

@testset "testing pdf of multivariate normal distribution with a RowVector σ" begin
	x = randn(2,3)
	c = randn(2,4)
	σ2 = RowVector(rand(4))
	o = pdf_normal(x, c, σ2)
	for i in 1:size(c,2)
		@test all(abs.(pdf(MvNormal(c[:,i],σ2[i]),x) .- o[i,:]) .< 1e-8)
	end
	@test Flux.Tracker.gradcheck((x,c,σ) -> sum(scaled_pairwisel2(c, x, σ)), x, c, σ2)
end


@testset "testing pdf of multivariate normal distribution with a matrix σ" begin
	x = randn(2,3)
	c = randn(2,4)
	σ2 = rand!(similar(c))
	o = pdf_normal(x, c, σ2)
	for i in 1:size(c,2)
		@test all(abs.(pdf(MvNormal(c[:,i],σ2[:,i]),x) .- o[i,:]) .< 1e-8)
	end
	@test Flux.Tracker.gradcheck((x,c,σ) -> sum(scaled_pairwisel2(c, x, σ)), x, c, σ2)
end

