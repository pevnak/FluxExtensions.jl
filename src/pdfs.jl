"""
	pairwisel2(x,y)
	pairwisel2(x, y::Y, σ::S)

	pairwise distances between `x` and `y` using Euclidean Distance 

"""
pairwisel2(x,y) = -2 .* x' * y .+ sum(x.^2,1)' .+ sum(y.^2,1)
function pairwisel2(x::M, y::M, σ::M) where {M<:Matrix}
	# assert((size(y, 1) = size(σ, 1) ) && (size(y, 2) == size(σ, 2)))
	o = zeros(size(x,2), size(y,2))
	for i in 1:size(y,2)
		for j in 1:size(x,2)
			for k in 1:size(x,1)
				o[j,i] += ((x[k,j] - y[k,i])/σ[k,i])^2
			end 
		end 
	end
	o
end


function pairwisel2_back(Δ, x::M, y::M, σ::M) where {M<:Matrix}
	∇x = zero(x)
	∇y = zero(y)
	∇σ = zero(σ)
	for i in 1:size(y,2)
		for j in 1:size(x,2)
			for k in 1:size(x,1)
				δ = x[k,j] - y[k,i]
				∇x[k,j] += 2*Δ[j,i]*δ/σ[k,i]^2
				∇y[k,i] -= 2*Δ[j,i]*δ/σ[k,i]^2
				∇σ[k,i] -= 2*Δ[j,i]*δ^2/σ[k,i]^3
			end 
		end 
	end
	(∇x, ∇y, ∇σ)
end
pairwisel2(x, y, σ) = Flux.Tracker.track(pairwisel2_back, x, y, σ)
Flux.Tracker.@grad function pairwisel2_back(x, y, σ)
  return(pairwisel2(Flux.data(x), Flux.data(y), Flux.data(σ)), Δ -> pairwisel2_back(Δ, Flux.data(x), Flux.data(y), Flux.data(σ)))
end



"""
	pdf_normal(x,c,σ2::T)

	probability density of Normal Distribution of samples in `x` (each column is 
	one sample) with respect to a series of Normal Distributions defined 
	by centers in `c` (each columns is one center) and standard deviation σ

"""
pdf_normal(x,c,σ ::T) where {T<:Real} = exp.(- 0.5 .* pairwisel2(c, x) ./ σ^2	 .- size(x,1)*log(2π*σ^2)/2)
function pdf_normal(x,c,σ::T) where {T<:AbstractMatrix} 
	assert(size(c,2) == size(σ,2))

	mapreduce(vcat,1:size(σ,2)) do i
		cc = c[:,i] ./ σ[:,i]
		xx = x ./ σ[:,i]
		exp.(- 0.5 .* pairwisel2(cc, xx) .-size(x,1)*log(2π)/2 .- sum(log.(σ[:,i])))
	end
end


"""
		kldiv(μ,σ2)

		kl-divergence of a Gaussian min mean `μ` and diagonal variance `σ^2`
		to N(0,I)
"""
kldiv(μ,σ2) = - mean(sum((@.log(σ2) - μ^2 - σ2), 1))

"""
		log_normal(x,μ,σ2 = I)

		log-likelihood of x to the Normal with centre at mu

"""
log_normal(x) = - sum((@. x^2), 1)/2 - size(x,1)*log(2π)/2
log_normal(x,μ) = - sum((@. (x - μ)^2), 1) / 2 - size(x,1)*log(2π)/2
log_normal(x,μ,σ2::T) where {T<:Real} = - sum((@. (x - μ)^2/σ2 + log(σ2)), 1)/2 - size(x,1)*log(2π)/2

log_bernoulli(x::AbstractMatrix,θ::AbstractVector) = log.(θ)' * x
log_bernoulli(x::AbstractMatrix,θ::AbstractMatrix) = sum(x .* log.(θ),1)