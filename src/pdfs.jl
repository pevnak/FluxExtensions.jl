using Flux.Tracker: TrackedArray, TrackedReal
"""
	pairwisel2(x,y)
	
	pairwise distances between `x` and `y` using Euclidean Distance.
	
"""
pairwisel2(x,y) = -2 .* x' * y .+ sum(x.^2, dims = 1)' .+ sum(y.^2, dims = 1)


"""
	scaled_pairwisel2(x::M, y::M, σ::M) where {M<:Matrix}
	when x,y, and σ are matrices, than ``o_{i,j} = \\frac{x_i - y_j}{σ_i}`` where `σ` are treated as a column vectors.

	if σ is:
		- Number means a σ shared by all `x`
		- Vector means a σ is diagonal shared by all `x`
		- Transpose means each column of `x` has its own scalar σ
		- Matrix means each column of `x` has its own scalar diagonal σ


"""
scaled_pairwisel2(x, y, σ::T) where {T<: Union{Real,TrackedReal}} = pairwisel2(x, y) ./ σ^2
scaled_pairwisel2(x, y, σ::T) where {T<:Union{AbstractVector, TrackedArray{T, N, A} where {T, N, A<: AbstractVector}}} = pairwisel2(x ./σ, y./σ)
scaled_pairwisel2(x, y, σ::T) where {T<:Union{Transpose, TrackedArray{T, N, A} where {T, N, A<: Transpose}}} = pairwisel2(x, y) ./ (σ'.^2)

function _scaled_pairwisel2(x, y, σ)
	# @assert ((size(y, dims = 1) == size(σ, dims = 1) ) && (size(x, dims = 2) == size(σ, dims = 2)) && (size(x, dims = 1) == size(y, dims = 1)))
	@assert ((size(y, 1) == size(σ, 1) ) && (size(x, 2) == size(σ, 2)) && (size(x, 1) == size(y, 1)))
	o = zeros(size(x,2), size(y,2))
	@inbounds for i in 1:size(y,2)
		for j in 1:size(x,2)
			for k in 1:size(x,1)
				o[j,i] += ((x[k,j] - y[k,i])/σ[k,j])^2
			end 
		end 
	end
	o
end

function _scaled_pairwisel2_back(Δ, x, y, σ)
	@assert ((size(y, 1) == size(σ, 1) ) && (size(x, 2) == size(σ, 2)) && (size(x, 1) == size(y, 1)))
	∇x = zero(x)
	∇y = zero(y)
	∇σ = zero(σ)
	@inbounds for i in 1:size(y,2)
		for j in 1:size(x,2)
			for k in 1:size(x,1)
				δ = x[k,j] - y[k,i]
				∇x[k,j] += 2*Δ[j,i]*δ/σ[k,j]^2
				∇y[k,i] -= 2*Δ[j,i]*δ/σ[k,j]^2
				∇σ[k,j] -= 2*Δ[j,i]*δ^2/σ[k,j]^3
			end 
		end 
	end
	(∇x, ∇y, ∇σ)
end

scaled_pairwisel2(x::Matrix, y::Matrix, σ::Matrix) = _scaled_pairwisel2(x, y, σ)
scaled_pairwisel2(x, y, σ::S) where {S<:Union{Matrix, TrackedArray{T, N, A} where {T, N, A<: Matrix}}} = Flux.Tracker.track(scaled_pairwisel2, x, y, σ)
Flux.Tracker.@grad function scaled_pairwisel2(x, y, σ)
  return(_scaled_pairwisel2(Flux.data(x), Flux.data(y), Flux.data(σ)), Δ -> _scaled_pairwisel2_back(Flux.data(Δ), Flux.data(x), Flux.data(y), Flux.data(σ)))
end

"""
	pdf_normal(x,c,σ2::T)

	probability density of Normal Distribution of samples in `x` (each column is 
	one sample) with respect to a series of Normal Distributions defined 
	by centers in `c` (each columns is one center) and standard deviation σ
	if σ is:

		- Number means a σ shared by all centers
		- Vector means a σ is diagonal shared by all centers
		- Transpose means each center has its own scalar σ
		- Matrix means each center has its own scalar diagonal σ

	This should be compatible with Flux
"""
pdf_normal(x, c, σ ::T) where {T<:Real} = exp.(- 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π*σ^2)/2)
pdf_normal(x, c, σ ::T) where {T<:Union{Transpose, TrackedArray{T, N, A} where {T, N, A<: Transpose}}} = exp.(- 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π)/2 .- size(x,1)*log.(σ'))
pdf_normal(x, c, σ ::T) where {T<:AbstractVector} = exp.(- 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π)/2 .- sum(log.(σ)))
pdf_normal(x, c, σ ::T) where {T<:AbstractMatrix} = exp.(- 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π)/2 .- sum(log.(σ), 1)')

log_normal(x, c, σ ::T) where {T<:Real} = - 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π*σ^2)/2
log_normal(x, c, σ ::T) where {T<:Union{Transpose, TrackedArray{T, N, A} where {T, N, A<: Transpose}}} = - 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π)/2 .- size(x,1)*log.(σ')
log_normal(x, c, σ ::T) where {T<:AbstractVector} = - 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π)/2 .- sum(log.(σ))
log_normal(x, c, σ ::T) where {T<:AbstractMatrix} = - 0.5 .* scaled_pairwisel2(c, x, σ) .- size(x,1)*log(2π)/2 .- sum(log.(σ), 1)'
log_normal(x) = - sum((@. x^2), 1)/2 - size(x,1)*log(2π)/2
log_normal(x,μ) = - sum((@. (x - μ)^2), 1) / 2 - size(x,1)*log(2π)/2

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

log_bernoulli(x::AbstractMatrix,θ::AbstractVector) = log.(θ)' * x
log_bernoulli(x::AbstractMatrix,θ::AbstractMatrix) = sum(x .* log.(θ),1)