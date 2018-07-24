adapt(T, x::Array) = T.(x)
adapt(T, m::Flux.Dense) = Flux.Dense(adapt(T,m.W),adapt(T,m.b),m.σ)

logit_cross_entropy(logit, y) = -sum(y.*(logit.-logsumexp(logit, 1))) / size(logit,2)

weighted_logit_cross_entropy(logit, y, w::AbstractVector) = weighted_logit_cross_entropy(logit, y, w')
weighted_logit_cross_entropy(logit, y::AbstractVector, w::AbstractMatrix) = weighted_logit_cross_entropy(logit, Flux.onehotbatch(y, 1:size(logit,1)), w)
weighted_logit_cross_entropy(logit, y::AbstractMatrix, w::AbstractMatrix) = -sum(y.*w.*(logit.-logsumexp(logit, 1)))

entropy(x,dim::Int=1) = -mean(sum(x .* log.(x),dim))

using StatsBase

function classweightvector(::Type{S},y::Vector{T}) where {S<:AbstractFloat,T<:Integer}
  w=ones(S,size(y,1))
  classsizes = StatsBase.countmap(y)
  for j in 1:size(y,1)
    w[j] = S(1/size(y,2))/classsizes[y[j]]
  end
  return(w./sum(w))
end

function classweightvector(y::AbstractArray{T},classweights::Vector{S}) where {T<:Integer,S<:AbstractFloat}
  classweights = classweights./sum(classweights)
  w=ones(S,size(y,1))
  classsizes = StatsBase.countmap(y)
  for j in 1:size(y,1)
    w[j] = classweights[y[j]]/classsizes[y[j]]
  end
  if sum(isnan.(w))>0 || sum(isinf.(w))>0
    save("error.jld","y",y,"classweights",classweights)
    error("nans or infs in classweights")
  end
  w
end

function confusion(y,pred,k)
	c = zeros(Int,k,k)
	foreach(i -> c[i[1],i[2]] += 1, zip(y,pred))
	c
end
