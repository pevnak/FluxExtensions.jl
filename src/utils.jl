adapt(T, x::Array) = T.(x)
adapt(T, m::Flux.Dense) = Flux.Dense(adapt(T,m.W),adapt(T,m.b),m.σ)

using FluxExtensions: logsumexp

logit_cross_entropy(logit, y) = -sum(y.*(logit.-logsumexp(logit, 1))) / size(logit,2)

weighted_logit_cross_entropy(logit, y::AbstractVector, w) = weighted_logit_cross_entropy(logit, Flux.onehotbatch(y, 1:size(logit,1)), w)
weighted_logit_cross_entropy(logit, y::AbstractMatrix, w) = -sum(y.*w.*(logit.-logsumexp(logit, 1)))

entropy(x,dim::Int=1) = -mean(sum(x .* log.(x),dim))

function confusion(y,pred,k)
	c = zeros(Int,k,k)
	foreach(i -> c[i[1],i[2]] += 1, zip(y,pred))
	c
end
