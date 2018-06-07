adapt(T, x::Array) = T.(x)
adapt(T, m::Flux.Dense) = Flux.Dense(adapt(T,m.W),adapt(T,m.b),m.σ)

function logitcrossentropy(logit, y) 
  logit = logit .-maximum(logit,1)
  ypred = logit .- log.( sum( exp.( logit),1))
  - sum(y .* ypred)/size(logit,2)
end

entropy(x,dim::Int=1) = -mean(sum(x .* log.(x),dim))

function confusion(y,pred,k)
	c = zeros(Int,k,k)
	foreach(i -> c[i[1],i[2]] += 1, zip(y,pred))
	c 
end
