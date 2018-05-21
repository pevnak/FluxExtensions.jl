adapt(T, x::Array) = T.(x)
adapt(T, m::Flux.Dense) = Flux.Dense(adapt(T,m.W),adapt(T,m.b),m.σ)





# function logitcrossentropy(logit, y) 
#   logit = logit .-maximum(logit,1)
#   ypred = logit .- log.( sum( exp.( logit),1))
#   -sum(mean(y .* ypred,2))
# end

function logitcrossentropy(logit, y) 
  logit = logit .-maximum(logit,1)
  ypred = logit .- log.( sum( exp.( logit),1))
  - sum(y .* ypred)/size(logit,2)
end


stable_softmax(x) = softmax(x .- maximum(x,1))