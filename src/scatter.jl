function startofbags(k::Vector{Int})
  s = zeros(Int,length(k))
  for i in 2:length(k)
    s[i] = s[i-1] + k[i - 1]
  end
  s
end

"""
    scatter(x::Matrix,k::Int)
    scatter(x::Matrix,k::Vector{Int})

    repeat each column of `x` k-times
    repeat j-th column of `x` k[j]-times

    ```juliadoc
    x = [1 3; 2 4]
    julia> FluxExtensions.scatter(x, 3)

    2×6 Array{Int64,2}:
     1  1  1  3  3  3
     2  2  2  4  4  4
     ```
"""
function scatter(x::Matrix,k::Int)
  xx = similar(x,size(x,1),k*size(x,2))
  @inbounds for j in 1:size(x,2)
    for b in 1:k
      for i in 1:size(x,1)
        xx[i,(j-1)*k+b] = x[i,j]
      end
    end 
  end
  xx
end

function scatter(x::Matrix,k::Vector{Int})
  @assert size(x,2) >= length(k)
  xx = similar(x,size(x,1),sum(k))
  starts = startofbags(k)
  for j in 1:length(k)
    for b in 1:k[j]
      for i in 1:size(x,1)
        xx[i,starts[j] + b] = x[i,j]
      end
    end 
  end
  xx
end


"""
    gather(x,k::Int) 
    gather(f,x,k::Vector{Int}) 
    gather(f,x,k::Int) 

    sum blocks of size k-columns
    reduce with function `f`

    ```juliadoc
    julia>  x = [ 1 1 1 3 3 3; 2 2 2 4 4 4]
    julia> FluxExtensions.gather(x,3)
    2×2 Array{Int64,2}:
     3   9
     6  12
    ```
"""
function gather(x,k::Int) 
  d, l = size(x, 1), div(size(x,2), k)
  reshape(sum(reshape(x, d, k, l), dims=2), d, l)
end

function gather(x,k::Vector{Int})
  xx = similar(x,size(x,1),length(k)) .= 0
  starts = startofbags(k)
  for j in 1:length(k)
    for b in 1:k[j]
      for i in 1:size(x,1)
        xx[i,j] += x[i,starts[j] + b]
      end
    end 
  end
  xx
end

function gather(f::Function, x, k::Int) 
  d, l = size(x,1), div(size(x,2),k)
  reshape(sum(reshape(x,d,k,l),2),d,l)
end

function logsumexp(x,k::Int)
  xm = maximum(x,dims = k)
  xm .+ log.( sum( exp.(x .- xm),dims = k))
end

function logsumexp(x)
  xm = maximum(x)
  xm .+ log.( sum( exp.(x .- xm)))
end

scatter(x::Flux.Tracker.TrackedMatrix, k) = Flux.Tracker.track(scatter,x, k )
Flux.Tracker.@grad function scatter(x,k)
  return(scatter(Flux.data(x), k), Δ -> (gather(Δ, k), nothing))
end
