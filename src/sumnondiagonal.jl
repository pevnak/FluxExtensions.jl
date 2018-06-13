function sumnondiagonal(x::Matrix{T}) where {T}
  assert(size(x,1) == size(x,2))
  s = zero(T)
  @inbounds for j in 1:size(x,2)
    for i in 1:size(x,1)
      s += (i == j) ? zero(T) : x[i,j]
    end 
  end
  s
end

function backsumnondiagonal(x::AbstractMatrix,Δ::T) where {T}
  assert(size(x,1) == size(x,2))
  s = zeros(T,size(x))
  @inbounds for j in 1:size(x,2)
    for i in 1:size(x,1)
      s[i,j] = (i == j) ? zero(T) : Δ
    end 
  end
  s
end

sumnondiagonal(x::Flux.Tracker.TrackedMatrix) = Flux.Tracker.track(sumnondiagonal,x)
back(::typeof(sumnondiagonal), Δ, x::AbstractMatrix) = Flux.Tracker.@back(x, backsumnondiagonal(x, Δ))