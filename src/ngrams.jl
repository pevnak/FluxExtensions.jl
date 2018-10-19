"""
  ngrams!(o,x,n::Int,b::Int)

  store indexes of `n` grams of `x` with base `b` to `o`

"""
function ngrams!(o,x::T,n::Int,b::Int) where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}}
  @assert b > maximum(x)
  @assert length(o) >= length(x) + n - 1
  idx = 0
  for (i,v) in enumerate(x) 
    idx = idx*b + v
    idx = (i>n) ? mod(idx,b^n) : idx 
    o[i] = idx
  end
  for i in 1:n-1
    idx = mod(idx,b^(n-i))
    o[length(x) + i] = idx
  end
  o
end

"""
  ngrams(x,n::Int,b::Int)

  indexes of `n` grams of `x` with base `b`

"""
ngrams(x::T,n::Int,b::Int) where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}} =   ngrams!(zeros(Int,length(x) + n - 1),x,n,b)
ngrams(x::T,n::Int,b::Int) where {T<:AbstractString} = ngrams(codeunits(x),n,b)

"""
  function countngrams!(o,x,n::Int,b::Int)

  counts number of of `n` grams of `x` with base `b` to `o` and store it to o

"""
function countngrams!(o,x::T,n::Int,b::Int) where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}}
  length(x) > 0 && @assert b > maximum(x)
  idx = 0
  for (i,v) in enumerate(x) 
    idx = idx*b + v
    idx = (i>n) ? mod(idx,b^n) : idx 
    o[mod(idx, length(o))+1] += 1
  end
  for i in 1:n-1
    idx = mod(idx,b^(n-i))
    o[mod(idx, length(o))+1] += 1
  end
  o
end

countngrams!(o,x::T,n::Int,b::Int) where {T<:AbstractString} = countngrams!(o,codeunits(x),n,b)

"""
  function countngrams(x,n::Int,b::Int)

  counts number of of `n` grams of `x` with base `b` to `o`

"""
countngrams(x,n::Int,b::Int,m) = countngrams!(zeros(Int,m),x,n,b)
function countngrams(x::Vector{T},n::Int,b::Int,m) where {T<:AbstractString}
  o = zeros(Int,m,length(x))
  for (i,s) in enumerate(x)
    countngrams!(view(o,:,i),x[i],n,b)
  end
  o
end


string2ngrams(x::T,n,m) where {T <: AbstractArray{I} where I<: AbstractString} = countngrams(Vector(x[:]),n,257,m)
string2ngrams(x::T,n,m) where {T<: AbstractString} = countngrams(x, n, 257, m)
string2ngrams(x,n,m) = x

# struct NGramIterator{T} where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}}
struct NGramIterator{T}
	s::T
	n::Int
	b::Int
end

# NGramIterator(s::AbstractString) = NGramIterator(codeunits(s), 3, 256)
# NGramIterator(s::AbstractString, n, b) = NGramIterator(codeunits(s), n, b)

Base.length(it::NGramIterator) = length(it.s) + it.n - 1
# length(x) > 0 && @assert b > maximum(x)

function Base.iterate(it::NGramIterator, s = (0, 1))
	idx, i = s
	b, n = it.b, it.n 
	if i <= length(it.s)
		idx = idx * b + it.s[i]
		idx = (i>n) ? mod(idx,b^n) : idx 
		return(idx, (idx, i + 1))
	elseif i < length(it.s) + n
		idx = mod(idx,b^(n - (i - length(it.s))))
		return(idx, (idx, i + 1))
	else 
		return(nothing)
	end
end

function mul(A::Matrix, B::Array{S}) where {S<:AbstractString}
  mA, nA = size(A)
  nB = length(B)
  C = zeros(eltype(A), mA, nB)
  @inbounds for (jB, s) in enumerate(B)
      for iB in NGramIterator(codeunits(B[jB]), 3, 257)
      	miB = mod(iB, nA) + 1
        for iA in 1:mA
            C[iA, jB] += A[iA, miB]
        end
      end
  end
  return C
end

function multrans(A::Matrix, B::Array{S}) where {S<:AbstractString}
  mA, nA = size(A)
  mB = length(B)
  C = zeros(eltype(A), mA, mB)
  @inbounds for (jB, s) in enumerate(B)
      for iB in NGramIterator(codeunits(B[jB]), 3, 257)
	      	miB = mod(iB, nA) + 1
          for iA in 1:mA
              C[iA, miB] += A[iA, jB]
          end
      end
  end
  return C
end

a::Flux.Tracker.TrackedMatrix * b::Array{S} where {S<:AbstractString} = Flux.Tracker.track(mul, a, b)
a::Matrix * b::Array{S} where {S<:AbstractString} = mul(a, b)
Flux.Tracker.@grad function mul(a::AbstractMatrix, b::Array{S}) where {S<:AbstractString}
  return mul(Flux.data(a),b) , Δ -> (multrans(Δ, b),nothing)
end