struct ProductLayer{I, P, S}
	i::I
  p::P
  s::S
end

function ProductLayer(in::Integer, out::Integer, σ = identity)
	i = in == out ? identity : Dense(in, out)
	ProductLayer(i, Dense(in, out, σ), Dense(in, out, NNlib.σ))
end

Flux.@treelike ProductLayer

(a::ProductLayer)(x::AbstractArray) = (a.i(x) .+  a.s(x)) .* a.p(x)

function Base.show(io::IO, l::ProductLayer)
  print(io, "ProductLayer(", l.i, ", ", l.p, ", ", l.s, ")")
end


# struct ResDense{I, S}
# 	i::I
#   S::S
# end

# function ResDense(in::Integer, out::Integer, σ = identity)
# 	i = in == out ? identity : Dense(in, out)
# 	ResDense(i, Dense(in, out, σ))
# end

# Flux.@treelike ResDense

# (a::ResDense)(x::AbstractArray) = a.i(x) .+  a.p(x)

# function Base.show(io::IO, l::ResDense)
#   print(io, "ResDense(", l.i, ", ", l.p, ")")
# end
