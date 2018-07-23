struct ResDense{A,B,D}
	a::A 		# Dense layer to align dimensions
	b::B 		# Matrix with parameters
	d::D 		# nothing or Matrix with parameters used to align dimensions
end

ResDense(d::Int,k::Int,σ = NNlib.relu) = (d == k) ? ResDense(Dense(k,k,σ),Flux.param(Flux.glorot_normal(k,k)),nothing) : ResDense(Dense(k,k,σ),Flux.param(Flux.glorot_normal(k,k)),Flux.param(Flux.glorot_normal(k,d)))

(m::ResDense{A,B,D})(x) where {A,B,D<:Void} = x + m.b*(m.a(x))
function (m::ResDense{A,B,D})(x) where {A,B,D<:Flux.TrackedArray}
	xx = m.d*x
	xx + m.b*(m.a(xx))
end

function Base.show(io::IO, l::ResDense)
  print(io, "ResDense(");
  print(io, l.d == nothing ? l.a : (size(l.d.data, 2), size(l.d.data, 1)))
  print(io, ")")
end

Flux.treelike(ResDense)
adapt(T, m::ResDense) = ResDense(adapt(T,m.a),adapt(T,m.b),adapt(T,m.d))
