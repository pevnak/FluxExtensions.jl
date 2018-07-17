"""
		function layerbuilder(d::Int,k::Int,o::Int,n::Int,ftype::String,lastlayer::String = "",ltype::String = "Dense")

		create a chain with `n` layers of with `k` neurons with transfer function `ftype`.
		input and output dimension is `d` / `o` 

		If lastlayer is no specified, all layers use the same function.
		If lastlayer is "linear", then the last layer is forced to be Dense.

```juliadoctest
julia> FluxExtensions.layerbuilder(4,11,1,3,"relu")
Chain(Dense(4, 11, NNlib.relu), Dense(11, 11, NNlib.relu), Dense(11, 1, NNlib.relu))

julia> FluxExtensions.layerbuilder(4,11,1,3,"relu","tanh")
Chain(Dense(4, 11, NNlib.relu), Dense(11, 11, NNlib.relu), Dense(11, 1, tanh))

julia> FluxExtensions.layerbuilder(4,11,1,3,"relu","tanh","ResDense")
Chain(ResDense(Dense(11, 11, NNlib.relu)), ResDense(Dense(11, 11, NNlib.relu)), ResDense(Dense(1, 1, tanh)))

julia> FluxExtensions.layerbuilder(4,11,1,3,"relu","linear","ResDense")
Chain(ResDense(Dense(11, 11, NNlib.relu)), ResDense(Dense(11, 11, NNlib.relu)), Dense(11, 1))
```

"""
layerbuilder(k::Vector{Int},l::Vector,f::Vector) = Flux.Chain(map(i -> i[1](i[3],i[4],i[2]),zip(l,f,k[1:end-1],k[2:end]))...)
function layerbuilder(d::Int,k::Int,o::Int,n::Int,ftype::String,lastlayer::String = "",ltype::String = "Dense")
	ftype = (ftype == "linear") ? "identity" : ftype
	ls = Array{Any}(fill(eval(:($(Symbol(ltype)))),n))
	fs = Array{Any}(fill(eval(:($(Symbol(ftype)))),n))
	ks = vcat(d,fill(k,n-1)...,o)
	if !isempty(lastlayer)
		fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
		ls[end] = (lastlayer == "linear") ? Dense : ls[end]
	end
	layerbuilder(ks,ls,fs)
end

