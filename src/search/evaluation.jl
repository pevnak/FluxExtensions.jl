using MLDataPattern
using DataFrames

function crossvalidate(data,target,createmodel,loss,bs=100,n=10000)
	function onerun(xTrn,xVal)
		m = createmodel()
		opt = Flux.Optimise.ADAM(params(m))
		Flux.train!((xx...) -> loss(m(getobs(xx[1])),getobs(xx[2])), RandomBatches(xTrn,bs,n), opt)
		(mean(Flux.argmax(m(getobs(xTrn[1]))) .!= Flux.argmax(getobs(xTrn[2]))),
		mean(Flux.argmax(m(getobs(xVal[1]))) .!= Flux.argmax(getobs(xVal[2]))))
	end
	map(x -> onerun(x[1],x[2]), kfolds(shuffleobs((data,target)),5))
end

function gridsearch(cr,parameters)
	errs = map(p -> (p,cr(p)),parameters)
	i = indmin(map(v -> mean(v[2]),errs))
	return(errs[i][1],errs[2][2],errs)
end

function gridsearch(cr,parameters,names::Vector{Symbol},ofname::String = "")
	df = []
	for p in parameters
		dff = DataFrame([cr(p)],[:errs])
		dff[:fold] = 1:size(dff,1)
		foreach(pn -> dff[pn[1]] = pn[2],zip(names,p))
		push!(df,dff)
		!isempty(ofname) && CSV.write(ofname, dff, append = true, header = true)
		display(dff)
	end
	vcat(df...)
end
gridsearch(cr,parameters,names::Vector{String},ofname::String = "") = gridsearch(cr,parameters,Symbol.(names),ofname)
