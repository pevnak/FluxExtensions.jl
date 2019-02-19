using Printf, Serialization
function default_cb(step, load_time, calculation_time, training_loss)
	println(@sprintf("%d: training loss = %.4g load time= %.0fs (%.2fs per step)  compute time %.0fs (%.2fs per step) ",step , training_loss, load_time, load_time / step, calculation_time, calculation_time / step))
end

function wronggrad(model)
	ps = params(model)
	a = any(map(p -> any(isnan.(Flux.Tracker.grad(p))), ps)) 
	a && @warn("nan in a gradient")
	b = any(map(p -> any(isinf.(Flux.Tracker.grad(p))), ps)) 
	b && @warn("inf in a gradient")
	a || b
end
correctgrad(model) = !wronggrad(model)


"""
		serializestate(filename, model, loss, data)

		serialize the model and data to filename. Used mainly for debugging nans

"""		
function serializestate(filename, model, loss, data)
	open(filename, "w") do fid 
		serialize(fid, (model, loss, data))
	end
end

"""
		deserializestate(filename)

		serialize the model and data to filename. Used mainly for debugging nans
		
"""		
function deserializestate(filename)
	open(filename, "r") do fid 
		deserialize(fid)
	end
end

"""
	meminfo()

	runs the garbage collector and output content of `/proc/meminfo` to track the memory usage. Use only on linux

"""	
function meminfo()
	gc();gc();gc();gc();gc();gc();
	run(pipeline(`/bin/cat /proc/meminfo`, `/bin/grep MemFree`))
end

"""
		learn(model, loss, opt, data_provider, max_steps;cb = default_cb, breaks = 100, state_filename = nothing)

		learns `model` with `loss` function using `opt` optimalization algorithm


		`model` is a model accepting data provided by data provider, which can be as simple as 

"""
function learn(model, loss, opt, data_provider, max_steps; cb = default_cb, breaks = 100, state_filename = nothing)
	step, load_time, calculation_time, training_loss =0, 0.0, 0.0, 0.0
	state = nothing
	while step < max_steps
	  # load_time += @elapsed  data = state == nothing ?  Base.iterate(data_provider) : Base.iterate(data_provider, state)
	  load_time += @elapsed  data = data_provider()
	  data == nothing && break
	  # data, state = data
	  calculation_time += @elapsed training_loss +=  ∇loss!(loss, model, data, state_filename)
	  calculation_time += @elapsed opt()
	  step+=1
	  if mod(step,breaks) == 0
	  	cb(step, load_time, calculation_time, training_loss / breaks)
	  	serializestate("state.ser", model, loss, data)
	  	GC.gc();GC.gc();GC.gc();GC.gc();GC.gc()
	    training_loss = 0.0
	  end
	end
end

function learn(model, loss, opt::Tuple, data_provider, max_steps; cb = default_cb, breaks = 100, state_filename = nothing)
	step, load_time, calculation_time, training_loss =0, 0.0, 0.0, 0.0
	state = nothing
	while step < max_steps
	  # load_time += @elapsed  data = state == nothing ?  Base.iterate(data_provider) : Base.iterate(data_provider, state)
	  load_time += @elapsed  data = data_provider()
	  data == nothing && break
	  # data, state = data
	  calculation_time += @elapsed training_loss +=  ∇loss!(loss, model, data, state_filename)
	  calculation_time += @elapsed Flux.Optimise._update_params!(opt...)
	  step+=1
	  if mod(step,breaks) == 0
	  	cb(step, load_time, calculation_time, training_loss / breaks)
	  	serializestate("state.ser", model, loss, data)
	  	GC.gc();GC.gc();GC.gc();GC.gc();GC.gc()
	    training_loss = 0.0
	  end
	end
end

function ∇loss!(loss, model, data, filename = nothing)
	fVal = loss(model, data)
	if filename != nothing && (isnan(Flux.data(fVal)) || isinf(Flux.data(fVal))) 
		serializestate(filename, model, loss, data)
		error("nan or inf in function evaluation")
  end
	Flux.back!(fVal)
  if filename != nothing && wronggrad(model) 
  	serializestate(filename, model, loss, data)
  	error("nan or inf in the functionloc(f::Function, types)ction gradient")
  end
  Flux.data(fVal)
end

∇loss!(loss, models::NTuple{N, A}, datas::NTuple{N, B}, filename)  where {N, A, B} = ∇loss!(loss, models, datas,  map(params, models),  filename)
function ∇loss!(loss, models::NTuple{N, A}, datas::NTuple{N, B}, mparams, filename) where {N, A, B}
	@assert length(models) == length(datas)
	n = length(models)
	fVals = fill(0.0, n)
	for i in 1:n
		i != 1 && copyvalues!(mparams[i], mparams[1])
		zerograds!(mparams[i])
		fVals[i] = ∇loss!(loss, models[i], datas[i], filename)
	end
	for i in 2:n
		addgrads!(mparams[1], mparams[i])
	end
	scalegrads!(mparams[1], 1/length(models))
	mean(fVals)
end