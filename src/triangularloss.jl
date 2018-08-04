function triangularloss(d::Vector{T},y::Vector{I}) where {T<:Number,I<:Integer}
	l = length(y)
	e = zero(T)
	for i in 1:l 
		for j in setdiff(find( y.== y[i]),i)
			for k in find( y.!= y[i])
				e += max(0, 1 + abs(d[i] - d[j]) - abs(d[i] - d[k]))
			end 
		end 
	end
	e
end

function triangularloss_back(d, y, Δ)
	l = length(y)
	e = 0.0
	g = zero(d)
	for i in 1:l 
		for j in setdiff(find( y.== y[i]),i)
			for k in find( y.!= y[i])
				if 1 + abs(d[i] - d[j]) - abs(d[i] - d[k]) > 0
					g[i] += sign(d[i] - d[j]) - sign(d[i] - d[k])
					g[j] += -sign(d[i] - d[j])
					g[k] += sign(d[i] - d[k])
				end
			end 
		end 
	end
	g.*Δ
end

triangularloss(d::Flux.Tracker.TrackedArray, y) = Flux.Tracker.track(triangularloss, d, y)
Flux.Tracker.@grad function triangularloss(d, y)
  return(triangularloss(Flux.data(d), y), Δ -> (triangularloss_back(d, y, Δ),nothing))
end
